# S2A DEV evaluation library (protocol sections 16/21/25/26).
#
# Every DEV window is evaluated under ALL reference ablations:
#   CORRECT_REFERENCE (aligned pristine segment), ZERO_REFERENCE, and
#   WRONG_REFERENCE (pristine segment of a DIFFERENT held-out TEST song) -
# plus the frozen S1R baseline forward on the identical augmented mixture, so
# the causal reference effect is measured against the teacher, not just
# internally. Deterministic per-window augmentation (fixed seed) keeps evals
# comparable across checkpoints.
import os
import sys
import json
import zlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402
import s2a_model as smod  # noqa: E402
import s2a_augment as sa  # noqa: E402
import s1r_common as rc  # noqa: E402
import clapsep_train_lib as ctl  # noqa: E402

ALPHA_EVAL = 0.30


def load_dev_windows():
    bank = json.load(open(os.path.join(sc.S2A_PRIVATE, "TRAIN_WINDOW_BANK.private.json"),
                          encoding="utf-8"))
    wins = [w for w in bank["windows"] if w["s2a_split"] == "DEV"]
    split = sc.load_split()
    rec_of = {r["recording_id"]: r for r in split["recordings"]}
    pairs = json.load(open(os.path.join(sc.S2A_PRIVATE, "S2A_SPLIT.private.json"),
                           encoding="utf-8"))["pairs"]
    pair_of = {p["recording_id"]: p for p in pairs}
    # wrong-reference pool: refs of held-out TEST songs (section 16)
    wrong_pool = [p for p in pairs if p["s2a_split"] == "TEST"]
    cache = {}

    def audio(rid):
        if rid not in cache:
            cache[rid] = rc.rec_audio(rec_of[rid])
        return cache[rid]

    def ref32(p):
        key = ("ref", p["ref_id"])
        if key not in cache:
            cache[key] = sc.ref_audio32k(p)
        return cache[key]

    items = []
    for i, w in enumerate(wins):
        p = pair_of[w["recording_id"]]
        wrong = wrong_pool[i % len(wrong_pool)]
        if wrong["recording_id"] == w["recording_id"]:
            wrong = wrong_pool[(i + 1) % len(wrong_pool)]
        items.append({
            **w, "pair": p, "wrong_pair": wrong,
            "x": audio(w["recording_id"]),
            "ref": ref32(p), "ref_wrong": ref32(wrong),
        })
    return items


def _forward(model, chunk, ref, e_pos, e_neg, device, grad=False):
    arr = np.asarray(chunk, dtype=np.float32)
    mx = float(np.max(np.abs(arr)))
    scale_back = 1.0
    xt = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
    rt = torch.tensor(np.asarray(ref, dtype=np.float32), device=device).unsqueeze(0)
    if mx > 1:
        xt = xt * (0.9 / mx)
        scale_back = mx / 0.9
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _mask, pred = smod.train_forward(model, xt, rt, e_pos, e_neg)
    if grad:
        return (pred.float().squeeze(0) * scale_back)
    return (pred.float().detach().squeeze(0) * scale_back).cpu().numpy()


def _s1r_forward(base_model, chunk, e_pos, e_neg, device):
    arr = np.asarray(chunk, dtype=np.float32)
    mx = float(np.max(np.abs(arr)))
    scale_back = 1.0
    xt = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
    if mx > 1:
        xt = xt * (0.9 / mx)
        scale_back = mx / 0.9
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _mask, pred, _ = ctl.train_forward(base_model, xt, e_pos, e_neg)
    return (pred.float().detach().squeeze(0) * scale_back).cpu().numpy()


def evaluate_model(model, emb, device, passthrough_div_l1, full=True):
    """Full DEV pass with the reference ablation triad. Returns (summary, per)."""
    e_pos, e_neg = emb
    items = load_dev_windows()
    smod.set_training_mode(model, training=False)
    per = []
    for it in items:
        w0, x, ref, ref_wrong = it["w0_s"], it["x"], it["ref"], it["ref_wrong"]
        v = rc.build_views(x, w0)
        cs = v["crop_starts"]

        def seg(entry, r32, margin):
            m, _cov = sc.ref_segment(entry, r32, w0 + 4.0 - margin, w0 + 14.0 - margin)
            return m[:rc.CHUNK]

        m_can = seg(it["pair"], ref, 2.0)
        m_early = seg(it["pair"], ref, 4.0)
        m_late = seg(it["pair"], ref, 0.0)
        m_w_can = seg(it["wrong_pair"], ref_wrong, 2.0)
        zero = np.zeros(rc.CHUNK, dtype=np.float32)

        t = np.load(os.path.join(sc.S2A, "work", "audio", "s1r_targets",
                                 it["recording_id"], it["files"]["canonical"]))

        pred_can = _forward(model, v["views"]["canonical"], m_can, e_pos, e_neg, device)
        pc = pred_can[cs["canonical"]:cs["canonical"] + rc.CENTRAL]
        pred_e = _forward(model, v["views"]["early"], m_early, e_pos, e_neg, device)
        pred_l = _forward(model, v["views"]["late"], m_late, e_pos, e_neg, device)
        pe = pred_e[cs["early"]:cs["early"] + rc.CENTRAL]
        pl = pred_l[cs["late"]:cs["late"] + rc.CENTRAL]

        seed = zlib.crc32(f'{it["recording_id"]}|{w0:.2f}'.encode("utf-8"))
        rng = np.random.default_rng(seed)
        x_aug_can, alpha = sa.make_augmented(v["views"]["canonical"], m_can, rng)
        x_aug_e, _ = sa.make_augmented(v["views"]["early"], m_early, rng)
        x_aug_l, _ = sa.make_augmented(v["views"]["late"], m_late, rng)

        res = {}
        res["correct"] = _forward(model, x_aug_can, m_can, e_pos, e_neg, device)
        res["zero"] = _forward(model, x_aug_can, zero, e_pos, e_neg, device)
        res["wrong"] = _forward(model, x_aug_can, m_w_can, e_pos, e_neg, device)
        s1r_aug = _s1r_forward(model.base_model, x_aug_can, e_pos, e_neg, device)
        pred_wrong_clean = _forward(model, v["views"]["canonical"], m_w_can,
                                    e_pos, e_neg, device)

        def rms_db(y):
            return rc.rms_db(y)

        rec = {
            "recording_id": it["recording_id"], "w0_s": w0, "alpha": round(float(alpha), 4),
            "labels": it.get("labels", []),
            "ratio_db": float(20 * np.log10(np.sqrt(np.mean(pc ** 2)) /
                                            (np.sqrt(np.mean(t ** 2)) + 1e-12) + 1e-9)),
            "anchor_div_l1": float(np.abs(pc - t).mean()),
            "click_retention_db": float(rc.click_band_db(pc) - rc.click_band_db(t)),
            "flat_delta": float(rc.flat_frac(pc) - rc.flat_frac(t)),
            "corr_student_raw": rc.wf_corr(pc, v["central"]),
            "corr_teacher_raw": rc.wf_corr(t, v["central"]),
            "consistency_l1": float((np.abs(pc - pe).mean() + np.abs(pc - pl).mean()) / 2),
            "consistency_mrstft": float((rc.mrstft_logmag_dist(pc, pe)
                                         + rc.mrstft_logmag_dist(pc, pl)) / 2),
            "resid_rms_db": {k: rms_db(res[k][cs["canonical"]:cs["canonical"] + rc.CENTRAL]
                                       - t) for k in res},
            "resid_s1r_rms_db": rms_db(s1r_aug[cs["canonical"]:cs["canonical"] + rc.CENTRAL] - t),
            "wrong_clean_ratio_db": float(20 * np.log10(
                np.sqrt(np.mean(pred_wrong_clean[cs["canonical"]:cs["canonical"] + rc.CENTRAL] ** 2))
                / (np.sqrt(np.mean(t ** 2)) + 1e-12) + 1e-9)),
        }
        c0 = rec["resid_rms_db"]["correct"]
        rec["suppression_adv_vs_s1r_db"] = rec["resid_s1r_rms_db"] - c0
        rec["causal_gap_zero_db"] = rec["resid_rms_db"]["zero"] - c0
        rec["causal_gap_wrong_db"] = rec["resid_rms_db"]["wrong"] - c0
        per.append(rec)

    def agg(key):
        vals = [r[key] for r in per]
        return {"median": round(float(np.median(vals)), 4),
                "mean": round(float(np.mean(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4)} if vals else None

    summary = {
        "n_dev_windows": len(per),
        "student_s1r_ratio_db": agg("ratio_db"),
        "collapse_rate_6db": round(float(np.mean([r["ratio_db"] < -6.0 for r in per])), 4),
        "collapse_rate_12db": round(float(np.mean([r["ratio_db"] < -12.0 for r in per])), 4),
        "anchor_divergence_l1": agg("anchor_div_l1"),
        "passthrough_margin_corr": {
            "median": round(float(np.median([r["corr_student_raw"] - r["corr_teacher_raw"]
                                             for r in per])), 4),
            "max": round(float(np.max([r["corr_student_raw"] - r["corr_teacher_raw"]
                                       for r in per])), 4)},
        "click_retention_db": agg("click_retention_db"),
        "friction_flat_delta": agg("flat_delta"),
        "consistency_l1": agg("consistency_l1"),
        "consistency_mrstft": agg("consistency_mrstft"),
        "suppression_adv_vs_s1r_db": agg("suppression_adv_vs_s1r_db"),
        "causal_gap_zero_db": agg("causal_gap_zero_db"),
        "causal_gap_wrong_db": agg("causal_gap_wrong_db"),
        "wrong_clean_ratio_db": agg("wrong_clean_ratio_db"),
    }
    summary["consistency_combined"] = round(
        summary["consistency_l1"]["median"] + summary["consistency_mrstft"]["median"], 4)
    summary["passthrough_div_l1"] = round(float(passthrough_div_l1), 4)
    torch.cuda.empty_cache()
    return summary, per


def check_validity(summary):
    """Protocol section 25 checkpoint validity. All must hold."""
    checks = {}
    checks["no_collapse"] = bool(
        summary["collapse_rate_12db"] == 0.0
        and summary["student_s1r_ratio_db"]["median"] >= -3.0
        and summary["collapse_rate_6db"] <= 0.10)
    checks["no_raw_passthrough"] = bool(
        summary["passthrough_margin_corr"]["median"] <= 0.05
        and summary["anchor_divergence_l1"]["median"] <= 0.5 * summary["passthrough_div_l1"])
    checks["s1r_anchor_preserved"] = bool(
        summary["anchor_divergence_l1"]["median"] <= max(0.5 * summary["passthrough_div_l1"], 1.0)
        and summary["click_retention_db"]["median"] >= -3.0)
    checks["correct_ref_causally_used"] = bool(
        summary["causal_gap_zero_db"]["median"] >= 1.0)
    checks["wrong_ref_safe"] = bool(
        summary["wrong_clean_ratio_db"]["median"] >= -3.0
        and summary["wrong_clean_ratio_db"]["min"] >= -6.0)
    checks["beats_s1r_on_injection"] = bool(
        summary["suppression_adv_vs_s1r_db"]["median"] > 0.0)
    checks["all"] = all(checks.values())
    return checks
