# S1R step 04 - REAL-DEV evaluation (protocol sections 31-33).
#
# Real mixtures ONLY: the frozen DEV recording identities, their Phase 0 candidate
# windows and (for anchors) precomputed frozen-teacher central crops. Axes kept
# separate; no single magical score. Used for: zero-shot reference, in-training
# monitoring, checkpoint selection support, and the machine-side anti-collapse gate.
import os
import sys
import json
import numpy as np
import torch

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402
import s1r_model as sm  # noqa: E402
import s1r_losses as sl  # noqa: E402

TEACHER_DIR = os.path.join(S1R, "work", "audio", "teacher")


def load_anchor_tables():
    with open(os.path.join(S1R, "work", "private", "phase0_windows.private.json"),
              encoding="utf-8") as f:
        results = json.load(f)
    with open(os.path.join(S1R, "work", "private", "teacher_precompute.private.json"),
              encoding="utf-8") as f:
        pre = json.load(f)
    pre_map = {(a["recording_id"], a["w0_s"]): a["files"] for a in pre["anchors"]}
    anchors, others = [], []
    for r in results:
        if r["split"] != "DEV" or "error" in r:
            continue
        if r.get("accepted"):
            files = pre_map.get((r["recording_id"], r["w0_s"]))
            if files:
                anchors.append({**r, "files": files})
        else:
            others.append(r)
    return anchors, others


def load_teacher_crop(rec_id, w0, files, name):
    return np.load(os.path.join(TEACHER_DIR, rec_id, files[name])).astype(np.float32)


def student_forward(model, chunk, emb, zeros, device, amp=True):
    """Official peak-0.9 protocol with grad-capable path; input-domain output.

    MUST mirror clapsep_lib.separate exactly (that is how the teacher crops were
    produced) so the student starts exactly on the teacher anchor surface.
    """
    max_arr = np.asarray(chunk, dtype=np.float32)
    max_val = float(np.max(np.abs(max_arr)))
    scale_back = 1.0
    x = torch.tensor(max_arr, dtype=torch.float32, device=device).unsqueeze(0)
    if max_val > 1:
        x = x * (0.9 / max_val)
        scale_back = max_val / 0.9
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        _mask, pred = sm.train_forward(model, x, emb, zeros)
    return (pred.float().detach().squeeze(0) * scale_back).cpu().numpy()


@torch.no_grad()
def evaluate_model(model, emb, device, limit=None, want_contexts=("early", "late")):
    """Full DEV diagnostic pass. Returns (summary_dict, per_anchor_list)."""
    anchors, others = load_anchor_tables()
    if limit:
        anchors = anchors[:limit]
    zeros = np.zeros((1, 512), dtype=np.float32)
    sm.set_training_mode(model, training=False)

    per = []
    # cache raw spans per recording
    raw_cache = {}
    for a in anchors:
        rec_key = a["recording_id"]
        if rec_key not in raw_cache:
            split_all = rc.load_split()
            fam = next((r.get("family") for r in split_all["recordings"]
                        if r["recording_id"] == rec_key), None)
            raw_cache[rec_key] = rc.rec_audio({"family": fam, "work32k": a.get("work32k")})
        x = raw_cache[rec_key]
        v = rc.build_views(x, a["w0_s"])
        raw_c = v["central"]

        preds = {}
        for name in ("canonical",) + want_contexts:
            preds[name] = student_forward(model, v["views"][name], emb, zeros, device)
        cs_can = v["crop_starts"]["canonical"]
        pc = preds["canonical"][cs_can:cs_can + rc.CENTRAL]
        for g in (-3.0, 3.0):
            chunk = v["views"]["canonical"] * (10.0 ** (g / 20.0))
            pred = student_forward(model, chunk, emb, zeros, device)
            preds[f"gain{int(g)}"] = pred[cs_can:cs_can + rc.CENTRAL]

        t_can = load_teacher_crop(rec_key, a["w0_s"], a["files"], "canonical")
        t_early = load_teacher_crop(rec_key, a["w0_s"], a["files"], "early")
        t_late = load_teacher_crop(rec_key, a["w0_s"], a["files"], "late")

        rms_s = float(np.sqrt(np.mean(pc ** 2)))
        rms_t = float(np.sqrt(np.mean(t_can ** 2)))
        ratio_db = 20 * np.log10(rms_s / (rms_t + 1e-12) + 1e-9)

        rec = {
            "recording_id": rec_key, "w0_s": a["w0_s"],
            "labels": a.get("interaction_labels", []),
            "anchor_div_l1": float(np.abs(pc - t_can).mean()),
            "anchor_div_mrstft": rc.mrstft_logmag_dist(pc, t_can),
            "ratio_db": ratio_db,
            "student_rms_db": rc.rms_db(pc),
            "teacher_rms_db": rc.rms_db(t_can),
            "raw_rms_db": rc.rms_db(raw_c),
            "corr_student_raw": rc.wf_corr(pc, raw_c),
            "corr_teacher_raw": rc.wf_corr(t_can, raw_c),
            "click_retention_db": float(rc.click_band_db(pc) - rc.click_band_db(t_can)),
            "flat_delta": float(rc.flat_frac(pc) - rc.flat_frac(t_can)),
        }
        # cross-context consistency (student) + same-quantity for teacher reference
        cons_s, cons_t = [], []
        for name, t_crop in (("early", t_early), ("late", t_late)):
            cs = v["crop_starts"][name]
            pb = preds[name][cs:cs + rc.CENTRAL]
            cons_s.append((np.abs(pc - pb).mean(), rc.mrstft_logmag_dist(pc, pb)))
            cons_t.append((np.abs(t_can - t_crop).mean(), rc.mrstft_logmag_dist(t_can, t_crop)))
        rec["consistency_l1"] = float(np.mean([c[0] for c in cons_s]))
        rec["consistency_mrstft"] = float(np.mean([c[1] for c in cons_s]))
        rec["teacher_consistency_l1"] = float(np.mean([c[0] for c in cons_t]))
        rec["teacher_consistency_mrstft"] = float(np.mean([c[1] for c in cons_t]))
        # gain equivariance (student), corrected
        ge = []
        for g in (-3.0, 3.0):
            pg = preds[f"gain{int(g)}"]
            ge.append((np.abs(pc - pg).mean(), rc.mrstft_logmag_dist(pc, pg)))
        rec["gain_equiv_l1"] = float(np.mean([q[0] for q in ge]))
        rec["gain_equiv_mrstft"] = float(np.mean([q[1] for q in ge]))
        per.append(rec)

    def agg(key):
        vals = [r[key] for r in per]
        return {"median": round(float(np.median(vals)), 4),
                "mean": round(float(np.mean(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4)} if vals else None

    # collapse accounting over ALL dev candidate windows (anchors + rejected)
    n_collapse12 = 0
    n_dev_windows = len(per)
    summary = {
        "n_dev_anchors": len(per),
        "anchor_divergence_l1": agg("anchor_div_l1"),
        "anchor_divergence_mrstft": agg("anchor_div_mrstft"),
        "student_teacher_ratio_db": agg("ratio_db"),
        "collapse_rate_6db": round(float(np.mean([r["ratio_db"] < -6.0 for r in per])), 4),
        "collapse_rate_12db": round(float(np.mean([r["ratio_db"] < -12.0 for r in per])), 4),
        "consistency_l1": agg("consistency_l1"),
        "consistency_mrstft": agg("consistency_mrstft"),
        "teacher_consistency_l1": agg("teacher_consistency_l1"),
        "teacher_consistency_mrstft": agg("teacher_consistency_mrstft"),
        "gain_equiv_l1": agg("gain_equiv_l1"),
        "gain_equiv_mrstft": agg("gain_equiv_mrstft"),
        "passthrough_margin_corr": agg("corr_student_raw") and {
            "median": round(float(np.median([r["corr_student_raw"] - r["corr_teacher_raw"]
                                             for r in per])), 4),
            "max": round(float(np.max([r["corr_student_raw"] - r["corr_teacher_raw"]
                                       for r in per])), 4),
        },
        "click_retention_db": agg("click_retention_db"),
        "friction_flat_delta": agg("flat_delta"),
        "out_raw_db": agg("student_rms_db") and {
            "median_student_minus_raw_db": round(float(np.median(
                [r["student_rms_db"] - r["raw_rms_db"] for r in per])), 4),
            "median_teacher_minus_raw_db": round(float(np.median(
                [r["teacher_rms_db"] - r["raw_rms_db"] for r in per])), 4),
        },
        "n_collapse12": n_collapse12,
        "n_dev_windows": n_dev_windows,
    }
    torch.cuda.empty_cache()
    return summary, per


def check_validity(summary, zs_summary, passthrough_div_l1):
    """Protocol section 32 checkpoint validity (DEV only). All must hold."""
    checks = {}
    checks["no_collapse"] = bool(
        summary["collapse_rate_12db"] == 0.0
        and summary["student_teacher_ratio_db"]["median"] >= -3.0
        and summary["collapse_rate_6db"] <= 0.10)
    pas_margin = summary["passthrough_margin_corr"]["median"]
    div = summary["anchor_divergence_l1"]["median"]
    checks["no_raw_passthrough"] = bool(pas_margin <= 0.05 and div <= 0.5 * passthrough_div_l1)
    zs_div = zs_summary["anchor_divergence_l1"]["median"]
    checks["anchor_selectivity_preserved"] = bool(
        div <= max(0.5 * passthrough_div_l1, 1.0)
        and summary["click_retention_db"]["median"] >= -3.0)
    zs_cons = (zs_summary["consistency_l1"]["median"]
               + zs_summary["consistency_mrstft"]["median"])
    cur_cons = summary["consistency_l1"]["median"] + summary["consistency_mrstft"]["median"]
    summary["consistency_combined"] = round(cur_cons, 4)
    summary["zeroshot_consistency_combined"] = round(zs_cons, 4)
    checks["consistency_improves"] = bool(cur_cons < zs_cons)
    zs_click = zs_summary["click_retention_db"]["median"]
    checks["fidelity_no_major_regression"] = bool(
        summary["click_retention_db"]["median"] >= zs_click - 1.0)
    checks["all"] = all(checks.values())
    return checks


def main():
    import argparse
    import clapsep_lib  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="S1R checkpoint to evaluate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sm.load_student_from_zero_shot(device, args.ckpt)
    emb = sm.embed_q1(model)
    summary, per = evaluate_model(model, emb, device)
    out = args.out or os.path.join(S1R, "logs", "04_dev_eval.json")
    rc.save_json(out, {"ckpt": args.ckpt, "summary": summary})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
