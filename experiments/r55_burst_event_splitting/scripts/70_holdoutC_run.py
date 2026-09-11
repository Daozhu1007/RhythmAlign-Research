# R5.5 Phase 8 - run the FROZEN R5.5 detector on Holdout C and evaluate vs the
# blind oracle. Uses detector_config_r55_frozen.json; no parameter changes.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common55 import (R55_OUT, R55_LOG, R55_WORK, load_json, save_json,
                      extract_features, FPS, match_one_to_one)
from detector55 import detect55
from r5_common import detect_candidates


def zone_of(w, t, i=None):
    k = int(round(t * FPS)) if i is None else i
    return "dense" if (0 <= k < w.n and w.dense_mask[k]) else "sparse"


def eval_full(cands, w):
    tp_t = np.array([c["t_video"] for c in cands])
    tc = w.tc()
    out = {"n_pred": len(cands), "n_oracle": len(tc)}
    for tol in (0.033, 0.050):
        mc, mp, pairs = match_one_to_one(tp_t, tc, tol)
        p = len(mc) / max(len(cands), 1)
        r = len(mc) / max(len(tc), 1)
        errs = [abs(e) * 1000 for _, _, e in pairs]
        out[f"tol{int(tol*1000)}"] = {
            "P": round(p, 3), "R": round(r, 3),
            "F1": round(2 * p * r / max(p + r, 1e-9), 3),
            "err_ms_median": round(float(np.median(errs)), 1) if errs else None,
            "err_ms_p95": round(float(np.percentile(errs, 95)), 1) if errs else None}
    for zone in ("sparse", "dense"):
        cs = [c for c in cands if zone_of(w, c["t_video"], c["i"]) == zone]
        es = [e for e in w.contacts if e["dense"] == (zone == "dense")]
        tcs = np.array([e["t_video"] for e in es])
        tps = np.array([c["t_video"] for c in cs])
        m, _, _ = match_one_to_one(tps, tcs, 0.050) if len(tcs) else (set(), set(), [])
        p = len(m) / max(len(cs), 1)
        r = len(m) / max(len(es), 1)
        out[zone] = {"n_pred": len(cs), "n_oracle": len(es),
                     "P": round(p, 3), "R": round(r, 3),
                     "F1": round(2 * p * r / max(p + r, 1e-9), 3)}
    strong = [e for e in w.contacts if e.get("confidence") == "high"]
    ts = np.array([e["t_video"] for e in strong])
    hit = [t for t in ts if len(tp_t) and np.min(np.abs(tp_t - t)) <= 0.050]
    out["strong"] = {"n": len(ts), "recall_50ms": round(len(hit) / max(len(ts), 1), 3),
                     "missed": [e["id"] for e in strong
                                if not (len(tp_t) and np.min(np.abs(tp_t - e["t_video"])) <= 0.050)]}
    mc, mp, pairs = match_one_to_one(tp_t, tc, 0.050)
    matched_o = {k for _, k, _ in pairs}
    out["fn"] = [{"id": w.contacts[k]["id"], "t": w.contacts[k]["t_video"],
                  "type": w.contacts[k]["type"], "conf": w.contacts[k].get("confidence"),
                  "zone": zone_of(w, w.contacts[k]["t_video"])}
                 for k in range(len(tc)) if k not in matched_o]
    tnc = np.array([e["t_video"] for e in w.noncontacts]) if w.noncontacts else np.array([])
    out["fp"] = [{"t": round(float(tp_t[j]), 3),
                  "near_noncontact": bool(len(tnc) and np.min(np.abs(tnc - tp_t[j])) <= 0.100)}
                 for j in range(len(tp_t)) if j not in mp]
    # region coverage (R5 definition)
    miss_tot, reg_tot = 0.0, 0.0
    for rg in w.continuous:
        a, b = rg["t0"], rg["t1"]
        holes, t = 0.0, a
        while t < b:
            if len(tp_t) and np.min(np.abs(tp_t - t)) <= 0.3:
                nxt = tp_t[np.argmin(np.abs(tp_t - t))] + 0.3
                t = max(t + 1 / FPS, min(nxt, b))
            else:
                t2 = t
                while t2 < b and not (len(tp_t) and np.min(np.abs(tp_t - t2)) <= 0.3):
                    t2 += 1 / FPS
                holes += min(t2, b) - t
                t = t2
        miss_tot += holes
        reg_tot += b - a
    out["region_coverage"] = round(1 - miss_tot / reg_tot, 4)
    return out


def main():
    frozen = load_json(os.path.join(R55_OUT, "detector_config_r55_frozen.json"))
    cfg = frozen["config"]
    n, series = extract_features(os.path.join(R55_OUT, "holdoutC_video.mp4"))
    np.savez(os.path.join(R55_WORK, "holdoutC_features.npz"), fps=FPS, **series)
    print("holdoutC frames:", n)

    score, mask, ivs, env, cands = detect55(series, FPS, cfg, rule="A")
    save_json(os.path.join(R55_OUT, "holdoutC_contacts_auto_visual.json"), {
        "experiment": "R5.5 Holdout-C automatic visual candidates (FROZEN detector)",
        "source_video": "outputs/holdoutC_video.mp4",
        "detector_config": "outputs/detector_config_r55_frozen.json",
        "n_bursts": len(ivs),
        "burst_intervals": [[round(a / FPS, 3), round(b / FPS, 3)] for a, b in ivs],
        "n_candidates": len(cands), "candidates": cands})
    print("candidates:", len(cands), "| bursts:", len(ivs))

    # R5 frozen baseline on the same window (identical eval code)
    _, cands0 = detect_candidates(series, FPS, {
        "smooth_frames": 3, "rise_frames": 2,
        "w_jz_text": 1.8, "w_jz_diff": 1.5, "w_glass_bright": 1.0,
        "w_hand_diff": 0.3, "threshold": 0.40, "slope_min": 0.15,
        "min_dist_frames": 3, "norm_pct": 99.5, "novelty_clip": 2.5})
    save_json(os.path.join(R55_OUT, "holdoutC_contacts_r5baseline_visual.json"),
              {"n_candidates": len(cands0), "candidates": cands0})

    from common55 import Window
    w = Window("holdoutC", os.path.join(R55_WORK, "holdoutC_features.npz"),
               os.path.join(R55_OUT, "holdoutC_oracle_blind.json"), 40.0, 58.0, 18.0)
    m = eval_full(cands, w)
    m0 = eval_full(cands0, w)
    save_json(os.path.join(R55_LOG, "holdoutC_metrics.json"),
              {"r55": m, "r5_baseline": m0})
    for name, mm in (("R5.5", m), ("R5  ", m0)):
        t5, t3 = mm["tol50"], mm["tol33"]
        print(f"\n{name}: pred {mm['n_pred']:3d} | @50 P {t5['P']:.3f} R {t5['R']:.3f} "
              f"F1 {t5['F1']:.3f} | @33 F1 {t3['F1']:.3f} | err med {t5['err_ms_median']} "
              f"p95 {t5['err_ms_p95']} ms")
        print(f"      sparse: {mm['sparse']['n_oracle']:2d} oracle / "
              f"P {mm['sparse']['P']:.3f} R {mm['sparse']['R']:.3f} | "
              f"dense: {mm['dense']['n_oracle']:2d} / P {mm['dense']['P']:.3f} "
              f"R {mm['dense']['R']:.3f} F1 {mm['dense']['F1']:.3f}")
        print(f"      strong {mm['strong']['recall_50ms']:.3f} ({mm['strong']['n']}) "
              f"missed {mm['strong']['missed']} | regcov {mm['region_coverage']:.3f} "
              f"| FP {len(mm['fp'])} FN {len(mm['fn'])}")


if __name__ == "__main__":
    main()
