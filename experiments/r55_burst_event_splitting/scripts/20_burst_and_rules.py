# R5.5 Phase 2+3 - burst detector vs oracle regions + rule A vs B selection.
# All tuning on Dev-A + Dev-B only (both are development data this round).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common55 import (R55_LOG, R55_OUT, load_windows, save_json, FPS,
                      match_one_to_one)
from detector55 import DEFAULT55, detect55, burst_mask, fused_score


def eval_cands(cands, w, rule_tag):
    tp_t = np.array([c["t_video"] for c in cands])
    tc = w.tc()
    res = {"rule": rule_tag, "n_pred": len(cands), "n_oracle": len(tc)}
    mc, mp, pairs = match_one_to_one(tp_t, tc, 0.050)
    p = len(mc) / max(len(cands), 1)
    r = len(mc) / max(len(tc), 1)
    res.update({"P": round(p, 3), "R": round(r, 3),
                "F1": round(2 * p * r / max(p + r, 1e-9), 3)})
    errs = [abs(e) * 1000 for _, _, e in pairs]
    res["err_ms_med"] = round(float(np.median(errs)), 1) if errs else None
    # dense-only F1 (predictions inside continuous regions vs dense contacts)
    d_c = [c for c in cands if w.dense_mask[min(int(c["i"]), w.n - 1)]]
    d_tc = np.array([e["t_video"] for e in w.contacts if e["dense"]])
    md_c, _, _ = match_one_to_one(np.array([c["t_video"] for c in d_c]), d_tc, 0.050)
    pd = len(md_c) / max(len(d_c), 1)
    rd = len(md_c) / max(len(d_tc), 1)
    res["dense_F1"] = round(2 * pd * rd / max(pd + rd, 1e-9), 3)
    res["dense_R"] = round(rd, 3)
    # strong recall
    strong = [e for e in w.contacts if e.get("confidence") == "high"]
    ts = np.array([e["t_video"] for e in strong])
    hit = sum(1 for t in ts if len(tp_t) and np.min(np.abs(tp_t - t)) <= 0.050)
    res["strong_recall"] = round(hit / max(len(ts), 1), 3)
    res["strong_n"] = len(ts)
    return res


def burst_eval(w, mask):
    """burst intervals vs oracle continuous regions (frame-time comparison)."""
    dense = w.dense_mask
    inter = (mask & dense).sum()
    cov = inter / max(dense.sum(), 1)
    false_burst_s = (mask & ~dense).sum() / FPS
    missed_burst_s = (dense & ~mask).sum() / FPS
    return {"region_coverage": round(float(cov), 3),
            "false_burst_s": round(float(false_burst_s), 2),
            "missed_burst_s": round(float(missed_burst_s), 2)}


def main():
    windows = load_windows()
    report = {"burst_grid": [], "rules": {}}

    # ---- Phase 2: small reasoned burst grid ----
    print("== burst detector grid (region match) ==")
    best = None
    for enter in (0.40, 0.50, 0.60):
        for stay in (0.22, 0.28, 0.35):
            for hold in (12, 20, 30):
                cfg = {**DEFAULT55, "burst_enter": enter, "burst_stay": stay,
                       "burst_hold_frames": hold}
                tot = {"cov": 0.0, "fb": 0.0, "mb": 0.0}
                for w in windows:
                    score, _ = fused_score(w.series, cfg)
                    mask, _, _ = burst_mask(score, cfg)
                    r = burst_eval(w, mask)
                    tot["cov"] += r["region_coverage"]
                    tot["fb"] += r["false_burst_s"]
                    tot["mb"] += r["missed_burst_s"]
                n = len(windows)
                row = {"enter": enter, "stay": stay, "hold": hold,
                       "cov": round(tot["cov"] / n, 3),
                       "false_burst_s": round(tot["fb"] / n, 2),
                       "missed_burst_s": round(tot["mb"] / n, 2)}
                report["burst_grid"].append(row)
                print(f"  enter {enter} stay {stay} hold {hold}: cov {row['cov']:.3f} "
                      f"false {row['false_burst_s']:.2f}s missed {row['missed_burst_s']:.2f}s")
                score_ = row["cov"] - 0.02 * row["false_burst_s"] - 0.02 * row["missed_burst_s"]
                if best is None or score_ > best[0]:
                    best = (score_, row)
    print("  chosen burst params:", best[1])
    report["burst_chosen"] = best[1]

    # ---- Phase 3: rule A vs rule B with chosen burst params ----
    print("\n== rule selection ==")
    for rule in ("A", "B"):
        per = {}
        for w in windows:
            cfg = {**DEFAULT55, "burst_enter": best[1]["enter"],
                   "burst_stay": best[1]["stay"],
                   "burst_hold_frames": best[1]["hold"]}
            score, mask, ivs, env, cands = detect55(w.series, FPS, cfg, rule=rule)
            per[w.key] = eval_cands(cands, w, rule)
            per[w.key]["n_bursts"] = len(ivs)
            print(f"  [{rule}] {w.key}: pred {per[w.key]['n_pred']} "
                  f"P {per[w.key]['P']:.3f} R {per[w.key]['R']:.3f} F1 {per[w.key]['F1']:.3f} "
                  f"| dense F1 {per[w.key]['dense_F1']:.3f} R {per[w.key]['dense_R']:.3f} "
                  f"| strong {per[w.key]['strong_recall']:.3f} "
                  f"| err {per[w.key]['err_ms_med']} ms | bursts {per[w.key]['n_bursts']}")
        report["rules"][rule] = per

    save_json(os.path.join(R55_LOG, "phase23_burst_and_rules.json"), report)


if __name__ == "__main__":
    main()
