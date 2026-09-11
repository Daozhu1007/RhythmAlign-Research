# R5.5 Phase 1c - feasibility ceiling of atomic rules in dense regions:
#   ruleA: local peaks of the R5 fused novelty score (NO rising-edge gate -
#          re-triggers on intra-platform bumps)
#   ruleB: 2-of-4 channel agreement of per-channel atomic peaks
#   ruleC: jz_diff peaks + glass/hand support
# Evaluated vs dense oracle contacts (dense-only, +/-50 ms, one-to-one).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import signal as sig

from common55 import (R55_LOG, load_windows, save_json, detect_candidates,
                      detector_score, R5_FROZEN_CFG, FPS, match_one_to_one)

CHANNELS = [("jz_text", "jz_warm", "jz_white"),
            ("jz_diff", "jz_diff", None),
            ("glass_bright", "glass_bright", None),
            ("hand_diff", "hand_diff", None)]
VALLEY = 9


def local_prom(x, pk):
    out = np.zeros(len(pk))
    for i, p in enumerate(pk):
        a, b = max(p - VALLEY, 0), min(p + VALLEY + 1, len(x))
        left = np.min(x[a:p + 1]) if p > a else x[p]
        right = np.min(x[p:b]) if b > p + 1 else x[p]
        out[i] = x[p] - max(left, right)
    return out


def eval_peaks(pk_frames, w, tag, store):
    tc = np.array([e["t_video"] for e in w.contacts if e["dense"]])
    pt = np.asarray(pk_frames) / FPS
    matched, matched_p, pairs = match_one_to_one(pt, tc, 0.050)
    prec = len(matched) / max(len(pt), 1)
    rec = len(matched) / max(len(tc), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    errs = [abs(e) * 1000 for _, _, e in pairs]
    store[tag] = {"peaks": int(len(pt)), "P": round(prec, 3), "R": round(rec, 3),
                  "F1": round(f1, 3),
                  "err_ms_median": round(float(np.median(errs)), 1) if errs else None}
    return f1


def main():
    summary = {}
    for w in windows_sel():
        score, _ = detect_candidates(w.series, FPS, R5_FROZEN_CFG)
        ch = {n: w.series[a] + (w.series[b] if b else 0.0) for n, a, b in CHANNELS}
        scale = {n: float(np.percentile(x, 99.5)) + 1e-9 for n, x in ch.items()}
        dense = w.dense_mask
        store = {}

        # ruleA: fused score local peaks (score already = weighted normalized novelty)
        best = None
        for prom in (0.05, 0.10, 0.15, 0.20, 0.30):
            for dist in (3, 4, 5):
                pk, _ = sig.find_peaks(score, distance=dist)
                proms = local_prom(score, pk) / (np.percentile(score, 99.5) + 1e-9)
                keep = pk[(proms >= prom) & dense[pk]]
                f1 = eval_peaks(np.array(keep), w, f"A_prom{prom}_d{dist}", store)
                if best is None or f1 > best[0]:
                    best = (f1, f"A_prom{prom}_d{dist}")
        # ruleB: 2-of-4 channel agreement
        from collections import Counter
        chan_pks = {}
        for name, x in ch.items():
            for h, pr in ((0.10, 0.03), (0.15, 0.06)):
                xk = x / scale[name]
                pk, _ = sig.find_peaks(xk, distance=4)
                prs = local_prom(xk, pk)
                keep = pk[(xk[pk] >= h) & (prs / scale[name] * 0 + prs >= pr * scale[name]) & dense[pk]]
                chan_pks[(name, h, pr)] = set(keep.tolist())
        for (h, pr) in ((0.10, 0.03), (0.15, 0.06)):
            sets = [chan_pks[(n, h, pr)] for n, _, _ in CHANNELS]
            for k_need in (2, 3):
                cnt = Counter()
                for s in sets:
                    for p in s:
                        for q in range(p - 3, p + 4):
                            cnt[q] += 1
                agree = sorted(p for p, c in cnt.items() if c >= k_need)
                # collapse runs
                coll = []
                for p in agree:
                    if coll and p - coll[-1][-1] <= 2:
                        coll[-1].append(p)
                    else:
                        coll.append([p])
                pk = np.array([g[int(np.argmax(len(g)))] for g in coll])
                f1 = eval_peaks(pk, w, f"B_{k_need}of4_h{h}_pr{pr}", store)
                if f1 > best[0]:
                    best = (f1, f"B_{k_need}of4_h{h}_pr{pr}")
        # ruleC: jz_diff peak, supported by glass or hand within +/-3 frames
        for h_jd in (0.10, 0.15, 0.20):
            xj = ch["jz_diff"] / scale["jz_diff"]
            pk, _ = sig.find_peaks(xj, distance=4)
            prs = local_prom(xj, pk)
            base = pk[(xj[pk] >= h_jd) & (prs >= 0.06 * scale["jz_diff"]) & dense[pk]]
            xg = ch["glass_bright"] / scale["glass_bright"]
            xh = ch["hand_diff"] / scale["hand_diff"]
            out = []
            for p in base:
                g_ok = xg[max(p - 3, 0):p + 4].max() >= 0.10
                h_ok = xh[max(p - 3, 0):p + 4].max() >= 0.10
                if g_ok or h_ok:
                    out.append(p)
            f1 = eval_peaks(np.array(out), w, f"C_jd{h_jd}_supp", store)
            if f1 > best[0]:
                best = (f1, f"C_jd{h_jd}_supp")

        top = sorted(store.items(), key=lambda kv: -kv[1]["F1"])[:8]
        summary[w.key] = {"best": {"rule": best[1], "F1": round(best[0], 3)},
                          "top8": {k: v for k, v in top}}
        print(f"\n=== {w.key} dense-only feasibility ceiling (best F1 {best[0]:.3f} @ {best[1]})")
        for k, v in top:
            print(f"  {k:24s} peaks {v['peaks']:3d}  P {v['P']:.3f} R {v['R']:.3f} "
                  f"F1 {v['F1']:.3f}  err_med {v['err_ms_median']} ms")
    save_json(os.path.join(R55_LOG, "phase1c_rule_ceiling.json"), summary)


def windows_sel():
    return load_windows()


if __name__ == "__main__":
    main()
