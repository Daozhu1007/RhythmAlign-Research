# R5.6 Phase 3a - calibration for burst-constrained audio completion (E2).
# Quantifies the "strong audio transient inside a burst, >=90 ms from any E1
# candidate" space on devA/B (trusted oracle): matched vs unmatched peaks,
# their band profiles (bass = raw incl. music; mid/hi1/hi2 = music-removed
# residual), hand-motion support, burst-edge distance. Feeds the ONE E2 rule.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import signal as sig

import common56 as c
from detector55 import fused_score


def strong_comb_peaks(comb, min_peak=0.35, min_ratio=4.0, min_dist_s=0.060):
    """Strong transient peaks mirroring the frozen 'strong' refinement level
    (peak >= 0.35 and peak > 4x local median of +-0.25 s excluding +-60 ms)."""
    pk, _ = sig.find_peaks(comb, distance=int(min_dist_s * c.SR))
    out = []
    for k in pk:
        a, b = max(k - int(0.25 * c.SR), 0), min(k + int(0.25 * c.SR), len(comb))
        excl = np.ones(b - a, bool)
        e0, e1 = k - a - int(0.06 * c.SR), k - a + int(0.06 * c.SR)
        excl[max(e0, 0):max(e1, 0)] = False
        sel = np.arange(a, b)
        med = float(np.median(comb[sel[excl]])) if excl.any() else 0.0
        ratio = float(comb[k] / (med + 1e-6))
        if comb[k] >= min_peak and ratio > min_ratio:
            out.append({"k": int(k), "t": k / c.SR, "peak": float(comb[k]),
                        "ratio": ratio})
    return out


def main():
    for w in c.load_dev_windows():
        score, ns = fused_score(w.series, c.CFG55_FROZEN)
        _, mask, ivs, env, cands = c.detect55(w.series, c.FPS, c.CFG55_FROZEN, rule="A")
        refined = c.refine(w, [cd["t_video"] for cd in cands],
                           [f"{w.key}{j:03d}" for j in range(len(cands))])
        e1_t = [r["t_audio_refined"] for j, r in enumerate(refined)
                if r.get("refined_confidence") in ("strong", "present", "weak")]
        comb = c.get_comb(w)
        bands = c.band_envelopes(w)
        bursts_s = [(a / c.FPS, b / c.FPS) for a, b in ivs]

        to = np.array(w.tc())
        hdn = np.asarray(w.series["hand_diff"])
        hs = np.percentile(hdn, 99.5) + 1e-9

        peaks = strong_comb_peaks(comb)
        rows = []
        for p in peaks:
            t = p["t"]
            in_b = any(a <= t <= b for a, b in bursts_s)
            if not in_b:
                continue
            d_cand = min([abs(t - x) for x in e1_t], default=10.0)
            d_orc = float(np.min(np.abs(to - t))) if len(to) else 10.0
            k = p["k"]
            edge = min([min(t - a, b - t) for a, b in bursts_s if a <= t <= b]) * 1000
            rows.append({
                **p, "in_burst": True, "burst_edge_ms": edge * 1.0,
                "d_e1_ms": d_cand * 1000, "d_oracle_ms": d_orc * 1000,
                "matched": d_orc <= 0.080,
                "hand": float(min(np.max(hdn[max(int(t * c.FPS) - 3, 0):
                                                 int(t * c.FPS) + 4]) / hs, 2.5)),
                "bass": float(bands["bass"][k]), "mid": float(bands["mid"][k]),
                "hi1": float(bands["hi1"][k]), "hi2": float(bands["hi2"][k]),
            })
        comp = [x for x in rows if x["d_e1_ms"] >= 90]   # completion space
        m = [x for x in comp if x["matched"]]
        u = [x for x in comp if not x["matched"]]

        def med(xs, k):
            v = [x[k] for x in xs]
            return round(float(np.median(v)), 3) if v else None

        print(f"\n===== {w.key} =====  burst strong peaks {len(rows)} | "
              f"completion space (d_e1>=90ms) {len(comp)} | matched-to-oracle {len(m)} "
              f"| unmatched {len(u)}")
        for lbl, xs in (("matched", m), ("unmatched", u)):
            if not xs:
                continue
            print(f"  {lbl}: peak {med(xs,'peak')} ratio {med(xs,'ratio')} | "
                  f"bass {med(xs,'bass')} mid {med(xs,'mid')} hi1 {med(xs,'hi1')} hi2 {med(xs,'hi2')} | "
                  f"resid_sum {med(xs, 'resid') if False else round(float(np.median([x['mid']+x['hi1']+x['hi2'] for x in xs])),3)} | "
                  f"hand {med(xs,'hand')} | burst_edge {med(xs,'burst_edge_ms')} | "
                  f"d_e1 {med(xs,'d_e1_ms')}")
        # candidate gates on the unmatched set (suspected music/taiko or
        # oracle-missing): how much does residual-band support cut?
        for resid_thr, hi_thr in ((1.0, 0.15), (1.2, 0.15), (1.0, 0.30), (1.4, 0.30),
                                  (1.6, 0.30), (1.0, 0.45), (1.4, 0.45)):
            keep_m = [x for x in m if x["mid"] + x["hi1"] + x["hi2"] >= resid_thr
                      and x["hi1"] >= hi_thr]
            keep_u = [x for x in u if x["mid"] + x["hi1"] + x["hi2"] >= resid_thr
                      and x["hi1"] >= hi_thr]
            print(f"  gate resid>={resid_thr} & hi1>={hi_thr}: matched kept "
                  f"{len(keep_m)}/{len(m)}, unmatched kept {len(keep_u)}/{len(u)}")
        # hand-motion threshold effect
        for h in (0.10, 0.15, 0.25):
            km = sum(x["hand"] >= h for x in m)
            ku = sum(x["hand"] >= h for x in u)
            print(f"  hand>={h}: matched {km}/{len(m)}, unmatched {ku}/{len(u)}")
        # spacing profile of unmatched after band gate (taiko rhythm check)
        c.save_json(os.path.join(c.R56_LOG, f"phase3_completion_space_{w.key}.json"),
                    {"peaks_in_burst": rows, "completion_space": comp})


if __name__ == "__main__":
    main()
