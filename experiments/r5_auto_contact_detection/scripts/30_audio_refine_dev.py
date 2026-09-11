# R5 Phase 4 - audio refinement of automatic visual candidates (dev).
# Reuses the R4-frozen algorithm verbatim (band envelope comb, +/-40 ms peak
# search, R4 confidence thresholds) via r5_common. AV offset +17.625 ms is the
# R1/R4-measured AAC residual for this clip.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from r5_common import (R1_WORK, R4_OUT, OUT, WORK, LOG, SR, load_json, save_json,
                       comb_envelope, refine_candidates)

AV_OFF_MS = 0.0  # empirical: R4 refined transients sit at t_video+0.2 ms median
CTX_START = 3.0


def main():
    comb, gain = comb_envelope(os.path.join(R1_WORK, "golden_ctx.wav"),
                               os.path.join(R1_WORK, "ref_warp_fixed.wav"),
                               CTX_START, 15.0)
    vis = load_json(os.path.join(OUT, "dev_contacts_auto_visual.json"))
    cands = vis["candidates"]
    ids = [f"a{i:03d}" for i in range(len(cands))]
    refined = refine_candidates([c["t_video"] for c in cands], comb,
                                AV_OFF_MS, 15.0, ids=ids)
    for c, r in zip(cands, refined):
        r["score"] = c["score"]
    save_json(os.path.join(OUT, "dev_contacts_auto_refined.json"),
              {"av_off_ms": AV_OFF_MS, "n_candidates": len(cands),
               "events": refined})
    conf = {}
    for r in refined:
        conf[r["refined_confidence"]] = conf.get(r["refined_confidence"], 0) + 1
    print("auto candidates refined:", len(refined), "confidence:", conf)

    # ---- compare vs R4 oracle refined ----
    # Matching: greedy one-to-one by ascending |dt| (tolerance 80 ms).
    # Also gate-level: fraction of oracle contacts covered by the merged
    # auto C1 windows (predicts mix similarity better than event error).
    oracle_ref = load_json(os.path.join(R4_OUT, "contacts_refined.json"))["events"]
    orc = [e for e in oracle_ref
           if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    to = np.array([e["t_audio_refined"] for e in orc])
    auto_used = [r for r in refined
                 if r.get("refined_confidence") not in (None, "no_transient")]
    ta = np.array([r["t_audio_refined"] for r in auto_used])
    pairs = sorted(((abs(a - b), i, j) for i, a in enumerate(ta)
                    for j, b in enumerate(to)), key=lambda x: x[0])
    used_a, used_o, errs = set(), set(), []
    for d, i, j in pairs:
        if d > 0.080:
            break
        if i in used_a or j in used_o:
            continue
        used_a.add(i)
        used_o.add(j)
        errs.append(ta[i] - to[j])
    errs = np.array(errs)
    print(f"oracle refined contacts: {len(to)} | matched by auto (80ms, 1-1): "
          f"{len(used_o)} ({len(used_o)/len(to):.0%})")
    errs_ms = errs * 1000.0
    print(f"refined timing error vs oracle (ms): mean {errs_ms.mean():+.1f} "
          f"median {np.median(errs_ms):+.1f} | |err| median {np.median(np.abs(errs_ms)):.1f} "
          f"p95 {np.percentile(np.abs(errs_ms), 95):.1f} | outliers >30ms: "
          f"{int(np.sum(np.abs(errs_ms) > 30))}/{len(errs_ms)}")
    missed = [orc[k]["id"] for k in range(len(to)) if k not in used_o]
    print("oracle contacts with no auto match:", missed)
    extra = [round(float(ta[i]), 3) for i in range(len(ta)) if i not in used_a]
    print(f"auto events with no oracle match (extra windows): {len(extra)}")

    # gate-level coverage of oracle contacts by auto windows
    spans = []
    for r in auto_used:
        t = r["t_audio_refined"]
        spans.append((max(t - 0.015, 0.0), min(t + 0.150, 15.0)))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    def covered(t):
        return any(s <= t <= e for s, e in merged)
    cov_strong = [orc[k] for k in range(len(to)) if orc[k]["refined_confidence"] == "strong"]
    gate_cov = np.mean([covered(t) for t in to])
    gate_cov_strong = np.mean([covered(e["t_audio_refined"]) for e in cov_strong]) if cov_strong else None
    print(f"gate-level: oracle contacts inside auto windows {gate_cov:.1%} | "
          f"strong-only {gate_cov_strong:.1%}")
    save_json(os.path.join(LOG, "dev_refine_comparison.json"),
              {"matched": len(used_o), "n_oracle": int(len(to)),
               "err_ms_median": float(np.median(errs_ms)),
               "abs_err_ms_median": float(np.median(np.abs(errs_ms))),
               "abs_err_ms_p95": float(np.percentile(np.abs(errs_ms), 95)),
               "outliers_gt30ms": int(np.sum(np.abs(errs_ms) > 30)),
               "missed_oracle_ids": missed,
               "n_extra_auto": len(extra),
               "gate_coverage_of_oracle": float(gate_cov),
               "gate_coverage_strong": float(gate_cov_strong)})

    # plot: comb envelope with auto vs oracle refined times
    fig, ax = plt.subplots(2, 1, figsize=(18, 7))
    t = np.arange(len(comb)) / SR
    ax[0].plot(t, comb, lw=0.4)
    for r in refined:
        if r.get("refined_confidence") in (None, "no_transient"):
            continue
        ax[0].axvline(r["t_audio_refined"], color="g", lw=0.6, alpha=0.6)
    for x in to:
        ax[0].axvline(x, color="b", lw=0.4, alpha=0.35)
    ax[0].set_title("comb envelope | green=auto refined, blue=oracle refined")
    ax[1].hist(np.abs(errs) * 1000, bins=30)
    ax[1].set_title("|auto - oracle| refined timing error (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "dev_refine_comparison.png"), dpi=80)
    print("saved outputs/dev_refine_comparison.png")


if __name__ == "__main__":
    main()
