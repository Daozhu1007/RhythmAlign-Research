# R5 Phase 4(holdout) + Phase 10 - audio refinement + frozen-C1 reconstruction
# for the holdout: AUTO (frozen detector) vs ORACLE (blind oracle) upper bound.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf

from r5_common import (OUT, WORK, LOG, SR, load_json, save_json, comb_envelope,
                       refine_candidates, build_c1)

AV_OFF_MS = 0.0
CTX_START = 3.0
DUR = 18.0


def main():
    comb, _ = comb_envelope(os.path.join(WORK, "holdout_ctx.wav"),
                            os.path.join(WORK, "holdout_ref_warp.wav"), CTX_START, DUR)

    # ---- refine auto candidates ----
    vis = load_json(os.path.join(OUT, "holdout_contacts_auto_visual.json"))["candidates"]
    ids = [f"a{i:03d}" for i in range(len(vis))]
    auto_ref = refine_candidates([c["t_video"] for c in vis], comb, AV_OFF_MS, DUR, ids=ids)
    for c, r in zip(vis, auto_ref):
        r["score"] = c["score"]
    save_json(os.path.join(OUT, "holdout_contacts_auto_refined.json"),
              {"av_off_ms": AV_OFF_MS, "n_candidates": len(vis), "events": auto_ref})
    conf = {}
    for r in auto_ref:
        conf[r.get("refined_confidence")] = conf.get(r.get("refined_confidence"), 0) + 1
    print("auto refined:", len(auto_ref), conf)

    # ---- refine oracle contacts (for the oracle upper bound) ----
    oracle = load_json(os.path.join(OUT, "holdout_oracle_blind.json"))
    contacts = [e for e in oracle["events"] if e["status"] == "ok"]
    oids = [e["id"] for e in contacts]
    orc_ref = refine_candidates([e["t_video"] for e in contacts], comb, AV_OFF_MS, DUR, ids=oids)
    for e, r in zip(contacts, orc_ref):
        r["type"] = e["type"]
        r["oracle_confidence"] = e["confidence"]
    save_json(os.path.join(OUT, "holdout_oracle_refined.json"),
              {"av_off_ms": AV_OFF_MS, "n": len(contacts), "events": orc_ref})
    oconf = {}
    for r in orc_ref:
        oconf[r.get("refined_confidence")] = oconf.get(r.get("refined_confidence"), 0) + 1
    print("oracle refined:", len(orc_ref), oconf)

    # ---- comparison: refined timing + gate coverage ----
    auto_used = [r for r in auto_ref
                 if r.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    ta = np.array([r["t_audio_refined"] for r in auto_used])
    to = np.array([r["t_audio_refined"] for r in orc_ref
                   if r.get("refined_confidence") not in (None, "no_transient", "out_of_range")])
    pairs = sorted(((abs(a - b), i, j) for i, a in enumerate(ta) for j, b in enumerate(to)),
                   key=lambda x: x[0])
    ua, uo, errs = set(), set(), []
    for d, i, j in pairs:
        if d > 0.080:
            break
        if i in ua or j in uo:
            continue
        ua.add(i); uo.add(j)
        errs.append((ta[i] - to[j]) * 1000)
    errs = np.array(errs) if errs else np.array([0.0])
    spans = []
    for r in auto_used:
        t = r["t_audio_refined"]
        spans.append((max(t - 0.015, 0.0), min(t + 0.150, DUR)))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    def covered(t):
        return any(s <= t <= e for s, e in merged)
    gate_cov = float(np.mean([covered(t) for t in to]))
    strong = [r for r in orc_ref if r.get("refined_confidence") == "strong"]
    gate_cov_strong = float(np.mean([covered(r["t_audio_refined"]) for r in strong])) if strong else None
    print(f"matched (80ms,1-1): {len(uo)}/{len(to)} | err med {np.median(errs):+.1f} ms "
          f"| |err| p95 {np.percentile(np.abs(errs), 95):.1f} ms")
    print(f"gate coverage of oracle contacts by AUTO windows: {gate_cov:.1%} "
          f"(strong-only {gate_cov_strong:.1%})")
    save_json(os.path.join(LOG, "holdout_refine_comparison.json"),
              {"matched": len(uo), "n_oracle": int(len(to)),
               "err_ms_median": float(np.median(errs)),
               "abs_err_ms_p95": float(np.percentile(np.abs(errs), 95)),
               "gate_coverage_of_oracle": gate_cov,
               "gate_coverage_strong": gate_cov_strong})

    # ---- C1 builds (frozen recipe) ----
    ref_full, _ = sf.read(os.path.join(WORK, "holdout_ref_warp.wav"),
                          dtype="float64", always_2d=True)
    ref = ref_full[int(CTX_START * SR):int((CTX_START + DUR) * SR)]
    raw_path = os.path.join(OUT, "holdout_raw.wav")
    d_auto = build_c1(raw_path, ref, [r["t_audio_refined"] for r in auto_used],
                      os.path.join(OUT, "holdout_auto_C1_interaction.wav"),
                      os.path.join(OUT, "holdout_auto_C1_final_mix.wav"))
    print("auto C1:", len(auto_used), "events |", d_auto)
    d_orc = build_c1(raw_path, ref, [r["t_audio_refined"] for r in orc_ref
                                     if r.get("refined_confidence") not in (None, "no_transient", "out_of_range")],
                     os.path.join(OUT, "holdout_oracle_C1_interaction.wav"),
                     os.path.join(OUT, "holdout_oracle_C1_final_mix.wav"))
    print("oracle C1:", len([r for r in orc_ref if r.get("refined_confidence") not in (None, "no_transient", "out_of_range")]),
          "events |", d_orc)
    save_json(os.path.join(LOG, "holdout_build_info.json"),
              {"auto": {"n_events": len(auto_used), **d_auto},
               "oracle": {"n_events": int(len(to)), **d_orc}})


if __name__ == "__main__":
    main()
