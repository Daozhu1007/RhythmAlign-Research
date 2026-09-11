# R5.6 Phase 2 - E1 precision cleanup (the ONE recipe) on Dev-A/B/C.
# E0 = R5.5 frozen detector (byte-identical recompute). E1 applies ONLY
# candidate post-filtering: refinement-confidence gating with extra visual
# support for weak candidates. Echo-ratio suppression audited and rejected.
# Outputs: dev*_E1_contacts.json + dev*_E1_C1_interaction.wav + final_mix.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf

import common56 as c
from detector55 import fused_score
from r5_common import SR
import r5_common as r5


def run_e1(w):
    score, ns = fused_score(w.series, c.CFG55_FROZEN)
    _, mask, ivs, env, cands = c.detect55(w.series, c.FPS, c.CFG55_FROZEN, rule="A")
    refined = c.refine(w, [cd["t_video"] for cd in cands],
                       [f"{w.key}{j:03d}" for j in range(len(cands))])
    for cd, r in zip(cands, refined):
        r["score"] = cd["score"]
        r["path"] = cd["path"]
    flags = c.e1_gate_flags(cands, refined, score, ns, ivs, c.E1_CFG)

    kept = [dict(cd, id=f"{w.key}{j:03d}", gate=flags[j][2],
                 candidate=True, window=flags[j][1])
            for j, cd in enumerate(cands)]

    # windows: candidates whose gate opens a window and whose refinement is usable
    used = [r for j, r in enumerate(refined)
            if flags[j][1] and
            r.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    ta = [r["t_audio_refined"] for r in used]

    out_int = os.path.join(c.R56_OUT, f"{w.key}_E1_C1_interaction.wav")
    out_mix = os.path.join(c.R56_OUT, f"{w.key}_E1_C1_final_mix.wav")
    ref_full, _ = sf.read(w.ref, dtype="float64", always_2d=True)
    ref = ref_full[int(c.CTX_START * SR):int((c.CTX_START + w.dur) * SR)]
    r5.build_c1(w.raw, ref, ta, out_int, out_mix)

    doc = {
        "experiment": "R5.6 E1 precision cleanup",
        "window": w.key, "e0_candidates": len(cands),
        "e1_config": c.E1_CFG,
        "n_candidates": len(kept),
        "n_windows": len(ta),
        "gate_counts": {
            "keep_strong_present": sum(1 for x in kept if x["gate"].startswith("keep_")),
            "weak_kept_visual": sum(1 for x in kept if "_kept_" in x["gate"]
                                    and not x["gate"].startswith("weak_kept_rescue")),
            "weak_kept_rescue": sum(1 for x in kept if x["gate"].startswith("weak_kept_rescue")),
            "weak_nowindow": sum(1 for x in kept if x["gate"].startswith("weak_nowindow")),
            "no_transient": sum(1 for x in kept if x["gate"].startswith("no_transient")),
        },
        "candidates": kept,
    }
    c.save_json(os.path.join(c.R56_OUT, f"{w.key}_E1_contacts.json"), doc)
    c.save_json(os.path.join(c.R56_OUT, f"{w.key}_E1_refined.json"),
                {"av_off_ms": c.AV_OFF_MS, "events": used})
    return {"n_e0": len(cands), "n_cand": len(kept), "n_windows": len(ta),
            **doc["gate_counts"]}


def main():
    for w in c.load_dev_windows():
        s = run_e1(w)
        print(f"{w.key}: E0 {s['n_e0']} cands -> E1 {s['n_cand']} candidates "
              f"(strong/present {s['keep_strong_present']}, weak visual {s['weak_kept_visual']}, "
              f"weak rescue {s['weak_kept_rescue']}, weak nowindow {s['weak_nowindow']}, "
              f"no_transient {s['no_transient']}) | windows {s['n_windows']}")


if __name__ == "__main__":
    main()
