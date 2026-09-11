# R5.5 Phase 8 (baseline audio) - R5 FROZEN detector audio pipeline on Holdout C
# for same-window comparison: refine + C1 + gate coverage + RMS vs oracle.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf

from common55 import R55_OUT, R55_LOG, R55_WORK, load_json, save_json
from r5_common import SR
import r5_common as r5

AV_OFF_MS = 0.0
CTX_START = 3.0
DUR = 18.0


def main():
    comb = np.load(os.path.join(R55_WORK, "holdoutC_comb.npy"))
    vis = load_json(os.path.join(R55_OUT, "holdoutC_contacts_r5baseline_visual.json"))["candidates"]
    ids = [f"b{i:03d}" for i in range(len(vis))]
    refined = r5.refine_candidates([c["t_video"] for c in vis], comb, AV_OFF_MS, DUR, ids=ids)
    for c, r in zip(vis, refined):
        r["score"] = c["score"]
    save_json(os.path.join(R55_OUT, "holdoutC_r5baseline_refined.json"),
              {"av_off_ms": AV_OFF_MS, "n_candidates": len(vis), "events": refined})
    used = [r for r in refined
            if r.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    ta = np.array([r["t_audio_refined"] for r in used])
    orc = load_json(os.path.join(R55_OUT, "holdoutC_oracle_refined.json"))["events"]
    to = np.array([e["t_audio_refined"] for e in orc
                   if e.get("refined_confidence") not in (None, "no_transient", "out_of_range")])
    spans = []
    for t in ta:
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
    gate = float(np.mean([covered(t) for t in to]))
    strong = [e for e in orc if e.get("refined_confidence") == "strong"]
    gate_s = float(np.mean([covered(e["t_audio_refined"]) for e in strong]))
    ref_full, _ = sf.read(os.path.join(R55_WORK, "holdoutC_ref_warp.wav"),
                          dtype="float64", always_2d=True)
    ref = ref_full[int(CTX_START * SR):int((CTX_START + DUR) * SR)]
    r5.build_c1(os.path.join(R55_OUT, "holdoutC_raw.wav"), ref, ta,
                os.path.join(R55_OUT, "holdoutC_r5baseline_C1_interaction.wav"),
                os.path.join(R55_OUT, "holdoutC_r5baseline_C1_final_mix.wav"))
    a, _ = sf.read(os.path.join(R55_OUT, "holdoutC_r5baseline_C1_final_mix.wav"),
                   dtype="float64", always_2d=True)
    o, _ = sf.read(os.path.join(R55_OUT, "holdoutC_oracle_C1_final_mix.wav"),
                   dtype="float64", always_2d=True)
    diff = a.mean(axis=1) - o.mean(axis=1)
    rms = round(float(20 * np.log10(np.sqrt(np.mean(diff ** 2)) /
                                    (np.sqrt(np.mean(o.mean(axis=1) ** 2)) + 1e-12))), 1)
    import librosa
    y = librosa.load(os.path.join(R55_OUT, "holdoutC_r5baseline_C1_interaction.wav"),
                     sr=SR, mono=True)[0]
    from r5_common import audio_onsets
    on = audio_onsets(y.astype(np.float64))
    refm = ref_full[int(3.0 * SR):int(21.0 * SR)].mean(axis=1).astype(np.float64)
    ron = audio_onsets(refm)
    leak = int(np.sum([len(on) and np.min(np.abs(on - t)) <= 0.030 for t in ron]))
    out = {"n_used": len(used), "gate_coverage_of_oracle": round(gate, 3),
           "gate_coverage_strong": round(gate_s, 3),
           "final_mix_rms_db_vs_oracle": rms, "music_onset_matched": leak}
    save_json(os.path.join(R55_LOG, "holdoutC_r5baseline_audio.json"), out)
    print("R5 baseline audio on holdoutC:", out)


if __name__ == "__main__":
    main()
