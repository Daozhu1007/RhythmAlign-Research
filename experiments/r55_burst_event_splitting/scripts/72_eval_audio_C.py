# R5.5 Phase 8 (audio comparison) - Holdout C auto C1 vs oracle C1 (objective),
# same metrics as R5 76_eval_audio_holdout.py.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf

from common55 import R55_OUT, R55_LOG, R55_WORK, load_json, save_json
from r5_common import SR, audio_onsets

MATCH_TOL = 0.030


def load_mono(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def gate_spans(stem, amp=1e-6):
    k = (np.abs(stem) > amp).astype(int)
    d = np.diff(np.concatenate([[0], k, [0]]))
    return [(s / SR, e / SR) for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]


def main():
    orc = load_json(os.path.join(R55_OUT, "holdoutC_oracle_refined.json"))["events"]
    events = [e for e in orc
              if e.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    tcts = np.array([e["t_audio_refined"] for e in events])
    strong = [e for e in events if e["refined_confidence"] == "strong"]
    ts = np.array([e["t_audio_refined"] for e in strong])

    stems = {"oracle_interaction": load_mono(os.path.join(R55_OUT, "holdoutC_oracle_C1_interaction.wav")),
             "auto_interaction": load_mono(os.path.join(R55_OUT, "holdoutC_auto_C1_interaction.wav"))}
    finals = {"oracle_final": load_mono(os.path.join(R55_OUT, "holdoutC_oracle_C1_final_mix.wav")),
              "auto_final": load_mono(os.path.join(R55_OUT, "holdoutC_auto_C1_final_mix.wav"))}

    ref_full, _ = sf.read(os.path.join(R55_WORK, "holdoutC_ref_warp.wav"),
                          dtype="float64", always_2d=True)
    ref = ref_full[int(3.0 * SR):int(21.0 * SR)].mean(axis=1).astype(np.float64)
    ref_on = audio_onsets(ref)

    rep = {}
    for name, y in {**stems, **finals}.items():
        on = audio_onsets(y)
        hit = [bool(len(on) and np.min(np.abs(on - t)) <= MATCH_TOL) for t in tcts]
        sh = [bool(len(on) and np.min(np.abs(on - t)) <= MATCH_TOL) for t in ts]
        tk = int(np.sum([len(on) and np.min(np.abs(on - t)) <= MATCH_TOL for t in ref_on]))
        r = {"n_onsets": int(len(on)),
             "contact_retention": f"{int(np.sum(hit))}/{len(tcts)}",
             "strong_retention": f"{int(np.sum(sh))}/{len(ts)}",
             "music_onset_matched": tk}
        if name in stems:
            spans = gate_spans(y)
            r["coverage"] = round(float(np.mean(np.abs(y) > 1e-6)), 3)
            r["n_spans"] = len(spans)
        else:
            path = {"oracle_final": os.path.join(R55_OUT, "holdoutC_oracle_C1_final_mix.wav"),
                    "auto_final": os.path.join(R55_OUT, "holdoutC_auto_C1_final_mix.wav")}[name]
            data, _ = sf.read(path, dtype="float64", always_2d=True)
            r["peak_dbfs"] = round(float(20 * np.log10(np.max(np.abs(data)) + 1e-12)), 2)
        rep[name] = r

    auto_spans = gate_spans(stems["auto_interaction"])
    orc_spans = gate_spans(stems["oracle_interaction"])
    fw = [(round(s, 2), round(e, 2)) for s, e in auto_spans
          if len(tcts) and np.min(np.abs(tcts - (s + e) / 2)) > 0.080]
    rep["auto_false_windows"] = {"count": len(fw), "total_s": round(sum(e - s for s, e in fw), 2),
                                 "spans_s": fw[:30]}

    def inside(t, spans):
        return any(s <= t <= e for s, e in spans)
    lost = [(round(s, 2), round(e, 2)) for s, e in orc_spans
            if not inside((s + e) / 2, auto_spans)]
    rep["oracle_windows_missing_in_auto"] = {"count": len(lost), "spans_s": lost[:25]}
    diff = finals["auto_final"] - finals["oracle_final"]
    rep["final_mix_abs_diff_rms_db_vs_oracle"] = round(
        float(20 * np.log10(np.sqrt(np.mean(diff ** 2)) /
                            (np.sqrt(np.mean(finals["oracle_final"] ** 2)) + 1e-12))), 1)
    save_json(os.path.join(R55_LOG, "holdoutC_audio_comparison.json"), rep)
    for k, v in rep.items():
        print(k, ":", v if not isinstance(v, dict) else
              {kk: vv for kk, vv in v.items() if kk != "spans_s"})


if __name__ == "__main__":
    main()
