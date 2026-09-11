# R5 Phase 5 - objective audio comparison: auto C1 vs oracle C1 (dev).
# Same onset-detector family as R4 30_evaluate.py. All numbers reported honestly.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf
import librosa

from r5_common import (R1_OUT, R1_WORK, R4_OUT, OUT, LOG, SR, load_json, save_json,
                       audio_onsets)

MATCH_TOL = 0.030


def load_mono(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def local_db(y, t, half=0.050):
    a, b = int(max(0, (t - half) * SR)), int(min(len(y), (t + half) * SR))
    if b - a < 32:
        return None
    return float(20 * np.log10(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-12))


def gate_spans(stem, amp=1e-6):
    k = (np.abs(stem) > amp).astype(int)
    d = np.diff(np.concatenate([[0], k, [0]]))
    return [(s / SR, e / SR) for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]


def main():
    raw = load_mono(os.path.join(R1_OUT, "golden_raw.wav"))
    orc = load_json(os.path.join(R4_OUT, "contacts_refined.json"))["events"]
    events = [e for e in orc
              if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    tcts = np.array([e["t_audio_refined"] for e in events])
    conf = [e["refined_confidence"] for e in events]
    gate = load_json(os.path.join(R1_WORK, "gate_diagnostics.json"))
    taiko = np.array([c["t"] for c in gate["stereo"]["taiko"]])

    stems = {
        "oracle_interaction": load_mono(os.path.join(OUT, "dev_oracle_C1_interaction.wav")),
        "auto_interaction": load_mono(os.path.join(OUT, "dev_auto_C1_interaction.wav")),
    }
    finals = {
        "oracle_final": load_mono(os.path.join(OUT, "dev_oracle_C1_final_mix.wav")),
        "auto_final": load_mono(os.path.join(OUT, "dev_auto_C1_final_mix.wav")),
    }
    rep = {}
    for name, y in {**stems, **finals}.items():
        on = audio_onsets(y)
        hit = [bool(np.min(np.abs(on - t)) <= MATCH_TOL) if len(on) else False for t in tcts]
        strong_hit = [h for h, c in zip(hit, conf) if c == "strong"]
        tk_hit = int(np.sum([np.min(np.abs(on - t)) <= MATCH_TOL for t in taiko])) if len(on) else 0
        r = {
            "n_onsets": int(len(on)),
            "contact_retention": f"{int(np.sum(hit))}/{len(tcts)}",
            "strong_retention": f"{int(np.sum(strong_hit))}/{len(strong_hit)}",
            "taiko_matched": tk_hit,
        }
        if name in stems:
            spans = gate_spans(y)
            r["coverage"] = round(float(np.mean(np.abs(y) > 1e-6)), 3)
            r["n_spans"] = len(spans)
        else:
            data, _ = sf.read({"oracle_final": os.path.join(OUT, "dev_oracle_C1_final_mix.wav"),
                               "auto_final": os.path.join(OUT, "dev_auto_C1_final_mix.wav")}[name],
                              dtype="float64", always_2d=True)
            r["peak_dbfs"] = round(float(20 * np.log10(np.max(np.abs(data)) + 1e-12)), 2)
        rep[name] = r

    # false windows: auto gate spans whose midpoint is >80ms from every oracle contact
    auto_spans = gate_spans(stems["auto_interaction"])
    orc_spans = gate_spans(stems["oracle_interaction"])
    fw = []
    for s, e in auto_spans:
        mid = (s + e) / 2
        if np.min(np.abs(tcts - mid)) > 0.080:
            fw.append((round(s, 2), round(e, 2)))
    rep["auto_false_windows"] = {"count": len(fw), "spans_s": fw,
                                 "total_s": round(sum(e - s for s, e in fw), 2)}
    # windows oracle has but auto doesn't (interaction loss): oracle span mid not inside any auto span
    def inside(t, spans):
        return any(s <= t <= e for s, e in spans)
    lost = [(round(s, 2), round(e, 2)) for s, e in orc_spans if not inside((s + e) / 2, auto_spans)]
    rep["oracle_windows_missing_in_auto"] = {"count": len(lost), "spans_s": lost[:20]}
    # final mix difference vs oracle final
    diff = finals["auto_final"] - finals["oracle_final"]
    rep["final_mix_abs_diff_rms_db_vs_oracle"] = round(
        float(20 * np.log10(np.sqrt(np.mean(diff ** 2)) /
                            (np.sqrt(np.mean(finals["oracle_final"] ** 2)) + 1e-12))), 1)
    save_json(os.path.join(LOG, "dev_audio_comparison.json"), rep)
    for k, v in rep.items():
        print(k, ":", v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "spans_s"})


if __name__ == "__main__":
    main()
