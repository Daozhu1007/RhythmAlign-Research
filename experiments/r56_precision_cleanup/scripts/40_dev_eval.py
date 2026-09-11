# R5.6 Phase 4 - development evaluation E0 (R5.5 frozen) vs E1 (cleanup) vs
# E2 (cleanup+completion) on Dev-A/B/C. Product metrics first:
# strong recall, strong gate coverage, confirmed false-window duration,
# interaction music leakage, taiko leakage, auto-vs-oracle final RMS diff,
# interaction coverage, headroom, USEFUL WINDOW RATIO; then P/R/F1 @33/50.
# Dev-C oracle is INCOMPLETE -> its precision/F1 carry a qualification and
# useful-window support uses high/med oracle contacts as anchors.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf
from scipy import signal as sig

import common56 as c
from detector55 import fused_score
from r5_common import SR, audio_onsets

E0_INT = {"devA": os.path.join(c.R55_OUT, "devA_r55auto_C1_interaction.wav"),
          "devB": os.path.join(c.R55_OUT, "devB_r55auto_C1_interaction.wav"),
          "devC": os.path.join(c.R55_OUT, "holdoutC_auto_C1_interaction.wav")}
E0_MIX = {"devA": os.path.join(c.R55_OUT, "devA_r55auto_C1_final_mix.wav"),
          "devB": os.path.join(c.R55_OUT, "devB_r55auto_C1_final_mix.wav"),
          "devC": os.path.join(c.R55_OUT, "holdoutC_auto_C1_final_mix.wav")}
ORC_MIX = {"devA": os.path.join(c.R55_OUT, "devA_oracle_C1_final_mix.wav"),
           "devB": os.path.join(c.R55_OUT, "devB_oracle_C1_final_mix.wav"),
           "devC": os.path.join(c.R55_OUT, "holdoutC_oracle_C1_final_mix.wav")}


def load_mono(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def gate_spans(stem, amp=1e-6):
    k = (np.abs(stem) > amp).astype(int)
    d = np.diff(np.concatenate([[0], k, [0]]))
    return [(s / SR, e / SR) for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]


def ref_bands_profile(w):
    """Band envelopes of the aligned MUSIC track itself, to label music onsets
    as bass-dominated (taiko-like) vs broadband."""
    from scipy import signal as sig
    data, _ = sf.read(w.ref, dtype="float64", always_2d=True)
    r = data[int(c.CTX_START * SR):int((c.CTX_START + w.dur) * SR)].mean(axis=1)

    def band_env(x, lo, hi):
        sosb = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
        xb = sig.sosfiltfilt(sosb, x)
        return sig.convolve(np.abs(xb), sig.windows.hann(49), mode="same")

    def norm(e):
        return e / (np.percentile(e, 99.9) + 1e-9)
    return {"bass": norm(band_env(r, 150, 500)), "mid": norm(band_env(r, 500, 2000)),
            "hi1": norm(band_env(r, 2000, 6000)), "hi2": norm(band_env(r, 6000, 14000))}


def eval_variant(w, tag, int_path, mix_path, windows_t):
    """windows_t: audio times whose C1 windows are open (None -> derive from stem)."""
    stem = load_mono(int_path)
    on = audio_onsets(stem)
    spans = gate_spans(stem)
    cov = float(np.mean(np.abs(stem) > 1e-6))

    orc_all = c.load_json(w.oracle_refined)["events"]
    orc = [e for e in orc_all if e.get("refined_confidence")
           not in (None, "no_transient", "out_of_range")]
    to = np.array([e["t_audio_refined"] for e in orc])
    ts_strong = np.array([e["t_audio_refined"] for e in orc
                          if e["refined_confidence"] == "strong"])

    def covered(t, arr):
        return any(s <= t <= e for s, e in spans)

    gate = float(np.mean([covered(t, to) for t in to])) if len(to) else None
    gate_s = (float(np.mean([covered(t, ts_strong) for t in ts_strong]))
              if len(ts_strong) else None)

    fw = [(s, e) for s, e in spans
          if len(to) == 0 or np.min(np.abs(to - (s + e) / 2)) > 0.080]

    # music + taiko leakage on the interaction stem
    ref_on = audio_onsets(_ref_slice(w))
    bands = ref_bands_profile(w)
    music_hits, taiko_hits = [], []
    for t in on:
        k = int(round(t * SR))
        d = np.min(np.abs(ref_on - t)) if len(ref_on) else 10
        if d <= 0.030:
            music_hits.append(k)
            if bands["bass"][k] >= 1.5 * (bands["mid"][k] + bands["hi1"][k] + bands["hi2"][k]):
                taiko_hits.append(k)

    # useful window ratio (support anchors depend on oracle trust)
    tc = w.tc()
    u_all, tot = c.useful_window_ratio(spans, list(tc))
    hi_med = [e["t_video"] for e in w.contacts
              if e.get("confidence") in ("high", "med")]
    u_hm, _ = c.useful_window_ratio(spans, hi_med)
    hi = [e["t_video"] for e in w.contacts if e.get("confidence") == "high"]
    u_hi, _ = c.useful_window_ratio(spans, hi)

    mix, _ = sf.read(mix_path, dtype="float64", always_2d=True)
    orc_mix, _ = sf.read(ORC_MIX[w.key], dtype="float64", always_2d=True)
    n = min(len(mix), len(orc_mix))
    diff = mix[:n].mean(axis=1) - orc_mix[:n].mean(axis=1)
    rms_db = 20 * np.log10(np.sqrt(np.mean(diff ** 2)) /
                           (np.sqrt(np.mean(orc_mix[:n].mean(axis=1) ** 2)) + 1e-12))
    peak_dbfs = 20 * np.log10(np.max(np.abs(mix)) + 1e-12)

    return {
        "tag": tag,
        "n_windows": len(windows_t) if windows_t is not None else len(spans),
        "interaction_spans": len(spans), "interaction_coverage": round(cov, 3),
        "gate_coverage": round(gate, 3) if gate is not None else None,
        "gate_coverage_strong": round(gate_s, 3) if gate_s is not None else None,
        "false_windows": {"count": len(fw), "total_s": round(sum(e - s for s, e in fw), 2)},
        "music_onset_matched": len(music_hits),
        "taiko_onset_matched": len(taiko_hits),
        "useful_window_ratio_all_oracle": round(u_all, 3),
        "useful_window_ratio_highmed": round(u_hm, 3),
        "useful_window_ratio_high": round(u_hi, 3),
        "auto_vs_oracle_final_rms_db": round(float(rms_db), 1),
        "final_mix_peak_dbfs": round(float(peak_dbfs), 2),
    }


def _ref_slice(w):
    data, _ = sf.read(w.ref, dtype="float64", always_2d=True)
    r = data[int(c.CTX_START * SR):int((c.CTX_START + w.dur) * SR)].mean(axis=1)
    return r


def eval_events(w, cand_t, label):
    tp_t = np.array(cand_t)
    tc = w.tc()
    out = {"label": label, "n_pred": len(cand_t), "n_oracle": len(tc)}
    for tol in (0.033, 0.050):
        mc, mp, pairs = c.match_one_to_one(tp_t, tc, tol)
        p = len(mc) / max(len(cand_t), 1)
        r = len(mc) / max(len(tc), 1)
        errs = [abs(e) * 1000 for _, _, e in pairs]
        out[f"tol{int(tol*1000)}"] = {
            "P": round(p, 3), "R": round(r, 3),
            "F1": round(2 * p * r / max(p + r, 1e-9), 3),
            "err_ms_p95": round(float(np.percentile(errs, 95)), 1) if errs else None}
    strong = [e for e in w.contacts if e.get("confidence") == "high"]
    ts = np.array([e["t_video"] for e in strong])
    hit = [t for t in ts if len(tp_t) and np.min(np.abs(tp_t - t)) <= 0.050]
    out["strong_recall"] = round(len(hit) / max(len(ts), 1), 3)
    out["strong_missed"] = [e["id"] for e in strong
                            if not (len(tp_t) and np.min(np.abs(tp_t - e["t_video"])) <= 0.050)]
    return out


def windows_of(w, variant):
    """(candidate_times, window_times) for each variant.
    Candidates follow the R5.5 visual convention (all detected candidates,
    including no_transient); windows are what actually opens C1 gates."""
    src_vis = {
        "devA": "devA_r55_contacts_visual.json",
        "devB": "devB_r55_contacts_visual.json",
        "devC": "holdoutC_contacts_auto_visual.json"}[w.key]
    src_ref = {
        "devA": "devA_r55auto_refined.json",
        "devB": "devB_r55auto_refined.json",
        "devC": "holdoutC_contacts_auto_refined.json"}[w.key]
    if variant == "E0":
        cands = [x["t_video"] for x in
                 c.load_json(os.path.join(c.R55_OUT, src_vis))["candidates"]]
        used = [r for r in c.load_json(os.path.join(c.R55_OUT, src_ref))["events"]
                if r.get("refined_confidence")
                not in (None, "no_transient", "out_of_range")]
        return cands, [r["t_audio_refined"] for r in used]
    if variant == "E1":
        doc = c.load_json(os.path.join(c.R56_OUT, f"{w.key}_E1_contacts.json"))
        cands = [x["t_video"] for x in doc["candidates"]]
        ref = c.load_json(os.path.join(c.R56_OUT, f"{w.key}_E1_refined.json"))["events"]
        return cands, [r["t_audio_refined"] for r in ref]
    if variant == "E2":
        doc = c.load_json(os.path.join(c.R56_OUT, f"{w.key}_E2_contacts.json"))
        from detector55 import fused_score as _fs
        score, ns = _fs(w.series, c.CFG55_FROZEN)
        _, mask, ivs, env, cands0 = c.detect55(w.series, c.FPS, c.CFG55_FROZEN, rule="A")
        ref = c.load_json(os.path.join(c.R55_OUT, src_ref))["events"]
        flags = c.e1_gate_flags(cands0, ref, score, ns, ivs, c.E1_CFG)
        cands = [r["t_video"] for j, r in enumerate(ref) if flags[j][0]]
        e1_t = [ref[j]["t_audio_refined"] for j in range(len(cands0))
                if flags[j][1] and ref[j].get("refined_confidence")
                not in (None, "no_transient", "out_of_range")]
        wins = e1_t + [a["t"] for a in doc["added"]]
        return cands + [a["t"] for a in doc["added"]], wins
    raise ValueError(variant)


def main():
    rep = {}
    for w in c.load_dev_windows():
        rep[w.key] = {}
        for variant in ("E0", "E1", "E2"):
            cands, wt = windows_of(w, variant)
            int_path = (E0_INT[w.key] if variant == "E0" else
                        os.path.join(c.R56_OUT, f"{w.key}_{variant}_C1_interaction.wav"))
            mix_path = (E0_MIX[w.key] if variant == "E0" else
                        os.path.join(c.R56_OUT, f"{w.key}_{variant}_C1_final_mix.wav"))
            m = eval_variant(w, variant, int_path, mix_path, wt)
            m["events"] = eval_events(w, cands, variant)
            m["n_candidates"] = len(cands)
            rep[w.key][variant] = m
        print(f"\n===== {w.key} =====")
        for v in ("E0", "E1", "E2"):
            m, e = rep[w.key][v], rep[w.key][v]["events"]
            print(f"  {v}: cand {m['n_candidates']:3d} win {m['n_windows']:3d} | "
                  f"gate {m['gate_coverage']:.3f} (strong {m['gate_coverage_strong']}) | "
                  f"false-win {m['false_windows']['count']} ({m['false_windows']['total_s']} s) | "
                  f"music-leak {m['music_onset_matched']} | taiko-leak {m['taiko_onset_matched']} | "
                  f"UWR(hi+med) {m['useful_window_ratio_highmed']:.3f} | "
                  f"rms-diff {m['auto_vs_oracle_final_rms_db']} dB | headroom {m['final_mix_peak_dbfs']} dBFS")
            t5, t3 = e["tol50"], e["tol33"]
            print(f"      events: P {t5['P']:.3f} R {t5['R']:.3f} F1 {t5['F1']:.3f} "
                  f"(@33 {t3['F1']:.3f}) | strong-R {e['strong_recall']:.3f} "
                  f"| timing p95 {t5['err_ms_p95']} ms | missed-strong {e['strong_missed']}")
    c.save_json(os.path.join(c.R56_LOG, "dev_eval_E0_E1_E2.json"), rep)


if __name__ == "__main__":
    main()
