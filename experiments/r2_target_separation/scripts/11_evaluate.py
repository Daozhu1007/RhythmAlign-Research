# R2 - evaluation of AudioSep runs on the R1 golden sample.
#
# Everything here is an OBJECTIVE PROXY. None of it replaces human listening;
# final judgment on artifacts/hallucination requires ears (flagged in the report).
#
# Checks per run (audiosep_{raw,a2}_p{1,2}):
#   1. player-click retention : how many of R1's 53 player-candidate transients
#      (gate_diagnostics.json "clicks") reappear as onsets in the target stem
#   2. transient-level drop   : energy of target vs mixture in a +-50 ms window
#      at each R1 click (how much click energy survives separation)
#   3. taiko leakage          : R1 "taiko" candidate onsets present in target
#   4. music leakage          : onsets of target matching aligned-reference
#      onsets; correlation of target with the aligned reference
#   5. hallucination          : target onsets with NO onset in the 32 kHz mono
#      mixture the model actually saw (new transients that do not exist in the
#      recording = severe failure per brief)
#   6. energy shares          : target/residual RMS vs mixture
#   7. spectrograms           : mixture/target/residual PNG per run
#
# Onset detection mirrors R1's audit: STFT N=1024 HOP=256, positive spectral
# flux summed over 2-12 kHz, peaks > median + 4*MAD, min distance ~60 ms.
import json
import os

import numpy as np
import librosa
import soundfile as sf
from scipy import signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(R2_DIR, "outputs")
LOG_DIR = os.path.join(R2_DIR, "logs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"

SR = 32000
N, HOP = 1024, 256
MATCH_TOL = 0.030          # s, onset matching tolerance
EV_HALF = 0.050            # s, half-window for local energy stats


def load32k(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def high_flux_onsets(y):
    """Positive spectral-flux onsets in 2-12 kHz (R1 audit method @ 32 kHz)."""
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=N, hop_length=HOP)) + 1e-10
    Sdb = librosa.amplitude_to_db(S)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N)
    dS = np.maximum(0.0, np.diff(Sdb, axis=1))
    fh = dS[(freqs >= 2000) & (freqs < 12000)].sum(axis=0)
    med = np.median(fh)
    mad = np.median(np.abs(fh - med)) + 1e-9
    pk, _ = sig.find_peaks(
        fh, height=med + 4.0 * mad, distance=max(1, int(0.06 * SR / HOP)))
    t_off = N / (2 * SR)
    return (pk + 1) / (SR / HOP) + t_off


def music_onsets(y):
    on = librosa.onset.onset_strength(y=y.astype(np.float32), sr=SR, hop_length=HOP)
    return librosa.onset.onset_detect(onset_envelope=on, sr=SR, hop_length=HOP,
                                      units="time", backtrack=False)


def match_to(times, targets, tol=MATCH_TOL):
    """For each t in times, nearest target time within tol (else None)."""
    out = []
    for t in times:
        if len(targets) == 0:
            out.append(None)
            continue
        d = np.min(np.abs(np.asarray(targets) - t))
        out.append(float(d) if d <= tol else None)
    return out


def local_db(y, t, half=EV_HALF):
    a, b = int(max(0, (t - half) * SR)), int(min(len(y), (t + half) * SR))
    if b - a < 32:
        return None
    return float(20 * np.log10(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-12))


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def spec_png(tag, mix, target, resid):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    vmax = -20
    for ax, (name, y) in zip(axes, [("mixture (32k mono)", mix), ("target", target), ("residual", resid)]):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-10)
        vmax = max(vmax, S.max())
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(y) / SR, 0, SR // 2], vmin=vmax - 90, vmax=vmax)
        ax.set_title(f"{tag}: {name}", fontsize=9)
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"spec_{tag}.png"), dpi=110)
    plt.close(fig)


def main():
    gate = json.load(open(os.path.join(R1_WORK, "gate_diagnostics.json")))
    clicks = np.array([c["t"] for c in gate["stereo"]["clicks"]])
    taiko = np.array([c["t"] for c in gate["stereo"]["taiko"]])
    print(f"R1 events: {len(clicks)} player-click candidates, {len(taiko)} taiko candidates")

    ref = load32k(os.path.join(R1_WORK, "ref_warp_fixed.wav"))[int(3.0 * SR):int(18.0 * SR)]
    ref_pk = music_onsets(ref)
    print(f"aligned reference: {len(ref_pk)} music onsets")

    mixtures = {"raw": load32k(os.path.join(R1_OUT, "golden_raw.wav")),
                "a2": load32k(os.path.join(R1_OUT, "A2_residual.wav"))}
    mix_onsets = {k: high_flux_onsets(y) for k, y in mixtures.items()}
    print("input baseline (onsets in 32k mono mixture):",
          {k: len(v) for k, v in mix_onsets.items()},
          "| clicks matched in mixture:",
          {k: int(np.sum([m is not None for m in match_to(clicks, v)])) for k, v in mix_onsets.items()})

    report = {"sr": SR, "n_clicks": len(clicks), "n_taiko": len(taiko),
              "n_ref_onsets": len(ref_pk), "inputs": {}, "runs": {}}
    for tag, mix in mixtures.items():
        m = match_to(clicks, mix_onsets[tag])
        report["inputs"][tag] = {
            "n_onsets": len(mix_onsets[tag]),
            "clicks_matched_in_mixture": int(np.sum([x is not None for x in m])),
        }

    names = sorted(f[: -len("_target.wav")] for f in os.listdir(OUT_DIR) if f.endswith("_target.wav"))
    for name in names:
        tag = name.split("_")[1]
        target = load32k(os.path.join(OUT_DIR, f"{name}_target.wav"))
        resid = load32k(os.path.join(OUT_DIR, f"{name}_residual.wav"))
        mix = mixtures[tag]
        tgt_on = high_flux_onsets(target)

        c_hit = match_to(clicks, tgt_on)
        t_hit = match_to(taiko, tgt_on)
        m_hit = match_to(ref_pk, tgt_on)
        mix_hit = match_to(tgt_on, mix_onsets[tag])
        halluc = [float(t) for t, h in zip(tgt_on, mix_hit) if h is None]

        drop = []
        for t, hit in zip(clicks, c_hit):
            if hit is None:
                continue
            et, em = local_db(target, t), local_db(mix, t)
            if et is not None and em is not None:
                drop.append(et - em)
        r = np.corrcoef(target, ref)[0, 1] if len(ref) == len(target) else None

        # music-heavy frames: reference RMS in top quartile of frames
        fl = len(ref) // HOP
        frms = np.array([db(ref[i * HOP:(i + 1) * HOP]) for i in range(fl)])
        thr = np.quantile(frms, 0.75)
        loud = [(i * HOP, (i + 1) * HOP) for i in range(fl) if frms[i] >= thr]
        corr_loud = float(np.mean([np.corrcoef(target[a:b], ref[a:b])[0, 1] for a, b in loud]))

        spec_png(name, mix, target, resid)
        run = {
            "n_target_onsets": len(tgt_on),
            "click_retention": {
                "matched": int(np.sum([x is not None for x in c_hit])),
                "rate": float(np.mean([x is not None for x in c_hit])),
                "matched_click_times": [float(t) for t, h in zip(clicks, c_hit) if h is not None],
            },
            "click_level_drop_db_at_matched": {
                "median": float(np.median(drop)), "min": float(np.min(drop)),
                "max": float(np.max(drop))} if drop else None,
            "taiko_leakage": {
                "matched": int(np.sum([x is not None for x in t_hit])),
                "times": [float(t) for t, h in zip(taiko, t_hit) if h is not None]},
            "music_onsets_in_target": int(np.sum([x is not None for x in m_hit])),
            "corr_target_vs_ref": float(r),
            "corr_target_vs_ref_musicheavy_frames": corr_loud,
            "hallucinated_onsets_vs_mixture": {
                "count": len(halluc), "times": halluc},
            "energy_db": {"mixture": db(mix), "target": db(target), "residual": db(resid)},
        }
        report["runs"][name] = run
        print(f"\n=== {name} ===")
        print(f"  onsets in target: {len(tgt_on)}")
        print(f"  click retention : {run['click_retention']['matched']}/{len(clicks)} "
              f"({run['click_retention']['rate']:.1%})")
        if drop:
            print(f"  click level in target vs mixture: median {run['click_level_drop_db_at_matched']['median']:+.1f} dB")
        print(f"  taiko leakage   : {run['taiko_leakage']['matched']}/{len(taiko)}")
        print(f"  ref-onset events in target: {run['music_onsets_in_target']}/{len(ref_pk)}")
        print(f"  corr(target, ref): {r:.3f} | music-heavy frames: {corr_loud:.3f}")
        print(f"  hallucinated onsets (no mixture onset): {len(halluc)} {halluc[:10]}")
        print(f"  energy dB: mix {run['energy_db']['mixture']:.1f} | tgt {run['energy_db']['target']:.1f} "
              f"| res {run['energy_db']['residual']:.1f}")

    with open(os.path.join(LOG_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nsaved logs/evaluation.json + spec_*.png")


if __name__ == "__main__":
    main()
