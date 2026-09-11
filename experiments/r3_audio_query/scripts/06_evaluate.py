# R3 evaluation - objective proxies for all target stems (Phase 2 + 4 + 5 + R2 baseline).
#
# Core idea of the selectivity score (brief requirement): compare how much of the
# OPERATION-event energy survives vs how much of the MUSIC energy survives:
#   R_op    = median over R1 player-click windows of 20log10(rms_target/rms_mixture)
#   R_music = same ratio over music-heavy frames (ref rms top quartile) that are
#             at least 0.15 s away from any click/taiko candidate
#   R_quiet = same over quiet non-operation frames (ref rms bottom quartile)
#   selectivity = R_op - R_music   (dB)
# A uniform attenuator (target = g * mixture) has selectivity == 0 dB by
# construction, so any clearly positive value means real preference for the
# operation events; negative means music is kept better than the operations.
#
# Also per run: click retention (+30 ms tolerance, R1 audit method), click level
# drop, taiko leakage, ref-onset leakage, hallucinated onsets vs mixture, frame
# energy regression target~mixture (slope 1 + tiny residual => attenuated copy),
# correlation with aligned reference, spectrogram PNG.
import json
import os

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal as sig

R3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(R3, "outputs")
LOG_DIR = os.path.join(R3, "logs")
R3_OUT = OUT_DIR
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
R2_OUT = r"D:\Code\RhythmAlign\experiments\r2_target_separation\outputs"

SR = 32000
N, HOP = 1024, 256
MATCH_TOL = 0.030
EV_HALF = 0.050
GUARD = 0.060          # around CLICKS only for music frames (click transient is
                       # ~<50 ms; taiko hits stay in the music set - they ARE music)


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def load32k(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def high_flux_onsets(y):
    S = np.abs(librosa.stft(y, n_fft=N, hop_length=HOP)) + 1e-10
    Sdb = librosa.amplitude_to_db(S)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N)
    dS = np.maximum(0.0, np.diff(Sdb, axis=1))
    fh = dS[(freqs >= 2000) & (freqs < 12000)].sum(axis=0)
    med, mad = np.median(fh), np.median(np.abs(fh - np.median(fh))) + 1e-9
    pk, _ = sig.find_peaks(fh, height=med + 4.0 * mad, distance=max(1, int(0.06 * SR / HOP)))
    return (pk + 1) / (SR / HOP) + N / (2 * SR)


def music_onsets(y):
    on = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    return librosa.onset.onset_detect(onset_envelope=on, sr=SR, hop_length=HOP,
                                      units="time", backtrack=False)


def match_to(times, targets, tol=MATCH_TOL):
    out = []
    for t in times:
        d = np.min(np.abs(np.asarray(targets) - t)) if len(targets) else 1e9
        out.append(float(d) if d <= tol else None)
    return out


def local_db(y, t, half=EV_HALF):
    a, b = int(max(0, (t - half) * SR)), int(min(len(y), (t + half) * SR))
    if b - a < 32:
        return None
    return db(y[a:b])


def ratio_db(target, mix, a, b):
    tm, mm = db(target[a:b]), db(mix[a:b])
    return tm - mm


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

    mix = load32k(os.path.join(R1_OUT, "golden_raw.wav"))
    ref = load32k(os.path.join(R1_WORK, "ref_warp_fixed.wav"))[int(3.0 * SR):int(18.0 * SR)]
    assert len(ref) == len(mix)
    ref_pk = music_onsets(ref)
    mix_on = high_flux_onsets(mix)

    # frame classes from the clean reference
    fl = len(ref) // HOP
    frms = np.array([db(ref[i * HOP:(i + 1) * HOP]) for i in range(fl)])
    q_hi, q_lo = np.quantile(frms, 0.75), np.quantile(frms, 0.25)
    t_edges = np.arange(fl + 1) * HOP / SR

    def frame_windows(mask):
        return [(t_edges[i], t_edges[i + 1]) for i in range(fl) if mask[i]]

    def far_from_events(mid):
        return np.min(np.abs(clicks - mid)) > GUARD

    music_heavy = [(a, b) for a, b in frame_windows(frms >= q_hi) if far_from_events((a + b) / 2)]
    music_quiet = [(a, b) for a, b in frame_windows(frms <= q_lo) if far_from_events((a + b) / 2)]

    report = {"sr": SR, "n_clicks": len(clicks), "n_taiko": len(taiko),
              "n_ref_onsets": len(ref_pk), "n_music_heavy_windows": len(music_heavy),
              "n_music_quiet_windows": len(music_quiet), "runs": {}}

    stems = {os.path.basename(p): p for p in [
        os.path.join(R2_OUT, "audiosep_raw_p2_target.wav"),
        os.path.join(R3_OUT, "audiosep_aq_q1_target.wav"),
        os.path.join(R3_OUT, "audiosep_aq_q2_target.wav"),
        os.path.join(R3_OUT, "clapsep_q1_target.wav"),
        os.path.join(R3_OUT, "clapsep_q2_target.wav"),
        os.path.join(R3_OUT, "clapsep_n1_target.wav"),
        os.path.join(R3_OUT, "clapsep_n2_target.wav"),
    ] if os.path.exists(p)}
    print(f"evaluating {len(stems)} stems: {sorted(stems)}")

    for name, path in sorted(stems.items()):
        target = load32k(path)
        resid = mix - target
        tgt_on = high_flux_onsets(target)

        c_hit = match_to(clicks, tgt_on)
        t_hit = match_to(taiko, tgt_on)
        m_hit = match_to(ref_pk, tgt_on)
        mix_hit = match_to(tgt_on, mix_on)
        halluc = [float(t) for t, h in zip(tgt_on, mix_hit) if h is None]

        drop = [ratio_db(target, mix, *wb) for t, h in zip(clicks, c_hit)
                if h is not None and (wb := (int(max(0, (t - EV_HALF) * SR)),
                                             int(min(len(target), (t + EV_HALF) * SR))))[1] - wb[0] > 32]
        r_op = [ratio_db(target, mix, int(a * SR), int(b * SR)) for a, b in
                ((max(0, t - EV_HALF), t + EV_HALF) for t in clicks)]
        r_mus = [ratio_db(target, mix, int(a * SR), int(b * SR)) for a, b in music_heavy]
        r_qui = [ratio_db(target, mix, int(a * SR), int(b * SR)) for a, b in music_quiet]

        # frame-energy regression: target ~ mixture over all frames
        fe_m = np.array([db(mix[i * HOP:(i + 1) * HOP]) for i in range(fl)])
        fe_t = np.array([db(target[i * HOP:(i + 1) * HOP]) for i in range(fl)])
        A = np.vstack([fe_m, np.ones(fl)]).T
        slope, intercept = np.linalg.lstsq(A, fe_t, rcond=None)[0]
        r2 = float(1 - np.sum((fe_t - A @ [slope, intercept]) ** 2) /
                   np.sum((fe_t - fe_m.mean()) ** 2))

        corr = float(np.corrcoef(target, ref)[0, 1])
        corr_loud = float(np.mean([np.corrcoef(target[int(a * SR):int(b * SR)],
                                               ref[int(a * SR):int(b * SR)])[0, 1]
                                   for a, b in music_heavy]))

        # target-vs-mixture spectrogram similarity: correlation of flattened
        # log-magnitude STFTs (a uniform attenuator scores ~1.0)
        Sm = librosa.amplitude_to_db(np.abs(librosa.stft(mix, n_fft=N, hop_length=HOP)) + 1e-10)
        St = librosa.amplitude_to_db(np.abs(librosa.stft(target, n_fft=N, hop_length=HOP)) + 1e-10)
        spec_sim = float(np.corrcoef(Sm.ravel(), St.ravel())[0, 1])

        spec_png(os.path.splitext(name)[0], mix, target, resid)
        run = {
            "n_target_onsets": len(tgt_on),
            "click_retention": {"matched": int(np.sum([x is not None for x in c_hit])),
                                "rate": float(np.mean([x is not None for x in c_hit]))},
            "click_level_ratio_db": {"median": float(np.median(drop)),
                                     "p25": float(np.quantile(drop, 0.25)),
                                     "p75": float(np.quantile(drop, 0.75))} if drop else None,
            "taiko_leakage": {"matched": int(np.sum([x is not None for x in t_hit])),
                              "rate": float(np.mean([x is not None for x in t_hit]))},
            "music_onsets_in_target": int(np.sum([x is not None for x in m_hit])),
            "hallucinated_onsets": {"count": len(halluc), "times": halluc[:20]},
            "selectivity": {
                "R_op_db": float(np.median(r_op)),
                "R_music_heavy_db": float(np.median(r_mus)),
                "R_music_quiet_db": float(np.median(r_qui)),
                "selectivity_score_db": float(np.median(r_op) - np.median(r_mus)),
                "op_vs_quiet_db": float(np.median(r_op) - np.median(r_qui)),
                "op_vs_taiko_note": "see taiko_leakage",
            },
            "frame_energy_regression": {"slope": float(slope), "intercept_db": float(intercept),
                                        "r2": r2},
            "spec_sim_vs_mixture_logmag": spec_sim,
            "corr_target_vs_ref": corr,
            "corr_target_vs_ref_music_heavy": corr_loud,
            "energy_db": {"mixture": db(mix), "target": db(target), "residual": db(resid)},
        }
        report["runs"][name] = run
        s = run["selectivity"]
        print(f"\n=== {name} ===")
        print(f"  click retention {run['click_retention']['matched']}/{len(clicks)} "
              f"| click level {run['click_level_ratio_db']['median']:+.1f} dB | taiko "
              f"{run['taiko_leakage']['matched']}/{len(taiko)} | ref onsets "
              f"{run['music_onsets_in_target']}/{len(ref_pk)} | halluc {len(halluc)}")
        print(f"  R_op {s['R_op_db']:+.1f} | R_music_heavy {s['R_music_heavy_db']:+.1f} "
              f"| R_quiet {s['R_music_quiet_db']:+.1f} | SELECTIVITY {s['selectivity_score_db']:+.1f} dB "
              f"| op_vs_quiet {s['op_vs_quiet_db']:+.1f}")
        print(f"  regression slope {slope:.3f} intercept {intercept:+.1f} dB r2 {r2:.3f} "
              f"| corr(ref) {corr:.3f}/{corr_loud:.3f}")

    with open(os.path.join(LOG_DIR, "06_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nsaved logs/06_evaluation.json + spec_*.png")


if __name__ == "__main__":
    main()
