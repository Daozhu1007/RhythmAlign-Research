# R3 Phase 1 - rank R1's 53 player-click candidates by "query cleanliness":
# strong isolated transient + low local music energy, then extract handcam
# frames around the top candidates for VISUAL confirmation of player contact
# (per brief: onset detector alone must not decide identity).
#
# Metrics per candidate t (golden_raw is 48 kHz stereo, 15 s):
#   flux_peak      max positive 2-12 kHz spectral flux within +-40 ms (at 32 kHz
#                  mono analysis, same method as R1/R2 audits)
#   evt_db         golden mono RMS in [t-0.02, t+0.07]   (the transient itself)
#   music_db       aligned clean reference RMS in [t-0.15, t+0.15] (local music)
#   bg_db          golden mono RMS in flanking [t+-0.10..0.30] (local background)
#   snr_vs_music   evt_db - music_db
#   crowd          other click/taiko candidates within +-0.35 s
import json
import os
import subprocess

import imageio_ffmpeg
import librosa
import numpy as np
from scipy import signal as sig

R3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
WORK = os.path.join(R3, "work")
LOGS = os.path.join(R3, "logs")
SR = 32000
N, HOP = 1024, 256
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def high_flux(y):
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=N, hop_length=HOP)) + 1e-10
    Sdb = librosa.amplitude_to_db(S)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N)
    dS = np.maximum(0.0, np.diff(Sdb, axis=1))
    fh = dS[(freqs >= 2000) & (freqs < 12000)].sum(axis=0)
    return fh, (pk := None)  # noqa - flux curve only; peaks come from R1 table


def flux_at(fh, t):
    i = int(round(t * SR / HOP))
    i0, i1 = max(0, i - 6), min(len(fh), i + 7)  # ~+-48 ms
    return float(np.max(fh[i0:i1]))


def local_db(y, t, half, sr=SR):
    a, b = int(max(0, (t - half) * sr)), int(min(len(y), (t + half) * sr))
    if b - a < 16:
        return None
    return db(y[a:b])


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    gate = json.load(open(os.path.join(R1_WORK, "gate_diagnostics.json")))
    clicks = [c["t"] for c in gate["stereo"]["clicks"]]
    taiko = [c["t"] for c in gate["stereo"]["taiko"]]

    mix32, _ = librosa.load(os.path.join(R1_OUT, "golden_raw.wav"), sr=SR, mono=True)
    fh, _ = high_flux(mix32)
    fh = fh[0] if isinstance(fh, tuple) else fh

    ref, _ = librosa.load(os.path.join(R1_WORK, "ref_warp_fixed.wav"), sr=SR, mono=True)
    ref = ref[int(3.0 * SR):int(18.0 * SR)]
    assert len(ref) == len(mix32), (len(ref), len(mix32))

    rows = []
    for t in clicks:
        others = [u for u in clicks + taiko if abs(u - t) > 1e-6 and abs(u - t) < 0.35]
        evt = local_db(mix32, t, 0.045)
        bg_l = local_db(mix32, t - 0.20, 0.10)
        bg_r = local_db(mix32, t + 0.20, 0.10)
        bg = np.mean([x for x in (bg_l, bg_r) if x is not None]) if (bg_l or bg_r) else None
        mus = local_db(ref, t, 0.15)
        rows.append({
            "t": round(float(t), 4),
            "flux_peak": round(flux_at(fh, t), 1),
            "evt_db": None if evt is None else round(evt, 1),
            "bg_db": None if bg is None else round(bg, 1),
            "music_db": None if mus is None else round(mus, 1),
            "snr_vs_music": None if (evt is None or mus is None) else round(evt - mus, 1),
            "evt_vs_bg": None if (evt is None or bg is None) else round(evt - bg, 1),
            "crowd": len(others),
        })

    # rank: want high snr_vs_music, high flux, low crowding
    for r in rows:
        r["score"] = round(
            (r["snr_vs_music"] or -99) + 0.02 * (r["flux_peak"] or 0) - 2.0 * r["crowd"], 2)
    rows.sort(key=lambda r: -r["score"])

    with open(os.path.join(LOGS, "01_event_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)

    print(f"{'rank':>4} {'t(s)':>8} {'snr':>6} {'flux':>7} {'evt':>6} {'bg':>6} {'mus':>6} {'crwd':>4} {'score':>7}")
    for i, r in enumerate(rows):
        print(f"{i:>4} {r['t']:>8.3f} {r['snr_vs_music']:>6} {r['flux_peak']:>7} "
              f"{r['evt_db']:>6} {r['bg_db']:>6} {r['music_db']:>6} {r['crowd']:>4} {r['score']:>7}")

    # frame strips for the top 14 candidates: 5 frames, -80..+80 ms (60 fps)
    top = [r["t"] for r in rows[:14]]
    fdir = os.path.join(WORK, "frames")
    os.makedirs(fdir, exist_ok=True)
    vid = os.path.join(R1_OUT, "golden_video.mp4")
    for k, t in enumerate(top):
        outs = []
        for j, dt in enumerate((-0.080, -0.040, 0.0, 0.040, 0.080)):
            fp = os.path.join(fdir, f"cand_{k:02d}_t{t:.3f}_f{j}.jpg")
            ts = max(0.0, t + dt)
            subprocess.run(
                [FFMPEG, "-y", "-v", "error", "-ss", f"{ts:.4f}", "-i", vid,
                 "-frames:v", "1", "-q:v", "3", fp], check=True)
            outs.append(fp)
        # 5-up strip via ffmpeg hstack
        strip = os.path.join(fdir, f"strip_{k:02d}_t{t:.3f}.jpg")
        subprocess.run(
            [FFMPEG, "-y", "-v", "error"] + sum([["-i", p] for p in outs], []) +
            ["-filter_complex",
             "[0][1][2][3][4]hstack=5,scale=1600:-2,drawbox=y=0:h=4:w=iw:color=red",
             "-q:v", "4", strip], check=True)
    print(f"\nframe strips for top {len(top)} candidates -> {fdir}")


if __name__ == "__main__":
    main()
