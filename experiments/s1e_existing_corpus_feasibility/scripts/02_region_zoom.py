# S1E step 02 - focused zoom diagnostics for candidate challenge windows + auto-scan
# for weak-tap / friction / speech-modulation candidates. Selection aid ONLY.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import signal as sig
from scipy.ndimage import uniform_filter1d, maximum_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s1e_common import WORK_DIR, DIAG_DIR, SR, load_wav, save_json, mmss

NATIVE = os.path.join(WORK_DIR, "handcam_native.wav")

FOCUSED = [
    ("amb_start",   2.5, 10.5),   # pre-reference ambience/attract, VAD bump ~2-5 s
    ("golden",      22.5, 37.5),  # known R1 golden window (dense, ref-covered)
    ("npc_a1",      86.0, 94.0),
    ("npc_a2",      94.0, 102.0),
    ("npc_a3",      102.0, 110.0),
    ("taiko_prompt", 125.5, 131.5),  # owner: loud taiko system prompt 2:08-2:09
    ("post_end",    148.5, 161.5),  # post-reference: announcer/jingle/ambience
]


def band_env(x, sr, lo, hi, hop_s=0.01):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    xb = sig.sosfiltfilt(sos, np.mean(x, axis=1))
    hop = int(hop_s * sr)
    n = len(xb) // hop
    return np.sqrt(np.mean(xb[: n * hop].reshape(n, hop) ** 2, axis=1)), hop_s


def vad_probs(x, sr):
    import torch, resampy
    from silero_vad import load_silero_vad
    model = load_silero_vad().to("cuda" if torch.cuda.is_available() else "cpu")
    y = np.mean(x, axis=1)
    sos = sig.butter(6, [80, 7500], btype="bandpass", fs=sr, output="sos")
    y = sig.sosfiltfilt(sos, y)
    y16 = resampy.resample(y.astype(np.float32), sr, 16000)
    t = torch.from_numpy(y16).cuda() if torch.cuda.is_available() else torch.from_numpy(y16)
    n = len(y16) // 512
    probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            probs[i] = float(model(t[i * 512:(i + 1) * 512], 16000).item())
    return 0.032, probs


def speech_modulation_score(x, sr, hop_s=0.25, win_s=2.0):
    """2-8 Hz AM energy in 300-3000 Hz band (speech-like modulation over music)."""
    sos = sig.butter(4, [300, 3000], btype="bandpass", fs=sr, output="sos")
    xb = sig.sosfiltfilt(sos, np.mean(x, axis=1))
    env = np.abs(sig.hilbert(xb))
    env = sig.sosfiltfilt(sig.butter(4, [2, 8], btype="bandpass", fs=sr, output="sos"), env)
    hop = int(hop_s * sr)
    n = (len(env)) // hop
    e = env[: n * hop].reshape(n, hop)
    am = np.sqrt(np.mean(e ** 2, axis=1))
    return am, hop_s


def zoom_window(name, t0, t1, y, vad_h, vad, outdir):
    i0, i1 = int(t0 * SR), int(t1 * SR)
    seg = y[i0:i1]
    m = (np.arange(len(vad)) * vad_h >= t0) & (np.arange(len(vad)) * vad_h < t1)
    v = vad[m]
    e_low, _ = band_env(seg, SR, 40, 150)
    e_mid, _ = band_env(seg, SR, 150, 2000)
    e_clk, _ = band_env(seg, SR, 2000, 9000)
    t = np.arange(len(e_low)) * 0.01

    fig, axes = plt.subplots(5, 1, figsize=(18, 12),
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1.2, 1.2]})
    f, ts, S = sig.spectrogram(np.mean(seg, axis=1), SR, nperseg=4096, noverlap=3584)
    Sd = 10 * np.log10(S + 1e-12)
    fm = f <= 12000
    axes[0].pcolormesh(ts + t0, f[fm], Sd[fm], vmin=-105, vmax=-25, shading="auto", cmap="magma")
    axes[0].set_ylabel("Hz")
    axes[0].set_title(f"{name} [{mmss(t0)}-{mmss(t1)}] video clock")
    axes[1].plot(t + t0, 10 * np.log10(e_low + 1e-12), lw=0.5)
    axes[1].set_ylabel("low dB")
    axes[2].plot(t + t0, 10 * np.log10(e_mid + 1e-12), lw=0.5)
    axes[2].set_ylabel("mid dB")
    axes[3].plot(t + t0, 10 * np.log10(e_clk + 1e-12), lw=0.5)
    axes[3].set_ylabel("clk dB")
    tv = np.arange(len(v)) * vad_h + t0
    axes[4].plot(tv, v, lw=0.8)
    axes[4].axhline(0.5, color="r", ls="--", lw=0.6)
    axes[4].set_ylabel("VAD")
    axes[4].set_xlabel("video time s")
    axes[4].set_ylim(0, 1)
    fig.tight_layout()
    p = os.path.join(outdir, f"zoom_{name}.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    stats = {
        "vad_s_gt02": float(np.sum(v > 0.2) * vad_h),
        "vad_s_gt05": float(np.sum(v > 0.5) * vad_h),
        "vad_mean": float(np.mean(v)) if len(v) else 0.0,
        "low_rms_db": float(10 * np.log10(np.mean(e_low ** 2) + 1e-12)),
        "mid_rms_db": float(10 * np.log10(np.mean(e_mid ** 2) + 1e-12)),
        "clk_rms_db": float(10 * np.log10(np.mean(e_clk ** 2) + 1e-12)),
    }
    print(name, stats)
    return stats


def main():
    y, sr = load_wav(NATIVE)
    vad_h, vad = vad_probs(y, SR)
    np.save(os.path.join(WORK_DIR, "vad_probs.npy"), vad)

    stats = {}
    for name, a, b in FOCUSED:
        stats[name] = zoom_window(name, a, b, y, vad_h, vad, DIAG_DIR)

    # auto-scan: speech modulation rank over 80-115 s (2 s bins)
    am, ah = speech_modulation_score(y, SR)
    t_am = np.arange(len(am)) * ah
    rows = []
    for t0 in np.arange(80, 113, 2.0):
        m = (t_am >= t0) & (t_am < t0 + 2.0)
        rows.append({"t0": float(t0), "am": float(np.mean(am[m]))})
    rows.sort(key=lambda r: -r["am"])
    print("top speech-AM bins 80-115s:", [(round(r["t0"], 1), round(r["am"], 5)) for r in rows[:8]])

    # auto-scan: sustained flatness stretches (friction candidates) 13-148 s
    sos = sig.butter(4, [500, 8000], btype="bandpass", fs=SR, output="sos")
    xf = sig.sosfiltfilt(sos, np.mean(y, axis=1))
    nfft, hopf = 4096, 2048
    win = sig.windows.hann(nfft, sym=False)
    nf = (len(xf) - nfft) // hopf + 1
    idx = np.arange(nfft)[None, :] + hopf * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(xf[idx] * win[None, :], axis=1)) + 1e-12
    flat = np.exp(np.mean(np.log(S), axis=1)) / np.mean(S, axis=1)
    tf = (np.arange(nf) * hopf + nfft / 2) / SR
    track = (tf > 15) & (tf < 147)
    fr = np.where(track, flat, 0)
    frs = uniform_filter1d(fr, 40)  # ~2 s
    cand = []
    tt = np.arange(len(frs)) * (hopf / SR)
    for t0 in np.arange(15, 145, 2.0):
        m = (tt >= t0) & (tt < t0 + 2.0)
        cand.append({"t0": float(t0), "flat2s": float(np.mean(frs[m]))})
    cand.sort(key=lambda r: -r["flat2s"])
    print("top flatness (friction) bins:", [(round(c["t0"], 1), round(c["flat2s"], 4)) for c in cand[:8]])

    # auto-scan: weak taps = click onsets whose peak is in the lowest quartile, 15-147 s
    from scipy.ndimage import maximum_filter1d
    xc = sig.sosfiltfilt(sig.butter(4, [2000, 9000], btype="bandpass", fs=SR, output="sos"),
                         np.mean(y, axis=1))
    hopn = int(0.01 * SR)
    n = len(xc) // hopn
    envc = np.sqrt(np.mean(xc[: n * hopn].reshape(n, hopn) ** 2, axis=1))
    tc = np.arange(n) * 0.01
    on = np.where((envc[1:-1] > envc[:-2]) & (envc[1:-1] >= envc[2:]) &
                  (envc[1:-1] > 3 * uniform_filter1d(envc, 300)[1:-1]) &
                  (tc[1:-1] > 15) & (tc[1:-1] < 147))[0] + 1
    peaks = envc[on]
    q1 = np.quantile(peaks, 0.25)
    weak = on[peaks <= q1]
    hist, edges = np.histogram(weak, bins=np.arange(15, 148, 5.0))
    print("weak-onset count per 5 s bin (15-147):", dict(zip(map(float, edges[:-1]), map(int, hist))))

    stats["scan_top_speech_am"] = rows[:8]
    stats["scan_top_flatness"] = cand[:8]
    stats["weak_onset_hist"] = {float(k): int(v) for k, v in zip(edges[:-1], hist)}
    save_json(os.path.join(WORK_DIR, "region_scan.json"), stats)
    print("saved region_scan.json")


if __name__ == "__main__":
    main()
