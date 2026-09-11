# S1E step 01 - decode sources once, then compute a full-timeline diagnostic map used
# ONLY to choose challenge clips objectively (not as a quality metric):
#   - band envelopes (taiko/bass, mid music/vocals, click band, friction flatness)
#   - Silero VAD speech probability (nuisance locator)
#   - click-band onset candidates
# Plus overview spectrogram strips (image inspection only, not evidence).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s1e_common import (
    HANCAM_MP4, REF_MP3, WORK_DIR, DIAG_DIR, LOG_DIR, SR,
    decode_audio, load_wav, save_json, db, mmss,
)

NATIVE = os.path.join(WORK_DIR, "handcam_native.wav")
REF_NATIVE = os.path.join(WORK_DIR, "ref_native.wav")

BANDS = {
    "low_40_150": (40, 150),      # taiko/bass + low music
    "mid_150_2000": (150, 2000),  # music body / vocals / speech
    "click_2k_9k": (2000, 9000),  # player clicks / taps / friction
}


def band_env(x, sr, lo, hi, hop_s=0.01):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    xb = sig.sosfiltfilt(sos, np.mean(x, axis=1))
    hop = int(hop_s * sr)
    n = len(xb) // hop
    e = np.sqrt(np.mean(xb[: n * hop].reshape(n, hop) ** 2, axis=1))
    return e, hop_s


def spectral_flatness_env(x, sr, lo=500, hi=8000, hop_s=0.05, nfft=2048):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    xb = sig.sosfiltfilt(sos, np.mean(x, axis=1))
    hop = int(hop_s * sr)
    win = sig.windows.hann(nfft, sym=False)
    n = max(0, (len(xb) - nfft) // hop + 1)
    flat = np.full(n, np.nan)
    for i in range(n):
        seg = xb[i * hop : i * hop + nfft] * win
        S = np.abs(np.fft.rfft(seg)) + 1e-12
        flat[i] = np.exp(np.mean(np.log(S))) / np.mean(S)
    return flat, hop_s


def vad_speech_prob(x, sr):
    """Silero VAD on 16k mono; returns (hop_s, probs) at 32 ms hop."""
    from silero_vad import load_silero_vad, get_speech_timestamps
    import torch
    model = load_silero_vad()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    y16 = np.mean(x, axis=1)
    sos = sig.butter(6, [80, 7500], btype="bandpass", fs=sr, output="sos")
    y16 = sig.sosfiltfilt(sos, y16)
    import resampy  # librosa dep, present
    y16k = resampy.resample(y16.astype(np.float32), sr, 16000)
    t = torch.from_numpy(y16k).to(dev)
    probs = np.zeros(int(np.ceil(len(y16k) / 512)), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(probs)):
            seg = t[i * 512 : (i + 1) * 512]
            if len(seg) < 512:
                break
            probs[i] = float(model(seg, SR_VAD).item())
    return 0.032, probs


SR_VAD = 16000


def main():
    # 1. decode once
    if not os.path.exists(NATIVE):
        decode_audio(HANCAM_MP4, NATIVE, sr=None, channels=2, subtype="f32le")
    if not os.path.exists(REF_NATIVE):
        decode_audio(REF_MP3, REF_NATIVE, sr=None, channels=2, subtype="f32le")
    y, sr_y = load_wav(NATIVE)
    r, sr_r = load_wav(REF_NATIVE)
    assert sr_y == SR and sr_r == SR
    dur = len(y) / SR
    print(f"handcam {len(y)/SR:.2f}s, ref {len(r)/SR:.2f}s")

    out = {"handcam_duration_s": dur, "bands": {}, "vad": {}}

    # 2. band envelopes
    envs = {}
    for name, (lo, hi) in BANDS.items():
        e, hop_s = band_env(y, SR, lo, hi)
        envs[name] = e
        out["bands"][name] = {"hop_s": hop_s, "rms_dbfs_curve_ref": db(e)}
    e_low = envs["low_40_150"] + 1e-12
    e_click = envs["click_2k_9k"] + 1e-12

    # 3. VAD speech probability
    hop_s, probs = vad_speech_prob(y, SR)
    out["vad"] = {"hop_s": hop_s, "n": int(len(probs)),
                  "speech_seconds_prob>0.5": float(np.sum(probs > 0.5) * hop_s)}

    # 4. click-band onsets (descriptive locator only)
    from scipy.ndimage import maximum_filter1d
    hop = int(0.01 * SR)
    xc = np.mean(y, axis=1)
    sos = sig.butter(4, [2000, 9000], btype="bandpass", fs=SR, output="sos")
    xclick = sig.sosfiltfilt(sos, xc)
    hopn = int(0.01 * SR)
    n = len(xclick) // hopn
    envc = np.sqrt(np.mean(xclick[: n * hopn].reshape(n, hopn) ** 2, axis=1))
    thresh = np.maximum(maximum_filter1d(envc, 150) * 0.25, np.median(envc) * 2)
    on = np.where((envc[1:-1] > thresh[1:-1]) & (envc[1:-1] >= envc[:-2]) & (envc[1:-1] > envc[2:]))[0] + 1
    out["click_onsets"] = [float(t) for t in (on * 0.01)]

    save_json(os.path.join(WORK_DIR, "timeline_diag.json"), out)

    # 5. overview plots (image inspection only)
    t = np.arange(len(envs["low_40_150"])) * 0.01
    fig, axes = plt.subplots(5, 1, figsize=(20, 14), sharex=True)
    axes[0].plot(t, 10 * np.log10(envs["low_40_150"] + 1e-12), lw=0.4)
    axes[0].set_ylabel("low 40-150 dB")
    axes[1].plot(t, 10 * np.log10(envs["mid_150_2000"] + 1e-12), lw=0.4)
    axes[1].set_ylabel("mid 150-2k dB")
    axes[2].plot(t, 10 * np.log10(envs["click_2k_9k"] + 1e-12), lw=0.4)
    axes[2].set_ylabel("click 2k-9k dB")
    tv = np.arange(len(probs)) * 0.032
    axes[3].plot(tv, probs, lw=0.5)
    axes[3].axhline(0.5, color="r", ls="--", lw=0.7)
    axes[3].set_ylabel("VAD speech prob")
    flat, fh = spectral_flatness_env(y, SR)
    tf = np.arange(len(flat)) * fh
    axes[4].plot(tf, flat, lw=0.4)
    axes[4].set_ylabel("flatness 0.5-8k")
    axes[4].set_xlabel("video time s")
    for ax in axes:
        for x in np.arange(0, dur, 10):
            ax.axvline(x, color="gray", lw=0.2, alpha=0.4)
    fig.suptitle("S1E full-timeline diagnostics (clip-selection aid only)")
    fig.savefig(os.path.join(DIAG_DIR, "timeline_overview.png"), dpi=110)
    print("saved", os.path.join(DIAG_DIR, "timeline_overview.png"))

    # spectrogram strips
    f, ts, S = sig.spectrogram(xc, SR, nperseg=4096, noverlap=2048)
    Sd = 10 * np.log10(S + 1e-12)
    strips = [(0, 60), (55, 115), (110, 163.1)]
    for i, (a, b) in enumerate(strips):
        m = (ts >= a) & (ts <= b)
        fig, ax = plt.subplots(figsize=(22, 7))
        ax.pcolormesh(ts[m], f, Sd[:, m], vmin=-100, vmax=-20, shading="auto", cmap="magma")
        ax.set_yscale("symlog", linthresh=1000)
        ax.set_ylabel("Hz")
        ax.set_xlabel("video time s")
        ax.set_title(f"spectrogram strip {a:.0f}-{b:.0f}s (video clock)")
        fig.savefig(os.path.join(DIAG_DIR, f"spec_strip_{i}_{a:.0f}_{b:.0f}.png"), dpi=100)
        plt.close(fig)
        print("saved strip", i)


if __name__ == "__main__":
    main()
