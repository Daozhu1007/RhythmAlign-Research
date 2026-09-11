# S1E step 08 - S0-Specialist multi-stage deterministic pipeline.
# Motivated by measured facts, not tuned:
#   - waveform cancellation of the known pristine music yields only ~0.4-0.8 dB
#     (phone transfer incoherent in phase, r1 ceiling 5.9% coherent share) -> use a
#     MAGNITUDE-domain music estimate + Wiener-style gain instead of subtraction.
#   - Silero VAD does not fire on music-masked NPC speech (~0 prob) but does on the
#     clear end-of-video announcer -> fixed soft gate G_B = 1 - 0.7*p (p = VAD prob).
# Stages:
#   A (ref-covered clips): local delay refine (+/-10 ms around r1 d*), warp ref,
#      per-bin |H(f)| = freq-smoothed median |Y|/|R| over ref-active frames,
#      music estimate M = |H|*|R_warp|, gain G_A = max(10^(-18/20),
#      sqrt(max(0, 1 - M^2/|Y|^2))). STFT 2048/512 hann. No phase subtraction.
#   B: VAD soft gate on 32 ms grid (all clips).
# Output: same sample rate as input master (48k mono), gain untouched.
# All parameters above are fixed a priori and disclosed; no listening-based tuning.
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf
import resampy
from scipy import signal as sig
from scipy.ndimage import median_filter

from s1e_common import WORK_DIR, CLIP_DIR, OUT_DIR, LOG_DIR, SR, load_wav, save_wav, save_json
from align_cancel import cancel_clip

NFFT, HOP = 2048, 512
GAIN_FLOOR_DB = -18.0
LAMBDA = 1.0
VAD_GATE_DEPTH = 0.7
REFINE_SPAN_MS = 10.0


def istft(S, nfft=NFFT, hop=HOP):
    win = sig.windows.hann(nfft, sym=False)
    n_frames = S.shape[1]
    n_out = (n_frames - 1) * hop + nfft
    out = np.zeros(n_out)
    wsum = np.zeros(n_out)
    frames = np.fft.irfft(S, n=nfft, axis=0)
    w2 = win ** 2
    for t in range(n_frames):
        s = t * hop
        out[s:s + nfft] += frames[:, t] * win
        wsum[s:s + nfft] += w2
    interior = wsum[nfft:n_out - nfft] if n_out > 2 * nfft else wsum
    cola = float(np.median(interior))
    return out / max(cola, 1e-12)


def stft_mag(x, nfft=NFFT, hop=HOP):
    win = sig.windows.hann(nfft, sym=False)
    n_frames = (len(x) - nfft) // hop + 1
    idx = np.arange(nfft)[None, :] + hop * np.arange(n_frames)[:, None]
    S = np.fft.rfft(x[idx] * win[None, :], axis=1).T  # (bins, frames)
    return S, n_frames


def vad_probs_16k(x, sr):
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad().to("cuda" if torch.cuda.is_available() else "cpu")
    dev = next(model.parameters()).device
    sos = sig.butter(6, [80, 7500], btype="bandpass", fs=sr, output="sos")
    y = sig.sosfiltfilt(sos, x)
    y16 = resampy.resample(y.astype(np.float32), sr, 16000)
    t = torch.from_numpy(y16).to(dev)
    n = len(y16) // 512
    p = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            p[i] = float(model(t[i * 512:(i + 1) * 512], 16000).item())
    return p  # 32 ms hop


def process(clip, y, r):
    t0, t1, refc = clip["t_start_video_s"], clip["t_end_video_s"], clip["reference_covered"]
    i0, i1 = int(round(t0 * SR)), int(round(t1 * SR))
    seg = y[i0:i1]
    xm = np.mean(seg, axis=1)
    info = {"stages": []}

    if refc:
        m = 1.0
        res = cancel_clip(y[int((t0 - m) * SR):int((t1 + m) * SR)], r, t0 - m,
                          span_ms=REFINE_SPAN_MS)
        rw = res["ref_warp"]
        # trim margin to exact clip length
        lo = int(m * SR)
        rw = rw[lo:lo + len(xm)]
        rm = np.mean(rw, axis=1)

        Y, nf = stft_mag(xm)
        R, _ = stft_mag(rm)
        magY = np.abs(Y) + 1e-12
        magR = np.abs(R) + 1e-12
        # ref-active frames per bin
        act = magR > np.percentile(magR, 60, axis=1, keepdims=True) * 1.0
        ratio = magY / magR
        Hf = np.array([np.median(ratio[b][act[b]]) if act[b].sum() > 8 else 0.0
                       for b in range(magY.shape[0])])
        Hf = median_filter(Hf, size=5)
        Hf = np.clip(Hf, 0.0, 4.0)
        M = Hf[:, None] * magR
        G = np.sqrt(np.maximum(0.0, 1.0 - LAMBDA * (M ** 2) / (magY ** 2)))
        floor = 10 ** (GAIN_FLOOR_DB / 20)
        G = np.maximum(G, floor)
        # temporal smoothing of the power gain (2 frames)
        G = np.sqrt(np.maximum(0.0, 0.5 * G + 0.25 * np.roll(G, 1, axis=1)
                               + 0.25 * np.roll(G, -1, axis=1)))
        Xh = G * Y
        xm_a = istft(Xh, nfft=NFFT, hop=HOP)
        xm_a = np.concatenate([xm_a, np.zeros(len(xm) - len(xm_a))]) if len(xm_a) < len(xm) \
            else xm_a[:len(xm)]
        info["stages"].append("A_reference_wiener")
        info["mean_gain_A_db"] = float(20 * np.log10(np.mean(G) + 1e-12))
    else:
        xm_a = xm
        info["stages"].append("A_skipped_no_reference")

    # Stage B: VAD soft gate
    p = vad_probs_16k(xm_a, SR)
    hop_v = 0.032
    n_frames_v = len(p)
    p_up = np.interp(np.arange(len(xm_a)) / SR, (np.arange(n_frames_v) + 0.5) * hop_v, p,
                     left=0, right=0)
    k = int(0.05 * SR)
    p_s = np.convolve(p_up, np.ones(k) / k, mode="same")
    G_B = 1.0 - VAD_GATE_DEPTH * np.clip(p_s, 0.0, 1.0)
    xm_b = xm_a * G_B
    info["stages"].append("B_vad_soft_gate")
    info["vad_gate_mean_gain_db"] = float(20 * np.log10(np.mean(G_B) + 1e-12))
    info["vad_max_prob"] = float(np.max(p))
    return xm_b, info


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def main():
    t00 = time.time()
    y, _ = load_wav(os.path.join(WORK_DIR, "handcam_native.wav"))
    r, _ = load_wav(os.path.join(WORK_DIR, "ref_native.wav"))
    manifest = json.load(open(os.path.join(CLIP_DIR, "clip_manifest.json"), encoding="utf-8"))
    runs = []
    for clip in manifest["clips"]:
        cid = clip["id"]
        t0 = time.time()
        out, info = process(clip, y, r)
        p48 = os.path.join(OUT_DIR, "audio", f"specialist_{cid}_target.wav")
        save_wav(p48, out.astype(np.float32), SR)
        x16 = resampy.resample(out.astype(np.float32), SR, 32000)
        sf.write(os.path.join(OUT_DIR, "audio", f"specialist_{cid}_target_32k.wav"),
                 x16, 32000, subtype="FLOAT")
        mix, _ = load_wav(os.path.join(CLIP_DIR, "audio", f"{cid}_48k_stereo.wav"))
        info.update({"run": f"specialist_{cid}", "clip": cid,
                     "mix_rms_db": db(np.mean(mix, axis=1)), "out_rms_db": db(out),
                     "out_gain_dev_db": db(out) - db(np.mean(mix, axis=1)),
                     "runtime_s": time.time() - t0})
        runs.append(info)
        print(f"{cid}: stages={info['stages']} gain_dev {info['out_gain_dev_db']:+.1f} dB "
              f"({time.time()-t0:.0f}s)")
    save_json(os.path.join(LOG_DIR, "08_specialist_runs.json"),
              {"params": {"nfft": NFFT, "hop": HOP, "gain_floor_db": GAIN_FLOOR_DB,
                          "lambda": LAMBDA, "vad_gate_depth": VAD_GATE_DEPTH,
                          "refine_span_ms": REFINE_SPAN_MS},
               "runs": runs})
    print(f"done in {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
