# R4 - core analysis: audio onsets (multi-band), judgment-text onsets, LED flashes,
# and empirical A/V offset from matched pairs. Detection aid only; no output product.
import json
import os
import subprocess

import numpy as np
import librosa
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FF = imageio_ffmpeg.get_ffmpeg_exe()
VID = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs\golden_video.mp4"
RAW = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs\golden_raw.wav"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(R4, "work")
SR = 48000
FPS = 60.04

def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))

# ---------------- audio ----------------
y, sr = librosa.load(os.path.join(R1_WORK, "golden_ctx.wav"), sr=SR, mono=True)  # ctx 21s, golden=3..18
g = y[int(3.0 * SR):int(18.0 * SR)]  # golden 15s
ref, _ = librosa.load(os.path.join(R1_WORK, "ref_warp_fixed.wav"), sr=SR, mono=True)
r = ref[int(3.0 * SR):int(18.0 * SR)]
assert len(g) == len(r) == 15 * SR

# detection residual: golden minus best-gain aligned reference (LS gain, robust)
w = np.ones(len(g))
gg = g - g.mean()
rr = r - r.mean()
gain = float(np.dot(gg * w, rr * w) / np.dot(rr * w, rr * w))
resid = g - gain * r

bands = [(40, 150), (150, 500), (500, 2000), (2000, 6000), (6000, 14000)]
from scipy import signal as sig
kern = sig.windows.hann(97)
band_env = {}
for lo, hi in bands:
    # high bands on the detection residual (music-removed -> interaction clicks stand out);
    # low bands on raw (bass thump lives there, residual bass is unreliable)
    x = resid if lo >= 500 else g
    sosb = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    xb = sig.sosfiltfilt(sosb, x)
    env = np.abs(xb)
    env = sig.convolve(env, kern, mode="same")
    env /= env.max() + 1e-9
    band_env[f"{lo}_{hi}"] = env
    print("band", lo, hi, "peak", float(env.max()))

# onset peaks per band: local maxima above adaptive threshold, min gap 60ms
def pick_peaks(env, thr, min_gap=0.05):
    hops = env[1:-1]
    loc = sig.find_peaks(env, height=thr, distance=int(min_gap * SR))[0]
    return loc

onsets = {}
for k, env in band_env.items():
    loc = pick_peaks(env, 0.18)
    onsets[k] = loc / SR
json.dump({k: [float(t) for t in v] for k, v in onsets.items()},
          open(os.path.join(WORK, "audio_onsets.json"), "w"), indent=1)
print("audio onsets per band:", {k: len(v) for k, v in onsets.items()})

np.save(os.path.join(WORK, "golden_raw_bandenvs.npy"),
        np.stack([band_env[k] for k in band_env]))
np.save(os.path.join(WORK, "detection_residual.npy"), resid.astype(np.float32))
print("ls gain", gain)
