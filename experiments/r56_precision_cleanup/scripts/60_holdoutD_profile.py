# R5.6 Holdout D - step 1: full-take onset density profile; forbidden = the
# three dev windows. Candidate zones shortlisted for visual inspection.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sig

from common56 import R1_WORK, R56_OUT, R56_WORK, save_json
from r5_common import SR

FORBIDDEN = [(22.5, 37.5), (40.0, 58.0), (84.0, 102.0)]
ZONES = [(2.0, 22.0), (60.0, 82.0), (104.0, 134.0)]   # allowed, mid-song


def main():
    hand, _ = sf.read(os.path.join(R1_WORK, "handcam_native.wav"),
                      dtype="float64", always_2d=True)
    y = hand.mean(axis=1)
    dur = len(y) / SR
    print(f"handcam duration {dur:.1f} s")

    hop = 480
    sos = sig.butter(4, [2000, 14000], btype="bandpass", fs=SR, output="sos")
    yb = sig.sosfiltfilt(sos, y)
    env = np.abs(yb)
    env = np.convolve(env, np.ones(49) / 49, mode="same")
    flux = np.maximum(0, np.diff(env))
    bins = np.arange(0, dur, 2.0)
    dens = []
    for t0 in bins:
        a, b = int(t0 * SR / hop), int(min((t0 + 2) * SR / hop, len(flux)))
        seg = flux[a:b]
        med = np.median(seg) + 1e-9
        mad = np.median(np.abs(seg - med)) + 1e-9
        dens.append(float(np.sum(seg > med + 4 * mad)))
    dens = np.array(dens)

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.bar(bins, dens, width=1.8, color="steelblue")
    for f0, f1 in FORBIDDEN:
        ax.axvspan(f0, f1, color="red", alpha=0.25)
    for z0, z1 in ZONES:
        ax.axvspan(z0, z1, color="green", alpha=0.10)
    ax.set_xlabel("handcam-native time (s)")
    ax.set_ylabel("onsets / 2 s bin")
    ax.set_title("Holdout-D density (red = forbidden dev windows, green = allowed zones)")
    fig.tight_layout()
    fig.savefig(os.path.join(R56_OUT, "holdoutD_density_profile.png"), dpi=100)
    for t0, d in zip(bins, dens):
        tag = " <dev>" if any(f0 - 2 <= t0 <= f1 for f0, f1 in FORBIDDEN) else ""
        print(f"  {t0:5.0f}s: {d:3.0f}{tag}")
    save_json(os.path.join(R56_WORK, "holdoutD_density.json"),
              {"bins_s": bins.tolist(), "onsets_per_2s": dens.tolist(),
               "forbidden": FORBIDDEN, "zones": ZONES, "duration_s": dur})


if __name__ == "__main__":
    main()
