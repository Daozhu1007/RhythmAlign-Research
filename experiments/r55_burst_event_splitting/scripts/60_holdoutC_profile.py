# R5.5 Phase 6 - Holdout-C selection. Audio onset-density profile over the full
# take + contact sheets for candidate zones; visual choice recorded with reason.
# Hard constraint: no overlap with [22.5, 37.5] (Dev-A) or [84, 102] (Dev-B).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sig

from common55 import R1_WORK, R55_OUT, R55_WORK, save_json
from r5_common import SR

FORBIDDEN = [(22.5, 37.5), (84.0, 102.0)]


def main():
    hand, _ = sf.read(os.path.join(R1_WORK, "handcam_native.wav"),
                      dtype="float64", always_2d=True)
    y = hand.mean(axis=1)
    dur = len(y) / SR
    print(f"handcam duration {dur:.1f} s")

    # onset envelope: 2-14 kHz flux (R4-family), 2 s bins
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
    ax.set_xlabel("handcam-native time (s)")
    ax.set_ylabel("onsets / 2 s bin")
    ax.set_title("onset density (red = forbidden dev windows)")
    fig.tight_layout()
    fig.savefig(os.path.join(R55_OUT, "holdoutC_density_profile.png"), dpi=100)
    for t0, d in zip(bins, dens):
        tag = " <dev>" if any(f0 - 2 <= t0 <= f1 for f0, f1 in FORBIDDEN) else ""
        print(f"  {t0:5.0f}s: {d:3.0f}{tag}")
    save_json(os.path.join(R55_WORK, "holdoutC_density.json"),
              {"bins_s": bins.tolist(), "onsets_per_2s": dens.tolist(),
               "forbidden": FORBIDDEN, "duration_s": dur})


if __name__ == "__main__":
    main()
