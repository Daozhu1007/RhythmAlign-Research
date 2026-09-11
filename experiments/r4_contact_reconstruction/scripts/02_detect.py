# R4 Phase 1 - semi-automatic contact candidate detection from precomputed frame metrics.
# Candidates: (a) bezel LED flash onsets per 8 sectors, (b) judgment-text onsets, (c) hand-motion spikes.
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(R4, "work")
FPS = 60.04

with open(os.path.join(WORK, "frame_metrics.json")) as f:
    M = json.load(f)
n = len(M)
t = np.arange(n) / FPS
diff = np.array([m["diff"] for m in M])
bezm = np.array([m["motion"] for m in M])
sector_keys = [f"b{(a % 360)}" for a in (-90 + 45 * k for k in range(8))]
# remap: our keys were int(round(deg)%360) of -90..135 -> 270,315,0,...,135
keys = ["b270", "b315", "b0", "b45", "b90", "b135", "b180", "b225"]
S = np.array([[m[k] for k in keys] for m in M])  # brightness per sector
names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# --- judgment text detector: re-decode judgment-zone crop is expensive; use diff spikes
# on the judgment zone only. Zone: upper-center-left region of rotated frame (480x270).
# We approximate using existing full-frame metrics only; judgment onsets come from
# the audio-side + visual strips. Here we rely on LED onsets as primary candidates.

def onsets(sig, thr_ratio=1.8, min_gap=0.10, floor=1.0):
    """local-rise onset detector: sig - median-filtered baseline > thr_ratio*noise"""
    med = np.array([np.median(sig[max(0, i - 30):i + 30]) for i in range(n)])
    resid = sig - med
    noise = np.array([np.std(sig[max(0, i - 60):i + 60]) + 1e-3 for i in range(n)])
    cand = np.where((resid > thr_ratio * noise) & (resid > floor))[0]
    ev = []
    last = -10**9
    for i in cand:
        if t[i] - last >= min_gap:
            ev.append(i)
            last = t[i]
    return ev

cands = {}
for si in range(8):
    ev = onsets(S[:, si], thr_ratio=1.5, min_gap=0.13, floor=2.0)
    cands[f"led_{names[si]}"] = [float(t[i]) for i in ev]

# motion spikes (hand activity bursts)
ms = onsets(bezm, thr_ratio=2.2, min_gap=0.18, floor=3.0)
cands["motion"] = [float(t[i]) for i in ms]

# LED sector time series plot
fig, ax = plt.subplots(8, 1, figsize=(16, 12), sharex=True)
for si in range(8):
    ax[si].plot(t, S[:, si], lw=0.7)
    ax[si].set_ylabel(names[si])
    for e in cands[f"led_{names[si]}"]:
        ax[si].axvline(e, color="r", lw=0.6, alpha=0.7)
ax[0].set_title("bezel sector brightness (red = flash-onset candidate)")
plt.tight_layout()
plt.savefig(os.path.join(WORK, "led_plot.png"), dpi=70)
plt.close()

fig2, ax2 = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
ax2[0].plot(t, diff, lw=0.7, label="full-frame diff")
ax2[0].plot(t, bezm, lw=0.7, label="bezel diff")
ax2[0].legend()
for e in cands["motion"]:
    ax2[0].axvline(e, color="r", lw=0.6, alpha=0.6)
ax2[1].plot(t, S.mean(axis=1), lw=0.7, color="g")
ax2[1].set_title("mean bezel brightness")
plt.tight_layout()
plt.savefig(os.path.join(WORK, "motion_plot.png"), dpi=70)
plt.close()

with open(os.path.join(WORK, "candidates.json"), "w") as f:
    json.dump(cands, f, indent=1)
for k, v in cands.items():
    print(k, len(v))
