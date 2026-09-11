# S1E step 03 - full-timeline music-residual scan (locates non-reference content:
# taiko prompt, NPC speech, interactions) and quality check of A2-style cancellation
# away from the r1 golden window. Clip-selection aid ONLY.
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s1e_common import WORK_DIR, DIAG_DIR, SR, load_wav, save_json, save_wav, mmss
from align_cancel import cancel_clip

NATIVE = os.path.join(WORK_DIR, "handcam_native.wav")
REFN = os.path.join(WORK_DIR, "ref_native.wav")

# 25 s chunks covering the reference-covered span [11.4 .. 150.9] with margins
CHUNKS = [(10.0, 35.0), (35.0, 60.0), (60.0, 85.0), (85.0, 110.0),
          (110.0, 135.0), (135.0, 150.5)]


def main():
    y, _ = load_wav(NATIVE)
    r, _ = load_wav(REFN)

    resid_chunks = []
    meta = []
    t_start = time.time()
    for (a, b) in CHUNKS:
        i0, i1 = int(a * SR), int(b * SR)
        res = cancel_clip(y[i0:i1], r, a)
        print(f"chunk {a:.0f}-{b:.0f}: local_delay={res['local_delay_ms']*1000:.2f} ms "
              f"ncc={res['local_ncc']:.3f}  ({time.time()-t_start:.0f}s)")
        y_en = float(10 * np.log10(np.mean(y[i0:i1] ** 2) + 1e-12))
        e_en = float(10 * np.log10(np.mean(res["residual"] ** 2) + 1e-12))
        meta.append({"chunk": [a, b], "local_delay_ms": res["local_delay_ms"] * 1000,
                     "ncc": res["local_ncc"], "y_rms_db": y_en, "resid_rms_db": e_en})
        resid_chunks.append((a, b, res["residual"]))
        np.save(os.path.join(WORK_DIR, f"resid_{a:.0f}_{b:.0f}.npy"),
                res["residual"].astype(np.float32))

    save_json(os.path.join(WORK_DIR, "residual_scan_meta.json"), {"chunks": meta})

    # plot residual envelopes + spectrograms of two key regions
    fig, axes = plt.subplots(len(resid_chunks) + 1, 1, figsize=(20, 12), sharex=True)
    for i, (a, b, res) in enumerate(resid_chunks):
        x = np.mean(res, axis=1)
        hop = int(0.01 * SR)
        n = len(x) // hop
        env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
        axes[i].plot(np.arange(n) * 0.01 + a, 10 * np.log10(env + 1e-12), lw=0.4)
        axes[i].set_ylabel(f"{a:.0f}-{b:.0f}s dB")
        axes[i].set_ylim(-80, 0)
    hop = int(0.01 * SR)
    xm = np.mean(y, axis=1)
    n = len(xm) // hop
    envm = np.sqrt(np.mean(xm[: n * hop].reshape(n, hop) ** 2, axis=1))
    axes[-1].plot(np.arange(n) * 0.01, 10 * np.log10(envm + 1e-12), lw=0.4)
    axes[-1].set_ylabel("mix dB")
    fig.suptitle("residual after pristine-reference FIR cancellation (per 25 s chunk)")
    fig.savefig(os.path.join(DIAG_DIR, "residual_scan.png"), dpi=110)
    print("saved residual_scan.png")

    # zoom spectrograms of residual around suspected taiko prompt 126-132 s
    res = cancel_clip(y[int(124 * SR):int(134 * SR)], r, 124.0)
    xr = np.mean(res["residual"], axis=1)
    print(f"taiko zoom chunk: delay {res['local_delay_ms']*1000:.2f} ms ncc {res['local_ncc']:.3f}")
    f, ts, S = sig.spectrogram(xr, SR, nperseg=4096, noverlap=3584)
    Sd = 10 * np.log10(S + 1e-12)
    fm = f <= 12000
    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    axes[0].pcolormesh(ts + 124, f[fm], Sd[fm], vmin=-105, vmax=-25, shading="auto", cmap="magma")
    axes[0].set_ylabel("Hz"); axes[0].set_title("RESIDUAL 124-134 s (taiko prompt hunt)")
    xm2 = np.mean(y[int(124 * SR):int(134 * SR)], axis=1)
    f2, ts2, S2 = sig.spectrogram(xm2, SR, nperseg=4096, noverlap=3584)
    S2d = 10 * np.log10(S2 + 1e-12)
    axes[1].pcolormesh(ts2 + 124, f2[fm], S2d[fm], vmin=-105, vmax=-25, shading="auto", cmap="magma")
    axes[1].set_ylabel("Hz"); axes[1].set_xlabel("video time s"); axes[1].set_title("MIXTURE 124-134 s")
    fig.savefig(os.path.join(DIAG_DIR, "taiko_residual_zoom.png"), dpi=100)
    save_wav(os.path.join(WORK_DIR, "resid_taiko_zoom_124_134.wav"), res["residual"], SR)
    print("saved taiko_residual_zoom.png")


if __name__ == "__main__":
    main()
