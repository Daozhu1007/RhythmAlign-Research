# S1E step 10 - comparison spectrograms for named/failure cases.
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf
import resampy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sig

from s1e_common import CLIP_DIR, OUT_DIR, DIAG_DIR, SR

PANELS = {
    "c1_speech_npc_a": ["raw", "clapsep", "audiosep", "specialist", "soloaudio"],
    "c3_taiko_prompt": ["raw", "clapsep", "audiosep", "specialist", "soloaudio"],
    "c5_announcer_end": ["raw", "clapsep", "specialist", "soloaudio"],
    "c6_dense_golden": ["raw", "clapsep", "audiosep", "specialist", "soloaudio"],
}

SRC = {
    "raw": ("{cid}_48k_stereo.wav", os.path.join(CLIP_DIR, "audio"), None),
    "clapsep": ("clapsep_aq_{cid}_target.wav", os.path.join(OUT_DIR, "audio"), 32000),
    "audiosep": ("audiosep_p2_{cid}_target.wav", os.path.join(OUT_DIR, "audio"), 32000),
    "soloaudio": ("soloaudio_aq_{cid}_target.wav", os.path.join(OUT_DIR, "audio"), 24000),
    "specialist": ("specialist_{cid}_target.wav", os.path.join(OUT_DIR, "audio"), None),
}

FIX_GAIN = json.load(open(os.path.join(os.path.dirname(__file__), "..", "logs",
                                       "listening_gains.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "logs",
                                   "listening_gains.json")) else {}


def main():
    manifest = json.load(open(os.path.join(CLIP_DIR, "clip_manifest.json"), encoding="utf-8"))
    for cid, methods in PANELS.items():
        fig, axes = plt.subplots(len(methods), 1, figsize=(16, 2.6 * len(methods)))
        for ax, meth in zip(axes, methods):
            fname, d, srt = SRC[meth]
            x, sr = sf.read(os.path.join(d, fname.format(cid=cid)), dtype="float64",
                            always_2d=True)
            x = np.mean(x, axis=1)
            if srt and sr != srt:
                x = resampy.resample(x.astype(np.float32), sr, srt)
                sr = srt
            if meth in FIX_GAIN.get(cid, {}):
                x = x * FIX_GAIN[cid][meth]["gain"]
            f, ts, S = sig.spectrogram(x, sr, nperseg=2048, noverlap=1536)
            Sd = 10 * np.log10(S + 1e-12)
            fm = f <= 12000
            ax.pcolormesh(ts, f[fm], Sd[fm], vmin=-105, vmax=-25, shading="auto", cmap="magma")
            ax.set_ylabel(f"{meth}\nHz", fontsize=8)
            ax.set_title(f"{meth} (gain x{FIX_GAIN.get(cid, {}).get(meth, {}).get('gain', 1.0):.3f})"
                         if FIX_GAIN else meth, fontsize=9)
        axes[-1].set_xlabel("clip time s")
        fig.suptitle(cid)
        fig.tight_layout()
        p = os.path.join(DIAG_DIR, f"panel_{cid}.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)
        print("saved", p)


if __name__ == "__main__":
    main()
