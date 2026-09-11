"""S1a 02 — lossless working PCM extraction.

Per audio input: (a) native-rate float32 PCM copy (lossless container decode),
(b) 32 kHz float32 working copy for the CLAPSep domain via polyphase resampling
(scipy resample_poly, Kaiser-window FIR; factors reduced by gcd; documented).
Originals are never modified. Fixed gain: samples are only ever scaled by the
documented polyphase filter normalization (DC gain 1), never normalized.

Usage: python 02_extract_pcm.py [--root <dir>] [--workdir <dir>]
"""
import argparse
import os
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def extract(path: str, work_native: str, work_32k: str) -> dict:
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    C.write_wav(work_native, x, sr)
    rec = {"native_pcm": os.path.relpath(work_native, C.S1A_DIR).replace("\\", "/"),
           "native_sr": sr}
    if sr != C.WORK_SR:
        g = gcd(sr, C.WORK_SR)
        y = resample_poly(x, C.WORK_SR // g, sr // g, axis=0,
                          window=("kaiser", 8.0))
        rec.update({
            "working_pcm": os.path.relpath(work_32k, C.S1A_DIR).replace("\\", "/"),
            "working_sr": C.WORK_SR,
            "resampler": f"scipy.resample_poly up={C.WORK_SR // g} down={sr // g} "
                         "kaiser8 (zero-phase, DC gain 1); no normalization",
            "peak_before": float(np.max(np.abs(x))),
            "peak_after": float(np.max(np.abs(y))),
        })
        C.write_wav(work_32k, y, C.WORK_SR)
    else:
        rec.update({"working_pcm": os.path.relpath(work_native, C.S1A_DIR).replace("\\", "/"),
                    "working_sr": sr, "resampler": "none (native 32 kHz)"})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.RAW_DIR)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    tag = os.path.basename(root)
    workdir = args.workdir or os.path.join(C.S1A_DIR, "work", f"pcm_{tag}")

    entries = []
    for dirpath, _dirs, filenames in os.walk(root):
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() not in {".wav", ".flac", ".aif",
                                                       ".aiff", ".mp3", ".m4a",
                                                       ".ogg", ".opus", ".aac"}:
                continue
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, root)
            base = rel if rel.lower().endswith(".wav") else rel + ".wav"
            try:
                rec = extract(src,
                              os.path.join(workdir, "native", base),
                              os.path.join(workdir, "32k", base))
                rec["source"] = os.path.relpath(src, C.S1A_DIR).replace("\\", "/")
                entries.append(rec)
                C.log(f"extracted {rel}")
            except Exception as e:
                entries.append({"source": src, "error": str(e)})
                C.log(f"ERROR extracting {rel}: {e}")
    out = {"root": os.path.relpath(root, C.S1A_DIR).replace("\\", "/"),
           "workdir": os.path.relpath(workdir, C.S1A_DIR).replace("\\", "/"),
           "gain_convention": "fixed; no normalization at any point",
           "entries": entries}
    C.save_json(out, os.path.join(C.MANIFEST_DIR, f"pcm_extraction_{tag}.json"))
    C.log(f"wrote manifests/pcm_extraction_{tag}.json ({len(entries)} entries)")


if __name__ == "__main__":
    main()
