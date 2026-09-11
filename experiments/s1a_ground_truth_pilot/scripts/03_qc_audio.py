"""S1a 03 — audio QC: clipping, DC, rest floor, RMS/spectrum bands, truth flags.

Usage: python 03_qc_audio.py [--pcm-json <manifests/pcm_extraction_*.json>]
QC is computed on the 32 kHz working PCM (CLAPSep domain).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FULLSCALE = 1.0


def qc_one(path: str, sr: int) -> dict:
    x, sr = C.read_wav(path)
    mono = C.to_mono(x)
    n = len(mono)
    peak = float(np.max(np.abs(mono))) if n else 0.0
    dc = float(np.mean(mono)) if n else 0.0
    k = max(1, int(0.05 * sr))
    m = n // k
    frame_rms = (np.sqrt(np.mean(mono[:m * k].reshape(m, k) ** 2, axis=1))
                 if m else np.array([0.0]))
    clip_s = float(np.mean(np.abs(mono) >= 0.999 * FULLSCALE))
    # codec full-scale = long runs of identical max samples
    runs = np.mean(np.abs(np.diff(np.abs(mono) >= 0.999 * FULLSCALE).astype(int)) == 0) if n > 1 else 0.0
    rest_floor_dbfs = float(C.db(np.percentile(frame_rms, 5) ** 2))
    rms_dbfs = float(C.db(np.mean(mono ** 2)))
    X = np.fft.rfft(mono)
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    bands = {}
    for lo, hi in ((0, 100), (100, 1000), (1000, 8000), (8000, sr / 2)):
        msk = (fr >= lo) & (fr < hi)
        bands[f"{lo}_{int(hi)}Hz"] = float(C.db(np.sum(np.abs(X[msk]) ** 2)))
    flags = []
    if clip_s > 1e-4:
        flags.append("clipped")
    if rest_floor_dbfs > -45:
        flags.append("high_floor")
    return {
        "path": os.path.relpath(path, C.S1A_DIR).replace("\\", "/"),
        "sr": sr, "n_samples": n, "channels": int(x.shape[1]),
        "peak": peak, "clip_fraction": clip_s, "codec_fullscale_runs_frac": float(runs),
        "dc_offset": dc, "rms_dbfs": rms_dbfs,
        "rest_floor_dbfs_p5": rest_floor_dbfs,
        "band_energy_db": bands, "flags": flags,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcm-json", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag
    pcm_json = args.pcm_json
    if pcm_json is None:
        cands = sorted(glob.glob(os.path.join(C.MANIFEST_DIR, "pcm_extraction_*.json")))
        assert cands, "no pcm extraction manifest found; run 02 first"
        pcm_json = cands[-1]
    tag = tag or os.path.basename(pcm_json).replace("pcm_extraction_", "").replace(".json", "")
    man = C.load_json(pcm_json)
    s1a_root = C.S1A_DIR
    out = {"tag": tag, "gain_convention": "fixed; no normalization", "files": []}
    for e in man["entries"]:
        if "error" in e:
            continue
        wp = os.path.join(s1a_root, e["working_pcm"])
        rec = qc_one(wp, e["working_sr"])
        rec["source"] = e["source"]
        out["files"].append(rec)
    path = os.path.join(C.MANIFEST_DIR, f"qc_audio_{tag}.json")
    C.save_json(out, path)
    C.log(f"QC: {len(out['files'])} files -> {path}")
    for f in out["files"]:
        C.log(f"  {os.path.basename(f['path'])}: peak {f['peak']:.3f}, "
              f"clip {f['clip_fraction']:.2e}, floor {f['rest_floor_dbfs_p5']:.1f} dBFS, "
              f"flags={f['flags']}")


if __name__ == "__main__":
    main()
