# S1E shared helpers. Standalone (adapted from r1_common.py patterns); read-only w.r.t.
# all source media. All writes go inside the S1E experiment directory.
import json
import os
import re
import subprocess

import numpy as np
import soundfile as sf

EXP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK_DIR = os.path.join(EXP_DIR, "work")
CLIP_DIR = os.path.join(EXP_DIR, "clips")
OUT_DIR = os.path.join(EXP_DIR, "outputs")
DIAG_DIR = os.path.join(EXP_DIR, "diagnostics")
LOG_DIR = os.path.join(EXP_DIR, "logs")

HANCAM_MP4 = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4"
REF_MP3 = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\共感觉.mp3"

SR = 48000  # native rates of both sources

for _d in (WORK_DIR, CLIP_DIR, OUT_DIR, DIAG_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

import imageio_ffmpeg  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

_NO_WINDOW = {}
if os.name == "nt":
    _NO_WINDOW["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_ffmpeg(args, timeout=1800):
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y"] + args
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", timeout=timeout, **_NO_WINDOW,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-4000:]}")
    return proc.stderr


def decode_audio(src, dst, sr=None, channels=2, subtype="f32le"):
    """Decode first audio stream to float PCM WAV. Original media untouched."""
    args = ["-i", src, "-vn", "-map", "0:a:0", "-c:a", "pcm_" + subtype, "-ac", str(channels)]
    if sr:
        args += ["-ar", str(sr)]
    args.append(dst)
    run_ffmpeg(args)
    return dst


def load_wav(path):
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    return data, sr  # data shape (n, ch)


def save_wav(path, data, sr, subtype="FLOAT"):
    """data shape (n, ch) or (n,). No normalization applied here — gain policy is
    pass-through: outputs preserve the input's absolute scale unless a stage documents
    a fixed, disclosed gain."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if data.ndim == 1:
        data = data[:, None]
    sf.write(path, data, sr, subtype=subtype)
    return path


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def db(x, eps=1e-12):
    return 10.0 * np.log10(max(float(np.mean(np.square(x))), eps))


def mmss(t):
    m, s = divmod(float(t), 60.0)
    return f"{int(m):d}:{s:04.1f}"
