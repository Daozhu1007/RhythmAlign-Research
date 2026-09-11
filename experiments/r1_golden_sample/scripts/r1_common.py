# R1 golden sample experiment - shared helpers.
# Read-only w.r.t. all source media; all writes go to the experiment work/outputs dirs.
import json
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(EXP_DIR, "outputs")
WORK_DIR = os.path.join(EXP_DIR, "work")

HANCAM_MP4 = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4"
REF_MP3 = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\共感觉.mp3"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

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


def probe_streams(path):
    """Parse `ffmpeg -i` output for format + stream info (no ffprobe available)."""
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-i", path]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", **_NO_WINDOW,
    )
    text = proc.stderr
    info = {"path": path, "format_start": None, "format_duration": None,
            "format_bit_rate": None, "streams": [], "raw": text}
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+).*?start:\s*([-\d.]+).*?bitrate:\s*([\d.]+\s*\w+)", text, re.S)
    if m:
        h, mn, s = m.group(1), m.group(2), m.group(3)
        info["format_duration"] = int(h) * 3600 + int(mn) * 60 + float(s)
        info["format_start"] = float(m.group(4))
        info["format_bit_rate"] = m.group(5)
    for sm in re.finditer(r"Stream #0:(\d+)(?:\[.*?\])?(?:\(\w+\))?: (.+)", text):
        idx, rest = sm.groups()
        entry = {"index": int(idx), "desc": rest.strip()}
        if "Audio:" in rest:
            entry["type"] = "Audio"
            am = re.search(r"(\d+)\s*Hz,\s*(\w+),\s*([a-z0-9|_]+?)(?:,\s*([\d.]+\s*kb/s))?", rest)
            if am:
                entry["sample_rate"] = int(am.group(1))
                entry["layout"] = am.group(2)
                entry["codec_fmt"] = am.group(3)
                entry["bit_rate"] = am.group(4)
        elif "Video:" in rest:
            entry["type"] = "Video"
            fm = re.search(r"([\d.]+)\s*fps", rest)
            if fm:
                entry["fps"] = float(fm.group(1))
        info["streams"].append(entry)
    return info


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
    """data shape (n, ch) or (n,)."""
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


def audio_stats(x):
    """x: (n, ch) float."""
    peak = float(np.max(np.abs(x)))
    clip999 = float(np.mean(np.abs(x) >= 0.999)) * 100.0
    clip99 = float(np.mean(np.abs(x) >= 0.99)) * 100.0
    return {
        "peak": peak,
        "peak_dbfs": 20 * np.log10(max(peak, 1e-12)),
        "rms_dbfs": db(x),
        "pct_at_fullscale": clip999,
        "pct_over_-0.1dB": clip99,
    }


def channel_relation(x):
    """Detect dual-mono / correlated L-R."""
    if x.shape[1] < 2:
        return {"channels": x.shape[1]}
    l, r = x[:, 0], x[:, 1]
    ln, rn = l - l.mean(), r - r.mean()
    denom = np.sqrt(np.mean(ln * ln) * np.mean(rn * rn))
    corr = float(np.mean(ln * rn) / denom) if denom > 0 else 0.0
    mid, side = 0.5 * (l + r), 0.5 * (l - r)
    return {
        "channels": 2,
        "lr_corr": corr,
        "mid_rms_dbfs": db(mid),
        "side_rms_dbfs": db(side),
        "side_mid_ratio_db": db(side) - db(mid),
        "l_r_level_db": db(l) - db(r),
    }
