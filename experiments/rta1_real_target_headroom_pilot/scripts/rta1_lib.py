# RTA1 shared library: STFT tools with EXPLICIT axis conventions, media
# probing, hashing, and manifest privacy helpers.
#
# Axis convention (tested in tests/test_stft_axes.py): for 1-D waveform input,
# torch.stft returns (freq, time). Every frequency-dependent operation here
# indexes axis 0 for frequency and carries an explicit fftfreq-style mapping.
# This exists because the RTA-0 audit (docs/reviews/RTA0_INTEGRITY_AUDIT.md
# section 3.3) found a shipped loss implementation that treated S.size(1) of a
# 1-D-input STFT as frequency when it is time. Do not "simplify" these helpers
# back to size-based inference.
import hashlib
import json
import os
import re
import subprocess

import numpy as np

RTA1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(RTA1, "work")
INBOX = os.path.join(WORK, "capture_inbox")
PRIVATE = os.path.join(WORK, "private")
EXTRACTED = os.path.join(WORK, "extracted")
ORACLE = os.path.join(WORK, "oracle")
LISTENING = os.path.join(WORK, "listening")

MEDIA_EXTS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".opus",
              ".webm", ".mp4", ".mov", ".mkv", ".avi"}

CLIP_LEVEL = 0.99          # |sample| >= CLIP_LEVEL counts as clipped
SR = 32000                 # working rate for analysis copies (documented)


# ---------------------------------------------------------------- hashing ----

def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------- STFT (torch-based) --

def stft(x, n_fft=1024, hop=256):
    """Waveform (n,) float -> complex STFT (freq, time). 1-D in, 2-D out."""
    import torch

    assert x.ndim == 1, f"stft expects 1-D waveform, got {x.shape}"
    window = torch.hann_window(n_fft, device=x.device if torch.is_tensor(x) else "cpu")
    xa = torch.as_tensor(x, dtype=torch.float32)
    S = torch.stft(xa, n_fft, hop, window=window, return_complex=True)
    assert S.ndim == 2 and S.size(0) == n_fft // 2 + 1, tuple(S.shape)
    return S


def istft(S, n_fft=1024, hop=256, length=None):
    """Complex (freq, time) -> 1-D waveform of `length` samples."""
    import torch

    assert S.ndim == 2, f"istft expects (freq, time), got {tuple(S.shape)}"
    window = torch.hann_window(n_fft, device=S.device)
    y = torch.istft(S, n_fft, hop, window=window, length=length)
    return y


def stft_mag(x, n_fft=1024, hop=256):
    return stft(x, n_fft, hop).abs()


def bin_frequencies(n_fft, sr, device=None):
    """Exact bin center frequencies (Hz), shape (n_fft//2+1,). Use this instead
    of linspace-over-arbitrary-axis."""
    import torch

    return torch.linspace(0, sr / 2, n_fft // 2 + 1, device=device)


def band_profile_db(x, sr=SR, lo_hz=2000, hi_hz=9000, n_fft=1024, hop=256):
    """2-9 kHz style band energy (dB) with EXPLICIT frequency-axis mapping.

    Returns (band_db, full_band_share). Raises if the mask is empty, which is
    the loud failure mode the audit bug needed.
    """
    import librosa

    S = np.abs(librosa.stft(np.asarray(x, dtype=np.float32),
                            n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    assert S.shape[0] == len(freqs), (S.shape, freqs.shape)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    assert band.any(), f"empty band [{lo_hz}, {hi_hz}] for n_fft={n_fft}"
    e_band = float(S[band].sum())
    e_all = float(S.sum()) + 1e-20
    return 10.0 * np.log10(e_band / e_all + 1e-12), e_band / e_all


# ------------------------------------------------------------ audio decode ---

def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_to_float32(path, sr=None, max_channels=8):
    """Decode ANY media file to float32 PCM (n_samples, n_channels).

    Derived in-memory analysis copy only; the source file is opened read-only.
    Mono returns shape (n, 1).
    """
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path),
           "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le"]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd += ["-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    # channel count is unknown to ffmpeg pipe; recover via soundfile info when
    # possible, else assume 1 (wav/m4a mono/stereo handled by soundfile path)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        ch = info.channels
        if sr is None:
            sr_decoded = info.samplerate
        else:
            sr_decoded = sr
    except Exception:
        ch, sr_decoded = 1, (sr or SR)
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch)
    else:
        x = x.reshape(-1, 1)
    return x, int(sr_decoded)


def probe_media(path):
    """Probe duration/sample-rate/codec/channels. Prefers ffprobe; falls back
    to `ffmpeg -i` header output; cross-checks WAV-family via soundfile."""
    info = {"path_os": str(path), "size_bytes": os.path.getsize(path)}

    ffprobe = _which("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout)
            stream = next((s for s in data.get("streams", [])
                           if s.get("codec_type") == "audio"), {})
            info.update({
                "duration_s": _f(stream.get("duration"),
                                 _f(data.get("format", {}).get("duration"))),
                "sample_rate": _i(stream.get("sample_rate")),
                "channels": _i(stream.get("channels")),
                "codec": stream.get("codec_name"),
                "probe_source": "ffprobe",
            })
            _sf_crosscheck(path, info)
            return info

    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True)
    text = out.stderr
    m = re.search(r"Audio: (\w+)", text)
    info["codec"] = m.group(1) if m else None
    m = re.search(r"(\d+) Hz", text)
    info["sample_rate"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) channels", text)
    info["channels"] = int(m.group(1)) if m else None
    m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", text)
    if m:
        h, mn, s, cs = (int(g) for g in m.groups())
        info["duration_s"] = h * 3600 + mn * 60 + s + cs / 100.0
    info["probe_source"] = "ffmpeg -i"
    _sf_crosscheck(path, info)
    return info


def _sf_crosscheck(path, info):
    try:
        import soundfile as sf

        i = sf.info(str(path))
        info["sf_samplerate"] = int(i.samplerate)
        info["sf_channels"] = int(i.channels)
        info["sf_frames"] = int(i.frames)
        info["sf_duration_s"] = round(i.frames / i.samplerate, 3)
        if info.get("sample_rate") in (None, 0):
            info["sample_rate"] = info["sf_samplerate"]
        if info.get("channels") in (None, 0):
            info["channels"] = info["sf_channels"]
        if info.get("duration_s") in (None, 0):
            info["duration_s"] = info["sf_duration_s"]
    except Exception:
        pass


def _which(name):
    from shutil import which

    return which(name)


def _f(v, alt=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return alt


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------- clipping / levels ---

def clipping_stats(x):
    """Fraction of |samples| >= CLIP_LEVEL, per channel + overall."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    frac = [(float(np.mean(np.abs(x[:, c]) >= CLIP_LEVEL)))
            for c in range(x.shape[1])]
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return {"clip_frac_per_channel": [round(v, 6) for v in frac],
            "clip_frac_max": round(max(frac), 6),
            "peak_abs": round(peak, 6)}


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2) + 1e-20))


def rms_db(x):
    return 20.0 * np.log10(rms(x) + 1e-12)


# --------------------------------------------------------- privacy helpers ---

_ABS_PATH_RE = re.compile(
    r"([A-Za-z]:\\\\?[^\"\s]*|/home/[^\"\s]*|/Users/[^\"\s]*|/mnt/[^\"\s]*)")
_DEVICE_TOKENS = ("serial", "imei", "gps", "latitude", "longitude",
                  "phone_number", "owner_name")


def scan_private_text(obj):
    """Scan a decoded JSON object for absolute paths / private tokens.
    Returns list of findings (empty = clean)."""
    findings = []
    for k, v in _walk(obj):
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        for m in _ABS_PATH_RE.finditer(s or ""):
            findings.append({"field": "/".join(k), "match": m.group(0)})
        low = (k[-1] if k else "").lower()
        if any(t in low for t in _DEVICE_TOKENS):
            findings.append({"field": "/".join(k), "match": "private-key-name"})
    return findings


def _walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (f"[{i}]",))
    else:
        yield path, obj


def anonymize_take_entry(entry):
    """Public manifest entry: anonymized IDs + numbers only."""
    keep = ("session", "take_id_public", "type", "duration_s", "sample_rate",
            "channels", "codec", "sha256", "clip_frac_max", "peak_abs",
            "probe_source")
    out = {k: entry[k] for k in keep if k in entry}
    return out


def mono(x):
    """Documented reproducible mono derivation: mean of native channels."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float32)
    return x.mean(axis=1).astype(np.float32)
