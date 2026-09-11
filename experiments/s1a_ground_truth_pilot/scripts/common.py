"""S1a shared environment: paths, CLAPSep signal-path configuration, IO, hashing.

The STFT configuration below is transcribed from the LOCAL vendored CLAPSep source
(inspected 2026-09-10, hashes recorded in 00_env_audit.py output):

  experiments/r3_audio_query/third_party/CLAPSep/model/CLAPSep.py
      self.stft  = STFT(n_fft=1024, hop_length=320, win_length=1024,
                        window='hann', center=True, pad_mode='reflect')
      self.istft = ISTFT(... same parameters ...)

  experiments/r3_audio_query/scripts/clapsep_lib.py
      MODEL_CONFIG = {..., "phase": False, ...}  -> bounded real sigmoid magnitude mask

Do not change these values from memory; they must match the vendored code, which is
verified numerically by scripts/92_stft_equivalence_check.py in the torch venv.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf

S1A_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_DIR = os.path.abspath(os.path.join(S1A_DIR, "..", ".."))
RAW_DIR = os.path.join(S1A_DIR, "raw")
FIXTURE_DIR = os.path.join(S1A_DIR, "fixtures")
LOG_DIR = os.path.join(S1A_DIR, "logs")
MANIFEST_DIR = os.path.join(S1A_DIR, "manifests")
LISTENING_DIR = os.path.join(S1A_DIR, "listening_pack")

WORK_SR = 32000  # CLAPSep working sample rate (Hz)

# Vendored CLAPSep signal path (torchlibrosa STFT/ISTFT), inspected 2026-09-10.
CLAPSEP_STFT = {
    "n_fft": 1024,
    "hop_length": 320,
    "win_length": 1024,
    "window": "hann",  # periodic (fftbins=True), as in librosa.filters.get_window
    "center": True,
    "pad_mode": "reflect",
}
CLAPSEP_MASK = {
    "type": "sigmoid_bounded_real_magnitude",
    "range": [0.0, 1.0],
    "phase": "mixture",
    "phase_config": False,  # MODEL_CONFIG["phase"] is False locally
}
# CLAPSep inference protocol (NOT part of the representation; oracles bypass it and
# evaluate whole clips; the peak rescale is documented in REPRESENTATION_DIAGNOSTICS.md).
CLAPSEP_INFERENCE_PROTOCOL = {
    "chunk_samples": 320000,  # 10 s at 32 kHz
    "peak_rescale": "if max(|x|)>1: x *= 0.9/max(|x|)",
}

# Vendored source files whose hashes pin the audited representation.
CLAPSEP_SOURCES = [
    os.path.join(REPO_DIR, "experiments", "r3_audio_query", "third_party", "CLAPSep",
                 "model", "CLAPSep.py"),
    os.path.join(REPO_DIR, "experiments", "r3_audio_query", "third_party", "CLAPSep",
                 "model", "CLAPSep_decoder.py"),
    os.path.join(REPO_DIR, "experiments", "r3_audio_query", "scripts", "clapsep_lib.py"),
]

EPS = 1e-10           # torchlibrosa magphase clamp
METRIC_EPS = 1e-10    # fixed metric floor (documented, never tuned)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath_safe(path: str, start: str) -> str:
    """relpath that survives cross-drive paths on Windows (falls back to absolute)."""
    try:
        return os.path.relpath(path, start).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read audio as float64 (native rate). Multi-channel stays multi-channel."""
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int, subtype: str = "FLOAT") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    sf.write(path, x.astype(np.float32), sr, subtype=subtype)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def db(x: float) -> float:
    return float(10.0 * np.log10(max(x, 1e-300)))


def log(*a) -> None:
    print("[s1a]", *a, flush=True)


TRUTH_TIERS = {
    "T0": "sensor/event evidence (video, contact channel); never exact waveform truth",
    "T1": "quiet isolated acoustic target recording at phone position",
    "T2": "exact constructed additive mixture y = s + n with stored components",
    "T3": "acoustic/device stress condition",
    "T4": "simultaneous real-world scene",
    "SYNTHETIC_FIXTURE": "synthetic tooling-validation media; no real-acoustic meaning",
}


def env_audit() -> dict:
    """Record the two environments used by S1a (primary + torch venv receipt)."""
    return {
        "primary_interpreter": sys.executable,
        "primary_python": sys.version,
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "torch_venv_python": os.path.join(
            REPO_DIR, "experiments", "r2_target_separation", ".venv", "Scripts",
            "python.exe"),
        "torch_venv_role": "runs 92_stft_equivalence_check.py only (torch+torchlibrosa)",
        "clapsep_env_note": (
            "laion_clap unavailable locally; NOT needed for S1a (no model inference, "
            "no training). Representation is audited from vendored source and "
            "numerically cross-validated against torchlibrosa in the R2 venv."),
    }
