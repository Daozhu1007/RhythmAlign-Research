# ISTFT/STFT roundtrip tests: analysis/resynthesis must be near-lossless.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rta1_lib as rl  # noqa: E402


def _signal(n, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rl.SR
    x = (0.3 * np.sin(2 * np.pi * 440 * t)
         + 0.1 * rng.standard_normal(n)
         + 0.2 * np.sin(2 * np.pi * 3000 * t) * (t > 1.0))
    return x.astype(np.float32)


def test_roundtrip_identity_various_windows():
    for n_fft in (512, 1024, 2048):
        x = _signal(4 * rl.SR)
        y = rl.istft(rl.stft(x, n_fft, n_fft // 4), n_fft, n_fft // 4,
                     length=len(x)).numpy()
        rel = float(np.max(np.abs(x - y)) / (np.max(np.abs(x)) + 1e-12))
        assert rel < 1e-4, (n_fft, rel)


def test_roundtrip_short_input():
    x = _signal(1000)
    y = rl.istft(rl.stft(x, 512, 128), 512, 128, length=len(x)).numpy()
    assert len(y) == len(x)
    assert float(np.max(np.abs(x - y))) < 1e-3


def test_target_roundtrip_oracle_floor():
    """The roundtrip oracle row must be essentially perfect (it is the floor
    every real method is measured against)."""
    from importlib import import_module

    bo = import_module("06_run_bounded_oracles")
    x = _signal(3 * rl.SR)
    row = bo.roundtrip_floor(x)
    assert row["recon_snr_db"] > 40.0, row


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("test_roundtrip: ALL PASS")
