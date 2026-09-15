# Source-path metric tests: with known p and n, verify the decomposition
# (same fixed mask applied separately to each path) against direct
# computation, plus the degenerate cases (identical sources, nuisance-only,
# zero signal, clipping-free gain).
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rta1_lib as rl  # noqa: E402


def _bo():
    from importlib import import_module

    return import_module("06_run_bounded_oracles")


def _material(seed=11, seconds=3.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rl.SR)) / rl.SR
    p = (0.35 * np.sin(2 * np.pi * 2800 * t)
         * (np.sin(2 * np.pi * 0.9 * t) > 0)).astype(np.float32)
    p[int(1.0 * rl.SR):int(2.0 * rl.SR)] += (
        0.02 * rng.standard_normal(int(1.0 * rl.SR))).astype(np.float32)
    n = (0.3 * np.sin(2 * np.pi * 100 * t)
         + 0.15 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    return p, n


def test_decomposition_matches_direct_masked_mixture():
    bo = _bo()
    p, n = _material()
    y = p + n
    M = bo.binwise_bounded_mask(p, y)
    p_hat, n_hat, lin_res = bo.source_path_analysis(M, p, n)
    y_hat = rl.istft(M * rl.stft(y), length=len(y)).numpy()
    assert float(np.max(np.abs((p_hat + n_hat) - y_hat))) / (
        np.max(np.abs(y_hat)) + 1e-12) < 1e-4


def test_identical_source_case():
    """p == n: a fixed mask must treat both paths identically."""
    bo = _bo()
    p, n = _material()
    y = p + p
    M = bo.binwise_bounded_mask(p, y)
    p_hat, n_hat, _ = bo.source_path_analysis(M, p, p)
    assert float(np.max(np.abs(p_hat - n_hat))) < 1e-5


def test_nuisance_only_case():
    """p == 0: the oracle mask is identically zero; both paths output silence."""
    bo = _bo()
    p, n = _material()
    y = n.copy()
    M = bo.binwise_bounded_mask(np.zeros_like(p), y)
    p_hat, n_hat, _ = bo.source_path_analysis(M, p, n)
    assert float(np.max(np.abs(p_hat))) == 0.0
    assert float(np.max(np.abs(n_hat))) == 0.0


def test_zero_signal_case():
    bo = _bo()
    z = np.zeros(rl.SR, np.float32)
    M = bo.binwise_bounded_mask(z, z)
    p_hat, _n_hat, lin_res = bo.source_path_analysis(M, z, z)
    assert float(np.max(np.abs(p_hat))) == 0.0
    assert lin_res < 1e-6


def test_clipping_free_fixed_gain():
    bo = _bo()
    rng = np.random.default_rng(5)
    p = (0.9 * np.tanh(3 * rng.standard_normal(rl.SR))).astype(np.float32)
    bad = (5.0 * p).astype(np.float32)          # would clip if played directly
    gained, g = bo.fixed_gain_no_clip(bad, p)
    assert float(np.max(np.abs(gained))) <= 1.0 + 1e-9, "clipping-free contract"
    assert bo.snr_db(p, gained) > bo.snr_db(p, bad), "gain must not hurt SNR"


def test_metrics_sanity():
    bo = _bo()
    rng = np.random.default_rng(2)
    p = rng.standard_normal(rl.SR).astype(np.float32) * 0.1
    assert bo.snr_db(p, p) > 200.0            # identity -> huge SNR
    assert bo.nmse_db(p, p.copy()) < -100.0   # identity; epsilon floor -120 dB
    n_hat = p * 0.1
    assert bo.nuisance_attenuation_db(p, n_hat) > 19.0  # 10x cut = 20 dB
    half = p * 0.5
    assert abs(bo.nuisance_attenuation_db(p, half) - 6.02) < 0.1


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("test_source_path_metrics: ALL PASS")
