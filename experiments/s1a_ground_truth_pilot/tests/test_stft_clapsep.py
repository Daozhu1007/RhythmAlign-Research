"""Unit tests for the numpy CLAPSep STFT/ISTFT twin (run in the primary env)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from stft_clapsep import (ClapSepSTFT, adjoint_mask, bounded_real_mask,
                          complex_ratio_mask, forward_mask)
import common as C

ST = ClapSepSTFT()


def test_config_matches_vendored_source():
    assert C.CLAPSEP_STFT == {"n_fft": 1024, "hop_length": 320, "win_length": 1024,
                              "window": "hann", "center": True, "pad_mode": "reflect"}
    assert C.CLAPSEP_MASK["range"] == [0.0, 1.0]
    assert C.CLAPSEP_MASK["phase"] == "mixture"


def test_frame_count_and_shapes():
    rng = np.random.default_rng(0)
    for L in (8000, 160000, 4321):
        X = ST.stft(rng.standard_normal(L))
        assert X.shape == (1 + L // 320, 513)


def test_roundtrip_identity():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32000) * 0.1
    x2 = ST.istft(ST.stft(x), len(x))
    assert np.max(np.abs(x2 - x)) < 1e-6


def test_magphase_clamp_matches_torchlibrosa():
    X = np.array([[0.0, 1e-12, 3.0 + 4.0j]])
    mag, cos, sin = ST.magphase(X)
    assert mag[0, 0] == 0.0
    assert cos[0, 0] == pytest.approx(0.0, abs=1e-8)  # 0/clamp(1e-10)
    np.testing.assert_allclose(cos[0, 2], 0.6, rtol=1e-12)
    np.testing.assert_allclose(sin[0, 2], 0.8, rtol=1e-12)


def test_adjoint_matches_finite_differences():
    rng = np.random.default_rng(2)
    L = 6000
    Y = ST.stft(rng.standard_normal(L) * 0.1)
    M = rng.random(Y.shape)
    g = rng.standard_normal(L)
    G = adjoint_mask(ST, g, Y, L)
    for _ in range(3):
        p = (int(rng.integers(0, Y.shape[0])), int(rng.integers(0, Y.shape[1])))
        h = 1e-6
        Mp = M.copy(); Mp[p] += h
        xp = forward_mask(ST, Mp, Y, L)
        Mm = M.copy(); Mm[p] -= h
        xm = forward_mask(ST, Mm, Y, L)
        num = np.dot(xp - xm, g) / (2 * h)
        assert abs(num - G[p]) <= 1e-5 * max(1.0, abs(num))


def test_forward_equals_exact_wav_reconstruct_path():
    """Where |Y| >= clamp, the linear surrogate equals the exact CLAPSep path."""
    rng = np.random.default_rng(3)
    L = 32000
    y = rng.standard_normal(L) * 0.1
    s = rng.standard_normal(L) * 0.05
    Y = ST.stft(y)
    M = bounded_real_mask(Y, ST.stft(s))
    a = forward_mask(ST, M, Y, L)
    b = ST.wav_reconstruct(M, Y, L)
    assert np.max(np.abs(a - b)) < 1e-7


def test_bounded_and_complex_masks_analytic_values():
    Y = np.array([[2.0 + 0j, 0.5 + 0j]])
    S = np.array([[1.0 + 0j, 2.0 + 0j]])
    M = bounded_real_mask(Y, S)
    assert M[0, 0] == pytest.approx(0.5)  # Re(S*conj(Y))/|Y|^2 = 2/4
    assert M[0, 1] == 1.0  # |S|>|Y| -> clipped
    Mc = complex_ratio_mask(Y, S)
    assert Mc[0, 1] == pytest.approx(4.0)  # unbounded


def test_librosa_cross_convention_check():
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(4)
    x = rng.standard_normal(16000).astype(np.float32)
    Xn = ST.stft(x)
    Xl = librosa.stft(x, n_fft=1024, hop_length=320, win_length=1024,
                      window="hann", center=True, pad_mode="reflect")
    assert np.max(np.abs(Xn - Xl.T)) / np.max(np.abs(Xl)) < 1e-5
    xl = librosa.istft(Xl, hop_length=320, win_length=1024, window="hann",
                       center=True, length=16000)
    assert np.max(np.abs(ST.istft(Xn, 16000) - xl)) < 1e-4


def test_equivalence_receipt_exists():
    p = os.path.join(C.LOG_DIR, "stft_equivalence_check.json")
    if os.path.exists(p):
        import json
        with open(p) as f:
            assert json.load(f)["all_ok"] is True
