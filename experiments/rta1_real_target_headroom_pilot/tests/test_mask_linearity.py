# Fixed-mask linearity tests: M(p+n) == Mp + Mn must hold in the STFT domain
# exactly and in the waveform domain to numeric tolerance. This underwrites the
# source-path decomposition (ORACLE_PROTOCOL.md).
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rta1_lib as rl  # noqa: E402

N_FFT, HOP = 1024, 256


def _pn(seed=3, seconds=3.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rl.SR)) / rl.SR
    p = (0.4 * np.sin(2 * np.pi * 2500 * t)
         * (np.sin(2 * np.pi * 1.3 * t) > 0)).astype(np.float32)
    n = (0.3 * np.sin(2 * np.pi * 120 * t)
         + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    return p, n


def test_stft_domain_linearity_exact():
    import torch

    p, n = _pn()
    Sp, Sn = rl.stft(p, N_FFT, HOP), rl.stft(n, N_FFT, HOP)
    rng = np.random.default_rng(0)
    M = torch.clamp(torch.as_tensor(
        rng.uniform(0, 1, tuple(Sp.shape)), dtype=torch.float32), 0, 1)
    left, right = M * (Sp + Sn), M * Sp + M * Sn
    scale = float(torch.max(torch.abs(Sp + Sn))) + 1e-12
    rel = float(torch.max(torch.abs(left - right))) / scale
    assert rel < 1e-6, rel  # exact algebra; float redistribution only


def test_waveform_domain_linearity_within_tolerance():
    from importlib import import_module

    bo = import_module("06_run_bounded_oracles")
    p, n = _pn()
    y = p + n
    M = bo.binwise_bounded_mask(p, y)
    _p_hat, _n_hat, lin_res = bo.source_path_analysis(M, p, n)
    assert lin_res < 1e-4, lin_res


def test_zero_and_full_masks():
    import torch

    p, n = _pn()
    y = p + n
    Sy = rl.stft(y, N_FFT, HOP)
    zero = torch.zeros_like(torch.abs(Sy))
    y0 = rl.istft(zero * Sy, N_FFT, HOP, length=len(y)).numpy()
    assert float(np.max(np.abs(y0))) == 0.0
    one = torch.ones_like(torch.abs(Sy))
    y1 = rl.istft(one * Sy, N_FFT, HOP, length=len(y)).numpy()
    assert float(np.max(np.abs(y1 - y))) < 1e-3


def test_mask_bounds_enforced_by_oracles():
    from importlib import import_module

    bo = import_module("06_run_bounded_oracles")
    p, n = _pn()
    y = p + n
    M = bo.binwise_bounded_mask(p, y)
    assert float(M.min()) >= 0.0 and float(M.max()) <= 1.0
    M2, _hist = bo.waveform_optimized_mask(y, p, iters=20)
    assert float(M2.min()) >= 0.0 and float(M2.max()) <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("test_mask_linearity: ALL PASS")
