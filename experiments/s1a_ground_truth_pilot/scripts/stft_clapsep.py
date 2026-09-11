"""Numpy reproduction of the LOCAL CLAPSep signal path + oracle diagnostics.

Reproduced from the vendored source (inspected 2026-09-10):

- torchlibrosa STFT: n_fft=1024, hop=320, win=1024, periodic hann, center=True,
  reflect padding -> (F, 513) complex (real/imag in the original code).
- torchlibrosa magphase: mag=sqrt(re^2+im^2); cos=re/clamp(mag,1e-10); sin=im/clamp(mag,1e-10).
- CLAPSep.wav_reconstruct: mag_y = relu(mag_x * mask); pred = ISTFT(mag_y*cos, mag_y*sin,
  length=input_length). torchlibrosa ISTFT: per-frame irfft * window, overlap-add,
  divide by clipped window^2 overlap sum (clamp 1e-11), trim n_fft//2 then take `length`.
- Decoder mask: torch.sigmoid -> bounded real magnitude mask in [0, 1] (phase=False).

Equivalence with the actual torchlibrosa modules is verified numerically in the R2 venv
by scripts/92_stft_equivalence_check.py (receipt: logs/stft_equivalence_check.json).

Oracle definitions (per stage instructions):

  BOUNDED_REAL_MASK_ORACLE      M* = clip(Re(S*conj(Y)) / (|Y|^2 + eps), 0, 1)
  OPTIMIZED_BOUNDED_MASK_ORACLE direct L-BFGS-B optimization of real M in [0,1] against
                                the target waveform loss, deterministic init, fixed budget
  COMPLEX_RATIO_ORACLE          M = S*conj(Y) / (|Y|^2 + eps) (unbounded complex; diagnostic)

All are attained diagnostics, not universal bounds and not deployable separators.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import get_window

import common as C


class ClapSepSTFT:
    """Exact-numpy twin of the torchlibrosa STFT/ISTFT pair used by local CLAPSep."""

    def __init__(self, n_fft=1024, hop_length=320, win_length=1024,
                 window="hann", center=True, pad_mode="reflect"):
        assert win_length == n_fft, "local config uses win_length == n_fft (1024)"
        assert pad_mode == "reflect"
        self.n_fft = int(n_fft)
        self.hop = int(hop_length)
        self.center = bool(center)
        # librosa.filters.get_window('hann', L, fftbins=True) == periodic hann
        w = get_window(window, win_length, fftbins=True).astype(np.float64)
        self.window = w
        self.ola = w ** 2

    # ---------------------------------------------------------------- STFT
    def n_frames(self, length: int) -> int:
        return 1 + length // self.hop

    def stft(self, x: np.ndarray) -> np.ndarray:
        """(L,) float -> (F, n_fft//2+1) complex64-in-float64, torchlibrosa convention."""
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
        n = self.n_fft
        xp = np.pad(x, n // 2, mode="reflect") if self.center else x
        F = 1 + (len(xp) - n) // self.hop
        idx = np.arange(n)[None, :] + self.hop * np.arange(F)[:, None]
        frames = xp[idx] * self.window[None, :]
        return np.fft.rfft(frames, n=n, axis=-1)

    # ------------------------------------------------- torchlibrosa magphase
    @staticmethod
    def magphase(X: np.ndarray):
        real, imag = X.real, X.imag
        mag = np.sqrt(real ** 2 + imag ** 2)
        denom = np.clip(mag, 1e-10, np.inf)
        cos = real / denom
        sin = imag / denom
        return mag, cos, sin

    # -------------------------------------------------- torchlibrosa ISTFT
    def _wsum(self, F: int) -> np.ndarray:
        out_len = (F - 1) * self.hop + self.n_fft
        wsum = np.zeros(out_len)
        for f in range(F):
            s = f * self.hop
            wsum[s:s + self.n_fft] += self.ola
        return wsum

    def istft(self, X: np.ndarray, length: int) -> np.ndarray:
        """Complex (F,K) -> (length,) waveform; torchlibrosa ISTFT convention."""
        F = X.shape[0]
        n = self.n_fft
        frames = np.fft.irfft(X, n=n, axis=-1) * self.window[None, :]
        out_len = (F - 1) * self.hop + n
        y = np.zeros(out_len)
        for f in range(F):
            s = f * self.hop
            y[s:s + n] += frames[f]
        y /= np.clip(self._wsum(F), 1e-11, None)
        start = n // 2 if self.center else 0
        return y[start:start + length]

    # --------------------------------------------- exact CLAPSep reconstruct
    def wav_reconstruct(self, mask: np.ndarray, X_mix: np.ndarray,
                        length: int) -> np.ndarray:
        """CLAPSep.wav_reconstruct with a real mask (mag*cos, mag*sin path)."""
        mag_x, cos_x, sin_x = self.magphase(X_mix)
        mag_y = np.maximum(mag_x * mask, 0.0)  # torch relu_
        return self.istft(mag_y * cos_x + 1j * (mag_y * sin_x), length)


# ------------------------------------------------------------------ oracles
def bounded_real_mask(Y: np.ndarray, S: np.ndarray, eps: float = C.METRIC_EPS) -> np.ndarray:
    """Per-bin least-squares real mask clipped to [0,1]."""
    num = np.real(S * np.conj(Y))
    return np.clip(num / (np.abs(Y) ** 2 + eps), 0.0, 1.0)


def complex_ratio_mask(Y: np.ndarray, S: np.ndarray, eps: float = C.METRIC_EPS) -> np.ndarray:
    """Unbounded complex ratio mask (diagnostic only)."""
    return (S * np.conj(Y)) / (np.abs(Y) ** 2 + eps)


# --------------------------------------- linear forward/adjoint (mask space)
# The exact CLAPSep path equals ISTFT(M .* Y) wherever |Y| >= 1e-10 (the magphase
# clamp); the optimizer uses this linear surrogate, and the FINAL reported
# reconstruction always goes through the exact wav_reconstruct path above.

def forward_mask(stft: ClapSepSTFT, M: np.ndarray, Y: np.ndarray, length: int) -> np.ndarray:
    return stft.istft(M * Y, length)


def adjoint_mask(stft: ClapSepSTFT, g: np.ndarray, Y: np.ndarray, length: int) -> np.ndarray:
    """Exact adjoint of forward_mask w.r.t. real M; validated by finite differences."""
    n, hop = stft.n_fft, stft.hop
    F = stft.n_frames(length)
    out_len = (F - 1) * hop + n
    g_full = np.zeros(out_len)
    g_full[n // 2:n // 2 + length] = g
    g_full /= np.clip(stft._wsum(F), 1e-11, None)
    idx = np.arange(n)[None, :] + hop * np.arange(F)[:, None]
    lam = g_full[idx] * stft.window[None, :]
    Lam = np.fft.rfft(lam, n=n, axis=-1)
    scale = np.full(n // 2 + 1, 2.0 / n)
    scale[0] = 1.0 / n
    scale[-1] = 1.0 / n
    GX = Lam * scale[None, :]
    return np.real(np.conj(GX) * Y)


def optimize_bounded_mask(stft: ClapSepSTFT, y: np.ndarray, s: np.ndarray,
                          max_iter: int = 150, maxcor: int = 20) -> dict:
    """OPTIMIZED_BOUNDED_MASK_ORACLE.

    Deterministic init = bounded per-bin oracle mask; fixed L-BFGS-B budget;
    objective = squared waveform error against the exact target.
    """
    Y = stft.stft(y)
    S = stft.stft(s)
    L = len(s)
    M0 = bounded_real_mask(Y, S)
    Yc = np.ascontiguousarray(Y)

    def fg(m_flat):
        M = m_flat.reshape(Y.shape)
        r = forward_mask(stft, M, Yc, L) - s
        f = float(np.dot(r, r))
        g = adjoint_mask(stft, r, Yc, L)
        return f, g.ravel()

    res = minimize(fg, M0.ravel(), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, 1.0)] * M0.size,
                   options={"maxiter": max_iter, "maxcor": maxcor,
                            "ftol": 1e-16, "gtol": 1e-14, "maxfun": 4 * max_iter + 40})
    return {
        "mask": res.x.reshape(Y.shape),
        "init_mask": M0,
        "Y": Yc,
        "loss_init": float(fg(M0.ravel())[0]),
        "loss_final": float(res.fun),
        "n_iter": int(res.nit),
        "n_feval": int(res.nfev),
        "converged": bool(res.success),
        "message": str(res.message),
    }


def run_oracles(stft: ClapSepSTFT, y: np.ndarray, s: np.ndarray,
                opt_max_iter: int = 150) -> dict:
    """All diagnostics for one exact mixture; returns waveforms + bookkeeping."""
    L = len(s)
    Y = stft.stft(y)
    S = stft.stft(s)
    M_b = bounded_real_mask(Y, S)
    s_b = stft.wav_reconstruct(M_b, Y, L)
    opt = optimize_bounded_mask(stft, y, s, max_iter=opt_max_iter)
    s_opt = stft.wav_reconstruct(opt["mask"], Y, L)
    M_c = complex_ratio_mask(Y, S)
    s_c = stft.istft(M_c * Y, L)
    s_round = stft.istft(S, L)  # TRUE_TARGET_STFT_ROUNDTRIP floor
    return {
        "Y": Y, "S": S,
        "masks": {"bounded": M_b, "optimized": opt["mask"], "complex": M_c},
        "outputs": {
            "RAW_MIXTURE": y,
            "BOUNDED_REAL_MASK_ORACLE": s_b,
            "OPTIMIZED_BOUNDED_MASK_ORACLE": s_opt,
            "COMPLEX_RATIO_ORACLE": s_c,
            "TRUE_TARGET_STFT_ROUNDTRIP": s_round,
            "TRUE_TARGET": s,
        },
        "optimizer": {k: opt[k] for k in
                      ("loss_init", "loss_final", "n_iter", "n_feval",
                       "converged", "message")},
    }
