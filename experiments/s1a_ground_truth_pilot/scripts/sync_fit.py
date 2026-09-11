"""S1a clock-map fitting: t_target = a + b * t_aux, with drift/propagation bookkeeping."""
from __future__ import annotations

import numpy as np


def find_onset(x: np.ndarray, sr: int, t_center: float, search_s: float = 5.0,
               frame_s: float = 0.005) -> dict:
    """Pick the strongest energy-onset in (t_center ± search_s).

    Returns absolute onset time + peakiness (max/median frame-RMS ratio), or None
    if the region is silent. Simple, auditable, same routine on every channel.
    """
    n = len(x)
    k = max(1, int(frame_s * sr))
    m = n // k
    if m < 4:
        return None
    env = np.abs(x[:m * k]).reshape(m, k).mean(axis=1)
    c = int(t_center * sr / k)
    lo, hi = max(0, c - int(search_s * sr / k)), min(m, c + int(search_s * sr / k))
    if hi - lo < 2:
        return None
    seg = env[lo:hi]
    i = int(np.argmax(seg))
    peakiness = float(seg[i] / (np.median(seg) + 1e-12))
    if seg[i] <= 0:
        return None
    # onset = first frame before the peak crossing 20% of peak
    thr = 0.2 * seg[i]
    j = i
    while j > 0 and seg[j] > thr:
        j -= 1
    t_onset = (lo + j + 1) * frame_s
    return {"t": float(t_onset), "peakiness": peakiness,
            "peak_frame_rms": float(seg[i])}


def fit_clock_map(anchor_pairs: list[tuple[float, float]]) -> dict:
    """Least-squares fit of t_target = a + b * t_aux from (t_aux, t_target) pairs."""
    t_aux = np.array([p[0] for p in anchor_pairs], dtype=np.float64)
    t_tgt = np.array([p[1] for p in anchor_pairs], dtype=np.float64)
    if len(t_aux) < 2:
        raise ValueError("need >= 2 anchors")
    A = np.column_stack([np.ones_like(t_aux), t_aux])
    coef, *_ = np.linalg.lstsq(A, t_tgt, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = t_tgt - (a + b * t_aux)
    out = {
        "a_s": a, "b_drift_ratio": b,
        "drift_ppm": (b - 1.0) * 1e6,
        "anchors_used": len(t_aux),
        "residuals_s": [float(r) for r in resid],
        "residual_rms_s": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs_residual_s": float(np.max(np.abs(resid))),
    }
    # leave-one-out validation when >= 3 anchors
    if len(t_aux) >= 3:
        loo = []
        for i in range(len(t_aux)):
            m = np.ones(len(t_aux), bool)
            m[i] = False
            Ai = np.column_stack([np.ones(m.sum()), t_aux[m]])
            ci, *_ = np.linalg.lstsq(Ai, t_tgt[m], rcond=None)
            pred = ci[0] + ci[1] * t_aux[i]
            loo.append(float(t_tgt[i] - pred))
        out["leave_one_out_abs_err_s"] = [abs(v) for v in loo]
        out["leave_one_out_max_abs_err_s"] = float(np.max(np.abs(loo)))
    return out


def apply_map(t_aux: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.asarray(t_aux)


def sync_verdict(sync_map: dict, tolerance_s: float) -> dict:
    err = sync_map.get("leave_one_out_max_abs_err_s",
                       sync_map["max_abs_residual_s"])
    return {"validation_err_s": err, "tolerance_s": tolerance_s,
            "sync_ok": bool(err <= tolerance_s)}
