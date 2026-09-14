# S2A controlled counterfactual music augmentation (protocol section 18).
#
# x_aug = x + alpha * T(m) where T(m) applies ONLY mild acoustic
# transformations: small delay consistent with acoustic playback, gain, gentle
# spectral tilt, mild short room-like coloration. The authentic raw mixture is
# NEVER removed or replaced; the only synthetic component is ADDITIONAL KNOWN
# music nuisance. No synthetic target stem is created (section 17).
import numpy as np
from scipy import signal as sig

import s1r_common as rc

ALPHA_RANGE = (0.12, 0.45)
DELAY_RANGE_S = (0.0, 0.020)
GAIN_RANGE_DB = (-4.0, 4.0)
TILT_RANGE = (0.0, 0.5)
IR_LEN_RANGE_S = (0.12, 0.30)
IR_WET_RANGE = (0.2, 0.5)


def perturb_reference(m, rng, sr=rc.SR):
    """Mild acoustic transformation chain T (section 18)."""
    m = np.asarray(m, dtype=np.float32)
    d = int(rng.uniform(*DELAY_RANGE_S) * sr)
    if d > 0:
        m = np.concatenate([np.zeros(d, dtype=np.float32), m[:-d]])
    m = m * (10.0 ** (rng.uniform(*GAIN_RANGE_DB) / 20.0))
    # gentle spectral tilt: blend toward a one-pole lowpass (darkening)
    a = rng.uniform(0.15, 0.85)
    lp = sig.lfilter([a], [1.0, -(1.0 - a)], m).astype(np.float32)
    tilt = rng.uniform(*TILT_RANGE)
    m = (1.0 - tilt) * m + tilt * lp
    # mild short room coloration: quiet exponentially-decaying noise IR
    n_ir = int(rng.uniform(*IR_LEN_RANGE_S) * sr)
    ir = (rng.standard_normal(n_ir).astype(np.float32)
          * np.exp(-np.arange(n_ir, dtype=np.float32) / (0.15 * sr)))
    ir /= (np.linalg.norm(ir) + 1e-9)
    wet = rng.uniform(*IR_WET_RANGE)
    m = sig.fftconvolve(m, ir)[:len(m)].astype(np.float32) * wet + m
    peak = float(np.max(np.abs(m)))
    if peak > 1.0:
        m = m * (0.95 / peak)
    return m.astype(np.float32)


def make_augmented(x, m, rng):
    alpha = rng.uniform(*ALPHA_RANGE)
    return x + alpha * perturb_reference(m, rng), alpha
