# S2A training signals (protocol section 20). Declared a priori.
#
# L = 1.00 * S1R_anchor            (A: on unmodified real mixture x, stay close
#                                   to the frozen S1R output; anti-forgetting)
#   + 0.75 * injection_invariance  (B: F(x + alpha*T(m), m) ~= S1R(x); the model
#                                   learns reference-explained excess music is
#                                   nuisance - THE new-information signal)
#   + 0.75 * cross_context_consistency (C: stable central output across real
#                                   context boundaries, S1R's successful axis)
#   + 1.00 * anti_collapse         (D: no wholesale suppression; protect
#                                   transients / weak taps / friction)
#   + 0.10 * weight_anchor         (L2-SP toward S1R trainable weights)
#
# Kernels are reused read-only from S1R's s1r_losses (identical formulations).
# Section 22 respected: nothing here penalizes onset coincidence with the song;
# the reference enters as spectral/content side information only.
from s1r_losses import (distill, consistency, anti_collapse, weight_anchor,  # noqa: F401
                        mr_stft_logmag, rms)

W_ANCHOR = 1.00
W_INVAR = 0.75
W_CONSIST = 0.75
W_COLLAPSE = 1.00
W_WEIGHT = 0.10
