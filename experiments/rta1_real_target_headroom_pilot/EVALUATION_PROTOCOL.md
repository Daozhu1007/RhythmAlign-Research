# RTA1 — Evaluation Protocol

**Waveform truth first.** With real target truth available, direct
reconstruction error against the true target is the primary evidence.
The 2–9 kHz band / weak-event / friction-continuity metrics below are
**supporting diagnostics**, never substitutes for waveform comparison, and
never again a training-time "protection" claim.

## Methods compared (frozen before the panel is scored)

| method | instance discipline |
|---|---|
| RAW | the mixture itself |
| Zero-shot CLAPSep | loaded once as an independent instance; parameters hashed; never shared with any other method |
| S1R (frozen) | the ORIGINAL selected checkpoint (`s1r_selected.ckpt`), loaded as an immutable independent instance, sha256 verified and recorded; the S1W artifact is refused by the loader |
| Bounded oracles (binwise, waveform-optimized) | receive the true target offline; deterministic/seeded |
| Complex oracle | diagnostic only |

**Baseline immutability is enforced and tested** (`tests/test_baseline_immutability.py`):
two models under comparison must never share mutable parameters, and running
one must not change another's weights. This directly addresses the RTA-0
finding that S2A's "S1R baseline" reused trained weights.

## Metrics (per mixture)

Primary (waveform truth):

- **Reconstruction SNR** of `p_hat` vs true `p` (dB).
- **Target-path NMSE** (dB) from the source-path decomposition.
- **Nuisance attenuation** (dB): in-mixture `n` level minus `n_hat_M` level.

Supporting diagnostics (all with explicitly tested STFT axis handling):

- **2–9 kHz target-path energy** retention (correct frequency-axis mapping).
- **Weak-event preservation**: scored weak events present in `p_hat_M`
  (level within ±1 dB of the same event in `p`); an event is "omitted" if its
  band level drops > 6 dB below its level in `p` at the recorded time.
- **Friction/tail continuity**: scored friction intervals must remain
  continuous (no new silence gaps > 50 ms; spectral flatness character
  preserved within the recorded tolerance).
- **Decay preservation**: per scored strong event, tail decay slope within
  a recorded tolerance of the true tail.

## Panels and holdout discipline

- 24 mixtures (8/session × 3). **Session 01 = tuning panel; sessions 02+03
  (16 mixtures) = held-out evaluation panel.** The split is declared before
  any scoring. The pre-registered success test (below) is computed on the 16
  held-out mixtures only.
- No method, mask, threshold, or protocol constant may be tuned on the
  held-out panel. Tuning happens on session 01 or is declared beforehand.

## Transient/friction metric implementation note

These metrics use the corrected band-energy implementation with explicit
frequency-axis tests (`tests/test_stft_axes.py`). The RTA-0 audit found the
historical S1R loss implementation indexed the time axis; that implementation
is **not** inherited here, and the tests assert the correct behavior directly
(sinusoid-based probes at known frequencies).

## Human listening

None by default. Only if the waveform evidence leaves the final decision
ambiguous per [DECISION_RULES.md](DECISION_RULES.md): at most 6 short A/B
comparisons, ≤ 10 minutes total, drawn from diverse sessions, no 18-item
packs, no repetitive same-recording evaluation. "No obvious difference" is a
valid recorded outcome. Any listening material goes to `work/listening/`
(ignored) and results are recorded as owner listening passes, not as
equivalence proofs.
