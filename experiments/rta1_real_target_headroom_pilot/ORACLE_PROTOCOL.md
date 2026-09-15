# RTA1 — Oracle Protocol

The oracle is **not a deployable model**. It receives the true target offline
and exists to measure **representational headroom**: how much better than the
frozen models a bounded real mask could do if supervision were perfect.

Implementation: `scripts/06_run_bounded_oracles.py`,
`scripts/07_run_complex_oracle.py`. All shared STFT/ISTFT helpers live in
`scripts/rta1_lib.py` with explicit, tested axis conventions (freq × time for
1-D inputs — the failure mode found in the RTA-0 audit cannot recur silently;
see `tests/test_stft_axes.py`).

## Oracle family

1. **BINWISE bounded real magnitude oracle.**
   `M = clamp( |S_p| / (|S_y| + ε), 0, 1 )` per TF bin (freq × time),
   `p_hat = ISTFT(M · S_y)`. The classical upper bound for a real, bounded
   magnitude mask given perfect target knowledge.

2. **WAVEFORM-OPTIMIZED bounded real-mask oracle.**
   A free real mask (one parameter per TF bin, passed through a hard sigmoid
   to enforce bounds [0,1]), initialized from oracle 1 and optimized by
   gradient ascent to maximize **waveform-domain** reconstruction SNR of
   `ISTFT(M · S_y)` against `p`, with an ISTFT roundtrip that is part of the
   optimization (phase-aware through the synthesis, not through the mask).
   A few hundred Adam steps at lr 0.01, fixed seed, convergence recorded.

3. **Complex-ratio / phase-capable diagnostic oracle** (`07`).
   `G = S_p / (S_y + ε)` (complex), `p_hat = ISTFT(G · S_y)`. Diagnoses only:
   if this succeeds where bounded real masks fail, phase freedom is implicated
   (Outcome B). It is not a bounded representation and not deployable.

4. **Target roundtrip reconstruction.** `ISTFT(STFT(p))` — the lossless
   floor; quantifies how much of every "error" is just analysis/resynthesis.

## Fixed-mask source-path analysis (mandatory for oracles 1–2)

Given `y = p + n` and mask `M` **estimated from y/target information**:

- `p_hat_M = ISTFT(M · STFT(p))` — target path,
- `n_hat_M = ISTFT(M · STFT(n))` — nuisance path.

The SAME fixed mask is applied separately to `p` and `n`. A nonlinear
separator is never rerun independently on the paths. This separates
**target damage** (`p_hat_M` vs `p`) from **nuisance leakage** (`n_hat_M` vs
`n`) under the selected mask, and is verified against direct masked-mixture
reconstruction within numeric tolerance (`tests/test_mask_linearity.py`).

## Fixed gain / clipping

Reconstructions may apply at most ONE fixed global scalar gain per mixture,
chosen without clipping (recorded). No per-item normalization is allowed
anywhere (a lesson preserved from the S1W listening-protocol mistake).

## Reporting

Per mixture and method: reconstruction SNR (vs true `p`), target-path NMSE,
nuisance attenuation (vs `n` level in-mixture), roundtrip floor, convergence/
sanity flags, and the source-path decomposition. All outputs under
`work/oracle/` (ignored). Metrics go to the headroom aggregator
(`08_evaluate_headroom.py`); audio stays local.
