# S1a representation diagnostics — CLAPSep bounded magnitude-mask audit

**2026-09-10.** Companion to [REPRESENTATION_RESULTS.json](REPRESENTATION_RESULTS.json).
Current status: oracle machinery implemented, unit-validated, and exercised end-to-end on
`SYNTHETIC_FIXTURE` media. **No real-capture representation feasibility number exists yet.**

## 1. What was audited (inspected, not remembered)

Vendored local source (hashes in [PILOT_MANIFEST.json](PILOT_MANIFEST.json), written by
`scripts/00_env_audit.py`):

- `experiments/r3_audio_query/third_party/CLAPSep/model/CLAPSep.py`
  - `self.stft = STFT(n_fft=1024, hop_length=320, win_length=1024, window='hann',
    center=True, pad_mode='reflect')`; identical `ISTFT`; torchlibrosa modules;
  - `wav_reconstruct`: `mag_y = relu_(mag_x * mask)`, mixture phase `cos, sin`,
    `pred = istft(mag_y*cos_y, mag_y*sin_y, length=mixed.size(-1))`;
  - `inference_from_data` runs at 32 kHz mono under `torch.no_grad()`.
- `experiments/r3_audio_query/third_party/CLAPSep/model/CLAPSep_decoder.py`
  - decoder returns `torch.sigmoid(mask)` (bounded real magnitude mask); local config
    `phase: False` → no phase head exists in this checkpoint path.
- `experiments/r3_audio_query/scripts/clapsep_lib.py`
  - `MODEL_CONFIG` confirms `phase: False`; 10 s chunking; peak rescale `x *= 0.9/max|x|`
    when `max>1`; `load_state_dict(strict=False)` with the local `best_model.ckpt`.

torchlibrosa internals (read from the installed source, R2 venv 0.1.0): periodic hann
window; center reflect-padding by `n_fft//2`; frame count `1 + L//320`; ISTFT = per-frame
irfft × window → overlap-add → divide by clipped window² sum (`clamp 1e-11`) → trim
`n_fft//2` and slice to `length`; `magphase` clamps at `1e-10`.

## 2. The exact representation under test

```
Ŝ = ISTFT( M ⊙ Y ),   M = f_θ(...) ∈ [0,1]^(F×513) real,  Y = STFT(y) at 32 kHz
STFT/ISTFT: n_fft 1024, hop 320, hann periodic, center reflect, length-preserving
```

Structural properties that follow immediately (confirmed numerically by the Part-15
cases in `scripts/oracle_validation.py`):

- the path is **linear in Y** for a fixed mask → exact per-path decomposition
  `R(M⊙Y) = R(M⊙S) + R(M⊙N)` is available (used for exact leakage diagnostics);
- with mixture phase and `M ∈ [0,1]`, **any bin where the target magnitude exceeds the
  mixture magnitude is amplitude-limited**: `|Ŝ_f| ≤ |Y_f|`. Destructive interference
  (`|S| > |Y|`) cannot be represented at that bin;
- output phase = mixture phase everywhere; any target/nuisance phase cancellation pattern
  is inherited by the estimate.

## 3. Oracle diagnostics

| Oracle | Definition | Role |
|---|---|---|
| `BOUNDED_REAL_MASK_ORACLE` | `M* = clip(Re(S·conj(Y)) / (|Y|² + ε), 0, 1)`, ε = 1e-10, applied through the exact `wav_reconstruct` path | per-bin least-squares solution — the best the real head's *output family* can do bin-by-bin |
| `OPTIMIZED_BOUNDED_MASK_ORACLE` | directly optimize `M ∈ [0,1]` (L-BFGS-B, deterministic init = M*, fixed budget 60–150 iters, objective = squared waveform error vs `s`) through an exact linear forward + finite-difference-validated adjoint | absorbs overlap-add/STFT-redundancy effects; a stronger attained diagnostic — still not a universal upper bound |
| `COMPLEX_RATIO_ORACLE` | `M = S·conj(Y)/(|Y|² + ε)` unbounded complex | diagnostic: separates representation limitation from data difficulty |
| `TRUE_TARGET_STFT_ROUNDTRIP` | `ISTFT(STFT(s))` | analysis/synthesis consistency floor |
| `TRUE_TARGET` | `s` itself | identity anchor |

None of these is a model, a proof, or a deployable separator. No oracle parameter was
tuned after seeing any result (fixed budgets declared in the plan before the fixture run).

## 4. Convention verification chain

1. numpy twin vs librosa (primary env): max rel err ≤ 1.7e-7 across lengths
   (`tests/test_stft_clapsep.py`).
2. numpy twin vs **the actual torchlibrosa modules** used by CLAPSep (R2 venv,
   torch 2.14.0+cu126): stft rel err ≤ 1.4e-6, full `wav_reconstruct` sequence rel err
   ≤ 3.8e-7 (`logs/stft_equivalence_check.json`, receipt from
   `scripts/92_stft_equivalence_check.py`).
3. Adjoint of the mask-space linear operator vs central finite differences:
   pointwise rel err ~1e-8 (`tests/test_stft_clapsep.py`).
4. Part-15 six required synthetic cases: **all pass**
   (`logs/synthetic_validation.json`) — see §6.

Differences declared: (a) the optimizer works on the surrogate `ISTFT(M⊙Y)`, which
equals the exact CLAPSep path wherever `|Y| ≥ 1e-10` (verified ≤ 1e-7 apart);
(b) torchlibrosa computes in float32, the twin in float64; comparisons use 1e-4
tolerances. Reported reconstruction outputs always go through the exact
`wav_reconstruct` path.

## 5. What S1a's oracle answers, precisely

- **Answered when real T2 exists:** "For a real recorded target mixed exactly with real
  nuisance, how much of the target can the current [0,1] magnitude-mask output family
  preserve at fixed gain, and how much nuisance can it remove?" Gate B vs A hinges on
  `OPTIMIZED_BOUNDED_MASK_ORACLE` perceptual adequacy vs `COMPLEX_RATIO_ORACLE`.
- **NOT answered by the oracles:** whether a trained network can *learn* to approach the
  oracle; whether any mask-based model is a universal upper bound; anything about S1b
  training outcomes.

## 6. Part-15 synthetic validation (required before real use)

All six cases pass (`scripts/oracle_validation.py`, results in
`logs/synthetic_validation.json`):

| Case | Expected | Result |
|---|---|---|
| Y = S | bounded oracle reproduces S | max err 1.2e-10 |
| S = 0 | target output exactly zero | 0.0 (bounded), 0.0 (optimized) |
| separate TF regions | near-perfect separation | recon SNR 52.9 dB |
| destructive interference | bounded fails visibly; complex recovers | bounded/optimized projection gain ≈ −14.0 dB (amplitude-limited to `0.2·s`); complex SNR 155.4 dB |
| continuous tone | no artificial chopping | bounded == roundtrip, max err 7e-9 |
| impulse | transient timing/amplitude sanity | peak offset 0 samples; amplitude ratio 1.0000 |

## 7. Fixture-panel exercise (SYNTHETIC_FIXTURE — tooling validation only)

The 12-case fixture panel (structure mirrors the future real T2 panel: −20/−10/0/+10 dB,
music/speech/ambience/impact/composite families, overlap cases, friction/dense/weak
strata) was run through oracles + evaluation to validate the pipeline. Highlights
(full numbers in [REPRESENTATION_RESULTS.json](REPRESENTATION_RESULTS.json)):

| Variant (median, n=12) | recon SNR dB | proj gain dB | SI-SDR dB |
|---|---|---|---|
| RAW_MIXTURE | −10.4 | +0.01 | −10.4 |
| BOUNDED_REAL_MASK_ORACLE | 15.2 | −0.35 | 15.0 |
| OPTIMIZED_BOUNDED_MASK_ORACLE | 15.9 | −0.33 | 15.8 |
| COMPLEX_RATIO_ORACLE | 107.0 | ~0 | 107.0 |

By requested level (optimized oracle recon SNR): −20 dB → 6.9; −10 dB → 13.7;
0 dB → 15.7; +10 dB → 20.1 dB. Exact path decomposition works (bounded nuisance-path
attenuation median ≈ 28.9 dB; linearity sum-check ≤ 1e-7). Friction envelope
correlation 0.96–1.00 on friction cases.

**Interpretation limits:** these numbers validate that the pipeline computes sensible
values; they say NOTHING about real maimai interaction audio. Synthetic targets are
easier than real handcam-domain acoustics; the −20 dB fixture case already shows the
expected bounded-mask strain, which is the kind of evidence the real panel must
quantify before the A/B gate can be decided.

## 8. Evaluation convention (frozen)

Fixed gain everywhere; no per-output normalization; no post-hoc alignment. Primary
preservation = fixed-gain reconstruction SNR from NMSE (eps 1e-10, never tuned);
SI-SDR secondary; projection gain signed; MR-STFT (256/1024/2048) spectral distance;
HF retention ratio; per-event transient peak/rise/decay; friction envelope correlation +
log-spectral distance. Suppression = exact per-path energies (mask linearity) + LSQ
projection diagnostics with condition-number flag (1e4). Failure taxonomy per S0:
invented event vs source misassignment vs acoustic distortion.
