# S1a plan — Ground-Truth Acquisition Pilot + Representation Feasibility

**2026-09-10.** This plan implements the S1a stage instructions on top of the S0 program
(see `../s0_research_reframing/`). It fixes scope, conventions and the decision gate
BEFORE any real data or model output exists.

## 1. Scope

In scope: one high-quality pilot session design; reusable ingest/QC tooling; sync
procedure; truth-tier bookkeeping; a small exact constructed (T2) mixture panel; oracle
diagnostics of the current CLAPSep bounded magnitude-mask representation; representation
evaluation; a small blind listening pack; the pre-training decision gate.

Out of scope (explicitly): S1b training of any kind; fine-tuning CLAPSep; creating
reference adapters or presence heads; training SAM Audio / AudioSep; tuning the R5
detector; visual contact-detection research; production integration; new separator
architectures; re-annotation or repair of historical Holdout-D artifacts.

## 2. Questions

- **Q1** Can trustworthy real target interaction audio be acquired close enough to the
  intended handcam-microphone domain to supervise/evaluate learned extraction?
- **Q2** Given exact target truth in a constructed mixture, can the current CLAPSep
  bounded magnitude-mask representation faithfully represent the desired target?

Primary representation feasibility evidence comes from T2 (exact constructed mixtures).
Never compute exact waveform recovery metrics against T0/T3/T4 as if they were exact
target stems.

## 3. Truth tiers (S0 convention)

| Tier | Meaning here | Use in S1a |
|---|---|---|
| T0 | sensor/event evidence (contact mic, video, logs) | independent event/timing evidence only |
| T1 | quiet isolated acoustic target recording (phone position) | candidate supervision target; QC'd with truth-quality flags |
| T2 | exact constructed additive mixture `y = s + n` (float PCM, stored components) | primary representation feasibility; exact waveform bookkeeping |
| T3 | acoustic/device stress (replay, AGC, clipping) | separate stress reporting, never exact-stem scoring |
| T4 | simultaneous real scene | perceptual/event evaluation only |

Every pilot asset is assigned a tier at manifest time. A contaminated capture is flagged,
never silently labeled clean. The synthetic fixtures generated in this stage while real
capture is missing are labeled `SYNTHETIC_FIXTURE` and are for tooling validation only.

## 4. Acquisition pilot design (Q1)

One session, ~5–8 minutes total per the protocol in
[ACQUISITION_PROTOCOL.md](ACQUISITION_PROTOCOL.md):

- 3–5 min quiet / lowest-practical-playback real interaction at the phone/camera position
  (same device/geometry as real handcam capture when practical);
- 1–2 min target-absent environment / machine playback / ambience;
- 1–2 min ordinary noisy gameplay (representative real mixture);
- strongly preferred auxiliary channels: close air mic; contact/piezo sensor (evidence
  only); optional linear recorder at the phone position;
- visible+audible sync events near start and end ([SYNC_PROTOCOL.md](SYNC_PROTOCOL.md)).

The interaction capture must cover the acoustic source diversity list (isolated taps,
physical buttons, palm hits, slides, slow/fast friction, weak/strong contacts, dense,
sparse, deliberate rests, natural combinations). The user does NOT need to reproduce a
precise chart. If cabinet music cannot be muted, the quietest practical alternative is
documented and the capture is truth-flagged accordingly.

Non-invasiveness rule: no sensor attachment, cabinet modification, or setting changes
without operator permission. Nothing in S1a requires invasive cabinet work.

## 5. Constructed T2 mixture panel (after real T1 exists)

- ~12–24 short examples (NOT hundreds), built from real recorded target segments:
  isolated tap, weak tap, palm hit, button impact, slide/friction, dense interaction,
  decay tail;
- additive construction with independently stored nuisance families: (A) pristine /
  acoustically transformed maimai music, (B) speech / announcer-like, (C) arcade ambience,
  (D) unrelated impact, (E) composite;
- mandatory overlap cases: weak target + loud speech, target onset + nuisance onset,
  friction + music, dense target + music, target + similar unrelated impact;
- target-to-nuisance levels −10 / 0 / +10 dB, plus a few −20 dB diagnostics;
- every component stored exactly (`s`, `n`, `y = s + n` in float PCM) with recipe, gains,
  hashes; gains frozen before any model output is seen;
- nuisance placed by explicit recipe (not tuned after results).

While real capture is missing, a fixture panel with the same structure (12 mixtures) is
generated from synthetic sources purely to validate the tooling; its numbers are labeled
`SYNTHETIC_FIXTURE` and carry no feasibility meaning about real acoustics.

## 6. Representation audit (Q2)

The local CLAPSep signal path was inspected in the vendored source
(`../r3_audio_query/third_party/CLAPSep/model/CLAPSep.py` and `CLAPSep_decoder.py`, with
runner config `../r3_audio_query/scripts/clapsep_lib.py`):

- STFT: `n_fft=1024, hop_length=320, win_length=1024, window='hann' (periodic),
  center=True, pad_mode='reflect'`, via torchlibrosa `STFT`/`ISTFT`, at 32 kHz mono;
- mask: `torch.sigmoid(...)` — a bounded **real** magnitude mask in [0, 1]
  (`phase: False` in the local config);
- reconstruction: `mag_y = relu(mag_x * mask)`, mixture phase (`cos, sin` from mixture
  STFT), `ISTFT(..., length=input_length)`;
- inference protocol (not part of the representation): 10 s chunks, peak rescale to 0.9
  when `max>1`, `torch.no_grad`.

Full details: [REPRESENTATION_DIAGNOSTICS.md](REPRESENTATION_DIAGNOSTICS.md).

### Oracles

For exact mixture `Y = STFT(y)` and exact target `S = STFT(s)`:

1. `BOUNDED_REAL_MASK_ORACLE` — per-bin least-squares real mask
   `M* = clip(Re(S·conj(Y)) / (|Y|² + ε), 0, 1)`, reconstructed with the exact CLAPSep
   STFT/ISTFT convention. This is an **attained oracle diagnostic**, not a universal
   upper-bound proof (STFT redundancy/overlap-add consistency matters).
2. `OPTIMIZED_BOUNDED_MASK_ORACLE` — directly optimize a real mask `M ∈ [0,1]` against
   the target waveform loss for each short exact mixture (deterministic initialization
   from M*, fixed L-BFGS-B budget, exact linear forward + validated adjoint). Stronger
   diagnostic that absorbs overlap-add/consistency effects.
3. `COMPLEX_RATIO_ORACLE` (diagnostic) — unbounded complex ratio mask
   `S·conj(Y)/(|Y|² + ε)`; separates representation limitation from data difficulty.
   Plus `TRUE_TARGET_STFT_ROUNDTRIP` (`ISTFT(STFT(s))`) as the analysis/synthesis
   consistency floor and the raw `TRUE_TARGET` itself.

All oracles are diagnostics, not models, not deployable separators. No oracle parameter
is tuned after seeing results.

## 7. Evaluation (separate axes, never one score)

For every exact T2 case: raw mixture, bounded-mask oracle, optimized bounded-mask oracle,
true target (+ complex oracle where informative).

Target preservation: fixed-gain waveform NMSE / reconstruction SNR (primary), SI-SDR
(secondary), target projection gain, multi-resolution spectral distance, high-frequency
retention, transient peak/rise/decay fidelity, friction/continuous-contact fidelity.

Nuisance suppression: exact nuisance leakage (exact per-path decomposition available
because the mask path is linear in Y), nuisance attenuation, target-vs-nuisance
projection diagnostics.

Full formulas: `scripts/evaluator.py` docstrings + §7 of
[REPRESENTATION_DIAGNOSTICS.md](REPRESENTATION_DIAGNOSTICS.md).

## 8. Pre-training decision gate (fixed in advance)

Exactly one of:

- **A. CAPTURE PASS + BOUNDED MASK PASS** → proceed to S1b Route A per S0.
- **B. CAPTURE PASS + BOUNDED MASK FAIL + COMPLEX PASS** → do NOT train the current head;
  design a compact complex-mask / RI-mapping adaptation stage.
- **C. CAPTURE FAIL** → redesign acquisition before any model work.
- **D. ALL REPRESENTATIONS FAIL PERCEPTUALLY** → revisit target definition/capture setup.
- **PENDING. CAPTURE REQUIRED** → tooling ready; no real recordings yet; no decision on
  Q1/Q2 real-data feasibility is possible.

Stop conditions (from the stage instructions) are enforced in scripts: mixture-sum
identity checks, clipping/contamination flags, sync adequacy, gain/sample conventions,
oracle unit tests, aliasing checks, no independent output normalization, no historical
file modification, no post-hoc gain tuning.

## 9. Environments

- Primary (all S1a scripts/tests): Python 3.10, numpy/scipy/soundfile/librosa/matplotlib.
  No torch dependency — the CLAPSep STFT/ISTFT convention is reproduced in
  `scripts/stft_clapsep.py` and numerically cross-validated.
- Secondary (exact-convention receipt only): the repo R2 venv (torch 2.14.0+cu126 +
  torchlibrosa 0.1.0) runs `scripts/92_stft_equivalence_check.py`, which compares
  torchlibrosa's STFT/ISTFT (the actual CLAPSep signal-path modules) against the numpy
  reproduction and writes `logs/stft_equivalence_check.json`.
- The CLAP model / query encoders are NOT needed for S1a (no model inference, no
  training); `laion_clap` being unavailable in the local envs does not block the stage.
