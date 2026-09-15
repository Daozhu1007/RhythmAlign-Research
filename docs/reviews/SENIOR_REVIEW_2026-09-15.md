# Senior Research Review — 2026-09-15 (public record)

An independent senior research review of the RhythmAlign research program was
supplied by the owner on 2026-09-15. This file is the **public-safe summary**
of that review combined with this repository's independent verification of it.
The full private review text is not reproduced (it contains local environment
detail); every code-level claim it made was re-verified here before any record
was changed — see `RTA0_INTEGRITY_AUDIT.md` for the evidence and
`RTA0_CLAIM_CORRECTION_MAP.md` for the wording repairs.

## Review scope

- The S2A reference-conditioning stage (adapter architecture, trainable scope,
  evaluation and comparator isolation, split/reference handling).
- The S1R real-mixture adaptation stage (loss implementation, interpretation
  wording, listening claims).
- The R1/S1E/S1W public conclusions and their scope.
- The program's research priority and next-step logic.

## Major verified findings (all independently confirmed in code)

1. **S2A adapter geometry:** the reference adapter's stride-2 transposed
   convolution doubles both STFT axes and the output is cropped back, so the
   implementation does not preserve one-to-one aligned time–frequency
   geometry. Initialization parity stays valid as a no-op check but does not
   validate learned TF alignment. S2A is not a clean test of aligned local
   reference fusion. (Causal contribution to the null result: not measured.)
2. **S2A comparator isolation:** the base model's final mask head was trainable
   during S2A, and the reported "S1R baseline" during DEV/TEST reused that
   post-training base model instead of an immutable original-S1R instance. The
   reported S2A-vs-S1R deltas are therefore **not** a valid frozen
   original-S1R comparison. Within-model correct/zero/wrong-reference
   comparisons remain valid descriptive evidence.
3. **Transient-protection loss axis:** the S1R loss library's "2–9 kHz"
   transient guard indexed the time axis of the STFT instead of the frequency
   axis for the actual 1-D inputs, so the documented guard was not implemented
   as intended. Used by S1R and S2A **training only**; all published
   evaluation metrics used a different, correct implementation. The total-RMS
   anti-collapse hinge is separate and valid.
4. **Split / reference exposure:** S2A DEV contains two ex-S1R-TRAIN recording
   identities and shares song identities with S2A TRAIN (both documented
   in-stage), and DEV wrong-reference selection drew on TEST songs'
   references. Correctly classified: **test-side reference-content exposure
   during model selection** — not gradient leakage; no TEST mixture entered
   training.
5. **Public wording:** several conclusions were stated more broadly than the
   evidence supports (cancellation, pretrained-frontier, S1W root cause, S1R
   general-law phrasing, S2A headline). All repaired via dated banners.

## Dominant current bottleneck

> **INSUFFICIENT TRUSTWORTHY REAL TARGET SUPERVISION / EVALUATION TRUTH.**

No stage so far has possessed a ground-truth recording of the actual target
(the player–machine physical interaction at the phone). Every learner to date
was trained and judged against proxies (teacher outputs, constructed mixtures,
descriptive metrics). Under those conditions, additional learners optimize an
unverified objective.

## Revised research priority

1. Acquire a small, controlled, trustworthy **real-target** capture set.
2. Establish truth quality through explicit QC gates (quiet ≠ ground truth).
3. Run **frozen** baselines (zero-shot CLAPSep; the original S1R checkpoint
   loaded as an immutable independent instance).
4. Run a **bounded-mask oracle headroom** diagnostic to separate
   target-selection/suppression failure from representation/phase limitation.
5. Only then decide whether any future learning is justified.

**Research training status: PAUSED PENDING TRUSTWORTHY REAL-TARGET EVIDENCE.**

## Methodological risks the review surfaced (now tracked)

- Comparator immutability: baselines must be loaded as independent, immutable
  instances with recorded checkpoint hashes (RTA1 tooling enforces + tests).
- STFT-axis correctness: axis-dependent code must carry explicit axis tests.
- Selection-time exposure: held-out content must not inform checkpoint
  selection, including as "wrong" inputs.
- Quiet-capture fallacy: a quiet recording is not automatically target truth;
  contamination must be measured against scored target activity (~20 dB
  separation target where measurable, else `TRUTH_QUALITY_INSUFFICIENT`).
- Scope discipline: negatives are recorded with their tested scope; no
  universal ceilings or class-level failures may be claimed from single-
  recording or single-configuration evidence.

## Next authorized experiment

**RTA1 — real-target acquisition + bounded-mask headroom pilot**
(`experiments/rta1_real_target_headroom_pilot/`). Preparation is complete and
the stage is `READY_FOR_REAL_TARGET_CAPTURE`: the owner's only task is a
small controlled physical capture (~15 minutes of usable material across 3
sessions). On trustworthy phone-domain recordings, the pilot asks whether the
existing bounded real-mask representation has substantial headroom over frozen
zero-shot CLAPSep and frozen S1R. Pre-registered decision rules and success
criteria are in the stage's `DECISION_RULES.md` and `EVALUATION_PROTOCOL.md`.

## What is NOT authorized

- No model training or fine-tuning of any kind before the RTA1 gate resolves.
- No new extraction research stage.
- No human listening for RTA-0 (none was performed); the future pilot uses
  waveform truth first, with at most 6 short A/B comparisons (≤10 minutes)
  only if a final decision stays ambiguous after waveform evidence.
- No modification of the product repository.
- No deletion or silent rewriting of historical records, including the
  mistakes documented here.
