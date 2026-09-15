# RTA1 — Real-Target Acquisition + Bounded-Mask Headroom Pilot

**Status:** `READY_FOR_REAL_TARGET_CAPTURE` (see [STATUS.json](STATUS.json))
**Nature:** PREPARATION COMPLETE; the experiment itself has NOT run — no real
capture exists yet. Nothing in this stage may fabricate or simulate the capture.

## The one open dependency: a controlled physical capture

The owner's **only** task in this stage is to physically record a small
controlled capture set at the arcade (~15 minutes of usable material across
3 sessions), following [OWNER_CAPTURE_CHECKLIST.md](OWNER_CAPTURE_CHECKLIST.md)
(one printed page, no ML knowledge needed) and the fuller
[CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md). Original files go, unedited, into
the local capture inbox (layout below). Everything after that is prepared
tooling.

## Scientific question

On trustworthy phone-domain interaction recordings, does the **existing bounded
real-mask representation** have substantial headroom over **frozen zero-shot
CLAPSep** and **frozen S1R**? This separates:

- **target-selection / supervision failure** (oracle wins → supervision was the
  problem), from
- **representation / phase limitation** (bounded oracle fails but the complex
  oracle succeeds → phase freedom is implicated).

No model training is required or authorized. See
[ORACLE_PROTOCOL.md](ORACLE_PROTOCOL.md), [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md),
[DECISION_RULES.md](DECISION_RULES.md).

## Directory layout (local; all ignored by Git)

```
work/
  capture_inbox/          <- owner copies ORIGINAL camera/phone files here, unedited
    session_01/  session_02/  session_03/
  private/                <- private manifests, hashes, QC details (never published)
  extracted/              <- derived analysis copies (reproducible; originals untouched)
  oracle/                 <- oracle/baseline audio outputs (never published)
  listening/              <- future A/B material ONLY if DECISION_RULES requires it
```

These directories are git-ignored (`work/` + all media patterns in the repo
`.gitignore`). Placeholder READMEs inside them explain what belongs where.
Originals in `capture_inbox/` are **never** modified, transcoded, renamed
destructively, or deleted by any script.

## Pipeline (prepared, in run order)

| script | purpose | runs now? |
|---|---|---|
| `scripts/01_ingest_capture.py` | discover inbox media, probe (ffmpeg), hash, duration/codec/channels, clipping stats, private + public manifests; `--dry-run` supported; never modifies originals | dry-run validated on synthetic fixtures only |
| `scripts/02_channel_diagnostic.py` | do the native channels carry independent spatial information? (correlation, level difference, mid/side, coherence) — measured, never assumed | validated on synthetic fixtures only |
| `scripts/03_truth_qc.py` | clipping / noise floor / scored-active vs rest / continuity / manual flags → PASS / REVIEW / FAIL (+ `TRUTH_QUALITY_INSUFFICIENT`) | validated on synthetic fixtures only |
| `scripts/04_build_controlled_mixtures.py` | build the 24 × ~6 s evaluation-fixture panel (8/session; strata weak / strong-button / friction-tail / dense; nuisance −10 dB and 0 dB) with exact recipe bookkeeping | plan mode only until real capture exists |
| `scripts/05_run_frozen_baselines.py` | zero-shot CLAPSep + ORIGINAL S1R checkpoint as **independent immutable instances**, checkpoint hashes recorded; S1W artifact explicitly refused | after capture + QC |
| `scripts/06_run_bounded_oracles.py` | binwise bounded real mask; waveform-optimized bounded mask; target roundtrip; fixed-mask source-path analysis (target damage vs nuisance leakage) | after capture + QC |
| `scripts/07_run_complex_oracle.py` | complex-ratio / phase-capable diagnostic oracle | after capture + QC |
| `scripts/08_evaluate_headroom.py` | aggregate into the pre-registered headroom table; apply DECISION_RULES | after 05–07 |

## What is already validated (TOOLING VALIDATION ONLY)

Unit tests (`tests/`, run with the S1R stage virtualenv) cover STFT axis
correctness, ISTFT roundtrip, fixed-mask linearity, baseline immutability
guards, source-path metrics, and manifest privacy. Ingest dry-run, manifest
generation, oracle sanity, and source-path linearity were executed on **tiny
synthetic fixtures created only to validate the software**. These fixtures and
their outputs are NOT scientific evidence of anything.

## Truth definition (short form)

The target is the acoustic contribution of the actual player–machine physical
interaction at the recording phone/microphone position — timing, source-image
level, room response, decay/tails, weak contacts, continuous friction.
Excluded: cabinet playback music, neighboring machines, NPC/system speech,
unrelated human speech, ambient unrelated noise. The target is never
synthesized or "cleaned" with a learned model. Full definition:
[TRUTH_DEFINITION.md](TRUTH_DEFINITION.md).

## Privacy boundary

Public in this directory: protocols, scripts, tests, anonymized schemas, this
README. Private/ignored: captured audio/video, absolute local paths, private
session mappings, oracle audio outputs, future listening audio, checkpoints,
personal metadata. `01_ingest_capture.py` emits a public anonymized manifest
whose structure is tested by `tests/test_manifest_privacy.py`.
