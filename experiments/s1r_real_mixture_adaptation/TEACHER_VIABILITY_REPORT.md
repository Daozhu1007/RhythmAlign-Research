# S1R — PHASE 0: REAL-MIXTURE TEACHER VIABILITY REPORT

**Date:** 2026-09-14 · **Teacher gate: PASS (preferred gate)**
Prerequisite: zero-shot parity re-reproduced bitwise (`logs/00_parity_check.json`,
max |Δ| = 0.0 vs the S1E stored output on c6_dense_golden under torch 2.14.0+cu126) —
the frozen teacher is exactly the model validated in S1E/S1W.

## What was audited

The frozen zero-shot CLAPSep teacher (audio query Q1, zero negative — the exact S1E/S1W
policy) was audited on **authentic raw handcam mixtures** from the frozen S1W
TRAIN + DEV recording identities ONLY (36 TRAIN / 7 DEV; SEALED and
TEST_GENERALIZATION identities untouched). 252 candidate windows (≤7 per recording)
were sampled deterministically in the song-active core, seeded by the S1W conservative
target-proxy event times (proxy times were used only to TAG windows — never as
targets, per protocol section 28).

For the SAME central ~6 s of each window the teacher ran under:

- three 10-s context views (early / canonical / late boundaries; exact trained chunk
  length, no zero padding), and
- ±3 dB global-gain views of the canonical context (known gain undone before
  comparison) — no semantic-changing augmentation of any kind.

Multi-metric consistency diagnostics (protocol section 13) were computed per window;
no single scalar decided acceptance.

## Results (252 windows)

| diagnostic (across-view, central crop) | p10 | median | p90 |
|---|---|---|---|
| pairwise waveform correlation (min over 3 pairs) | 0.914 | **0.976** | 0.993 |
| log-mag spectral correlation (min pair) | 0.985 | 0.993 | 0.996 |
| MR-STFT log-mag distance (max pair) | 0.020 | 0.038 | 0.074 |
| output RMS spread across 5 views (dB) | 0.17 | 0.49 | 1.18 |
| worst view vs canonical output (dB) | −0.72 | −0.20 | 0.0 |
| teacher output vs raw window (dB) | −5.5 | −4.6 | −3.9 |
| click-band retention vs raw (dB) | −1.15 | −0.48 | −0.23 |

Reading: on real handcam material the ZERO-SHOT teacher is strongly chunk/context
robust — the collapse and context sensitivity recorded in S1W were properties of the
S1W ADAPTED model, not of zero-shot. Worst-view behavior stays within ~0.7 dB of
canonical at p10, with no window showing a >12 dB context collapse and none
amplifying >6 dB. Gain equivariance is likewise benign (the spread column includes
the corrected ±3 dB views).

## High-confidence teacher anchors

240 / 252 windows (95.2%) met ALL conservative acceptance criteria simultaneously
(waveform corr ≥ 0.85, spectral corr ≥ 0.90, MR-STFT ≤ 0.35, RMS spread ≤ 2 dB,
no ≤−12 dB view, no ≥+6 dB view, onset lag ≤ 20 ms, transient-envelope corr ≥ 0.80,
teacher output within [−15, +3] dB of raw). The 12 rejections were: 7 RMS-spread,
7 waveform-corr, 1 spectral-corr, 1 MR-STFT distance (some windows fail multiple).

Accepted anchors: **201 TRAIN anchors = 1206 s** (every one of the 36 TRAIN
recording identities; largest single-recording share 3.0%) + **39 DEV anchors**
(DEV is used for evaluation only, never for training).

Interaction coverage among TRAIN anchors: transient-rich 172, weak 201, dense 116,
friction 2. Friction is genuinely scarce in the corpus (S1W audit: 8 friction cuts
across 7 recordings) but IS represented; all four required interaction types are
covered, and anchors span all 36 recordings (36 different songs) — high-confidence
behavior is not restricted to one song.

## Gate decision

| requirement (protocol §15) | threshold | measured | pass |
|---|---|---|---|
| total anchor duration (TRAIN) | ≥ 120 s | **1206 s** | ✓ |
| independent TRAIN recordings | ≥ 8 | **36** | ✓ |
| max single-recording share | ≤ 20% | **3.0%** | ✓ |
| transient-rich / weak / dense / friction coverage | all present | 172 / 201 / 116 / 2 | ✓ |
| not one song | multiple | 36 songs | ✓ |

**TEACHER GATE: PASS** under the preferred (not reduced) thresholds. No threshold was
relaxed; the acceptance criteria are exactly the protocol's suggested starting values.

## Interpretation guard

Consistency is not correctness: these anchors certify that the frozen teacher is
ROBUST on these real windows, not that it is RIGHT. The teacher remains an imperfect
anchor (S1E/S1W defects: muffling, underwater texture, discontinuity). S1R training
uses these windows as conservative preservation anchors — the student must not drift
from verified zero-shot selectivity there — and explicitly treats them as pseudo-label
evidence, never ground truth.

## Artifacts

- `logs/00_parity_check.json` — bitwise parity reproduction
- `logs/01_phase0_summary.json` — full machine-readable summary
- `TEACHER_ANCHOR_MANIFEST.public.json` — anonymized anchor manifest (stable IDs only)
- `work/private/phase0_windows.private.json` — per-window diagnostics (local paths)
- `work/private/teacher_precompute.private.json` — stored teacher crops manifest
- Scripts: `scripts/00_parity_check.py`, `scripts/01_phase0_audit.py`,
  `scripts/02_teacher_precompute.py`, `scripts/s1r_common.py`
