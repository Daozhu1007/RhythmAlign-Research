# RTA1 — Controlled Mixture Protocol

**These constructed mixtures are evaluation fixtures with exact bookkeeping.
They are NOT ordinary real arcade mixtures and must never be described as
such.** Real-mixture claims belong to the historical S-stage records.

Tool: `scripts/04_build_controlled_mixtures.py` (supports `--plan` /
`--dry-run`; refuses to build until ingested, QC-passed capture exists).

## Composition

Each mixture: `y = p + g · n_shifted`, where

- `p` = a QC-passed REAL target region (from take A, quiet interaction),
- `n` = a QC-passed REAL nuisance region (take B playback-only preferred;
  take C regions only where documented), time-shifted by a recorded offset,
- `g` = recorded scalar gain giving the required target/nuisance ratio.

No resampling of stored material (all derives from the same ingested domain);
no learned processing anywhere; integer-sample alignment with the shift
recorded.

## Required panel (pre-registered)

- **24 mixtures × ~6 s**, 8 per session × 3 sessions.
- **Target strata (mandatory coverage):** `weak`, `strong_button`,
  `friction_tail`, `dense` — every session contributes all four strata
  (2 mixtures per stratum per session: one at −10 dB, one at 0 dB nuisance).
- **Nuisance levels (active-target):** nuisance RMS at **−10 dB** and **0 dB**
  relative to the target region RMS (measured on the actual selected regions).
- Sessions 02 and 03 (16 mixtures) are the **held-out evaluation panel**;
  session 01 (8 mixtures) is the tuning/development panel. The holdout split
  is by session — declared before any baseline or oracle runs on the panel.

## Bookkeeping (every mixture records)

target take content hash + region [t0, t1]; nuisance take content hash +
region + shift; target RMS and nuisance RMS after gain; gain `g`; sample rate;
channel layout (working mono derivation + its formula); output file SHA-256.
Recipes are written to `work/private/mixture_recipes.private.json` with a
public anonymized mirror (hashes and numbers only, no paths/devices).

## Scoring metadata

For each target region, the recipe links the stratum and (when provided by
the manual pass) the scored event lists: weak-event times, strong-event times,
friction/tail intervals. These drive the event-preservation criteria in
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). Mixtures without scored
events for their stratum are excluded from event-based criteria (recorded, not
silently dropped).
