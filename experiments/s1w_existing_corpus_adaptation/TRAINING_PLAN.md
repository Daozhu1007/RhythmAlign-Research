# S1W — TRAINING_PLAN

**Date:** 2026-09-12 · status: executed after ALL implementation sanity tests passed
(`logs/10_sanity.json`, 13/13 PASS including split isolation and 5.14 GB VRAM profile).

## Goal

Domain-adapt the S1E CLAPSep to RhythmAlign's target — authentic player–machine
physical interaction audio — using ONLY existing-corpus weak supervision, while
REDUCING the S1E human failure modes (muffling, weak-hit deletion, friction loss,
target deletion). Preservation matters as much as suppression.

## Substrate and scope (fixed)

- Same CLAPSep checkpoint as S1E (sha256-verified against `S1E/RUN_MANIFEST.json`);
  zero-shot parity vs S1E's stored output is EXACT (max |Δ| = 0.0 on c6_dense_golden,
  `logs/04_parity_check.json`), so the training comparison shares a trusted baseline.
- Trainable: `decoder_model` (HTSAT decoder + mask/output head), 149 tensors.
- Frozen: CLAP text/query encoder, audio_branch (incl. LoRA), BatchNorm running state.
- Representation unchanged: bounded real magnitude mask + mixture phase (S1a-audited).
  If muffling persists despite better suppression, the bottleneck hypothesis is the
  representation, not the training (protocol section 42).
- Query policy: S1E's fixed audio query Q1, zero negative. No sweep anywhere.

## Data (weak supervision; exact bookkeeping)

- Target proxies: 156 conservative events / 238 s from 32 of 36 TRAIN recordings
  (strong 36 / ordinary 81 / weak 31 / friction 8) — see `TARGET_PROXY_AUDIT.md`.
- Nuisance: pre-song attract music, arcade ambience, unrelated impacts from TRAIN
  (and DEV-recording variants for DEV mixtures), pristine reference music (independent
  source, LOCAL ONLY). No VAD-detectable speech exists in TRAIN/DEV — documented
  absence, not an omission.
- Mixtures: y = guard·(s + gain·n); TNR solved exactly; per-example recipe stored
  (files, placements, gains, measured TNR). Distribution 50/20/20/10
  (mix / target-only / nuisance-only / hard), TNR grid −20/−10/0/+10 dB,
  hard negatives −30/−25/−20 dB incl. system-audio-over-weak-target overlaps.
- Splits frozen BEFORE mining (seed 20260912): TRAIN 36 / DEV 7 / TEST_GEN 14 /
  SEALED 1 recording identities; byte-identical duplicate excluded; all derived
  `_synced` files excluded; leakage checked programmatically.

## Objective (protocol section 26)

L = 1.00·waveform L1 + 0.50·MR-STFT(256/1024/2048, log-magnitude) + 0.25·pre-emphasis HF L1
(x[n]−0.95·x[n−1]); target-absent examples: L1+L2² energy penalty; target-only:
identity. No visual/chart/semantic/leakage-proxy objectives.

## Run plan (protocol sections 30–32)

1. Primary run: seed 20260912, AdamW lr 1e-4 (decoder only), microbatch 1 × grad-accum 8,
   ≤ 2000 optimizer updates, bf16 autocast, DEV eval every 250 updates on 160 fixed
   constructed DEV examples with separate preservation/suppression axes.
2. Pause-and-diagnose if no meaningful DEV movement by ~1000 updates.
3. One replication seed (20260913, identical config) only if the primary run is clearly
   stable and promising. Max 2 runs, ≤ 24 GPU-hours total.
4. Checkpoint selection: DEV only; composite = preservation ratio + suppression gain
   (components always shown); guardrail: weak-stratum preservation must not regress
   > 10% vs zero-shot. Sealed test untouched until everything is frozen.
5. Only after freezing: sealed primary test (6 challenge groups × RAW/zero-shot/adapted),
   generalization test, fixed-gain blind pack (18 items), owner listening.
