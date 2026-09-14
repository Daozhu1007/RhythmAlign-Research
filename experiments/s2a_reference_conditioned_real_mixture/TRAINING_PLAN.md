# S2A — TRAINING_PLAN (declared a priori; protocol sections 20/23-26)

**Date declared:** 2026-09-15, before the micro-pilot.

## Objective

Train ONLY the reference adapter + final mask head on top of the frozen S1R
checkpoint so that the model (a) remains S1R on unmodified real mixtures, and (b)
uses the aligned pristine reference to reject KNOWN reference-correlated excess
music — while never deleting authentic interaction, never collapsing, never
passing the mixture through, and never merely memorizing song identities (TEST is
song-disjoint).

## Loss (all four section-20 signals; kernels reused verbatim from S1R)

```
L = 1.00 * anchor_distill(F(x_can, m_can), S1R(x_can))
  + 0.75 * injection_invariance(F(x_aug_can, m_can), S1R(x_can))
  + 0.75 * cross_context_consistency(F(x_aug_view, m_view) vs F(x_aug_can, m_can))
  + 1.00 * anti_collapse_hinge(vs S1R target, incl. transient-band floor)
  + 0.10 * L2-SP weight anchor (toward frozen S1R trainable weights)
```

Per update: one randomly chosen window; augmentation fresh each step; three real
context views (early/canonical/late) with per-view aligned reference segments.

## Data

82 TRAIN windows (492 s central crop) across the 16 reference-verified TRAIN
recordings; 15 DEV windows across 3 DEV recordings; all windows require reference
coverage ≥ 0.9 of the 14-s span. No sealed or TEST material anywhere.

## Schedule (section 24)

- optimizer AdamW, lr 2e-5 (conservative end of the 2e-5..1e-4 adapter range),
  weight decay 0.01, grad clip 5.0, accumulation 8 (matches S1R machinery)
- micro-pilot: 200 updates, eval every 50 (section 23 conditions)
- primary: 1,500 updates, eval every 250; hard cap 2,000; ≤ 2 full runs;
  aggregate GPU budget ≤ 24 h on the RTX 4060 8 GB
- in-loop guards: non-finite loss/grad, collapse, passthrough drift

## DEV selection (section 25) — all must hold, not a single composite

zero collapse (12-dB rate 0, 6-dB rate ≤ 0.10, median ratio ≥ −3 dB) · no raw
passthrough (margin ≤ 0.05, divergence ≤ 0.5× raw-vs-S1R scale) · S1R anchor
preserved (divergence ≤ max(0.5×pt_div, 1.0), click retention ≥ −3 dB) · correct
reference causally affects output (causal_gap_zero ≥ 1.0 dB median) · wrong
reference safe (ratio ≥ −3 dB median, ≥ −6 dB min) · injected nuisance suppressed
better than S1R (advantage > 0 dB median) — plus NO material regression of
transient retention and cross-context consistency vs S1R's own DEV numbers.

## Causal reference-use gate before TEST (section 26)

On DEV: `CORRECT_REFERENCE` must beat `WRONG_REFERENCE` and `ZERO_REFERENCE` on
known-injection suppression by ≥ 1.5 dB median, with median transient-retention
regression ≤ 1 dB vs S1R and no catastrophic individual cases. Thresholds may be
adjusted only on TRAIN/DEV evidence and only before TEST, with the adjustment
documented. If correct/wrong/zero are indistinguishable → STOP:
`REFERENCE_ADAPTER_IGNORED`, no human listening.

## Freeze before TEST (section 27)

Weights + reference preprocessing (frozen STFT + per-chunk standardization) +
alignment settings (accepted offsets; production threshold) + inference protocol
(exact 10-s chunk overlap-add, per-chunk aligned reference) + query + checkpoint
hashes recorded; no tuning after TEST begins.
