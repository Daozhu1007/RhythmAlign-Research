# S2A — DEV_RESULTS (corrected pipeline, run 2 of 2)

**Date:** 2026-09-15 · DEV = 3 reference-verified recordings / 15 windows, all with
accepted alignments. Full per-checkpoint tables: `TRAINING_LOG.public.md`; raw
numbers in `logs/06_train_full_log.json` and `logs/08_selection.json`.

## Training trajectory (DEV, medians)

| upd | adv vs S1R on injection (dB) | causal gap zero (dB) | causal gap wrong (dB) | click ret (dB) | consistency | ratio vs S1R (dB) |
|---|---|---|---|---|---|---|
| 250 | −0.0003 | 0.0002 | 0.0001 | −0.016 | 0.0380 | −0.187 |
| 500 | +0.0016 | 0.0007 | 0.0001 | −0.029 | 0.0373 | −0.138 |
| 750 | −0.0014 | −0.0003 | 0.0001 | −0.008 | 0.0373 | −0.272 |
| 1000 | +0.0074 | 0.0012 | 0.0002 | +0.001 | 0.0374 | −0.115 |
| 1250 | +0.0007 | −0.0002 | 0.0000 | −0.019 | 0.0374 | −0.270 |
| 1500 | −0.0004 | 0.0003 | 0.0000 | −0.037 | 0.0379 | −0.266 |

S1R reference row under the IDENTICAL evaluation code (frozen S1R, zero adapter):
consistency 0.0415, click retention −0.002 dB, suppression advantage exactly 0
(sanity check of the harness itself).

## What the DEV evidence says

1. **Safety axes all hold everywhere** (section 25): zero collapse at every eval,
   no passthrough drift (median margin ≤ 0.01), S1R anchors preserved
   (divergence ≈ 0.006 vs the 0.053 half-passthrough bound), wrong reference
   safe (median ratio ≥ −0.27 dB, min ≥ −0.63 dB).
2. **Consistency stays slightly better than S1R** (0.0373–0.0381 vs S1R's 0.0415)
   — inherited from the base, not improved further by the adapter.
3. **The causal reference effect is absent.** The median advantage over S1R on
   known-reference nuisance oscillates within ±0.008 dB with no upward trend
   after update 500; correct-vs-wrong separation never exceeds 0.0002 dB. The
   section-25 check `correct_ref_causally_used` (≥ 1.0 dB) fails at EVERY
   checkpoint; best observed gap 0.0012 dB (upd 1000), i.e. ~1000× below the
   gate and ~100× below anything a listener could hear.

## Selection outcome

**NO checkpoint passed all section-25 checks.** The only failing check is the
causal-use one; every safety/no-regression check passes. The section-26 gate
therefore returns **REFERENCE_ADAPTER_IGNORED** before any held-out test.
`full_upd01000` (best advantage AND best causal gap among candidates, all safety
checks true) is recorded as the DOCUMENTATION checkpoint for the held-out
negative result — not a product checkpoint.
