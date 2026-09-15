# S2A — GENERALIZATION_RESULTS (held-out, song-disjoint; protocol sections 12/28)

> **POST-HOC INTEGRITY NOTE (2026-09-15, RTA-0 audit).** "vs S1R" quantities
> below used the post-training S2A base model (its mask head trained with the
> adapter), not an immutable original-S1R instance; they are not a valid
> frozen-original-S1R comparison. The within-model CORRECT/WRONG null effect
> stands. Evidence: `docs/reviews/RTA0_INTEGRITY_AUDIT.md` §3.2.

**Date:** 2026-09-15 · Documentation checkpoint `full_upd01000.ckpt` (see
DEV_RESULTS for the section-25 failure). Held-out set: 5 recordings, each with an
accepted pristine reference, every song identity ABSENT from TRAIN (4 windows per
recording, deterministic spread inside the reference-covered span; exact 10-s
chunk OLA; frozen protocol).

Per-window numbers: `logs/09_real_test.json` (RAW / ZERO-SHOT / S1R / S2A
CORRECT / S2A ZERO / S2A WRONG).

## Result

- **The null effect transfers completely.** Median suppression advantage vs S1R
  under controlled injection: +0.013 dB (max +0.027 dB; 0/20 windows ≥ 1.5 dB).
  Correct and wrong references remain indistinguishable (median gap 0.000 dB).
- **No instability anywhere.** Zero collapse flags across 20 windows × 3 S2A
  reference modes; output RMS within ±0.02 dB of S1R on every window; transient
  retention identical to S1R (median −0.386 vs −0.386 dB vs raw).
- Per-recording: 19/20 windows show a positive but negligible (+0.006…+0.027 dB)
  advantage over S1R — uniform in magnitude across all five recordings and never
  specific to the correct reference, consistent with a residual "any reference
  present" artifact of the adapter rather than content use.

## Interpretation

Held-out generalization was never in doubt — the adapter barely changes the
model, so it "generalizes" trivially. The meaningful generalization question
(does reference-driven suppression transfer to unseen songs?) is answered
negatively: the reference effect does not EXIST on held-out material, so there
is nothing to transfer. TEST songs were fully song-disjoint from TRAIN, so
memorized song identities cannot explain even the residual ±0.02 dB deltas.
