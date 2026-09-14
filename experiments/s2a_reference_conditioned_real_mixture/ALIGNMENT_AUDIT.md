# S2A — ALIGNMENT_AUDIT

**Date:** 2026-09-15 · **Method:** production RhythmAlign analysis path, read-only
(`auto_sync._align_hybrid` / `_align_onset`, 22050 Hz mono, hop 512 — the exact
functions the product and R1 use), plus three conservative S2A additions.
Convention: `offset = t_handcam − t_ref`; the handcam covers
`ref[offset .. offset + duration]`.

## Acceptance criteria (declared a priori; protocol section 10)

| Check | Threshold | Purpose |
|---|---|---|
| z_hybrid (production confidence) | ≥ 2.0 | production-grade global alignment |
| offset stability | densest cluster of ≥3 estimates (full pass + up to 5 independent 60-s segment passes) containing the full-pass value, spread ≤ 0.05 s | drift & repetition-trap resistance |
| overlap | ≥ 60 s | usable song-active material |
| song identity | aligned log-mel agreement ≥ 0.25 AND ≥ shifted-position baseline + 0.10 | catch wrong-song / red-white-variant pairings |

## Why the extra checks exist (failure modes actually observed)

1. **Repetition traps.** Two pairs (分诊4, 海底谭2) had individual 60-s segments
   whose best hybrid alignment landed 40–119 s away from the true offset — the
   same song at a different section (maimai songs are highly repetitive). A
   max-residual drift rule would have rejected one correct pair (分诊4: full pass
   + 2 segments agree within 0.02 s) and accepted nothing safer; the densest-
   cluster rule with full-pass membership accepts the verified cluster and ignores
   trapped outliers. 海底谭2's cluster never formed across repeated runs
   (near-tied segment peaks flip under numba parallel scheduling) → correctly
   REJECTED as ambiguous, never forced.
2. **Wrong-song risk.** Red/white variant pairs (红宙天↔宙天, 白39↔39,
   白妄想↔妄想感伤代偿联盟) were candidates precisely because names suggest
   different versions of one song. The identity check (aligned log-mel agreement
   vs a ±55-s shifted-position baseline of the same reference) separates them:
   accepted red/white pairs score 0.29–0.54 vs baselines ≤0.09, while the rejected
   分诊1/海底谭3/4/5/零对话/共感怪物AP sit at ≤0.25 agreement or below-margin
   separation.
3. **Onset-only verification is unreliable here** (repetitive rhythm self-aligns at
   measure offsets); it was tried and replaced by the segment-stability +
   identity design.

## Results (24 accepted / 31 candidates)

- offsets ≈ +6.6 … +32.7 s (maimai attract screen before song start; the +32.7 s
  case is a recording that starts mid-attract), residuals ≤ 0.04 s.
- ACCEPTED: 16 TRAIN + 3 DEV + 5 TEST recordings (see REFERENCE_CORPUS_AUDIT for
  the split view; per-pair numbers in `REFERENCE_MANIFEST.public.json`).
- REJECTED (7): 共感怪物AP (SEALED — identity 0.19), 分诊1 (identity), 海底谭2
  (unstable segment cluster), 海底谭3/4/5 (identity margin), 零对话 (unstable +
  identity). Each rejection is a conservative decision recorded with its failed
  checks; none was silently forced.

## Honest limitation

The song-identity check is a research-grade heuristic (median smoothed log-mel
cosine against a shifted-baseline control), not a perceptual verification. It
cleanly separates the observed distribution (accepted ≥0.27 vs rejected ≤0.25 with
failed margins), but two accepted pairs sit near 0.27–0.29 — their alignment is
trusted because segment stability, z-score, and identity margin ALL agree, not
because any single number is large.
