# S2A — FAILURE_CASES and honest negatives

**Date:** 2026-09-15. The stage's headline negatives, stated first and preserved.

## 1. The reference adapter does not use reference content (primary negative)

Across two independent training runs, all DEV checkpoints, and 20 held-out
windows over 5 song-disjoint recordings, the model's behavior is identical for
CORRECT, ZERO, and WRONG references (median correct-vs-wrong gap 0.000 dB; best
suppression advantage over S1R 0.027 dB — 50× below the section-26 evidence bar
of 1.5 dB). Verdict C via the section-26 stop condition
REFERENCE_ADAPTER_IGNORED. No human listening was requested (section 30 gate
fails; section 44 followed).

## 2. Run 1 was invalidated by a reference-position sign bug (recorded honestly)

`s2a_common.ref_position_s` originally returned `t_hand + offset` instead of the
production convention `t_ref = t_hand − offset` — every reference segment in run
1 was ~20 s misaligned. The bug was caught by a held-out window-picker assertion
(windows landing beyond the reference's covered span). Consequences, stated
exactly:

- Run 1's injection task was INTERNALLY consistent (the injected nuisance was
  always T(m_shift) and the conditioning input was the same m_shift), so run 1
  is valid evidence that the adapter does not content-match even a
  self-consistent reference. Its DEV trajectory (+0.010 dB best advantage,
  ~0.002 dB causal gap at upd 1500) is preserved in git history.
- All reference-dependent artifacts of run 1 (targets kept — they are
  sign-independent — but window-coverage decisions, training, DEV eval,
  selection) were REDONE from scratch as run 2 with identical declared
  hyperparameters (lr 2e-5, α ∈ [0.12, 0.45], 1500 updates). This was a bug fix,
  not tuning; run 2 is the stage's record run. GPU total ≈ 2.8 h of the 24 h
  budget; the 2-full-run maximum was reached (run 2 = final).

## 3. The conservative adapter CAN open — it just has no incentive to

The zero-init output head receives gradient from step 0 and the gate does move
(DEV losses shift; small uniform mask dips appear). The failure is not
mechanical. It is structural: with the S1R anchor pinning clean-mixture behavior
to the teacher (section 20A by design) and S1R's mask already suppressing music
strongly, the learnable margin from reference-explained excess music is tiny in
absolute loss terms (~0.01 L1), and the cheapest way to claim it is a
reference-presence prior, not content matching. Nothing in the section-20 loss
set rewards content specificity specifically — a design gap this stage
demonstrates empirically.

## 4. What S2A does NOT establish

- That NO reference-conditioning design can work — only that the minimal
  near-no-op adapter + section-20 objective (the protocol's prescribed smallest
  defensible design) extracts no usable signal from the aligned reference under
  this training signal. Larger adapters, different fusion points, or objectives
  that directly reward content-specific suppression remain untested and are NOT
  recommended lightly: they would trade away the anti-catastrophe guarantees
  that S1W/S1R established as necessary.
- That alignment quality is the limiting factor: accepted-pair residuals are
  ≤ 0.05 s with multi-window stability, so conditioning inputs were accurate.
- Any listening claim: no human evaluation occurred (machine gate failed first).

## 5. Smaller defects found and fixed during the stage (none affected decisions)

- First held-out run selected windows beyond the reference-covered span
  (zero-reference segments → −240 dB residual artifacts) and had a boolean-
  indexing bug in the ref-attenuation metric; both fixed before the recorded
  run. All superseded artifacts overwritten.
- The Python `hash()` salt made one DEV augmentation seed process-dependent;
  replaced with crc32 before any recorded evaluation.
- A zero-init scalar gate on the adapter created a gradient deadlock
  (gate=0 × zero-init head ⇒ zero grads); replaced by the zero-init output head
  itself acting as the section-15 residual gate (gradient flows from step 0).
