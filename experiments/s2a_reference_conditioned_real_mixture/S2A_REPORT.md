# S2A — REPORT: Reference-Conditioned Real-Mixture Extraction (final)

> **POST-HOC INTEGRITY NOTE (2026-09-15, RTA-0 audit).** The verdict headline
> below predates the audit. Confirmed post-hoc: (1) the adapter's stride-2
> transposed conv + crop does not preserve one-to-one TF geometry, so this is
> not a clean test of aligned local reference fusion (init parity stays a valid
> no-op check only); (2) the "S1R" comparator in §2 was the post-training S2A
> base model (its mask head trained with the adapter), not an immutable
> original S1R — those deltas are not a valid frozen-original-S1R comparison.
> Current-status label: **NO USEFUL REFERENCE-CONTENT EFFECT WAS DEMONSTRATED
> BY THE RECORDED S2A IMPLEMENTATION.** The within-model CORRECT/ZERO/WRONG
> null effect stands as descriptive evidence. Evidence:
> `docs/reviews/RTA0_INTEGRITY_AUDIT.md` §3.1–3.2.

**Verdict (original 2026-09-15 wording): C — REFERENCE CONDITIONING DOES NOT
HELP**
(decision record: [S2A_DECISION.json](S2A_DECISION.json); causal evidence:
[REFERENCE_ABLATION_RESULTS.md](REFERENCE_ABLATION_RESULTS.md); held-out:
[GENERALIZATION_RESULTS.md](GENERALIZATION_RESULTS.md),
[REAL_TEST_RESULTS.md](REAL_TEST_RESULTS.md))

S2A asked the decisive follow-up to S1R's verdict B: RhythmAlign natively
possesses the time-aligned pristine song reference — does that genuinely NEW
information source let a model move beyond the teacher where teacher-only
distillation could not? The answer, established with a clean pipeline and full
causal ablations, is **no for the minimal adapter this stage was allowed to
build**: the model ends up behaviorally identical with the correct reference,
with a wrong song's reference, and with no reference at all.

## 1. What was built (worked exactly as designed)

- **Corpus:** 22 pristine song references found inside the existing handcam tree
  (owner-collected, class-C assets; no downloads, no `_synced` output, no
  product artifacts). Conservative production alignment accepted 24/31 pairs
  (residuals ≤ 0.05 s, multi-window stability, song-identity checks); 7
  rejected, including the sealed recording's only candidate. Data gate PASSED
  at preferred minimums: TRAIN 16 recs/14 songs, DEV 3/3, TEST 5/5
  song-disjoint ([REFERENCE_CORPUS_AUDIT.md](REFERENCE_CORPUS_AUDIT.md),
  [ALIGNMENT_AUDIT.md](ALIGNMENT_AUDIT.md)).
- **Model:** minimal near-no-op adapter (12k-param TF CNN, zero-init output head
  as the residual gate) fused at the S1R mask logits; trainable = adapter +
  final mask head = 4.85% of the model. **At initialization the model is EXACTLY
  S1R (max|Δ| = 0.0)** for all three reference modes
  ([ARCHITECTURE.md](ARCHITECTURE.md)).
- **Training:** all four section-20 signals on authentic real mixtures with
  controlled reference-correlated injection; two full runs (the first invalidated
  by a reference-position sign bug — recorded honestly in
  [FAILURE_CASES.md](FAILURE_CASES.md) §2 — the second with the corrected
  pipeline and identical hyperparameters). Total 2.8 GPU-h of the 24 h budget.

## 2. The result (stated plainly)

With the reference genuinely able to help — correct song, correctly aligned,
held-out songs disjoint from training — the adapter extracts nothing:

| evidence | value |
|---|---|
| suppression advantage over S1R (controlled injection, held-out) | **+0.013 dB median, +0.027 max** (bar: ≥ 1.5 dB) |
| correct vs ZERO reference | 0.004 dB median gap |
| correct vs WRONG reference | **0.000 dB median gap** |
| stability across training | no upward trend after update 500; oscillates within ±0.008 dB |
| safety | zero collapse, no passthrough, transients identical to S1R, wrong-ref harmless |

Every safety axis held perfectly at every checkpoint — the stage failed only on
the axis it was testing. Section-25 selection found no checkpoint passing the
causal-use check; section-26 returned **REFERENCE_ADAPTER_IGNORED**; the
section-30 promotion gate failed, so per sections 26/44 there was **no human
listening** — nothing audible exists to compare.

## 3. Why it failed (mechanism, not excuse)

The section-20 objective set anchors clean-mixture behavior to the teacher (by
design, to prevent another S1W) and teaches reference use only through
invariance to injected nuisance. But S1R's mask already suppresses music
strongly, so the learnable margin — the residual injected music after the frozen
mask — is worth only ~0.01 L1. The cheapest way to claim that margin is a
"some reference exists" prior (a uniform mask dip), not content matching; the
data show exactly that signature. Content-specific suppression was never
rewarded specifically, and it never emerged for free.

## 4. Consequence

Record the negative cleanly and stop spending effort on this direction in this
form (protocol section 46). The S1R checkpoint remains the stable baseline; the
aligned reference — the program's last obvious new information source — does
not yield product gain through the minimal-adapter route. A future attempt would
need an objective that explicitly rewards reference-CONTENT specificity (e.g.,
correct-vs-wrong contrastive terms) and/or a larger trainable surface with
stronger anchoring; both are recorded as hypotheses in S2A_DECISION.json, not
recommendations. The alternative frontier remains the non-adaptive path:
product-side DSP around the frozen zero-shot/S1R substrate.

No SI-SDR/recall/attenuation claims are made anywhere (no ground truth exists);
the product repository was never modified.
