# S1R — REPORT: Real-Mixture Adaptation (final)

> **POST-HOC INTEGRITY NOTE (2026-09-15, RTA-0 audit).** Three scope
> corrections; the stage's core result is unchanged.
>
> 1. **Transient guard:** the anti-collapse term's documented "2–9 kHz
>    transient-band floor" indexed the wrong STFT axis for its 1-D inputs
>    (`s1r_losses.py:52-58`; it degenerated into a per-bin spectral floor over
>    a middle time block). The total-RMS anti-collapse hinge was separate and
>    valid, and ALL published evaluation metrics (click retention, RMS,
>    consistency) used a different, correct implementation — so the stage's
>    measurements and verdict stand, but the training signal did not include
>    the documented 2–9 kHz transient protection.
> 2. **Interpretation scope:** §3's "a student cannot outlearn a teacher it
>    only copies" is supported for THIS strongly anchored distillation recipe
>    only, not as a general scientific claim. The recipe improved context
>    consistency without demonstrating a clear product gain over the teacher.
> 3. **Listening wording:** "perceptually indistinguishable" overstates the
>    evidence. Supported form: one owner listening pass found no clear overall
>    preference over zero-shot (0/6 clear preferences; no equivalence test was
>    run).
>
> Evidence: `docs/reviews/RTA0_INTEGRITY_AUDIT.md` §3.3.

**Verdict: B — REAL-MIXTURE ADAPTATION IS VIABLE BUT NO CLEAR PRODUCT GAIN**
(decision record: [S1R_DECISION.json](S1R_DECISION.json); primary listening evidence:
[listening_pack/HUMAN_LISTENING_RESULTS.md](listening_pack/HUMAN_LISTENING_RESULTS.md))

S1R asked one question: can CLAPSep be adapted using authentic real handcam mixtures
WITHOUT destroying the useful source selectivity already present in the zero-shot
model? The answer is **yes** — and that alone closes S1W's failure mode. But the
adapted model is perceptually indistinguishable from its zero-shot teacher in
blinded listening, so the stage delivers a **stable training substrate, not a
product-level win**. This report records both halves honestly.

## 1. The stage's primary goal was achieved: real-domain adaptation without collapse

S1W's catastrophic failure (near-silent output on every real group, ~28–33 dB below
RAW) is directly repaired by changing only the training domain:

- **Training data:** authentic raw handcam mixtures only — 240 HIGH_CONFIDENCE
  teacher anchors (201 TRAIN / 39 DEV; 1,440 s) across 43 recordings, each with
  multi-context (early/canonical/late) and ±3 dB gain views. No synthetic
  target+nuisance example existed anywhere in the stage.
- **Conservative adaptation:** student = exact zero-shot init; 5.03% trainable
  (16.08 M params: mask head + upper decoder layers); teacher-anchor distillation +
  cross-context consistency + scale equivariance + anti-collapse hinge + L2-SP;
  1,500 updates at lr 2e-5, 0.91 GPU-h, 3.42 GB peak VRAM.
- **Outcome:** zero collapse and zero passthrough drift at every checkpoint eval;
  DEV cross-context inconsistency −25% (0.0609 → 0.0455); on the 14-recording
  TEST_GENERALIZATION set: zero collapse (S1W: −27.7 dB) and better click retention
  than zero-shot (−0.96 vs −1.20 dB); on the six sealed product groups: output
  within ±0.2 dB RMS of zero-shot, transient retention marginally better on all six
  ([REAL_TEST_RESULTS.md](REAL_TEST_RESULTS.md)).

## 2. Human blind listening: stability confirmed, product gain not found

One headphone pass, 18 items, fixed per-group gains, then unblinding (mapping
verified against the sealed key — 18/18 match):

- **Stability by ear:** interaction described as "very obvious" / "excellent and
  appears complete"; the c3 Taiko-prompt selectivity pattern audibly survives
  ("similar positive behavior" between zero-shot and S1R items); no deletion, no
  passthrough, no instability heard anywhere.
- **No product win:** ZERO-SHOT vs S1R blinded comparison → S1R clearly preferred
  in **0 of 6 groups** (bar: ≥3/6). Explicit tie in c2 ("same conclusion"),
  mirror-tie in c3, inconclusive in c7, tie in c8; at most a faint c6 edge for S1R
  (interaction naturalness matching RAW; zero-shot retains a small underwater
  quality there) and an ambiguous c1.
- **Remaining nuisances unchanged:** neighboring Taiko machines (c7/c8) stay
  obvious under BOTH separation methods; NPC speech stays obvious where present.

Protocol caveats recorded in the results file: headphones this stage (S1W:
speakers) → no cross-stage timbre claims; one pass only (substantial owner
fatigue).

## 3. Interpretation: a student cannot outlearn a teacher it only copies

S1R's supervision was the frozen zero-shot teacher's own behavior (anchor
distillation) plus self-consistency of the teacher's outputs. Nothing in that
objective contains information the teacher does not already have — so the student
converged to the teacher. The conservative machinery did its preservation job so
well that the measured deltas over zero-shot stayed small (±0.2 dB RMS; +0.02…+0.08
dB click retention). The result is exactly what the theory of
distillation-without-new-signals predicts, now demonstrated on the real domain:
**real-mixture adaptation is a valid FOUNDATION, but teacher-only adaptation does
not add product-level capability.**

What S1R does NOT establish: that adaptation cannot help at all — only that THIS
information source (the teacher itself) is insufficient. That is the gap S2A
attacks with a genuinely new signal.

## 4. Honest negatives and limitations (details in [FAILURE_CASES.md](FAILURE_CASES.md))

1. No suppression improvement over zero-shot was achieved (primary negative).
2. Consistency improvement is real but modest and saturates by update 250–500.
3. Gain equivariance unchanged (zero-shot was already near-equivariant).
4. Friction coverage is the thinnest axis (only 2 labeled TRAIN anchors).
5. Metric-implementation drift vs S1W limits flatness cross-stage comparison.
6. No speech-suppression claims for c1/c2 (in-music NPC speech is VAD-invisible).

## 5. Consequence and next stage

S1R is frozen as the stable real-mixture baseline
(`checkpoints/s1r_selected.ckpt`, sha256 `e1fade5e…`, anti-collapse gate PASS).
The recommended next stage — **S2A: reference-conditioned real-mixture
extraction** — adds the one new information source RhythmAlign natively has and
S1R never used: the **time-aligned pristine song reference** (a product-native
input, not an oracle). The S2A design constraints follow directly from this
report: start from the frozen S1R checkpoint, add the smallest near-no-op
reference adapter, and require a causal correct-vs-wrong-reference effect before
any human listening.

No SI-SDR/recall/attenuation claims are made anywhere (no ground truth exists);
the product repository was never modified.
