# S1W — Final Stage Report

**Date:** 2026-09-14 · **Verdict: C — WEAK-SUPERVISION ADAPTATION FAILS**
(decision record: [S1W_DECISION.json](S1W_DECISION.json); primary listening evidence:
[listening_pack/HUMAN_LISTENING_RESULTS.md](listening_pack/HUMAN_LISTENING_RESULTS.md))

S1W asked one question: does weak supervision mined conservatively from existing real
handcam recordings move CLAPSep toward "the physical interaction that actually
happened" and away from "everything else in the arcade" — WITHOUT deleting authentic
interaction? The answer is **no on real material**, and the stage is closed with that
negative result. This report separates what was learned from what failed; the
pre-listening report, the gain-protocol history (v1 mistake, v2 audit), and the
failure analysis are preserved unchanged.

## 1. The model successfully learned the constructed weak-supervision task

On the constructed DEV mixtures (2,560 ten-second mixtures built with exact
bookkeeping from conservatively mined target proxies + nuisance banks), training ONLY
the CLAPSep decoder (43.5M params, 2,000 updates, two seeds, ~1.05 GPU h) produced
large, checkpoint-reload-verified wins over the zero-shot baseline:

- nuisance leakage −0.8 → −10.3 dB (~9.4 dB improvement)
- target reconstruction 3.3× better, weak-hit preservation 3.3×, friction
  preservation 4×
- the zero-shot HF-muffling signature flipped positive

This is a real, replicable learning result: the optimization, the loss, and the data
pipeline work as designed **on the distribution they were trained on**.

## 2. The learned behavior failed to transfer to authentic raw handcams

On the frozen real sealed test (the S1E recording, never used in training), the
adapted model outputs ≈ −50 dB RMS on every group — near-total suppression that
removes the authentic player interaction together with the nuisance
([REAL_TEST_RESULTS.md](REAL_TEST_RESULTS.md),
[FAILURE_CASES.md](FAILURE_CASES.md) §1). The human blind listening pass confirmed
this by ear: **all six ADAPTED items (1 / 5 / 7 / 11 / 15 / 17) were perceived as
near-silent / severely suppressed**
([HUMAN_LISTENING_RESULTS.md](listening_pack/HUMAN_LISTENING_RESULTS.md)). Under the
v2 fixed-gain pack protocol (no per-item normalization; relative levels validated to
≤0.0004 dB — [gain audit](listening_pack/LISTENING_PACK_GAIN_AUDIT.md)), that
loudness loss is genuine model behavior, not packaging.

The transfer failure is a distribution gap, not a save/load or inference artifact
(checkpoint reload reproduces DEV numbers exactly; the exact-10-s overlap-add
protocol was applied identically to both models).

## 3. Zero-shot CLAPSep remains the only useful substrate — with known defects

The owner's listening attests that **zero-shot CLAPSep retains useful source
selectivity, especially on c3 (the loud Taiko system prompt window)** — the case
where S1E/S1W machine diagnostics also show its cleanest behavior (transients within
~1–2 dB of raw while attenuating music ~4–9 dB). At the same time, the listening
confirms zero-shot's standing quality defects: **muffling / underwater texture /
discontinuity**. Zero-shot is therefore a genuine but imperfect anchor: source
selectivity worth building on, fidelity not yet product-grade.

## 4. Primary failure mechanism

**Proxy/synthetic-distribution overfit plus chunk/context sensitivity.**

- *Distribution overfit (dominant):* the decoder learned the constructed-mixture
  distribution, not the product target. Supporting evidence from the preserved
  negatives (FAILURE_CASES §3): proxy music bleed plausibly taught the model that
  quiet music belongs to the target; speech nuisance is provably absent from
  TRAIN/DEV (c1/c2 outcomes were pure generalization); friction proxies — the most
  real-like material in the bank — are the only class that partially survived
  adaptation (c8, the least-suppressed group, −14.3 dB vs −27..−33 dB elsewhere).
- *Chunk/context sensitivity (independent robustness failure):* the adapted masker's
  behavior flips between zero-padded and exact-10-s chunk inference (mask p95 = 1.0
  vs near-total suppression, FAILURE_CASES §2). Zero-shot CLAPSep is robust to both.
  All reported real numbers use exact-chunk overlap-add applied identically to both
  models.

**What S1W does NOT establish:** the bounded real-valued mask representation is NOT
implicated as the cause of the adaptation collapse — the mask design was deliberately
left unchanged this stage (S1a's oracle ceiling still applies to any future mask
work). No SI-SDR/recall/attenuation claims are made anywhere (no real-capture ground
truth exists).

## 5. Consequence: do NOT continue tuning this recipe

S1W must not continue with more tuning of the same synthetic-proxy remixing recipe.
The constructed task is already learned to saturation (section 1); the failure is the
gap between that distribution and authentic handcams (section 2), which more of the
same remixing does not close.

## 6. Next-stage recommendation (recommendation only — NOT started)

A **new stage focused on REAL-MIXTURE adaptation**: learning from authentic handcam
acoustic distributions rather than better synthetic remixing. Candidate approaches
for that stage to evaluate:

- **zero-shot CLAPSep as teacher / anchor** (distill its selectivity, correct its fidelity)
- **real-mixture consistency learning** (same-recording perturbation consistency, no synthetic stems)
- **conservative pseudo-labeling / self-training** on real recordings, seeded by the zero-shot anchor
- **anti-collapse / identity constraints** — explicitly penalize wholesale output suppression so "near-silence" can never be an optimum
- **learning directly from authentic handcam acoustic distributions** (incl. the S1a exactly-matched capture path when scheduled)

Explicitly out of scope until that stage is designed and approved: starting the
experiment at all, complex-mask redesign, and aligned-reference conditioning.

## 7. Preserved record

Nothing below was deleted or rewritten at finalization:

- [S1W_PRELISTENING_REPORT.md](S1W_PRELISTENING_REPORT.md) — pre-listening machine-side prediction (direction C), unchanged
- [listening_pack/LISTENING_INSTRUCTIONS.md](listening_pack/LISTENING_INSTRUCTIONS.md) — includes the v1 gain-protocol mistake record (per-item −3 dBFS peak normalization; invalid; superseded)
- [listening_pack/LISTENING_PACK_GAIN_AUDIT.md](listening_pack/LISTENING_PACK_GAIN_AUDIT.md) — v2 gain audit, unchanged
- [FAILURE_CASES.md](FAILURE_CASES.md) — all negative findings, unchanged
- [DEV_RESULTS.md](DEV_RESULTS.md) / [REAL_TEST_RESULTS.md](REAL_TEST_RESULTS.md) / [GENERALIZATION_RESULTS.md](GENERALIZATION_RESULTS.md) — unchanged

Stage status is recorded in [S1W_STATUS.json](S1W_STATUS.json).
