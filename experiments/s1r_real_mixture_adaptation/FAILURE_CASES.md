# S1R — FAILURE_CASES and honest negatives

**Date:** 2026-09-14. The stage's headline negatives and limitations, stated first
and preserved unchanged.

## 1. No measurable suppression improvement was achieved (primary negative)

Machine-side, S1R's outputs sit within ±0.2 dB RMS of zero-shot on every sealed
group, with equal mid-band attenuation (REAL_TEST_RESULTS.md). The stage
successfully adapted WITHOUT destroying zero-shot — the question it was asked — but
it did not produce a model that removes nuisance more aggressively than zero-shot.
The conservative machinery (frozen-teacher anchoring, L2-SP weight anchor, 5%
trainable surface, lr 2e-5) did its preservation job so well that the student stayed
close to its teacher. Verdict A requires the human pass to prefer S1R on ≥3/6
groups; if that fails, the honest outcome is B (viable adaptation, no clear product
gain) — not A by construction.

## 2. Cross-context consistency improvement is real but modest and saturates early

DEV combined consistency improved 0.0609 → 0.0455 (−25%); most of the gain appeared
by update 250-500 and plateaued. The teacher's own context-consistency was already
high (Phase 0: median pairwise correlation 0.976), leaving little headroom; the
adaptation narrowed the remaining gap rather than transforming behavior. No claim is
made that S1R "fixed" context sensitivity — zero-shot barely had the problem; S1W's
adapted model did, and S1R's training recipe prevents it from developing.

## 3. Gain-equivariance did not improve

The ±3 dB equivariance error stayed at the zero-shot level (0.0321 → 0.0323 L1).
Zero-shot was already near scale-equivariant on these real views, and the 0.25 loss
weight at lr 2e-5 left it unchanged. The term acted as a guard against
loudness shortcuts, not as an improvement axis.

## 4. Friction coverage is the thinnest axis

The corpus contains only 8 friction cuts across 7 recordings (S1W audit); after the
Phase 0 acceptance criteria, only 2 TRAIN anchors carry the friction label (plus
flatness-derived friction-like windows among the 201). The friction stability
proxy (flatness delta vs teacher) never regressed, but "friction adaptation" rests
on little evidence. This is a corpus limitation, not fixable inside S1R
(no new capture allowed, protocol section 43).

## 5. Metric implementation drift vs S1W (comparability caveat)

The sustained-friction flatness fraction in S1R scripts (n_fft 1024, hop 512,
threshold 0.25) is stricter than S1W's, giving much lower absolute values (raw
≈ 0.0001 vs S1W's ≈ 0.30). Within-stage RAW/ZS/S1R comparisons are valid; the
flatness rows are NOT comparable to the S1W record. Window identity, click-band
convention (2–9 kHz), and OLA protocol ARE identical, so the generalization and
sealed comparisons stand.

## 6. Implementation failures during the stage (recorded, all fixed)

- The deleted S1W-era virtualenv forced a full environment reconstruction; a first
  torchvision install silently downgraded torch 2.14 → 2.7.0 and broke torchaudio's
  C extension until the matched set (torch 2.14.0 / torchvision 0.29.0 /
  torchaudio 2.11.0) was restored. Bitwise parity (max|Δ| = 0.0) proved the final
  environment faithful before any Phase 0 use.
- Four small training-harness bugs (numpy/tensor mixing, missing batch dim, missing
  device move, an undefined filename variable) crashed launches BEFORE any training
  effects; each was fixed and the affected run restarted from scratch. No eval,
  gate, or checkpoint decision was made from a broken state.
- First Phase 0 audit run missed the corpus' friction events (sibling JSON key, not
  a member of the impacts map) and mislabeled `weak_contact`; it was fixed and the
  audit re-run from scratch (240 anchors; friction present, weak corrected). The
  superseded first-run results were fully overwritten.

## 7. What S1R deliberately did NOT do

No synthetic remixing, no proxy targets, no S1W mixtures, no complex mask, no
phase-capable reconstruction (reserved for a future stage per protocol section 41),
no pristine-reference conditioning (section 42), no new capture (section 43), no
sealed/test feedback into training or selection.
