# S1R — GENERALIZATION_RESULTS (post-freeze; descriptive only)

**Date:** 2026-09-14 · The exact 28 song-active windows S1W used (2 per recording
across the 14 frozen TEST_GENERALIZATION identities) were re-run post-freeze so the
S1W and S1R records are directly comparable. Methods: RAW / ZERO-SHOT / S1R, exact
10-s chunk overlap-add applied identically. No ground truth exists — every number is
a descriptive proxy; these results were NOT used to tune anything.
Per-window data: `logs/06_generalization.json` (local).

| statistic (mean over 28 windows) | raw | zero-shot | S1R | S1W-adapted (record) |
|---|---|---|---|---|
| 2–9 kHz click-band retention vs raw (dB) | 0.0 | −1.20 | **−0.96** | −27.72 |
| output RMS (dB) | −13.41 | −18.38 | −18.29 | ≈ −50 (collapsed) |
| windows with >12 dB output suppression | — | 0 | **0** | all |

- **The S1W catastrophe is gone.** S1W's adapted model sat 27.7 dB under raw on
  unseen recordings; S1R tracks zero-shot within +0.10 dB RMS and improves click-band
  retention by +0.24 dB on average.
- **No passthrough drift.** Mean correlation to raw: zero-shot 0.870, S1R 0.885 —
  S1R is NOT systematically closer to the raw mixture than zero-shot (18/28 windows
  show S1R no closer to raw than zero-shot within 0.02 correlation); its mean
  MR-STFT distance from zero-shot is 0.049, i.e., a small refinement, not a return
  to identity.
- Sustained-friction flatness fraction (this stage's stricter implementation):
  raw 0.0001, zero-shot 0.0008, S1R 0.0007 — no texture-class change.

Reading: the conservative real-mixture adaptation generalized to 14 completely
unseen recordings without collapsing, without passing the mixture through, and
without regressing zero-shot's transient behavior — the exact three failure modes
that disqualified S1W.
