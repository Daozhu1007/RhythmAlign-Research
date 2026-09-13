# S1R — PRE-LISTENING REPORT

**Date:** 2026-09-14 · Status at writing: **AWAITING_HUMAN_LISTENING**. This document
answers the protocol section 52 questions from the frozen machine-side record, before
any human evidence exists. It makes no prediction about listening preference.

1. **Did the teacher gate pass?** YES — preferred thresholds (not the reduced pilot):
   1206 s of TRAIN anchors across all 36 TRAIN recordings, max single-recording share
   3.0%, coverage of transient-rich/weak/dense/friction (TEACHER_VIABILITY_REPORT.md).
2. **How many recordings/windows became anchors?** 240 of 252 audited windows
   (201 TRAIN = 1206 s + 39 DEV = 234 s) across 43 TRAIN+DEV recordings.
3. **How stable was zero-shot under context shifts?** Very stable: median min-pair
   waveform correlation 0.976 across early/canonical/late views; worst accepted window
   ≥ 0.85; only 16/252 windows rejected overall (7 RMS-spread, 7 waveform-corr, 1
   spectral, 1 MR-STFT).
4. **How stable under ±3 dB scaling?** Stable: corrected-output RMS spread including
   gain views had median 0.49 dB (p90 1.18 dB); no view ever breached the −12 dB
   collapse or +6 dB amplification lines.
5. **What trainable subset was chosen?** Protocol §18 option B: final mask head
   (`mask_net`) + upper decoder layers (`layers.3`, `skip.3`, `inverse_patch_embed`).
6. **How many parameters were trainable?** 16,082,566 = 5.03% of the model (S1W: the
   entire 43.5 M decoder).
7. **Did the micro-pilot avoid collapse?** YES — 200 updates, zero anchors below
   −6 dB vs teacher, finite loss/gradients, checkpoints reload-verified.
8. **Did training reduce real cross-context inconsistency?** YES — DEV anchors
   0.0609 → 0.0455 combined (−25%); TRAIN anchors 0.0100 → ~0.0096 (log section
   Micro-pilot/Primary).
9. **Did any collapse occur?** NO — collapse rate 0.0 at every DEV eval of every
   checkpoint; worst single anchor −1.0 dB vs teacher (guard −6 dB, line −12 dB).
10. **Did any raw-passthrough drift occur?** NO — passthrough margin ≤ +0.024
    throughout (guard 0.05); anchor divergence ≤ 0.0081 vs 0.089 full-passthrough
    scale; generalization: S1R not systematically closer to raw than zero-shot.
11. **Was zero-shot anchor behavior preserved?** YES — teacher-anchor divergence
    median 0.0068; click retention +0.10 dB vs teacher on DEV; sealed groups within
    ±0.2 dB RMS of zero-shot.
12. **Did transient-retention proxies regress?** NO — improved: DEV +0.10 dB;
    generalization −0.96 vs −1.20 dB (S1R better); sealed groups: S1R closer to raw
    on all six.
13. **Did friction proxies regress?** NO — flatness delta vs teacher 0.000 on DEV
    anchors; c3 flatness slightly lower than zero-shot (less smearing). Evidence base
    is thin (corpus has only 8 friction cuts) — recorded as a limitation.
14. **What happened on TEST_GENERALIZATION?** 14 unseen recordings, S1W's exact 28
    windows: zero collapse (S1W: −27.7 dB), click retention −0.96 dB (zero-shot
    −1.20), no passthrough drift (GENERALIZATION_RESULTS.md).
15. **What happened machine-side on six sealed groups?** S1R ≈ zero-shot with
    slightly better transient retention on all six; equal suppression; zero collapse
    (REAL_TEST_RESULTS.md). No machine-side failure; also no large machine-side
    separation win — that question goes to the pack.
16. **Is S1R still fundamentally different from S1W's synthetic training?** YES —
    training inputs were authentic raw TRAIN mixtures under multiple context/gain
    views; no synthetic target+nuisance example existed anywhere in the stage; no
    S1W mixtures, proxies, or labels were used; teacher outputs are precomputed
    frozen pseudo-label evidence, never truth.
17. **Was the fixed-gain listening protocol validated?** YES — one common gain per
    group from RAW only; relative levels preserved to 0.00001 dB (tolerance 0.05);
    role-keyed public audit, mapping sealed (LISTENING_PACK_GAIN_AUDIT.md).
18. **Where is the local 18-item pack?** `listening_pack/wavs/item_01.wav …
    item_18.wav` with `PLAYBACK_ORDER.json` (anonymous; no method identities).
19. **Is the blind key sealed?** YES — `work/private/listening_key.private.json`,
    gitignored, not published; not read into any public artifact.
20. **Research commit SHA?** See S1R_STATUS.json / the stage commit (recorded at
    finalization of this stage state).
21. **Push status?** main → origin/main pushed at stage finalization; verified
    HEAD == origin/main, clean tree (S1R_STATUS.json).
22. **Product repo untouched/clean?** YES — D:\Code\RhythmAlign never modified;
    pre-flight and post-flight checks show clean tree, HEAD == upstream.
