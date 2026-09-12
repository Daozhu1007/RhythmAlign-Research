# S1W — REAL_TEST_RESULTS (frozen primary test)

**Date:** 2026-09-12 · run AFTER training and checkpoint selection were completely frozen.
Source: the SEALED S1E recording (original raw, 48 kHz stereo; never used in training,
never substituted with a `_synced` version). Exactly three methods: RAW, ZERO-SHOT
CLAPSep (audio query Q1, zero negative), ADAPTED CLAPSep (selected checkpoint, same
query). No additional model zoo. Fixed gain: no normalization of any method output;
resample only. Inference protocol: exact 10-s chunk overlap-add applied IDENTICALLY to
both models (see FAILURE_CASES §2 for why zero-padding was not used).

## PRIMARY RESULT (stated plainly)

**The adapted model fails on real recordings.** On every group its output is ~-50 dB
RMS — near-total suppression that removes the authentic player interaction together
with the nuisance. The zero-shot baseline keeps transients within ~1-2 dB of raw while
attenuating music ~4-9 dB, exactly as recorded in S1E. Machine-side, the S1W
adaptation does NOT beat zero-shot on real material; it is catastrophically worse in
preservation. The full interpretation, the verified checkpoint reload, and the
constructed-DEV-vs-real gap analysis are in FAILURE_CASES.md §1.

| group | zero-shot click / mid (dB) | adapted click / mid (dB) | RMS raw→zs→ad (dB) |
|---|---|---|---|
| group | zero-shot click / mid (dB) | adapted click / mid (dB) | RMS raw→zs→ad (dB) |
|---|---|---|---|
| c1_speech_npc_a | -0.9 / -4.0 | -29.4 / -30.6 | -17.7 → -22.0 → -50.4 |
| c2_speech_npc_b | -0.8 / -5.9 | -30.6 / -29.9 | -19.4 → -23.5 → -50.2 |
| c3_taiko_prompt | -1.5 / -9.4 | -33.4 / -29.6 | -20.3 → -24.5 → -47.7 |
| c6_dense_golden | -0.9 / -5.6 | -31.3 / -28.8 | -20.3 → -23.9 → -51.5 |
| c7_weak_taps | -1.6 / -3.7 | -26.7 / -29.7 | -18.0 → -23.5 → -50.2 |
| c8_slide_friction | -1.1 / -4.0 | -17.7 / -13.5 | -16.8 → -21.8 → -31.1 |

**Honesty notes:** real clips have no target stems, so every number is a DESCRIPTIVE
proxy (protocol section 39); the in-music NPC speech (c1/c2) is VAD-invisible (S1E
finding), so no speech-suppression proxy is claimed. No SI-SDR, no recall, no source
attenuation claims. The 18-item blind pack carries the product verdict; given the
adapted items are near-silent, listeners should find little to rate there — that
near-silence IS the result to verify by ear. Secondary LOCAL product remixes (aligned
pristine at identical gain + stem, song-active interval only) were generated for
selected cases and stay local.
