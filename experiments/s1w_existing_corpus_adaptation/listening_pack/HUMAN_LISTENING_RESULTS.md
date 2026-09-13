# S1W — Human Blind Listening Results (unblinded)

**Listening pass:** one pass, per `LISTENING_INSTRUCTIONS.md`, against the v2
gain-corrected pack (18 items, 6 groups × RAW / ZERO-SHOT / ADAPTED, common
per-group gain, no per-item normalization).
**Unblinding:** the sealed key (`work/private/listening_key.private.json`) was opened
by the owner after the pass. The mapping below is reproduced from that key and is no
longer secret. **Final verdict: C — WEAK-SUPERVISION ADAPTATION FAILS.**

## Provenance of the observations (recorded honestly)

The owner completed the blind listening pass out-of-band and relayed the observations
together with the unblinded verdict in the stage-finalize brief. **No separate
per-item rating-sheet file was deposited in the repository** (verified 2026-09-14:
no ratings file exists on disk; the research session record ends at
`AWAITING_HUMAN_LISTENING`). This document therefore records the relayed observations
exactly as attested — summary-level attestations are marked as such, and no per-item
quotes, ranks, or preference marks are invented. Machine-measured pack levels are
included next to each item as objective context (from the v2 gain audit; packaging
verified to preserve relative levels to ≤0.0004 dB).

## Item → method mapping (from the opened key)

| Item | Group | Method | Pack RMS (dBFS, measured) | Owner observation (attestation) |
|---|---|---|---|---|
| item_01 | c1_speech_npc_a | **ADAPTED** | −51.33 | near-silent / severely suppressed (attested) |
| item_02 | c1_speech_npc_a | ZERO-SHOT | −22.96 | covered by stage-summary attestation (see below) |
| item_03 | c1_speech_npc_a | RAW | −18.61 | — |
| item_04 | c2_speech_npc_b | ZERO-SHOT | −24.58 | covered by stage-summary attestation (see below) |
| item_05 | c2_speech_npc_b | **ADAPTED** | −51.23 | near-silent / severely suppressed (attested) |
| item_06 | c2_speech_npc_b | RAW | −20.49 | — |
| item_07 | c3_taiko_prompt | **ADAPTED** | −48.88 | near-silent / severely suppressed (attested) |
| item_08 | c3_taiko_prompt | ZERO-SHOT | −25.69 | useful source selectivity retained — especially this group (attested); quality defects remain (summary) |
| item_09 | c3_taiko_prompt | RAW | −21.52 | — |
| item_10 | c6_dense_golden | ZERO-SHOT | −25.37 | covered by stage-summary attestation (see below) |
| item_11 | c6_dense_golden | **ADAPTED** | −52.99 | near-silent / severely suppressed (attested) |
| item_12 | c6_dense_golden | RAW | −21.86 | — |
| item_13 | c7_weak_taps | ZERO-SHOT | −24.28 | covered by stage-summary attestation (see below) |
| item_14 | c7_weak_taps | RAW | −18.80 | — |
| item_15 | c7_weak_taps | **ADAPTED** | −51.04 | near-silent / severely suppressed (attested) |
| item_16 | c8_slide_friction | RAW | −17.80 | — |
| item_17 | c8_slide_friction | ADAPTED | −32.09 | severely suppressed relative to group (attested within the adapted-items finding) |
| item_18 | c8_slide_friction | ZERO-SHOT | −22.73 | covered by stage-summary attestation (see below) |

## Attested findings (the human evidence of record)

1. **ADAPTED = items 1 / 5 / 7 / 11 / 15 / 17.** The owner perceived all six adapted
   items as **near-silent / severely suppressed**. This confirms, by ear, the frozen
   machine-side result: adapted outputs sit 27.4–32.7 dB below their group's RAW on
   five groups (c8: 14.3 dB) — the model deletes the authentic player interaction
   together with the nuisance. Under the v2 protocol this loudness loss is genuine
   model behavior, not packaging (per-item normalization would have hidden it; that
   invalid v1 procedure was corrected before listening — see
   `LISTENING_PACK_GAIN_AUDIT.md`).
2. **ZERO-SHOT CLAPSep retains useful source selectivity, especially on c3
   (Taiko system prompt).** Attested at stage-summary level, with c3 explicitly
   named as the strongest case. Consistent with S1E listening and the S1W frozen
   real test (zero-shot transients within ~1–2 dB of raw; music ~4–9 dB relative
   attenuation).
3. **ZERO-SHOT quality defects remain: muffling / underwater texture /
   discontinuity.** Attested at stage-summary level; consistent with the S1E
   listening record and the HF-distortion observations in DEV_RESULTS /
   GENERALIZATION_RESULTS.
4. **Verdict C** follows: the adapted model's deletion of authentic interaction is
   disqualifying, while zero-shot CLAPSep remains the only useful substrate.

## What this document deliberately does NOT claim

- No per-item preference marks, rankings, or verbatim notes exist in the repository;
  none are reconstructed here.
- No speech-suppression claims for c1/c2 (in-music NPC speech is VAD-invisible — S1E
  finding); the c1/c2 listening value is preservation comparison, which the
  near-silent adapted items fail outright.
- No new machine measurements were taken for this document; levels are read back from
  the v2 gain audit of the frozen pack.
