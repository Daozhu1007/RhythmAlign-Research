# S1W — Blind Listening Instructions

Thank you — this is the primary decision evidence for stage S1W. One pass is enough.

## What you have

18 anonymous WAV items in this folder (`item_01.wav` … `item_18.wav`) and
`PLAYBACK_ORDER.json` (playback order, group labels, durations — no method identities).
Each item belongs to one of six challenge groups (3 items per group, methods shuffled):
recommended listening order is by group, but the sheet lists the actual sequence.

Groups (same sealed recording as S1E):

| Group | Content |
|---|---|
| c1_speech_npc_a | NPC/announcer speech region + interaction (part 1) |
| c2_speech_npc_b | NPC/announcer speech region + interaction (part 2) |
| c3_taiko_prompt | loud neighboring-machine system prompt + simultaneous interaction |
| c6_dense_golden | dense interaction under music (R1/R2/R3 golden window) |
| c7_weak_taps | weakest taps under music |
| c8_slide_friction | slide/friction + ordinary interaction |

## How to listen

1. Within each group, compare the three items against each other and against your
   memory of the raw recording. Take your time; rewind freely.
2. Free-form notes are fine. Naturally note anything you perceive of:
   - missing interaction / deleted hits
   - weak-hit preservation
   - crispness vs muffling ("闷")
   - slide/friction continuity
   - decay/tails naturalness
   - NPC/system-prompt suppression
   - other arcade nuisance suppression
   - artifacts or invented content
3. Then give each item one overall preference mark within its group.

## The rules (fixed, disclosed)

- One listening pass; no repeated scoring rounds required.
- Gain: **no per-item normalization of any kind** — no peak, RMS, or loudness
  normalization, and no method-specific gain. Within each group exactly one common
  gain was derived from the raw source only (raw peak brought to −1 dBFS headroom)
  and applied identically to all three items, so every item keeps its frozen
  inference scale relative to the same raw source. Loudness differences you hear
  between items of the same group are therefore REAL content differences: large
  loudness loss is itself meaningful evidence (suppression/deletion magnitude).
  Expect some items to be near-silent — that is a result, not a packaging artifact.
  Numeric validation: `LISTENING_PACK_GAIN_AUDIT.md`.
- A method that removes nuisance but noticeably deletes authentic interaction
  DOES NOT WIN.

## Pack history (the v1 mistake is recorded, not hidden)

- v1 (2026-09-12): **INVALID — superseded.** Every item was independently
  peak-normalized to −3 dBFS, violating the preregistered rule of no per-item
  normalization (independent normalization can amplify near-silent residue and hide
  the actual magnitude of target deletion). The v1 audio was overwritten in place;
  the mistake is kept on record here and in `LISTENING_PACK_GAIN_AUDIT.md`.
- v2 (2026-09-12, current): the 18 items were regenerated from the same frozen
  inference outputs under the single-common-gain rule above; anonymous method
  assignment and playback order are unchanged; relative loudness between methods is
  validated to be preserved (`LISTENING_PACK_GAIN_AUDIT.md`).

Please do not open or ask about the unblinding key before rating — it stays sealed
until your pass is complete. Rating sheet: any free format you like; the follow-up
task will unblind and record your observations verbatim.
