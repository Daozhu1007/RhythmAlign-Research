# S1R — Blind Listening Instructions

Thank you — this is the primary decision evidence for stage S1R. One pass is enough.

## What you have

18 anonymous WAV items in this folder's `wavs/` directory (`item_01.wav` …
`item_18.wav`) and `PLAYBACK_ORDER.json` (playback order — no method identities).
Each item belongs to one of six challenge groups (3 items per group, methods
shuffled): recommended listening order is by group, but the sheet lists the actual
sequence.

Groups (same sealed recording as S1E/S1W):

| Group | Content |
|---|---|
| c1_speech_npc_a | NPC/announcer speech region + interaction (part 1) |
| c2_speech_npc_b | NPC/announcer speech region + interaction (part 2) |
| c3_taiko_prompt | loud neighboring-machine system prompt + simultaneous interaction |
| c6_dense_golden | dense interaction under music (R1/R2/R3 golden window) |
| c7_weak_taps | weakest taps under music |
| c8_slide_friction | slide/friction + ordinary interaction |

The three items in each group are: the raw handcam, the ZERO-SHOT CLAPSep output,
and the S1R real-mixture-adapted model output — in shuffled order.

## How to listen

1. Within each group, compare the three items against each other and against your
   memory of the raw recording. Take your time; rewind freely.
2. Free-form notes are fine. Naturally note anything you perceive of:
   - missing interaction / deleted hits
   - weak-hit preservation
   - crispness vs muffling ("闷") / underwater texture
   - discontinuity or stutter
   - slide/friction continuity
   - decay/tail naturalness
   - NPC/system-prompt suppression
   - neighboring-machine / music suppression
   - artifacts or invented content
3. Then give each item one overall preference mark within its group. No numerical
   scoring is required.

## The rules (fixed, disclosed)

- One listening pass; no repeated scoring rounds required.
- Gain: **no per-item normalization of any kind** — no peak, RMS, or loudness
  normalization, and no method-specific gain. Within each group exactly one common
  gain was derived from the raw source only (raw peak brought to 0.7 ≈ −3.1 dBFS)
  and applied identically to all three items, so every item keeps its frozen
  inference scale relative to the same raw source. Loudness differences you hear
  between items of the same group are therefore REAL content differences.
  Numeric validation: `LISTENING_PACK_GAIN_AUDIT.md` (tolerance ≤ 0.05 dB).
- A method that removes nuisance but noticeably deletes authentic interaction
  DOES NOT WIN. A model that simply returns the raw mixture also does not win.

Please do not open or ask about the unblinding key before rating — it stays sealed
(sealed copy: `work/private/listening_key.private.json`, local only) until your pass
is complete. Rating sheet: any free format you like; the follow-up task will unblind
and record your observations verbatim.
