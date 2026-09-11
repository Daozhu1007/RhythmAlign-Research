# S1E — Challenge Clips

**Stage:** S1E existing-corpus feasibility · **Date:** 2026-09-10
**Source:** `D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4` (163.07 s, 48 kHz stereo,
60 fps phone handcam, Xiaomi 14 Pro, `com.xiaomi.ai_audio: 1` — phone DSP active).
All timestamps are **original-video clock**. Clips were cut by
`scripts/04_cut_clips.py` (48 kHz stereo masters + 32 kHz mono copies in `clips/audio/`,
hashes in `clips/clip_manifest.json`).

## Selection method (disclosed)

Clips were selected from **owner guidance + objective diagnostics only** (band envelopes,
Silero-VAD speech probability, speech-AM modulation scan, spectral-flatness scan,
click-band onset-level scan — `scripts/01`/`02`, `diagnostics/timeline_overview.png`,
`diagnostics/zoom_*.png`). **No listening was available at selection time**, so clips are
diagnostic candidates, not auditioned exemplars. Favorable-only selection was avoided:
the set includes nuisance-only clips (negative controls), weak-signal clips, and both
reference-covered and reference-uncovered material.

Known reference coverage from R1: aligned pristine music covers video ≈ [11.4, 150.9] s
(static delay d* = −11.3747 s, drift negligible per R1 fine-alignment).

## Clip list

| ID | Video time | Dur | Ref? | Challenge | Selection basis |
|---|---|---|---|---|---|
| `c1_speech_npc_a` | 1:26.0–1:34.0 | 8 s | yes | NPC/announcer speech + interaction (owner-named region A, part 1) | owner guidance; top speech-AM bins at 86/90/92 s |
| `c2_speech_npc_b` | 1:36.0–1:44.0 | 8 s | yes | NPC/announcer speech + interaction (owner-named region A, part 2) | speech-AM bins 96/98/104 s |
| `c3_taiko_prompt` | 2:06.5–2:11.5 | 5 s | yes | **loud neighboring-machine Taiko system prompt (owner-named, 2:08–2:09)** + simultaneous interaction | owner guidance; clip = named event + context |
| `c4_ambience_start` | 0:02.5–0:08.5 | 6 s | **no** | pre-track arcade ambience/attract only (negative control for interaction) | VAD bump 2–5 s; before reference coverage |
| `c5_announcer_end` | 2:32.0–2:39.0 | 7 s | **no** | post-song announcer/results speech, no interaction (negative control + speech-suppression probe) | VAD: 7.1 s > 0.2, 3.3 s > 0.5 |
| `c6_dense_golden` | 0:22.5–0:37.5 | 15 s | yes | dense interaction + music (R1/R2/R3 golden window — historical comparability) | R1 golden selection |
| `c7_weak_taps` | 0:38.0–0:44.0 | 6 s | yes | **weakest click-band taps** + music | onset scan: 35–45 s bin lowest median onset level (−24.8 dB, p25 −27.0) |
| `c8_slide_friction` | 0:51.5–0:57.5 | 6 s | yes | slide/friction candidate + ordinary interaction | spectral-flatness scan: 51–53 s among top sustained-flatness bins outside NPC/weak regions |

Total: 61 s of material, 8 clips (within the 6–10 requested), 5–15 s each except the
owner-named speech region which was split into two 8 s parts.

## Known limitations of this selection (honesty notes)

- The owner-named nuisances were **not** independently confirmed by listening; their
  identity inside each clip rests on the owner's timestamps + diagnostics.
- Silero VAD probability is ~0 across the in-music NPC regions (c1/c2): the masked
  announcer speech is VAD-invisible. VAD therefore cannot serve as a speech proxy inside
  c1/c2; it only works on c5 (clear speech). This is itself a recorded finding.
- c4/c5 lie outside pristine-reference coverage; reference-informed processing is
  impossible there by construction (models that don't need a reference still run).
- No clip was chosen to be "easy". The dense golden window (c6) is the hardest
  target+music overlap case and is included for comparability with R2/R3.
