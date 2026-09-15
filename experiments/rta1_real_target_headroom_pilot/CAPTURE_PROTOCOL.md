# RTA1 — Capture Protocol (minimal version)

**Goal:** ~15 minutes of usable, well-documented material across **3
independently set-up sessions**, sufficient to build a 24-mixture diagnostic
panel. This is a **diagnostic measurement set**, NOT a training dataset.

The owner-facing, one-page version is
[OWNER_CAPTURE_CHECKLIST.md](OWNER_CAPTURE_CHECKLIST.md). This document is the
full protocol; the owner does not need to read it.

## Sessions (×3, each independently set up)

Per session, record three takes in this order:

| take | type | duration | content |
|---|---|---|---|
| **A** | quiet physical interaction (playback muted or quiet room) | ≈ 3 min | the target material: deliberate, complete coverage of the action strata below |
| **B** | playback-only / target-absent nuisance | ≈ 1 min | phone in the same position, NO intentional interaction: music playing (or ambient alone), nobody touching the machine |
| **C** | ordinary simultaneous gameplay | ≈ 1 min | normal play with music ON — the realistic simultaneous condition |

Prefer **A with cabinet playback muted/volume-off if the arcade permits**; if
not possible, choose the quietest available setup and record what B sounds
like under the identical setup so the noise floor is measurable.

## Take A — required action strata (the collector's checklist)

Deliberately cover ALL of these during the ~3 minutes; the checklist groups
them into a comfortable order:

1. **weak taps** — light finger touches, grazes, barely-pressed buttons
2. **normal button hits** — ordinary gameplay presses at usual force
3. **strong / palm impacts** — deliberate firm hits, palm strikes
4. **releases / tails** — let the machine ring out; record the decay after
   impacts; include slow button releases
5. **screen contacts** — touches/taps on the screen surface if the cabinet has one
6. **slide / friction** — slider travel both directions, sustained palm drag
7. **dense interaction** — fast continuous play-like action for ~20–30 s
8. **sparse interaction** — isolated single events with silence between (~20–30 s)
9. **deliberate rests** — several explicit "nobody touches anything" spans of
   5–10 s (these calibrate the noise floor; announce nothing, just wait)

A suggested spoken-free structure for take A: rests (1 min) → weak/normal/strong
isolated hits with tails (1 min) → slides/friction/screen (0.5 min) → dense
block (0.5 min). Never narrate during recording.

## Hardware rules

- **Primary recording device:** the same class of phone used for the product's
  handcam recordings. Do NOT substitute a camera's audio, a close mic, or any
  other device as the primary file.
- **Phone position:** the same mounting/position family used for handcams
  (e.g., chest/phone holder at usual distance and orientation). Note the
  position in the session file.
- **Optional synchronized secondary sensors** (close/contact mic, second
  phone, video camera) are welcome as **event/timing evidence only**. Start
  them together with the primary; they are never phone-domain waveform truth.
- **Disable post-processing where optional**: turn OFF "noise reduction",
  "spatial audio", "wind reduction", auto-gain if a switch exists. If a phone
  forces AGC, record that fact in notes — the QC will check gain behavior.
- **Airplane mode** recommended (no notification sounds); screen brightness
  and haptics as silent as possible.

## File handling

- Record in the **highest-quality native format the phone offers** (prefer
  WAV/LPCM if available; otherwise the default AAC/M4A is acceptable — codec
  is recorded and its integrity checked, never converted at capture time).
- **Preserve native channels** (stereo stays stereo; no downmix).
- **Never edit, trim, convert, "enhance", or rename destructively.** Copy the
  original files (e.g., `REC_001.m4a`) as-is into
  `work/capture_inbox/session_XX/`. File names stay original; session JSON
  provides the mapping.
- Fill one `CAPTURE_SESSION.json` per session (template provided) — every
  field is short; nothing personally identifying is required.

## Metadata recorded per session

`session_id`, `device_id` (e.g., "phoneA"), `player_id_anonymized`,
`date`, `room_type`, `phone_position_description`, `orientation`,
`native_channels`, `sample_rate`, `codec`, and per take: `take_id`, `type`
(A/B/C), `target_strata` covered, `playback_state`, `nuisance_state`,
`start_time`, `duration`, `notes`.

## Validity gate (upstream of any use)

A take is NOT ground truth merely because it is quiet. Every take passes
through `03_truth_qc.py` + manual review: clipping, codec/sample-rate
integrity, usable noise floor, silence/rest contamination, channel integrity,
recording continuity, and the ~20 dB contamination-separation target where
measurable. Outcomes: PASS / REVIEW / FAIL / `TRUTH_QUALITY_INSUFFICIENT`.
Details: [TRUTH_QC_PROTOCOL.md](TRUTH_QC_PROTOCOL.md).
