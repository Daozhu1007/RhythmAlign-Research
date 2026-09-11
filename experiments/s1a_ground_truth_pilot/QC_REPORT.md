# S1a QC report

**2026-09-10 · data class: SYNTHETIC_FIXTURE (session0001).** This report documents the
ingest/QC tooling run end-to-end on synthetic fixtures because no real capture exists
yet. Re-running scripts 01–04 on `raw/<session_id>` regenerates this section for real
media with the identical format. Gain convention everywhere: **fixed, no normalization**.

## 1. Inventory (script 01)

- `fixtures/`: 13 files, 10 audio (48 kHz float WAV), each with format/subtype/rate/
  channels/duration/sha256 in `manifests/media_inventory_fixtures.json`.
- Non-audio files (`.csv`, `.txt`, recipe JSON) inventoried by size + hash.
- `raw/`: **empty — awaiting real capture** (`manifests/media_inventory.json` absent).
- Video files are inventoried by container only; decoding requires ffmpeg, which is not
  installed on this machine. If a real session ships video-only sync evidence, extract
  its audio track externally or provide `phone_audio` separately (the checklist already
  requires a phone audio file).

## 2. Working PCM extraction (script 02)

- Native-rate float32 PCM copies written under `work/pcm_session0001/native/`
  (lossless decode of the WAV fixtures).
- 32 kHz working copies under `work/pcm_session0001/32k/` via `scipy.resample_poly`
  (polyphase Kaiser-8 FIR, zero-phase, DC gain 1; documented per file in
  `manifests/pcm_extraction_session0001.json`). This exercises the 48 kHz → 32 kHz path
  real phone recordings will take.
- Peaks before/after resampling recorded (identical to ~1e-7 on these fixtures).

## 3. Audio QC (script 03) — fixture session0001

| File | peak | clip frac | rest floor p5 (dBFS) | flags |
|---|---|---|---|---|
| phone/phone_audio.wav | 0.891 | 0 | −49.0 | — |
| phone/phone_playback_only.wav | 0.126 | 0 | −45.5 | — |
| phone/phone_noisy_gameplay.wav | 0.407 | 0 | −44.0 | `high_floor` |
| aux/closemic.wav | 1.077 | 2.6e-06 | −53.4 | — |
| aux/contact.wav | 0.895 | 0 | (< −120, digital near-silence floor) | — |
| pristine/fixture_track_a.wav | 0.503 | 0 | −33.5 | `high_floor` (music, expected) |
| pristine/fixture_rir_a.wav | 0.661 | 0 | −35.4 | `high_floor` (impulse response, expected) |

Flag rules (applied identically to real files): `clipped` if ≥ 0.01% samples at
full scale; `high_floor` if the 5th-percentile 50 ms-frame RMS exceeds −45 dBFS
(informative — music/RIR files legitimately trigger it). Band energies
(0–100/100–1k/1k–8k/8k–16k Hz), DC offset and codec full-scale run statistics are in
`manifests/qc_audio_session0001.json`. Truth-quality assignment (clean_quiet /
playback_present_low / contaminated) is a human step on REAL takes, informed by these
flags + `session_notes.txt`; the fixture generator's labels are synthetic by
construction.

## 4. Synchronization (script 04)

Fixture aux channels share the phone clock by construction, so the expected exact fit is
recovered: both `closemic` and `contact` fits give `a = 0.0 ms, drift = 0.0 ppm,
max residual = 0.0 ms → sync_ok` (`manifests/sync_map_session0001.json`). This validates
the anchor-picking + fitting machinery; real separately-recorded devices will show
non-zero offsets/drift and must satisfy the ≤ 2 ms leave-one-out tolerance to be
`sync_ok`. Contact-channel 0.8 ms sensor lead is part of fixture ground truth and is
NOT corrected away (drift-vs-propagation separation rule, SYNC_PROTOCOL.md §3).

## 5. T2 mixture construction (script 05)

12 fixture cases built (`fixtures/mixtures/`, recipes + hashes in
[MIXTURE_RECIPES.json](MIXTURE_RECIPES.json), `data_class: SYNTHETIC_FIXTURE`):

- every case passes the **stored-float32 sum identity check**
  `max|y − (s+n)| ≤ 2.4e-07` (stop condition);
- requested vs measured levels agree within 0.1 dB (level defined on annotated active
  target regions; floor 1e-8, frozen in the plan);
- components stored independently (s / n / y float32) with per-file sha256;
- nuisance families: music (pristine → RIR path transform, documented), speech-like,
  ambience, unrelated impacts, composite; levels −20/−10/0/+10 dB.

## 6. Stop-condition dashboard

| Condition | Status |
|---|---|
| Exact mixture components sum numerically | PASS (fixtures) |
| Part-15 oracle synthetic validation | PASS (6/6) |
| Sync sufficient for waveform work | PASS (fixtures; real verdict pending) |
| Sample/gain conventions consistent | PASS (fixed-gain, documented) |
| Oracle implementation unit-tested vs finite differences & torchlibrosa | PASS |
| Train/test / target/nuisance aliasing | N/A (no training; fixture nuisance sources disjoint from target takes by construction) |
| Independent output normalization | NOT USED |
| Historical files modified | NO |
| Post-hoc parameter tuning | NONE (budgets/eps frozen in S1A_PLAN.md before runs) |

**Nothing here authorizes S1b.** The stage decision remains PENDING pending real capture
([CAPTURE_REQUIRED.md](CAPTURE_REQUIRED.md)).
