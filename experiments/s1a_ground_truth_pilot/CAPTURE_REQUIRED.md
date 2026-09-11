# CAPTURE REQUIRED — minimum files to unlock the S1a decision

**Status: the S1a decision is PENDING solely because no real pilot recording exists.**
All tooling, protocols, and oracle diagnostics are built and validated on synthetic
fixtures. The moment the files below exist in `raw/`, the same scripts produce the real
QC, sync map, T2 mixtures, oracle results, and the A/B/C/D decision.

## Minimum required (one session, one continuous take preferred)

Place under `experiments/s1a_ground_truth_pilot/raw/<session_id>/`
(layout and templates in [raw/README.md](raw/README.md)):

| # | File | Where | Notes |
|---|---|---|---|
| 1 | `phone_audio.<ext>` (with sync claps at both ends) | `phone/` | **REQUIRED.** 3–5 min quiet interaction, plus 1–2 min playback-only, plus 1–2 min ordinary gameplay. Highest quality the device allows; AGC/NR off if possible; do not change gain mid-take. |
| 2 | `phone_video.<ext>` | `phone/` | **REQUIRED.** Video of the same take (phone's own recording is fine). |
| 3 | `anchors.csv` | session root | **REQUIRED.** ~-times of start/end claps (template provided). |
| 4 | `session_notes.txt` | session root | **REQUIRED.** Devices, settings, positions, music state. |
| 5 | `closemic.<ext>` | `aux/` | Strongly preferred. Any mic near the machine, same take. |
| 6 | `contact.<ext>` | `aux/` | Strongly preferred (evidence only). Only if non-invasive placement is permitted. |
| 7 | `linear.<ext>` | `aux/` | Optional. Second recorder at the phone position. |
| 8 | pristine music file(s) | `pristine/` | Optional but valuable (exact nuisance identity for T2). |

## What is NOT needed

- No chart reproduction, no special playing skill — the 12-item sound menu in
  [ACQUISITION_PROTOCOL.md](ACQUISITION_PROTOCOL.md) is enough.
- No cabinet modification, no sensor attachment without operator permission.
- No long campaign: one session of ~5–8 recorded minutes.

## What happens automatically after you drop the files

```bash
cd experiments/s1a_ground_truth_pilot
python scripts/01_media_inventory.py            # inventory + hashes
python scripts/02_extract_pcm.py                # lossless working PCM (48k->32k etc.)
python scripts/03_qc_audio.py                   # clipping/floor/bands/flags
python scripts/04_sync_fit.py --session <id>    # clock map + verdict
python scripts/05_build_mixtures.py --recipe <panel_recipe.json> --out work/mixtures
python scripts/06_oracle_diagnostics.py
python scripts/07_evaluate_representation.py
python scripts/08_listening_pack.py
python scripts/09_pilot_manifest.py
```

Then the report and `S1A_DECISION.json` are updated to exactly one of:
A (capture pass + bounded mask pass) / B (capture pass + bounded fail + complex pass) /
C (capture fail) / D (all representations fail perceptually).

## Recording tips that most affect usability

1. **Loud clap ON CAMERA at start and end** — this is the sync backbone.
2. **Fixed gain, peaks below −6 dBFS**, never touch the recorder afterwards.
3. Quiet take honesty: if the music cannot be muted, record at the lowest practical
   volume and SAY SO in `session_notes.txt` — flagged truth is usable; mislabeled
   "clean" is not.
4. Do not trim/convert/rename anything; raw means raw.
