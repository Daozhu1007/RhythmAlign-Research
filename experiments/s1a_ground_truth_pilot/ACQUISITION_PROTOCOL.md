# S1a acquisition protocol — one-session pilot

**Purpose:** acquire ONE high-quality feasibility pilot session of REAL player–machine
interaction audio near the intended handcam microphone position, plus synchronization and
QC material. This is a feasibility pilot, not the full ten-session S1 dataset and not a
statistical study.

## 0. Rules

- **No invasive cabinet work.** Do not attach sensors, open equipment, alter cabinet
  settings, or modify arcade hardware without the operator's explicit permission.
- **Fixed gains.** Once recording starts, do not change recorder gain; leave headroom
  (peaks below −6 dBFS); note the phone's recording settings and disable any "noise
  reduction"/"enhancer" options if the device allows.
- **Lossless first.** Prefer uncompressed PCM/WAV (48 kHz / 24-bit if available). If the
  phone only records lossy codec audio, still use it (that IS the product domain), but
  ALSO run the optional linear recorder if available.
- **Fixed truth quality.** Whatever background the "quiet" take contains, it is flagged
  honestly (see §4). A contaminated capture is never silently labeled clean.

## 1. Channels

| ID | Channel | Status | Purpose |
|---|---|---|---|
| A | Phone/camera-position air recording (the device that would shoot handcam video) | REQUIRED | candidate T1 target; product-domain acoustics |
| B | Video synchronized to A | REQUIRED | event/timing evidence (T0) |
| C | Pristine music file(s) | REQUIRED | exact nuisance identity for constructed mixtures |
| D | Close air mic near the interaction area | STRONGLY PREFERRED | better-target-ratio corroboration |
| E | Contact/piezo sensor | STRONGLY PREFERRED (evidence only) | independent event/timing (T0) |
| F | Separate linear recorder at/near the phone mic position | OPTIONAL | near-linear reference |

Close-mic and contact-mic tracks are **proxies/evidence, never exact phone-domain target
truth**. Nothing is attached to the cabinet without permission; D/E can be placed on a
stand or held near the machine instead.

## 2. Session blocks (~5–8 minutes + overhead)

| Block | Duration | Content |
|---|---:|---|
| 0. Setup | — | place devices, set gains, check headroom, note positions |
| 1. Sync-start | ~20 s | visible + audible sync event (see SYNC_PROTOCOL.md) |
| 2. Quiet interaction | 3–5 min | lowest-practical music playback; real interaction only (see §3) |
| 3. Playback-only / ambience | 1–2 min | no interaction: machine attract/music + room ambience |
| 4. Ordinary gameplay | 1–2 min | normal noisy gameplay (representative real mixture) |
| 5. Sync-end | ~20 s | second visible + audible sync event |
| 6. Room note | ~30 s | spoken note: date, devices, positions, settings, operator OK |

Do NOT stop recordings between blocks 1–5 on the phone; one continuous take is best. If
the phone must stop (file-size limits), add a brief sync event at each restart.

If the cabinet music cannot be muted: use the quietest practical alternative (lowest
volume setting, attract screen, or a known quiet jingle loop), **document exactly what
was audible**, and flag the take `playback_present_low` rather than clean.

## 3. Interaction content checklist (block 2)

Cover each item at least a few times, in any order, with deliberate pauses between
groups. Natural gameplay-like combinations at the end. There is no chart to follow —
the objective is acoustic source diversity:

1. isolated screen taps (light);
2. isolated screen taps (firm);
3. isolated physical button presses (each button used in real play);
4. button release/decay sounds (finger lifts);
5. palm hits;
6. slides: slow friction, fast friction, long continuous slide (≥3 s);
7. weak contacts (barely audible touches);
8. strong contacts;
9. dense sequences (rapid alternation, 5–10 s);
10. sparse sequences (one event per ~2 s);
11. deliberate rests (hands away, ≥3 s, several times);
12. natural gameplay-like combination of the above.

## 4. Truth-quality flags (assigned at ingest, not in the field)

| Flag | Meaning |
|---|---|
| `clean_quiet` | interaction take with no audible music/NPC/speech; rest floor measured |
| `playback_present_low` | quiet take contains low-level cabinet playback (documented) |
| `contaminated` | audible speech/NPC/loud external events in the take |
| `clipped` | clipping detected in QC (percentage reported) |
| `sync_ok` / `sync_poor` | per SYNC_PROTOCOL.md residual thresholds |

Blocks 2 (quiet interaction) yields candidate T1. Block 3 yields target-absent
calibration + nuisance source material. Block 4 yields T4 real-mixture stress material.
Sync markers are never scored content.

## 5. After the session

1. Copy files unchanged into `experiments/s1a_ground_truth_pilot/raw/<session_id>/`
   (see `raw/README.md` for the exact layout and filenames).
2. Fill `raw/<session_id>/session_notes.txt` (template in raw/README.md).
3. Run the ingest scripts (they compute the manifest, QC report, sync map, and flag
   everything above automatically).

Printable one-page version: [CAPTURE_CHECKLIST.md](CAPTURE_CHECKLIST.md).
