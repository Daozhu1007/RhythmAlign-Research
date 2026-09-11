# Third-Party Dependencies

This repository does **not** vendor third-party code. During the research, three
upstream systems were used locally (downloaded clones + checkpoints); the relevant
directories (`**/third_party/`, `**/checkpoints/`, `*.pt`, `*.ckpt`) are excluded from
Git by `.gitignore`. This document records what was used, where it comes from, and how
to recreate the local setup.

| Project | Used in | Upstream | License | Local modifications | Vendored here? |
|---|---|---|---|---|---|
| **AudioSep** | R2, R3, S1E | [github.com/Audio-AGC/AudioSep](https://github.com/Audio-AGC/AudioSep) | MIT (© Xubo Liu), present in the local clone | none identified — the clone was used as downloaded | No |
| **CLAPSep** | R3, S1E | CLAPSep (research code by Hao Ma @SDU, Waveformer-derived; file headers in the local copy reference the Waveformer project) | **No license file was present in the local copy — licensing unclear** | 3 model files (`model/CLAPSep.py`, `model/CLAPSep_decoder.py`, `model/__init__.py`) were copied locally for the R3 experiment; modification status vs upstream not established | No — excluded from the initial public commit because licensing is unclear |
| **SoloAudio** | S1E | [github.com/WangHelin1997/SoloAudio](https://github.com/WangHelin1997/SoloAudio) | MIT (© 2024 Helin Wang), present in the local clone | none identified | No — the clone also carried its own `.git` and `pretrained_models/`; neither is redistributed |

## Checkpoint provenance (local-only, not redistributed)

| Checkpoint | Size | Used by | Source |
|---|---|---|---|
| `audiosep_base_4M_steps.ckpt` | 1.26 GB | R2, S1E | AudioSep release (per AudioSep upstream instructions) |
| `music_speech_audioset_epoch_15_esc_89.98.pt` | 2.35 GB | R2 | CLAP checkpoint (LAION CLAP music_audioset branch), per AudioSep instructions |
| `music_audioset_epoch_15_esc_90.14.pt` | 2.35 GB | R3 | LAION CLAP checkpoint, per CLAPSep instructions |
| `best_model.ckpt` (CLAPSep) | 178 MB | R3 | CLAPSep release, per upstream instructions |
| `soloaudio_v2.pt` / `audio-vae.pt` | 592 MB / 549 MB | S1E | SoloAudio pretrained models, per SoloAudio upstream instructions |

Model checkpoints are **not** redistributed from this repository; each is downloadable
from its upstream project under the conditions those projects define. Hashes for the
local copies are recorded in `migration/MIGRATION_MANIFEST.public.json`.

## Recreating the environment

Each experiment's scripts are plain Python plus the upstream projects above. The
per-stage environment audits (`logs/00_env_audit.md`, `logs/manifest_env_inventory.json`)
record the interpreter and package versions actually used. In short:

1. Clone the upstream project(s) for the stage you care about (table above).
2. Follow that project's installation instructions (AudioSep and SoloAudio ship
   `environment.yml` files upstream).
3. Download the listed checkpoint(s) from the upstream release.
4. Point the stage's path constants at your local layout (historical scripts carry
   absolute local defaults; see `migration/MIGRATION_NOTES.md`).
