# R2 Phase 0 — Environment Audit (2026-09-07)

## Hardware
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB total, ~7947 MiB free at audit time
- Driver: 591.59
- Disk: D: 452G total / 221G free; C: 501G total / 223G free

## Software
- System Python: 3.10.11 (no torch, no huggingface_hub — clean)
- py launcher: 3.8-64, 3.10-64 only. No Python >= 3.11 installed.
- uv available at C:/Users/Daozh/.local/bin/uv (will provide standalone Python + fast wheels)
- conda: not installed
- Repo production env: D:\Code\RhythmAlign\.venv (Python 3.10.11, numpy 2.2.6, scipy 1.15.3) — NOT touched, per instructions

## Hugging Face authentication
- No HF_TOKEN env var, no ~/.cache/huggingface/token → machine has NO HF authentication at all
- HF cache hub is empty (only version.txt)

## MODEL 1 — SAM-Audio (Meta official): BLOCKED
Evidence:
- Model page https://huggingface.co/facebook/sam-audio-small publicly viewable (HTTP 200),
  model card reports `"gated": "manual"`, license `sam-license`, with extra gated fields
  (manual approval required after submitting gated access form).
- Anonymous file resolve https://huggingface.co/facebook/sam-audio-small/resolve/main/config.json
  → HTTP 401 (weights require authenticated + approved access).
- Calibration: nonexistent repos also return 401 anonymously, but sam-audio-small's public
  model card page confirms the repo exists and is gated.
- No HF account/token exists on this machine; the brief forbids bypassing the gate.
Additional blocker (moot given above): SAM-Audio requires Python >= 3.11; only 3.8/3.10 present.

Action per brief: record BLOCKED, do not bypass, continue with AudioSep.

## MODEL 2 — AudioSep: AVAILABLE
- Official checkpoint lives in public HF Space tree:
  https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint
  - checkpoint/audiosep_base_4M_steps.ckpt (1,264,844,076 bytes) — anonymous HTTP 302 to public CDN (downloadable)
- transformers-native conversion `nielsr/audiosep-demo` is publicly accessible (HTTP 200)
- AudioSep processes audio at 32 kHz mono (model limitation, to be documented in outputs)

## R2 isolated env plan
- venv: D:\Code\RhythmAlign\experiments\r2_target_separation\.venv (uv-managed, Python 3.12 standalone)
- torch cu126 wheels for sm_89 (RTX 4060), transformers with AudioSep support, librosa, soundfile

## Post-audit addendum — what actually ran (2026-09-07)
- transformers-native AudioSep: NOT VIABLE — AudioSep was never merged into any tagged
  transformers release (checked main + v4.36–v4.57 tags: no src/transformers/models/audiosep)
  → used the official Audio-AGI repo code (vendored under third_party/AudioSep).
- Actual env: Python 3.12.13 (uv standalone), torch 2.14.0+cu126, torchvision 0.29.0+cu126,
  transformers<5 (RobertaTokenizer), lightning 2.x, librosa/soundfile/scipy/matplotlib.
- Compatibility fixes required in scripts/10_audiosep_run.py (no third_party file modified):
  1. torch>=2.6 weights_only default → CLAP ckpt needs weights_only=False (official ckpt
     contains numpy scalars; trusted source = official Audio-AGI Space)
  2. old CLAP ckpt "position_ids" buffer dropped for strict load_state_dict
  3. CLAP_Encoder singleton factory (models/audiosep.py instantiates CLAP at import time
     with a relative ckpt path)
- Checkpoints (both verified byte-exact vs HF Space listing):
  checkpoints/audiosep_base_4M_steps.ckpt (1,264,844,076 B)
  checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt (2,352,471,003 B)
- Runtime: model load + 4 separation runs (2 prompts x 2 inputs, 15 s each) in 23.8 s on GPU.

