# R3 Phase 0 — Environment Audit (2026-09-08)

## Hardware
- GPU: RTX 4060 Laptop 8 GB（CLAPSep 推理峰值 1.4 GiB，AudioSep audio-query probe 复用 R2 环境）
- Disk: 充足

## Model 1 — AudioSep audio-query probe: RUN（复用 R2，未重建环境）
- 直接使用 `experiments/r2_target_separation/.venv`（Python 3.12.13, torch 2.14.0+cu126）、
  vendored AudioSep 与两份 checkpoint，未改动任何文件。
- audio-query 路径：`query_encoder.get_query_embed(modality='audio', audio=query32k[None,:])`，
  vendored CLAP encoder 内部 32k→48k 后 `repeatpad` 到 10 s（官方代码路径，短 query 可用）。
- 性质：**zero-shot cross-modal probe** —— released `audiosep_base` config `use_text_ratio=1.0`，
  训练实际是 text-conditioned，audio query 属 off-distribution。

## Model 2 — CLAPSep: RUN（独立环境）
- 来源：官方 repo `Aisaka0v0/CLAPSep`，README 指向公开 HF Space `AisakaMikoto/CLAPSep`。
- Checkpoints（Space → `r3_audio_query/checkpoints/`，字节级校验）：
  - `model/best_model.ckpt` 177,818,986 B（sha256 4c5e… LFS oid 匹配）
  - `model/music_audioset_epoch_15_esc_90.14.pt` 2,352,471,003 B，sha256 `fae3e9c0…` 与
    HF LFS oid 完全一致；与 R2 的 `music_speech_…89.98.pt`（2,352,471,003 B）大小相同但
    sha256 不同 —— 是不同权重，不可复用。
- 官方推理协议（Space `app.py`，已 vendor `model/*.py` 到 `third_party/CLAPSep/`）：
  - 32 kHz mono，按 320000 样本（10 s）硬分块；mixture 峰值 >1 时缩放到 0.9；
  - positive/negative = text embedding + audio embedding 相加（缺省项为零向量），
    两者的 normalize(concat([pos, neg])) 一起送 decoder；
  - 官方 audio-query 接口：`clap_model.get_audio_embedding_from_filelist([path])`（laion_clap）；
  - checkpoint 加载 `load_state_dict(strict=False)`。
- 依赖：torch/torchaudio/torchvision cu126、`laion-clap`、torchlibrosa、einops、loralib、
  librosa、soundfile、scipy、matplotlib、imageio-ffmpeg。README pin `transformers==4.30.2`，
  实际用 transformers 4.57.6 成功（R2 已验证 transformers<5 的 RobertaTokenizer 兼容性）。
- Runner-side 兼容处理（仅 `scripts/clapsep_lib.py`，未改 vendor）：
  1. torch>=2.6 `weights_only=False`（官方 ckpt 含 numpy 标量）；
  2. 丢弃旧 CLAP ckpt 的 `position_ids` buffer（与 R2 相同的已知问题）。
- VRAM：加载后 1.2 GiB allocated，推理峰值 1.4 GiB —— 8 GB 显卡余量充足。
- 运行耗时：模型加载约 1 min，单次 15 s 分离 <10 s。

## 未测试
- SAM-Audio（沿用 R2 的 BLOCKED 结论：gated checkpoint，无 HF 账号）。
