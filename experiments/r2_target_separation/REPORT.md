# RhythmAlign Phase R2 — Experiment Report

日期: 2026-09-07 · 环境: `logs/00_env_audit.md` · 运行数据: `logs/audiosep_runs.json`, `logs/evaluation.json`, `logs/prompt_selection.json`

## 结论一句话

**通用 text-conditioned 分离（AudioSep 官方预训练）在本素材上没有产生真正的分离**：
target stem ≈ 整体混音被宽带衰减 ~8–10 dB，玩家操作瞬态虽然完整保留，但本机音乐、
太鼓泄漏也同样保留。**无幻觉瞬态**。SAM-Audio 因 checkpoint 权限被 BLOCKED，未能测试。

## 1. Environment

- GPU: RTX 4060 Laptop, 8 GB VRAM；torch 2.14.0+cu126（独立 venv，Python 3.12.13，uv 管理，
  未触碰 RhythmAlign 正式环境）；磁盘充足。
- 实际成功运行: **AudioSep（官方 Audio-AGI 预训练 base, 4M steps）**。
- BLOCKED: **SAM-Audio** — HF checkpoint `gated: manual`（SAM License，需提交表单人工审批），
  本机无任何 HF 账号/token；按约定不绕过权限。次要阻碍（已无意义）: SAM-Audio 要求 Python ≥3.11。
- 注: transformers 从未发布过 AudioSep 集成（main 及 v4.36–v4.57 tag 均无），因此走官方仓库代码
  （vendor 到 `third_party/AudioSep`），兼容性修复共 3 处，见 env audit addendum，均未改动 vendor 文件。

## 2. SAM-Audio

被 BLOCKED，0 个 prompt 实际运行。brief 中的 4 个 prompt 已记录，其中语义被 AudioSep 复用
（"finger taps and button presses on a rhythm game cabinet" 在 CLAP 相似度排序中列第 2）。
predict_spans / temporal span 实验：未做（前置条件 text-only 结果不存在 + checkpoint 被阻塞）。

## 3. AudioSep

- 2 个 prompt 由 CLAP audio-text 相似度从 9 个候选中确定性选出（`logs/prompt_selection.json`）:
  - **p1 = "percussive taps"**（cos 0.400）
  - **p2 = "finger taps and button presses on a rhythm game cabinet"**（cos 0.282）
- 对 golden_raw 与 A2_residual 各跑 2 prompt，共 4 次分离（`chunk_inference` 官方长音频路径）。
- **32 kHz mono 限制如实生效**：所有 target/residual 均为 32 kHz 单声道 float32，
  不得当作 48 kHz 立体声结果使用。
- 关键数字（详见 `logs/evaluation.json`）：
  - target RMS 仅比输入低 ~4 dB（能量保留 ~40%）
  - 音乐重帧 target 比输入仅低 −6.8 ~ −10.3 dB，音乐安静帧 −9.4 ~ −13.9 dB
    → **衰减是非特异性的**（对音乐帧与静音帧几乎一视同仁）
  - target 与对齐干净参考的相关: raw 输入 0.115 → target 0.013–0.019（相干音乐成分被部分抑制，
    但能量上音乐仍大量留在 target）
  - target 保留 77/78 个输入 onset —— 分离器输出 ≈ 被衰减的输入拷贝
  - 频谱图（`outputs/spec_audiosep_*.png`）目视确认: target 与 mixture 结构相同

## 4. Raw vs A2

**基本无影响。** 两者 click retention 相同（49/53），音乐衰减差异在 ±1–2 dB 噪声范围内
（p1: raw −8.1 vs a2 −6.8 dB；p2: raw −10.3 vs a2 −9.4 dB），瞬态电平保留 raw 略优
（p2: −2.9 vs −2.4 dB）。A2 的 FIR 前处理既没有明显帮助 AudioSep，也没有因瞬态损伤而明显变差。
不预设的结论得到确认：对神经网络分离器而言，A2 preprocessing ≈ 中性。

## 5. Target audit（objective proxy）

| 指标 | 结果 | 判读 |
|---|---|---|
| R1 玩家瞬态保留 | 48–49/53（90.6–92.5%） | ✅ 但主要因为 target≈全量混音，非选择性提取 |
| 瞬态电平 vs 输入 | 中位 −2.1 ~ −2.9 dB | ✅ 操作声电平基本无损 |
| 幻觉瞬态（onset 级） | **0** | ✅ 无录音中不存在的新瞬态（严重失败项未触发） |
| 本机音乐残留 | target 能量 ~40%；音乐重帧仅 −7~−10 dB | ❌ 音乐未移除 |
| 太鼓泄漏 | 9–10/39 taiko 候选出现在 target | ⚠️ 部分泄漏 |
| 人声/环境 | 未做专门审计 | ⚠️ 需人工试听 |
| 与 R1 瞬态时间一致性 | 全部命中事件均在 R1 click 表 ±30 ms 内 | ✅ |

**requires human listening（objective proxy 不能回答）**: 音质 artifact / musical noise、
target 是否"听起来像操作声"、太鼓泄漏的主观显著性、人声存在性。
频谱图层面未见新结构（支持 onset 级无幻觉结论），但这只是必要非充分。

## 6. Best candidate

- **最值得试听的 target**: `outputs/audiosep_raw_p2_target.wav`
  （prompt "finger taps and button presses on a rhythm game cabinet"；49/53 瞬态保留、
  最佳音乐重帧衰减 −10.3 dB、零幻觉）
- **对应 final mix**: `outputs/audiosep_raw_p2_final_mix.wav`
  （0.5×R1 对齐干净参考 + 0.5×target 上采样 48k，无 limiter/compressor；
  电平偏差 −4.6 dB < 6 dB 阈值，故未生成 gain preview，raw stem 原样保留）
- 备选: `audiosep_a2_p2_target.wav` / `audiosep_a2_p2_final_mix.wav`（安静帧衰减略好 −13.9 dB）
- 诚实预期: 由于 target≈衰减的混音，final mix ≈ 干净参考 + 半份原混音；试听时预期听到
  音乐变"糊"而非消失。操作声本身从未是问题（R1 已证明它们明确存在于录音中）——
  失败点始终是**去音乐**。

## 7. 下一步：选择 C — audio-query conditioning

依据本轮实际输出（非理论）：
- A 否决: AudioSep 的失败不是调参能救的（衰减非特异、prompt 排名第 1 与第 9 的输出差异 <2 dB）。
- B 否决: SAM temporal 路线的 checkpoint 本轮被 BLOCKED，无法验证价值；不应基于无法运行的模型选方向。
- **C 选择**: 现有 checkpoint 已支持 audio query（AudioSep 的 CLAP 编码器接受 audio embedding，
  本轮基建可直接复用）。用一段真实操作声作为 query（替代文本）是下一个低成本、可立即执行的实验；
  若 audio query 也无法产生选择性，再进入 D。
- D 保留: 仅在 C 失败后触发。

## 交付物

```
experiments/r2_target_separation/
├── outputs/  audiosep_{raw,a2}_p{1,2}_{target,residual}.wav (32k mono float32)
│             audiosep_{raw,a2}_p2_final_mix.wav (48k stereo)
│             spec_audiosep_*.png
├── logs/     00_env_audit.md, audiosep_runs.json, evaluation.json,
│             prompt_selection.json, final_mix.json, run_10.log
├── scripts/  10_audiosep_run.py, 11_evaluate.py, 12_final_mix.py
├── third_party/AudioSep/  (官方仓库 vendor，未修改)
└── checkpoints/           (官方 Space 下载，字节级校验)
```

源文件只读 ✓ · 生产代码未动 ✓ · 未 commit ✓
