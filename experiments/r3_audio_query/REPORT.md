# RhythmAlign Phase R3 — Experiment Report

日期: 2026-09-08 · 环境: `logs/00_env_audit.md` · 运行数据: `logs/06_evaluation.json`（全部指标）、
`logs/0{3,4,5}_*.json`、`queries/query_events.json`

## 结论一句话

**audio query 没有打败 R2 的 text query**：AudioSep audio-query（zero-shot probe）比 text 更差
（target ≈ mixture 的 0.98+ 相关"衰减拷贝"）；CLAPSep（官方多模态 query 预训练模型）产生了
+4~6 dB 的相对选择性、但音乐绝对抑制只有 −7~−9 dB，且太鼓泄漏与 R2 基线完全相同（10/39），
target 频谱相似度仍高达 0.93–0.95。**没有方法达到"音乐被真正移除"的水平**，
最好的 stem 仍然是 R2 的 text-query 基线 `audiosep_raw_p2_target.wav`。

## 1. Query construction

- 方法：R1 的 53 个 stereo-gate click 候选 → 按局部音乐能量 / 瞬态强度 / 拥挤度排序 →
  对前 14 名抽取 handcam 帧条（60 fps，±80 ms 五帧）**逐条目视确认**后入选
  （`work/frames/strip_*.jpg`，判定记录在 `queries/query_events.json`）。全部命中均有可见的
  手套-按键接触证据（1 个 medium：t=0.389 手套贴在 6 点位边缘）。
- **Q1**（`queries/query_q1.wav`，0.113 s）：单个 t=14.907 双手同按事件 —— 全场音乐局部
  能量最低（−19.4 dB，SNR +3.0 dB）、PERFECT 判定画面清晰；窗口被 15 s 文件末端截短但
  完整覆盖击打瞬态。
- **Q2**（`queries/query_q2.wav`，1.713 s）：9 个确认事件（t=0.389/1.237/2.437/3.477/4.485/
  10.187/12.960/14.571/14.907）各取 [t−20ms, t+150ms] 原始切片、5 ms 边界淡入出、30 ms
  间隔拼接。无任何合成/神经生成成分。
- Q3（transient-focused）未制作：Q1/Q2 已能回答本轮问题，无需第三个 query。

## 2. AudioSep audio-query（zero-shot probe）

明确性质：**zero-shot cross-modal experiment** —— released base checkpoint 训练时
`use_text_ratio=1.0`（纯 text-conditioned），audio query 是 off-distribution 输入。

| run | target vs mixture | music-heavy 帧抑制 | selectivity | 频谱相似度 |
|---|---|---|---|---|
| audiosep_aq_q1 | −3.0 dB | −5.6 dB | +3.4 dB | 0.982 |
| audiosep_aq_q2 | −1.9 dB | −2.7 dB | +1.4 dB | 0.991 |

target 能量保留 50–65%，音乐重帧只压低 2.7~5.6 dB，频谱相似度≈1 —— **比 R2 text query 更差
（更接近均匀衰减器），不是更有选择性**。按 brief 立即判定：AudioSep audio-query 无价值，
未做任何调参。

## 3. CLAPSep（官方 checkpoint 一次性跑通）

- 环境/推理 100% 按官方 Space `app.py` 协议（32 kHz mono、10 s 硬分块、峰值 0.9 重缩放、
  零向量负条件），仅 runner 侧 2 处兼容 shim（详见 env audit），vendor 未改。
- RTX 4060 8 GB：峰值 1.4 GiB，单次分离 <10 s。
- positive-only audio query（negative = 零向量）：

| run | click 保留 | click 电平 | music-heavy 抑制 | selectivity | taiko | 幻觉 | 频谱相似度 |
|---|---|---|---|---|---|---|---|
| clapsep_q1 | 48/53 | −2.2 dB | −7.6 dB | +5.2 dB | 10/39 | **1** | 0.947 |
| clapsep_q2 | 48/53 | −2.4 dB | −6.7 dB | +4.2 dB | 10/39 | 0 | 0.946 |

- **Q1 > Q2**（音乐多压 ~1 dB、selectivity +1 dB）：单事件干净 exemplar 优于多事件拼接
  （Q2 混入了事件间音乐底）。但差距小，且两者都低于 R2 text 基线。
- **是否产生真正选择性？部分相对选择性，但达不到合格线**：
  - 操作瞬态留下 ✅（48/53，电平 −2.2 dB）；
  - 音乐比操作声多压 5 dB ⚠️（相对是，绝对不行：音乐重帧仍保留 ~21% 能量，人耳必然可闻）；
  - 太鼓与基线同比例保留 ❌（10/39，与 R2 text、AudioSep aq 完全相同 —— 该指标在本素材上
    不具区分度，10 个命中疑似与 click 时间重合）；
  - clapsep_q1 出现 1 个幻觉 onset（13.408 s，孤例，疑似重构伪影）。
- **判定：没有"明显懂得我要玩家操作声"。** 频谱图（`outputs/spec_clapsep_q1_target.png`）
  目视：target 仍是 mixture 的结构 + 低频偏亮，residual 只有中高频内容。

## 4. Positive/negative（N1 执行，N2 跳过）

Decision Gate 处于中间地带（有 +5.2 dB 部分选择性、非零），未触发"完全没有选择性 → 停止"
的硬条件；但按预算只执行了 1 个有明确动机的控制：

- **N1**（positive=Q1，negative=对齐干净音乐参考 8.0–10.0 s 片段 `queries/neg_music.wav`）：
  music-heavy 抑制 −7.6 → **−8.9 dB**（+1.3 dB），selectivity 5.2 → **+6.4 dB**；
  代价：click 保留 48→47，幻觉 onset 1 → **2**。改善边际且引入新伪影，**远不足以改变结论**。
- **N2 未执行**：素材中不存在能高置信度标注"纯太鼓"的片段（39 个太鼓候选均为歌曲内容的一部分，
  与音乐/其他乐器重叠），brief 禁止猜标签。

## 5. Human-listening candidates

| 试听文件 | 内容 | 预期 |
|---|---|---|
| `outputs/clapsep_n1_target.wav` | R3 最佳音乐抑制（−8.9 dB）| 音乐变"薄"，但预期仍清晰可闻；注意 13.4 s 附近疑似伪影 |
| `outputs/clapsep_q1_final_mix.wav` | 0.5×干净参考 + 0.5×target（固定 0.5/0.5，无限制器）| 音乐≈原声 + 半份"变糊的混音" |
| `outputs/clapsep_n1_final_mix.wav` | 同上配方 | 同上 |
| 参照系：`r2/…/audiosep_raw_p2_final_mix.wav` | R2 text 基线（仍为全场最佳）| 对照"最好的 text query 也只能这样" |

原始 target stem 全部原样保留（`*_target.wav`，32 kHz mono float32，CLAPSep 输出按官方
峰值协议逆缩放回 mixture 域）。

## 6. 与 R2 text baseline 的明确比较

| 指标 | R2 text p2 | AudioSep aq 最佳 | CLAPSep 最佳(q1) | CLAPSep +neg(n1) |
|---|---|---|---|---|
| click 保留 | 49/53 | 49/53 | 48/53 | 47/53 |
| click 电平保留 | −2.9 dB | −1.3~−2.2 dB | −2.2 dB | −2.4 dB |
| music-heavy 抑制 | **−9.8 dB** | −5.6 dB | −7.6 dB | −8.9 dB |
| music-quiet 抑制 | **−13.6 dB** | −4.0 dB | −6.0 dB | −7.8 dB |
| selectivity score | **+6.9 dB** | +3.4 dB | +5.2 dB | +6.4 dB |
| 太鼓泄漏 | 10/39 | 10/39 | 10/39 | 10/39 |
| 幻觉 onset | 0 | 0 | 1 | 2 |
| target-mixture 频谱相似度 | **0.893** | 0.982/0.991 | 0.947 | 0.935 |
| 与对齐干净参考相关 | 0.013 | 0.029 | 0.007 | 0.014 |

（selectivity = click 窗口能量保留中位数 − music-heavy 非操作帧能量保留中位数；纯均匀衰减器
该值为 0 dB。太鼓命中 10/39 对所有方法恒定，判定为与 click 时间重合，不具区分度。）

**结论：本轮两种 audio-query 路线都没有超过 R2 的 text query。** R2 text 基线在
music-heavy（−9.8 vs −7.6）与 quiet（−13.6 vs −6.0）两个音乐维度上都更优，且零幻觉；
其"target≈衰减混音"的缺陷依然成立（频谱相似度 0.893 仍偏高）。

## 7. 下一步：选择 C —— audio query 不够，转向 temporal/visual contact conditioning

依据本轮实际输出（非理论）：
- A 否决：CLAPSep 未超过 R2 text 基线，绝对音乐抑制（−7~−9 dB）远低于可用水平；
  official checkpoint 在本素材上没有展示出"实际价值"。
- B 否决：Q1 优于 Q2 仅 ~1 dB —— exemplar 质量影响存在但天花板太低，
  不值得把 R4 押在 query acquisition 上。
- **C 选择**：通用音频-query 分离（无论 text 还是 audio、无论 zero-shot 还是官方多模态
  训练模型）在同一个 8 GB 消费级 GPU 生态内已经系统性到达天花板：全部方法的
  spec-similarity ≥ 0.89、音乐绝对抑制 ≤ −9.8 dB。而 R1/R3 已证明 **60 fps handcam 视频
  可以逐帧确认每次玩家接触**（本轮 9/9 命中无一错判）。把"何时有接触"从音频 query 转交给
  temporal/visual 条件（例如以视频帧时间戳驱动 mask/门控，或在分离模型外做 time-gated
  处理），是下一个信息量更大的实验方向。
- D 保留：仅在 C 失败后触发（专用模型/采集侧，如手部麦克风、按键走线拾音）。

## 交付物

```
experiments/r3_audio_query/
├── queries/   query_q1.wav, query_q2.wav, neg_music.wav, query_events.json（9 事件证据链）
├── outputs/   audiosep_aq_q{1,2}_{target,residual}.wav (32k mono float32)
│              clapsep_q{1,2}_{target,residual}.wav, clapsep_n1_{target,residual}.wav
│              clapsep_{q1,n1}_final_mix.wav (48k stereo, 固定 0.5/0.5)
│              spec_*.png（全部 stem 的 mixture/target/residual 频谱图）
├── logs/      00_env_audit.md, 01_event_candidates.json, 01_queries_spec.png,
│              03_audiosep_audio_query_runs.json, 04_clapsep_runs.json,
│              05_clapsep_n1.json, 06_evaluation.json, 07_final_mix.json, run_03.log
├── scripts/   01_select_events.py, 02_build_queries.py, 03_audiosep_audio_query.py,
│              clapsep_lib.py, 04_clapsep_positive.py, 05_clapsep_negative.py,
│              06_evaluate.py, 07_final_mix.py
├── third_party/CLAPSep/model/  (官方 Space model/*.py vendor，未修改)
├── checkpoints/               (官方 Space 下载，sha256 校验)
└── work/frames/               (60fps 帧条，事件视觉确认证据)
```

源文件只读 ✓ · 生产代码未动 ✓ · 未 commit ✓ · 最多预算约束满足 ✓
（2 AudioSep audio queries / 2 CLAPSep positive queries / 1 of 2 negative controls）
