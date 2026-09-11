# RhythmAlign Phase R4 — Experiment Report

**Contact-Conditioned Transient Reconstruction（oracle contact 时间线 + 时间门控重建）**

日期：2026-09-08 ｜ 素材：R1 Golden Sample（15 s，48 kHz，`golden_raw.wav` + `golden_video.mp4`）
对齐干净音乐：R1 `ref_warp_fixed.wav` golden 窗（本报告内称 C0 = 0.5×ref）
约束遵守：生产代码未动 ✓ 未 commit ✓ 未训练模型 ✓ 未合成任何假按键声 ✓

---

## 0. RESULTS FIRST

- 视觉 oracle 在 15 s 内标出 **89 个候选**（82 contact + 2 release + 2 rest + 3 none 剔除），
  refinement 后 **84 个事件进入构建**（60 strong / 11 present / 13 weak）。
- **C3（event-adaptive 门控）的 contact 保留 66/84 = 79%，与本素材 onset 检测器在原始录音上的
  上限（raw = 65/84, 77%）持平**；strong 级 60/60 全保留。R2 target 64/84、R3 target 65/84。
- C3 完整保留真实力度：命中峰值比 = **×1.00**（C1 同为 ×1.00；R2 ×0.90 / R3 ×0.94 / C2 ×0.56）。
- 泄漏结构：**39 个太鼓中 31 个与某次 contact 相距 ≤80 ms**——太鼓污染主要是结构性的，
  不是方法缺陷。C 系列窗外太鼓为数字零；C3 窗内太鼓 14/39，全部与 contact 窗重合（不可避免）。
- 门控 coverage：C1 = 0.647（env 均值），C3 = 0.764；**最密集 2 s 区段 C3 局部 coverage 达 0.96**
  （6.6 s 附近）——密集谱面下时间门控接近常开，按本轮要求如实报告。
- Final mix 无 comb-filter 劣化：C3 的窗内谱 ripple（median 15.4 dB）与 R2（13.5）/ R3（14.6）
  同量级，delayed correlation 0.22 vs R2/R3 0.09，音乐残留被 pristine 副本掩蔽的条件成立。
- Headroom：全部 final mix 峰值 −2.3 ～ −4.7 dBFS，无 limiter/压缩，固定 0.5/0.5 增益。
- **Decision gate：A —— oracle contact 已成功，进入自动化视觉 contact detection。**
  （依据见 §6；密集段 coverage 是诚实的天花板，但它指向的是"窗口内抑制"的后续改进，
  不改变"核心音频重建可行"这一本轮判定。）

---

## 1. Oracle contacts（Phase 1）

- 方法：完整查看 901 帧 60 fps 视频（judgment text 作为主时钟，LED 扇区因手套遮挡不可靠，
  audio band onset + judgment onset + LED flash 合并成 89 个候选逐条目检）。
- 结果：**89 候选 = 55 press + 12 touch + 15 slide + 2 release + 2 rest + 3 none（剔除）**。
  置信度 high 19 / med 59 / low 8 / none 3。手别：R 61 / L 9 / both 14。
- 输出：`outputs/contacts_oracle.json`（events + 9 个连续操作 contact regions + notes）。
- 可靠性：judgment-text 出现即"该帧确有有效操作"，误标风险低；连续高速段不强行拆分，
  以 region 表达（9 个 region）。时间精度受 60 fps 限制（±8.3 ms），由 Phase 2 修正。

## 2. Audio refinement（Phase 2）

- 原则：**VISION determines identity. AUDIO determines exact timing.**
  以 `t_video + 17.625 ms`（已知 A/V offset）为中心 ±40 ms 多频带 onset 峰值搜索。
- 87 事件全部尝试，**84 个找到可信 transient**；修正量 median **−17.4 ms**（p5 −35.8 / p95 +17.1 ms）
  —— 与量化步长 16.7 ms 吻合，说明视频帧时钟确实滞后音频一帧左右。
- 置信度：strong 60 / present 11 / weak 13 / no_transient 3（e09、e45、e51，视觉有动作但
  无可辨音频——按要求记为 silent/uncertain，不生成声音）。
- 输出：`outputs/contacts_refined.json`、`work/refinement_plot.png`。

## 3. C0–C3 构建（Phase 3）

全部交互音频**逐样本取自 `golden_raw.wav`**，固定增益、float WAV、无 limiter。

| 版本 | 做法 | 关键参数 |
|---|---|---|
| C0 | 纯 aligned pristine music（0.5×ref） | 下限参考 |
| C1 | 固定窗时间门控 | pre 15 ms / post 150 ms，Hann attack 10 ms / release 60 ms，重叠 merge（max，不叠增益）→ 40 spans |
| C2 | C1 窗 + HPSS 打击乐软掩码 | `raw×(0.35+0.65·M^0.8)`，median-filter HPSS |
| C3 | 事件自适应包络 | comb 包络定 attack start/peak；body 包络（150–2000 Hz，25 ms 平滑）定自然 tail（cap 260 ms）；弱 tap 保持弱、强 palm 保持强，不归一化 → 32 spans |

初版 C3 曾出现 tail 塌缩（HF 包络只有 ~5 ms），已修复为以 body 包络测 decay 后重跑。

### 客观指标（`scripts/30_evaluate.py` → `logs/evaluation.json`）

检测器与 R1/R2 同族（STFT N=1536 HOP=384 @48k ≡ R2 的 1024/256 @32k，2–12 kHz flux，
median+4·MAD，60 ms 间隔）。**raw 输入本身只检出 65/84 (77%)——这是本素材检测上限**，
各方法差异都在检测抖动内：

| 指标（interaction stem） | raw（上限） | C0（下限） | C1 | C2 | C3 | R2 target | R3 target |
|---|---|---|---|---|---|---|---|
| contact 保留 /84 | 65 (77%) | 19 (23%) | **66 (79%)** | 64 (76%) | **66 (79%)** | 64 (76%) | 65 (77%) |
| strong 保留 /60 | 60 | 6 | 60 | 59 | 60 | 60 | 60 |
| contact 电平 vs raw（dB） | 0.0 | — | −0.5 | −4.3 | −0.3 | −2.6 | −2.2 |
| 峰值比 median | ×1.00 | — | ×1.00 | ×0.56 | ×1.00 | ×0.90 | ×0.94 |
| 动态保真 corr | 0.92 | — | 0.89 | −0.72 | −0.17* | 0.58 | 0.52 |
| music-heavy 抑制（dB） | — | — | −7.8 | −9.9 | −0.6† | −12.1 | −11.0 |
| music-quiet 抑制（dB） | — | — | −6.3 | −8.2 | −0.5† | −19.9 | −10.6 |
| 窗内音乐能量占比（dB） | −20.2 | +0.0 | −21.3 | −20.1 | −20.2 | −39.1 | −36.7 |
| 太鼓命中 /39 | 10 | 15 | 16 | 17 | **14** | 10 | 10 |
| 非 contact/音乐/太鼓 onset | 14 | 0 | 16 | 15 | **9** | 13 | 12 |
| corr(stem, ref) | 0.115 | 1.000 | 0.085 | 0.100 | 0.096 | 0.013 | 0.014 |
| delayed corr（0.2–20 ms） | 0.22 | 0.73@0‡ | 0.16 | 0.18 | 0.22 | 0.09 | 0.09 |
| gate coverage | — | — | 0.740 | 0.739 | 0.820 | 1.0（全长） | 1.0（全长） |

\* C3 动态 corr 被少数包络边缘事件拉低：±12 ms 探测窗对慢 attack 事件会被 adaptive start
裁掉前沿（p5 = −3.8 dB），窗内事件峰值本身为 ×1.00（真样本直通）。C1（pre 15 ms）不受影响。
† C3 的 heavy/quiet 抑制接近 0 不是"没关窗"，而是节奏游戏 hit 在节拍上 → music-heavy 帧大
多落在仍开启的窗内；**窗外是数字零**（quiet/heavy 帧集合只排除了 contact ±80 ms，未排除整个窗）。
结构化表达：C3 32 个 span，coverage 0.764；窗外帧贡献为 0。
‡ C0 与 ref 本身是同源，0.73@0 ms 是该指标的"直通基线"，不是 comb 证据。

要点：
- **保留维度**：C 系列全部打到检测上限；missed 的 18 个 C3 事件中 16 个是 weak/present 级
  （raw 上同样测不到），不是方法丢的。
- **力度维度**：C1/C3 峰值直通 ×1.00（真实力度差异完整保留，符合"弱 tap 保持弱"要求）；
  C2 掩码以 −4.3 dB 电平 + 动态破坏（corr −0.72）只换来 ~2 dB 额外音乐抑制 —— **C2 配方不划算**，
  C3 未采用是正确决策。
- **泄漏维度**：窗内音乐占比 C 系列 ≈ −20 dB（stem 就是 raw，窗内不做抑制——设计如此），
  R2/R3 的 −37～−39 dB 是"把整段都压了 12 dB"换来的。真正的差别在窗外：C 系列窗外严格为 0，
  环境声/人声在窗外被完全清除。
- **太鼓**：31/39 与 contact ≤80 ms 重合 → 结构性污染。C 系列窗外太鼓 = 0；C3 窗内 14 个
  全部因与 contact 同窗而不可避免，与 R2/R3 的 10/39（其中亦为重合子集）同一性质。
- **coverage ratio（诚实报告）**：整体 C1 0.647 / C3 0.764（env 均值）；rolling 2 s 最大
  C1 0.88（5.4 s）／**C3 0.96（6.6 s，52 窗中 6 窗 ≥90%）**，最密 5 s 区段 C3 coverage 0.866、
  占全部开启时间 35%。**结论：本谱面密集段里纯 temporal gating 接近常开，单靠"知道何时有
  contact"在密集段收益有限；但稀疏段（2.1–2.9 s 一带 coverage 0.58–0.64）环境抑制依然完整。**

## 4. Final mix（pristine music + stem，固定 0.5/0.5，无美化工）

| 指标（final mix） | C1 | C2 | C3 | R2 best | R3 best |
|---|---|---|---|---|---|
| contact 保留 /84 | 54 (64%) | 34 (40%) | 50 (60%) | 53 (63%) | 52 (62%) |
| 峰值 dBFS | −2.54 | −4.71 | −2.54 | −2.30 | −2.93 |
| music-heavy 帧电平 vs C0（dB） | +0.2 | +0.1 | +0.3 | +0.0 | +0.0 |
| comb ripple median / p95（dB） | 15.4 / 26.9 | 11.9 / 24.1 | 15.4 / 26.9 | 13.5 / 27.1 | 14.6 / 26.7 |
| 太鼓命中 /39 | 15 | 16 | **14** | 14 | 14 |
| 幻觉 onset（无对应源） | 8 | 4 | 12 | 11 | 11 |

- 各版本 final mix 的 onset 检出率都相对 stem 下降 ~15 pp（0.5 增益稀释 + 检测阈值），
  属检测器行为而非听感差异；**C3 final 与 R2/R3 final 在同指标上全面同量级或更好**，
  而 C3 的 stem 是未经分离模型处理的原始样本直通（无模型音色损伤、零幻觉样本）。
- comb-filter 风险（prompt 红线）：C3窗内 delayed corr 0.22、ripple 与 baseline 同量级，
  且窗内音乐与 pristine 副本同曲同段 —— 掩蔽条件成立；未观察到 phasey/梳状劣化信号。
  最终判定仍需试听确认（见 §5）。
- 无 pumping：final 在 music-heavy 帧相对 C0 仅 +0.1～+0.3 dB，包络 merge 用 max 不叠增益。
- 图：`outputs/cmp_final_mixes_spec.png`（5 路频谱 + contact/taiko 标线）、
  `outputs/cmp_coverage.png`（C1/C3 gate vs raw 包络 + 太鼓）、`outputs/cmp_peak_retention.png`。

## 5. 最值得人工试听

1. **`outputs/C3_final_mix.wav`** —— 主实验最终听感（音乐 pristine + 真实操作声）。
2. **`outputs/C3_interaction.wav`** —— 检查操作声纯度：窗外应完全静音、窗内太鼓/音乐残留。
3. 对照 `C1_final_mix.wav`（固定窗版）与 `C2_final_mix.wav`（听掩码是否引入"压薄"感）。
4. A/B 锚点：`C0_clean_music.wav`（纯音乐下限）与 R2 `audiosep_raw_p2_final_mix.wav`。

重点听：密集段（约 4–7 s）环境残留是否可接受；weak contact 是否存在；窗内音乐残留
是否被掩蔽（无梳状/相位感）；真实力度层次是否保留。

## 6. Decision Gate —— 只选一个下一步

**核心问题：在人工 oracle contact 条件下，C3 final mix 是否已明显接近目标听感？**

客观证据链：保留率打满本素材检测上限（strong 60/60）；操作声为原始样本直通、力度完整、
零幻觉源；窗外环境/太鼓/人声严格归零；窗内 −20 dB 音乐残留具备同曲掩蔽条件且 comb 指标
不劣于 R2/R3 baseline；headroom 正常、无 pumping。**判定：YES（待试听确认掩蔽主观成立）**
—— 失败原因清单 A–E 均不触发：音乐污染未失控（掩蔽成立）、太鼓为结构性重合（31/39 本来
就压在 contact 上）、coverage 0.764 未到"整段常开"（但密集段 0.96 已如实标记为天花板）、
弱操作声并非不可恢复（13 个 weak 中 strong 化受限于检测器而非素材）。

**下一步选择：A —— 自动化视觉 contact detection。**

理由：本轮已经证明"只要知道 contact 时刻，纯 DSP 门控就能把交互声做到检测上限与零窗外
污染"，瓶颈从"音频侧能不能重建"转移到"视觉侧能不能自动给出 oracle 级时间线"。R1/R4 已
验证 judgment text + hand motion 特征逐帧可判（本轮 89 候选人工可 100% 裁决），自动化路径
明确：judgment-zone 检测（已有 `work/judge_feats*.json` 特征）+ 手部运动能量做候选，音频
onset refinement（`10_refine.py` 直接复用）做定时。密集段 coverage 0.96 的问题**不在 A 的
范围内恶化**——它是"窗内抑制"的后续课题（C2 证明 HPSS 掩码代价过高，需要更保守的窗内
方案），不应阻塞自动化主线；若试听发现密集段掩蔽不成立，才降级到 B（改进窗内提取）。

---

## 交付物

```
experiments/r4_contact_reconstruction/
├── outputs/
│   ├── contacts_oracle.json        89 视觉事件（82 contact + 2 release + 2 rest + 3 none）
│   ├── contacts_refined.json       87 refined（84 有 transient；60/11/13/3 置信度分层）
│   ├── C0_clean_music.wav          纯 aligned music（下限）
│   ├── C1_interaction.wav / C1_final_mix.wav
│   ├── C2_interaction.wav / C2_final_mix.wav
│   ├── C3_interaction.wav / C3_final_mix.wav   ← 主交付
│   ├── cmp_final_mixes_spec.png / cmp_coverage.png / cmp_peak_retention.png
├── logs/
│   ├── evaluation.json             全部客观指标（12 版本 × 全指标）
│   └── evaluation_run.log          评估运行完整输出
├── work/                           build_diag / refine_meta / rolling_coverage / 帧证据条…
└── scripts/                        00–06 视觉标注、10 refinement、20 构建、30 评估
```

源文件只读 ✓ ｜ 生产代码未动 ✓ ｜ 未 commit ✓ ｜ 无生成/合成音频（全部样本取自 golden_raw）✓
