# RhythmAlign Phase R5.5 — Burst-Aware Automatic Contact Detection Report

**两层结构验证：LEVEL-1 burst 状态（hysteresis）+ LEVEL-2 爆发内原子事件分裂。
Dev 调参（Dev-A golden + Dev-B 旧 holdout，本轮均声明为开发数据）→ 冻结 → 全新 blind Holdout-C 验证。**

日期：2026-09-09 ｜ 素材：golden [22.5,37.5] 15 s + 旧 holdout [84,102] 18 s（开发）+ 共感怪物AP.mp4 [40,58] 18 s（Holdout C，全新盲测）
约束遵守：C1 reconstruction / clean alignment / audio refinement ±40ms / A-V mapping / R5 视觉 feature 定义全部未动 ✓ 无 source separation / heavy tracking / 训练模型 ✓ 生产代码未动 ✓ 未 commit ✓

---

## 0. RESULTS FIRST

- **Phase 1 audit 部分推翻了本轮核心假设**：R5 的 FN 时刻 fused score 中位数只有 **0.07–0.08**（阈值 0.40），仅 4–8% 的 FN 时刻 score 高于阈值——"score 高位平台缺上升沿"只描述少数 FN；多数 FN 是 **score 在该 contact 处根本没有脉冲**。locked-average 的 −116 ms "提前尖峰" 是密集间隔（中位 90–116 ms）下邻居贡献 + 自身沉默的合成。同时证实：dense 内有相当比例 oracle contact 在 60 fps + 现有特征下**没有可分离的自发瞬态**——任何单通道/融合局部峰规则的 dense F1 上限 ≈ **0.59**（dev 双窗实测）。
- **检测器**：LEVEL-1 burst 状态（fused activity 平滑后 hysteresis：enter 0.40 / stay 0.22 / 退出 hold 30 帧，dev 区域覆盖 0.947）；burst 外保留 R5 冻结上升沿规则（一字未改）；burst 内用 **rule A**（fused score 的每个显著局部峰直接成为候选 = 平台期 re-trigger）。**rule B**（jz_diff 原子峰+通道同意+burst 自适应阈值）在两个 dev 窗口全面更差，被否决。
- **Dev（可信 oracle）**：devA F1@50 **0.619** / strong **0.947**；devB F1@50 **0.586** / strong **0.818**（R5 基线：0.605 / 0.842；0.488 / 0.455）。gate coverage devA **86.9%** / devB **83.7%**（R5：63.8% / 66.3%）。
- **Holdout C（[40,58] 全新盲测，dense-dominant）**：冻结参数下 F1@50 **0.417**（同窗 R5 基线 0.367），**strong recall 0.810 vs 0.429**（≥0.80 目标达成），dense recall 0.632 vs 0.485，region coverage **1.000**，gate coverage **76.5%**（同窗基线 73.5%；R5 旧 holdout 口径 66.3%）。
- **Holdout C 的 oracle 本身被证明不完整**：同一盲注管线在 devB 产出 92 事件（中位间隔 90 ms），在密度剖面相当的 Holdout C 只产出 68（中位 203 ms）——comb 阈值在该音乐段漏列真实 hit。事件级 precision 双向不可靠；95 个 FP 中 47 个距最近 oracle ≤100 ms（burst echo/关联），多数带可信音频瞬态。
- 音频产品口径（Holdout C）：auto final mix 的 **strong contact 保留 27/34，oracle 自己是 28/34**（基本追平）；但泄漏上升（interaction stem 匹配音乐 onset 33 vs 基线 19），final mix RMS 差 −8.8 dB vs 同窗基线 −9.3 dB（略差）。
- 四项成功标准：strong ≥0.80 ✓（0.810）；dense recall 显著改善 ✓（+0.147）；gate 显著高于 66.3% ✓（76.5%）；**overall F1 显著高于 0.488 ✗**（0.417，且 0.488 这个数字来自已被本轮污染、oracle 更完整的旧窗口，不可直接横比——但按字面未过）。

**判定：B —— burst 分裂带来明显、可复现的提升（strong/dense/gate 三项达标），但事件 F1 与精度仍不够；再做一轮窄范围轻量规则改进（echo 抑制 + 瞬态置信度门控），若失败则转 C。**

---

## 1. Phase 1 — Trace audit（先证据，后改代码）

对 Dev-A/Dev-B 全部 TP/FN/FP ±300 ms 画了特征 trace（`outputs/phase1_traces_*.png`、`phase1_locked_avg.png`），量化结论（`outputs/phase1_audit_findings.md`、`logs/phase1*.json`）：

| 问题 | 回答 | 证据 |
|---|---|---|
| Q1 哪个通道 burst 内最尖锐 | jz_diff / hand_diff（TP 对齐中位偏差 16–33 ms）；glass 滞后 33–50 ms 且拖尾 | locked avg + 原子峰扫描 |
| Q2 jz_diff 适合作 atomic trigger 吗 | 定时最准，但单通道固定阈值漏 37/82（devA）、40/90（devB）的 dense contact | 通道原子峰 P/R 扫描 |
| Q3 glass_bright 能补遮挡吗 | 部分能（补 17/37、10/40），但滞后大，更适合当第二票 | 互补性统计 |
| Q4 hand_diff 适合作 support/veto 吗 | **veto 无鉴别力**（TP 0.250 vs FP 0.253，FP 是 echo，手同样在动）；做触发器反而 devB 最佳（F1 0.607） | R5 candidate 时刻 hand_diff 对照 |
| Q5 jz_text 只适合 burst-state 吗 | 是。contact 后 300 ms 仍是平台，burst 内全程高位 | locked avg + decay 曲线 |

**假设裁定**："R5 失败不是看不见 contact，而是不会在 burst 内数 contact"——**部分成立**：上升沿规则确实欠数（改局部峰后 dense recall 0.456→0.689）；**部分被推翻**：一部分 dense contact 视觉上就是沉默的，规则上限 dense F1 ≈ 0.59（devA 上 rule A 甚至 0.591 < R5 的 0.605）。

## 2. 检测器与冻结配置

`outputs/detector_config_r55_frozen.json`，frozen_at = **2026-09-09 03:51:51**，早于 Holdout C 提取（03:55）与盲注（04:06:48）。

- 视觉特征与融合权重 = R5 冻结原样（jz_text 1.8 / jz_diff 1.5 / glass 1.0 / hand 0.3，p99.5 归一，clip 2.5）。
- LEVEL-1：activity = fused score 5 帧平滑；enter ≥0.40，hysteresis stay ≥0.22，连续 30 帧（≈500 ms）低于 stay 退出。dev 区域覆盖 0.947 / 假 burst 0.77 s/窗 / 漏 burst 0.89 s/窗（3×3×3 网格，仅 dev）。
- burst 外：R5 上升沿规则原样（threshold 0.40 / slope 0.15 / min dist 3）。
- burst 内（rule A）：fused score 局部峰，local prominence ≥ 0.03×p99.5，min distance 3 帧（50 ms，与 dev oracle p10 间隔 61 ms 匹配），无上升沿要求——**平台期每个显著子峰直接成候选**。
- rule B（被否决）：jz_diff 原子峰 + glass/hand ±3 帧同意 + burst 内 p90 自适应阈值 + 振幅贪心接纳。dev dense F1 0.523/0.545 vs rule A 0.609/0.611，strong recall 也全面更低。

## 3. Dev 结果（dev-only 调参后，同一评估代码算 R5 基线）

| 指标 | devA R5.5 | devA R5 | devB R5.5 | devB R5 |
|---|---|---|---|---|
| candidates | 128 | 113 | 123 | 80 |
| P / R / F1 @±50 | 0.508 / 0.793 / **0.619** | 0.522 / 0.720 / 0.605 | 0.512 / 0.685 / **0.586** | 0.525 / 0.457 / 0.488 |
| F1 @±33 | 0.419 | 0.410 | 0.465 | 0.407 |
| timing med / p95 | 22.9 / 46.2 ms | 26.1 / 44.3 | 18.0 / 47.3 | 21.6 / 43.2 |
| dense P / R / F1 | 0.504 / 0.768 / 0.609 | 0.514 / 0.695 / 0.591 | 0.549 / **0.689** / 0.611 | 0.539 / 0.456 / 0.494 |
| strong recall | **0.947** (18/19) | 0.842 | **0.818** (18/22) | 0.455 |
| region coverage | 1.000 | 1.000 | 0.978 | 0.889 |
| FP / FN | 63 / 17 | 54 / 23 | 60 / 29 | 38 / 50 |
| **gate coverage** | **86.9%**（strong 83.3%） | 63.8% | **83.7%**（strong 89.4%） | 66.3% |
| final mix RMS 差 | **−15.7 dB** | −15.0 dB | **−10.8 dB** | −9.8 dB |

- sparse：devA 无 sparse contact；devB 2 个 sparse contact 两个检测器各命中 1 个，R5.5 在区域边缘多出少量过渡性 FP（n=2，样本过小只作记录）。**离散近完美表现没有实质退化**（R5 离散路径原样保留）。
- 泄漏代价（诚实记录）：devA interaction stem 匹配音乐 onset 27（R5 8，窗口变多音乐透过变多）；devB 17（R5 41，大幅改善）——泄漏随 candidate 数变化，是 precision 问题的音频面。
- 最值得试听：`devB_r55auto_C1_final_mix.wav` vs `devB_oracle_C1_final_mix.wav`（爆发内"少的那几下"基本补齐）。

## 4. Holdout C（全新盲测）

- **选段**：handcam-native **[40.0, 58.0]**，18 s。由全曲 onset 密度剖面 + 候选区 [134,158]/[38,60] 10 fps contact sheet 目检选出；[134,158] 因 ~143 s 进结算界面被否决。含密集双手爆发（~48–51、~56–58）、fan slide（~50）、chevron slide、yellow slide band N（~44–45）、连续 bottom/side press、中密度间歇（46–48、52–54）、手臂横穿遮挡。与 [22.5,37.5]/[84,102] 零重叠。`holdoutC_selection.json` 记录 selection_reason。
- **对齐**：3 个 4 s 锚点（41/49/57 s）一致性 0.26 ms（无窗口内漂移）；一个 12 s 窗差 7.3 ms 但其覆盖 [56,68] 超出窗口 10 s（窗口外内容所致），已记录，采用锚点共识 d = −11366.727 ms。AV offset −12.3 ms 仅为重编码文件音轨伪影（R5 同理），refinement 仍用 0.0（帧时钟=raw 音频时钟，R4 实测 +0.2 ms）。
- **盲注**：R5 同款三遍法（7×10fps 全览 → 19×30fps → 68 个音频候选逐个 60 fps 三帧条带目检；边缘候选 #34/#35/#55/#57 用 30fps 表复核）。68 contact / 0 rest（21 high / 44 med / 3 low）。**annotation_completed 04:06:48，晚于冻结 03:51:51、早于任何 detector 运行**。
- **oracle 完整性警告（重要）**：该窗口 R1 区声明全程连续（手从未离玻璃）。同一盲注管线的 comb 候选在 devB 产出 95 个（92 contact，中位间隔 90 ms），在 Holdout C 只产出 68（中位 203 ms），而两窗密度剖面相当——comb 阈值在此音乐段漏列真实 hit。因此本窗事件 precision 双向失真：R5.5 的 138 候选 vs 68 oracle 的巨大 P 差距不能全记为过触发。

| 指标 | R5.5（冻结） | 同窗 R5 基线 |
|---|---|---|
| candidates | 138 | 112 |
| P / R / F1 @±50 | 0.312 / 0.632 / **0.417** | 0.295 / 0.485 / 0.367 |
| F1 @±33 | 0.320 | 0.289 |
| timing med / p95 | 21.7 / 44.2 ms | 23.0 / 39.7 |
| dense P / R / F1 | 0.312 / **0.632** / 0.417 | 0.295 / 0.485 / 0.367 |
| sparse | 无 sparse 区（窗口全程连续） | 同 |
| **strong recall** | **0.810** (17/21) | 0.429 (9/21) |
| region coverage | **1.000** | 0.990 |
| FP / FN | 95 / 25 | 79 / 35 |

- FP 结构：95 个中 47 个距最近 oracle ≤100 ms（burst echo/关联），65 个 ≤150 ms；60 个有可信音频瞬态（51 weak + 9 strong/present）——与"oracle 漏列 + echo"混合的图景一致（`holdoutC_fp_hist.png`）。
- **音频（冻结 refinement + C1）**：auto 138 候选 → 110 有可信瞬态（26 strong）；oracle 68 全部有瞬态（34 strong）；80 ms 一对一匹配 51/68，配对误差中位 0.0 ms。**gate coverage 76.5%（strong 79.4%）** vs 同窗基线 73.5%（76.5%）；strong 保留 27/34 vs oracle 自身 28/34；interaction 泄漏 33 vs 基线 19 个音乐 onset；final mix RMS 差 −8.8 dB（基线 −9.3 dB）。假窗 6.17 s。
- 试听：`holdoutC_auto_C1_final_mix.wav` vs `holdoutC_oracle_C1_final_mix.wav`——strong contacts 段几乎一致，差距集中在密集弱事件段与音乐泄漏。

## 5. 对四项成功标准的逐条裁定

1. **strong recall ≥ ~0.80**：0.810 ✓（17/21；漏掉的 c19/c27/c54/c62 均为视觉沉默事件）。
2. **dense recall 显著改善**：0.632 vs 0.485 ✓（同 oracle 同窗）。
3. **gate coverage 显著高于 66.3%**：76.5% ✓（但同窗基线已有 73.5%，增益主要在 strong）。
4. **overall F1 显著高于 0.488**：0.417 ✗。按字面未达标；但 0.488 来自 oracle 更完整的旧窗口，与 Holdout C 的 0.417 不可直接横比（同窗口径 R5.5 比基线 +0.05）。

## 6. FAILURE DECISION：**B**

burst 分割带来了明显、跨窗口可复现的提升（dev 双窗 strong/dense/gate 大幅改善；全新盲测 holdout 上 strong recall 0.429→0.810、dense recall +0.147、region coverage 1.0），方向正确但整体 F1/precision 仍不够。剩余失败模式已经定位且**规则层仍有明确的下一步**：

- **echo/关联 FP**（47/95 距 oracle ≤100 ms）：可用"峰-峰比值 refractory"（新峰需 ≥ 此前 120 ms 内最强峰的 X%）抑制；
- **弱候选泄漏**（67 个 weak 瞬态开窗 → 音乐透过）：生产路径按 refinement 置信度门控（strong+present 开窗，weak 降级合并）。

这两项都是轻量规则，可在 dev 上快速验证。**若该轮后 Holdout 级 F1 仍 <0.5 且 strong <0.85，转 C（轻量 learned temporal detector）**——Phase 1 已证明规则层 dense F1 上限 ≈0.59，继续堆 heuristic 边际收益有限。不建议 D（回退）：strong contact 口径的自动化收益已被盲测证实（27/34 ≈ oracle 的 28/34）。

---

## 交付物

```
experiments/r55_burst_event_splitting/
├── outputs/
│   ├── detector_config_r55_frozen.json        (frozen 03:51:51)
│   ├── phase1_audit_findings.md / phase1_traces_devA|devB.png / phase1_locked_avg.png
│   ├── devA_r55_contacts_visual.json / devA_r55auto_refined.json / devA_r55auto_C1_*.wav / devA_oracle_C1_*.wav
│   ├── devB_（同上一套）
│   ├── holdoutC_density_profile.png / holdoutC_selection.json
│   ├── holdoutC_video.mp4 / holdoutC_raw.wav
│   ├── holdoutC_oracle_blind.json             (盲注 04:06:48，先于检测)
│   ├── holdoutC_contacts_auto_visual.json / holdoutC_contacts_auto_refined.json
│   ├── holdoutC_contacts_r5baseline_visual.json / holdoutC_r5baseline_refined.json
│   ├── holdoutC_auto_C1_interaction.wav / holdoutC_auto_C1_final_mix.wav
│   ├── holdoutC_oracle_C1_interaction.wav / holdoutC_oracle_C1_final_mix.wav
│   ├── holdoutC_r5baseline_C1_interaction.wav / holdoutC_r5baseline_C1_final_mix.wav
│   ├── holdoutC_timeline.png / r55_vs_r5_summary.png / holdoutC_fp_hist.png
├── work/   (holdoutC 特征 npz / comb / 对齐 / 盲注材料 7+19 sheets + 17 strips)
├── logs/   (phase1 audit / burst grid / rule 选型 / dev+holdoutC 全部指标 json)
└── scripts/ (common55 + detector55 / 10-12 audit / 20 burst+rules / 40 dev eval /
              50 freeze / 60-63 holdoutC 选段+提取+盲注材料+oracle / 70-73 运行+音频+基线 / 80 plots)
```

源文件只读 ✓ ｜ 生产代码未动 ✓ ｜ 未 commit ✓ ｜ 无合成音频（全部样本取自 holdoutC_raw / golden_raw）✓
