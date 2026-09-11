# RhythmAlign Phase R5.6 — Precision Cleanup + Burst-Constrained Audio Completion Report

**在 R5.5 冻结检测器之上，只做两件事：候选后过滤（E1）+ burst 内受限音频补全（E2）。
Dev 调参（Dev-A/B/C，Dev-C oracle 不完整只作锚点）→ 冻结 → 全新 Holdout-D [60,78] 盲测（本轮改为 VIDEO-FIRST 盲注，注解先于检测）。**

日期：2026-09-09 ｜ 素材：Dev-A [22.5,37.5] + Dev-B [84,102] + Dev-C [40,58]（开发）+ Holdout-D [60,78] 18 s（全新盲测）
约束遵守：R5.5 视觉特征 / burst 检测器 / C1 / ±40 ms refinement / clean alignment / gain convention 全部未动 ✓ 未训练模型 / 未动生产代码 / 未 commit ✓ 冻结（13:13:37）早于任何 Holdout-D 检测（13:52+），盲注完成（13:39:22）早于检测、晚于冻结 ✓

---

## 0. RESULTS FIRST

**本轮核心问题的回答：在 Dev 上三项全部成立，但在最严格的盲测（Holdout-D，video-first 完整 oracle）上不成立。E2 与 R5.5 (E0) 在 Holdout-D 上统计上打平——precision cleanup 的收益没有迁移出样本。**

- **Phase 1 审计再次推翻本轮出发点的一半**："echo FP 与真实相邻 contact 可用峰-峰比值区分" **被否证**：echo FP 的 echo_ratio 中位 **1.09**，反而高于真实相邻 contact 的 0.75–0.84；任何能压 echo 的比值阈值都会杀掉更多真实 90–120 ms 双击（devA thr=0.5：去 10 echo / 杀 4 双击）。视觉 echo 抑制被数据否决，未进入配方。**refinement 置信度本身已经覆盖 echo**：99 个 echo FP 中 87 个是 weak/no_transient/present。
- **E1（置信度门控+弱候选视觉支持+音频拯救层）**：Dev 三窗假窗时长 **-15/-26/-19%**，音乐泄漏 **-22/-35/-55%**，UWR 全升，strong recall 一分不掉（rescue 层保住 7 个"视觉沉默但音频可证"的弱候选，其中 5 个是 high-conf oracle 唯一匹配——视觉侧永远救不了它们）。
- **E2（E1 + burst 内强音频补全）**：Dev 上全面占优 E0——strong gate **+6.7/+4.2/+5.9 pt**、strong recall devC **0.810→0.905**、假窗/泄漏/UWR/RMS 全改善；20 个补全事件 18 个匹配 oracle、0 个 bass 主导。
- **Holdout-D（[60,78] 18 s，VIDEO-FIRST 盲注 86 事件——迄今最完整 oracle）**：
  - **strong recall 0.857（≥0.85 目标达成）**，timing p95 44 ms，taiko 泄漏 **0**（太鼓防线有效）；
  - 但 **E2 vs E0（R5.5）几乎打平**：F1@50 0.533 vs 0.534，strong recall 持平 0.857，假窗 5.07 vs 5.22 s（**-3%**），音乐泄漏 38 vs 40（**-5%**），strong gate 0.886 vs 0.914（**-2.8 pt**），UWR 0.712 vs 0.743（-3.1 pt），mix RMS -13.6 vs -13.8 dB。
  - 补全 4 事件：1 真（11.65，oracle 亦有）、1 无法证实（0.24）、**2 落在 G1 rest 区内**（6.10/6.43，burst hysteresis 的 500 ms hold 让 rest 前 0.5 s 仍是"活动 burst"，音频补全把音乐 onset 当成了接触）——这正是本轮警告过的静息污染模式，实际发生了。
- **为什么 Dev 赢、Holdout 平**：Dev 的精度收益主要来自清掉 devC 式"weak-FP 长尾"（devC 假窗 6.18 s）；Holdout-D 是 AP 密集段+干净 rest，弱候选大多是真 hits（E1 在此窗反而比 E0 多开 9 窗）。**规则层在密集真密集段的 precision-explore tradeoff 已经到顶**（R5.5 已测 dense F1 规则上限 ~0.59；本轮 F1 0.533 未越过）。

**判定：A —— strong-contact 口径的产品级重建已达标（0.857 recall / 0.886 strong gate / mix 接近 oracle / taiko 0），进入 R6；剩余精度差距不再用 heuristic 追（后续研究方向 = C：轻量 learned temporal detector，用本轮 4 个窗口的 video-first oracle 当训练/验证数据）。**

---

## 1. Phase 1 — Echo / weak-candidate audit（先量化，后规则）

对 Dev-A/B/C 全部 389 个候选采集 9 类特征（`logs/phase1_audit_*.json`），四个问题的回答：

| 问题 | 回答 | 证据 |
|---|---|---|
| Q1 echo FP 与真实相邻在 peak-ratio 上可分？ | **否（被否证）** | echo FP echo_ratio 中位 1.09（devA/devB/devC 1.095/1.095/1.087）vs 真实相邻 0.754/0.843/0.538——echo 不比真信号弱；thr=0.5 时 devA 去 10 echo 杀 4 真双击、devC 去 14 杀 9。**规则被否决** |
| Q2 weak refinement 富集于音乐泄漏窗？ | **是** | weak 候选 TP 率：devA 31%、devB 27%、devC 24%（strong 为 75%/100%/65%）；devB 44 个 weak 中 32 个 FP |
| Q3 strong/present refinement 够安全？ | **是** | devB strong 35/35 全 TP；present 47–62% TP 可接受，按原逻辑开窗 |
| Q4 视觉沉默真 contact 有强音频吗？ | **是（E2 的依据）** | devA/B/C 视觉沉默 oracle 事件 5/12/10 个中，refinement 为 strong 的有 4/6/4 个；devA 补全空间内 4 个匹配事件全部 residual-band 支撑（mid 0.83/hi1 0.91） |

附加发现（决定 E1 rescue 层）：被弱门控误杀的 7 个候选（devA e30 + devC 4×high-conf 唯一匹配等）全部 comb_peak 0.52–1.23 / ratio 1.6–2.5 且视觉特征平坦——**它们就是"视觉沉默"事件本身，只有音频证据能保住它们**。rescue（peak≥0.80 且 ratio≥1.8）救回 11 TP / 复活约 6 echo 邻近窗。

## 2. E1 — Precision cleanup（唯一配方，`detector_config_r56_frozen.json`）

- **echo-ratio 抑制：否决**（Q1 数据），echo 由置信度门控间接覆盖（87/99）。
- strong / present：照常开窗；no_transient：留名不开窗（E0 惯例）。
- weak：需额外支持——(a) 视觉：prominence ≥0.30×p99.5，或 ≥2/4 通道 novelty ≥0.10×p99.5，或（burst 边 400 ms 内且距 strong/present 候选 ≤250 ms）；(b) 音频拯救：comb peak ≥0.80 且 >1.8× 局部噪声。
- 候选列表保持 R5.5 全集口径（含 no_transient），开窗决策单独记录。

## 3. E2 — Burst-constrained audio completion（唯一配方）

- 来源：冻结 comb 的强瞬态峰（peak ≥0.35 且 >4×±0.25 s 局部中位、55 ms 合并——镜像冻结 "strong" 档）。
- **burst 外绝对不补**；补全须距每个 E1 开窗候选 ≥90 ms、彼此 ≥90 ms。
- **太鼓防线**：要求去音乐残差支撑 mid+hi1+hi2 ≥1.2 且 hi1 ≥0.15（bass-only 的太鼓型 onset 过不了 hi1）；hand motion 只作记录不作为条件。
- Dev 校准（`logs/phase3_completion_space_*.json`）：补全空间共 18 峰，15 个匹配 oracle；unmatched 也非 bass 主导——resid 门控是针对 devC 沉默事件 bass 0.62/hi1 0.08 的泄露模式设的。

## 4. Dev 结果（E0 = R5.5 冻结，同评估代码）

| 指标 | devA E0→E1→E2 | devB E0→E1→E2 | devC E0→E1→E2（oracle 不完整*） |
|---|---|---|---|
| 候选 / 开窗 | 128/104 → 128/96 → 134/102 | 123/100 → 123/80 → 128/85 | 138/110 → 138/87 → 147/96 |
| strong recall | .947 → .947 → .947 | .818 → .818 → .818 | .810 → .810 → **.905** |
| strong gate | .833 → .817 → **.900** | .894 → .872 → **.936** | .794 → .735 → **.853** |
| 假窗 (s) | 2.72 → **1.96** → 2.14 | 3.44 → **2.53** → 2.65 | 6.18 → **5.03** → 5.23 |
| 音乐泄漏 | 27 → **21** → 24 | 17 → **11** → **10** | 33 → **15** → 19 |
| 太鼓泄漏 | 0/0/0 | 0/0/0 | 3/2/2 |
| UWR (hi+med) | .774 → .818 → .806 | .755 → .822 → **.833** | .569 → .631 → **.654** |
| mix RMS 差 (dB) | -15.7 → -15.9 → **-16.9** | -10.8 → -11.6 → **-12.3** | -8.8 → -9.6 → **-10.2** |
| F1@50 | .619 → .619 → **.630** | .586 → .586 → **.609** | .417 → .417 → **.474** |

\* Dev-C 的 precision/F1 带 qualification：oracle 已知漏标（R5.5 结论），unmatched 候选不可当 FP；其 strong recall 只对 21 个 high-conf 事件成立。E2 补全质量：20 事件 / 18 匹配 oracle≤80 ms / 0 bass 主导 / 5 个与音乐 onset 重合但均为残差支撑的在拍真实接触（含被找回的 2 个 strong oracle）。

## 5. FREEZE

`outputs/detector_config_r56_frozen.json`，frozen_at = **2026-09-09 13:13:37**（Dev-only 证据；含 E1/E2 全部参数与否决记录）。此后未看任何新 holdout 调参。

## 6. Holdout-D 选段与 VIDEO-FIRST 盲注

- **选段 [60.0, 78.0] 18 s**（`holdoutD_selection.json`）：全曲密度剖面（三个 dev 窗全部屏蔽）+ 两允许区 [58,84]/[102,136] 10 fps sheet 目检。内容：密集双手爆发 [60,65.5]、**真实 rest ~[66,67.8]**（玻璃近空、手落 bezel）、黄 slide 带、cyan fan slide、双压、手臂遮挡+柜体 LED 干扰。对齐：锚点 61/69/77 s，69 s 锚点吸附邻峰（离群 +10 ms 已记录），median **d = -11374.776 ms** 与两个 12 s 波形相关一致到 0.5 ms；AV offset -13.6 ms 为重编码伪影，refinement 仍用 0。
- **盲注顺序（video-first，oracle 不再由音频候选决定全集）**：
  1. Pass 1：10 fps 全览 8 sheets（结构 + rest 区判定）；
  2. Pass 2：30 fps 全窗 18 sheets（judgment text / hit effect 帧级定时）→ 75 事件；
  3. Pass 3：8 个遮挡/歧义时刻 60 fps 五帧条带（3.25→3.20、11.53→11.50 修正；6.8 的黄丝带判为"未上手 trace 的 note 显示"→ 删除并立 G1 rest 区）；
  4. Pass 4（最后）：107 个 comb 音频候选仅作查漏——密集段 18 个 gap 逐个条带核验：**11 确认增补**（多为 fresh PERFECT text 铁证）、6 拒绝、1 并入；**rest 段 13 个音频 onset 全部拒绝**（视频证明手离玻璃——纯音乐泄漏）。
- 最终 oracle：**86 contacts + G1 rest 区**，`annotation_completed_utc = 13:39:22`，晚于冻结、早于任何检测器运行。refinement 置信度分布：strong 35 / present 14 / weak 23 / no_transient 14。

## 7. Holdout-D 冻结运行（盲注之后）

| 指标 | E0 = R5.5 冻结 | E2 = R5.6 冻结 |
|---|---|---|
| 候选 / 开窗 | 150 / 123 | 154 / **116** |
| P / R / F1 @±50 | .420 / .733 / .534 | .416 / .744 / .533 |
| F1 @±33 | .339 | .342 |
| timing med / p95 | 23.3 / 44.0 ms | 23.0 / 44.0 ms |
| **strong recall** (21 high) | **.857** (18/21) | **.857** (18/21) |
| gate coverage / strong gate | **.931** / **.914** | .917 / .886 |
| 假窗 | 5.22 s | **5.07 s** |
| 音乐泄漏 / 太鼓泄漏 | 40 / 1 | **38 / 0** |
| UWR (all / hi+med / high) | **.764 / .743 / .195** | .733 / .712 / .189 |
| interaction coverage | .740 | .809 |
| mix RMS 差 / headroom | **-13.8 dB** / -0.75 dBFS | -13.6 dB / -0.75 dBFS |

- E2 strong 漏检 3：d42（4.27 遮挡融合事件）、d56（8.21，音频查漏发现的真实 SE 压——检测器 ±50 ms 未命中）、d71（14.30，同前）。
- 补全 4 事件：11.65 真（oracle 同有）、0.24 不可证实、**6.10/6.43 落入 G1 rest**（burst hold 滞后把 rest 边缘当活动段；太鼓防线挡住了 bass-only，但挡不住有残差支撑的音乐 onset）。E1-only 口径下此窗假窗 4.74 s / 泄漏 36——E1 是本窗最干净的点，但 strong gate 同样 0.886。
- 结论：**Dev 上 E2 全面优于 E0；Holdout-D 上 E2≈E0**（增益 -3% FW / -5% 泄漏 / 太鼓 1→0，代价 strong gate -2.8 pt / UWR -3.1 pt）。R5.6 的 dev 收益主要是清掉了 devC 式弱 FP 长尾，而 AP 密集段的弱候选大多是真 hits，清理空间不在那里。

## 8. 成功标准逐条裁定（Holdout-D 口径）

1. **strong recall ≥0.85**：0.857 ✓（但 E0 也是 0.857——非本轮增益）。
2. 假窗时长明显下降：**-3% ✗**。
3. 音乐泄漏明显下降：**-5% ✗**。
4. strong gate 不下降：**-2.8 pt ✗**。
5. final mix 更接近 oracle：**-13.6 vs -13.8 dB ✗**（噪声级差异）。

产品口径：strong contacts + final mix 已接近 oracle（RMS -13.6 dB、strong gate 0.886、taiko 0）→ 产品路线本身成立；**本轮的 precision-cleanup 增量没有通过盲测**。

## 9. FINAL DECISION：**A**（进入 R6 integration），后续精度研究 = C

- **A 的依据**：strong-contact 重建的产品级指标在最完整的 video-first 盲测上全部达标（recall 0.857≥0.85、strong gate 0.886、mix 接近 oracle、太鼓 0、假窗/泄漏处于可控水平）。冻结管线 = **E2**（R5.6 frozen），R6 可直接集成；因 E2 与 R5.5-E0 在 Holdout-D 上统计打平，R6 若求最保守可回退 E0——两者产品表现等价，E2 的补全受 burst 门控与太鼓防线约束、从未在 burst 外产出。
- **为什么不是 B**：剩余问题中唯一"全新、具体、低风险"的规则修复是"补全峰处要求 activity ≥ stay 阈值（排除 burst-hold 滞后）"，但它只解释 Holdout-D 上 2 个窗 / 0.33 s——不足以兑现"显著减少假窗和泄漏"的轮目标；其余全部是再调阈值，按本轮纪律直接排除。
- **为什么不是现在就 C**：R5.5 预设的转 C 触发条件（strong <0.85）未触发；且本轮已产出 4 个窗口的 video-first oracle（Dev-A/B + Dev-C/D 共 89+92+68+86 事件）——这正是轻量 learned temporal detector 需要的训练/验证资产。
- 不选 D：自动化收益已在两个独立盲测上被证实（Holdout-C 27/34 vs oracle 28/34；Holdout-D strong-R 0.857）。

## 交付物

```
experiments/r56_precision_cleanup/
├── outputs/
│   ├── detector_config_r56_frozen.json      (frozen 13:13:37，含否决记录)
│   ├── dev{A,B,C}_E1_contacts.json / _E1_refined.json / _E1_C1_*.wav
│   ├── dev{A,B,C}_E2_contacts.json / _E2_C1_*.wav          (E0 = R5.5 复用)
│   ├── dev_summary.png
│   ├── holdoutD_density_profile.png / holdoutD_selection.json
│   ├── holdoutD_video.mp4 / holdoutD_raw.wav
│   ├── holdoutD_oracle_blind.json           (video-first 盲注 13:39:22，先于检测)
│   ├── holdoutD_oracle_refined.json
│   ├── holdoutD_contacts_auto_visual.json / holdoutD_contacts_auto_refined.json
│   ├── holdoutD_auto_C1_interaction.wav / holdoutD_auto_C1_final_mix.wav
│   ├── holdoutD_oracle_C1_interaction.wav / holdoutD_oracle_C1_final_mix.wav
│   ├── holdoutD_e0_C1_*.wav (参考) / holdoutD_summary.png
├── work/   (选段 sheets、pass1/2 盲注 sheets、pass3/4 核验 strips、特征 npz、comb)
├── logs/   (phase1 audit ×3 / weak-gate grid / completion calibration ×3 /
│            dev eval E0_E1_E2 / holdoutD metrics)
└── scripts/ (common56 + 10-12 audit/设计 / 20 E1 / 30-31 校准+E2 / 40 dev eval /
              50 freeze / 60-64 选段+提取+盲注材料+oracle / 70 holdoutD 运行 / 80 plots)
```

源文件只读 ✓ ｜ 生产代码未动 ✓ ｜ 未 commit ✓ ｜ 无合成音频（全部样本取自 holdoutD_raw / golden_raw / holdout_raw）✓
