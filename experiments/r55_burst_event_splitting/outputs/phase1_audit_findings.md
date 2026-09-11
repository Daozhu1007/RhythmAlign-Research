# R5.5 Phase 1 — Trace Audit 结论（先证据，后改代码）

数据：Dev-A golden [22.5,37.5]（82 contacts，全部 dense）+ Dev-B 旧 holdout [84,102]
（92 contacts，90 dense / 2 sparse）。视觉 feature 定义与 R5 完全相同（frozen）。

## 基线（R5 frozen 检测器，dense-only 口径）

| 窗口 | dense P | dense R | dense F1 | 备注 |
|---|---|---|---|---|
| devA | 0.522 | 0.720 | **0.605** | 全部 82 个 contact 都在 continuous region 内 |
| devB | 0.512 | 0.456 | **0.482** | 90/92 dense |

## 关键新证据（R5 报告没有的）

1. **FN 处 fused score 并不高**：FN contact 时刻 fused score 中位数仅 0.07–0.08
   （阈值 0.40），只有 4–8% 的 FN 时刻 score 高于阈值（TP 中位 0.43–0.54）。
   → "score 长期高位平台导致无上升沿" 只描述少数 FN；多数 FN 是 **fused score
   在该 contact 处根本没有响应**（无脉冲），不是"有脉冲但缺上升沿"。
2. **locked average 的 −116 ms 尖峰是邻居污染 + 自身沉默的合成**：
   dense 内 contact 中位间隔仅 90–116 ms（p10≈61 ms）。对齐到 contact k 时，
   −116 ms 的 jz_diff 峰多数属于 contact k−1；TP 自身 0 ms 处有峰（0.30–0.35），
   FN 自身 0 ms 处接近波谷（0.17–0.20）。
3. **存在"视觉沉默"的 dense contact**：FN 的 jz_diff 自身响应显著弱于 TP。
   按 ±50ms 一对一匹配，任何单通道原子峰规则（固定阈值+prominence）的 dense F1
   上限 ≈ **0.58–0.59**（devA 0.591 / devB 0.584，recall 0.60–0.73）——
   高于 R5 规则的 devB 0.482，但 devA 上不去（0.591 < 0.605）。
   即：**爆发内还有相当一部分 oracle contact 在 60 fps + 这些 feature 下不可分辨**
   （sub-tap / 被邻居 text-aftereffect 掩盖 / oracle 在 dense 段本身按 region 包络声明的局限）。
   devB 音频侧 ±40ms 内 0/90 缺 transient —— 但间隔 65–90 ms 时音频窗口同样互相重叠，
   不能证明每个 oracle contact 都有独立声源。

## 五个问题的回答

**Q1 哪个通道在 burst 内保持最尖锐？**
jz_diff 与 hand_diff 最尖锐（TP 对齐中位 |offset| 16–33 ms，≈1–2 帧）；
glass_bright 滞后更大（33–50 ms，devB 部分 +100–200 ms）；jz_text 无事件定时能力。
单通道原子触发最强者窗口依赖：devA glass_bright（F1 0.592），devB hand_diff（0.607）、
jz_diff 次之（0.551–0.560）。没有全能单通道。

**Q2 jz_diff 是否最适合做 atomic trigger？**
做"定时源"最合适（最尖锐、偏差最小），但作为唯一触发器不够：固定阈值下
jz_diff 漏掉 devA 37/82、devB 40/90 的 dense contact（prominence 3% 宽松口径）。

**Q3 glass_bright 能否补 jz_diff 被遮挡的事件？**
部分能：jz_diff 阈值漏掉的 dense contact 中，glass_bright 峰命中 17/37（devA）、
10/40（devB）。但 glass 滞后大且 devB 后段拖尾长，作 timing 源噪声大；
更适合做"第二票/支持通道"。

**Q4 hand_diff 更适合 support/veto？**
在 R5 candidate 时刻测 hand_diff：TP 中位 0.250–0.281 vs FP 中位 0.253–0.261
—— **在候选时刻无鉴别力**（FP 是爆发内 echo，手同样在动），做不了硬 veto；
只能否决"手完全静止"的罕见情形。有趣的是 hand_diff 单独做原子触发器在 devB
F1 0.607（最佳），说明它有触发信息量，但与 jz_diff 高度重叠。

**Q5 lingering text 是否让 jz_text 只适合 burst-state？**
是。jz_text locked average 在 contact 后 300 ms 仍是平台（无回落）；
burst 内连续多个 contact 时 jz_text 全程高位（见 phase1_traces h05/h06）。
jz_text → burst-state 指示器；其 novelty（文本出生 2 帧上升）仍参与事件定时，
但作为 raw 计数只回答"是否处于操作区"。

## 对核心假设的裁定

"R5 失败不是看不见 contact，而是不会在 burst 内数 contact"——
**部分成立，部分被推翻**：

- 成立部分：R5 的上升沿规则确实在爆发内欠触发（devB dense recall 仅 0.456），
  改成"fused score 局部峰 / 原子峰"即可把 dense recall 提到 0.60–0.73，
  strong recall 与 gate coverage 预期大幅改善。
- 被推翻部分：并非每个 contact 都"仍然存在独立尖峰"。FN 中相当一部分
  （估计 1/4–1/3）在 60 fps 视觉特征下没有可分离的自发瞬态，
  任何简单阈值规则都到不了它们（规则上限 dense F1 ≈ 0.59）。

→ 策略：Phase 3 按"爆发内局部峰分裂"实现并冻结；评估重点放在
**strong recall、gate coverage、sparse 不退化**；dense event F1 的结构性
上限如实报告，不再追求把它推过 0.85。

交付图：`outputs/phase1_traces_devA.png` / `phase1_traces_devB.png` /
`phase1_locked_avg.png`；数据：`logs/phase1_audit_*.json`、
`logs/phase1b_lead_lag.json`、`logs/phase1c_rule_ceiling.json`。
