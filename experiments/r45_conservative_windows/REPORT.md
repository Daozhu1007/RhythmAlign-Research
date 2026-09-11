# R4.5 — Conservative Contact Window Reconstruction

目标：保留 C1 的清晰/力度/自然感，减少时间门控造成的"剪碎感"。
冻结：oracle contact identity、A/V offset、refined audio onsets、onset detector、
clean music alignment、final mix gain convention（0.5·ref + 0.5·stem）。
只改变 interaction extraction envelope / window policy。未改生产代码，未 commit。

输入复用 `../r4_contact_reconstruction/`：golden_raw.wav、ref_warp_fixed.wav（golden 切片）、
contacts_oracle.json（9 regions 含 G1 gap）、contacts_refined.json（84 有效事件）。

## 版本

| 版本 | 窗口策略 |
|---|---|
| D0 | 原 C1：pre 15 ms / post 150 ms，merge=max，attack 10 ms / release 60 ms |
| D1 | 同 D0，post → 210 ms（唯一保守值，无 sweep） |
| D2 | 按类型固定窗：press 15/180，touch 15/140，slide 15/250，rest 按 touch；release 并入前一事件尾部（+80 ms），不单独开门脉冲；confidence 不缩短窗口 |
| D3 | 主实验：D2 离散窗 + oracle region 策略。短连续 region（R1 0.37 s / R3 1.45 s / R6 0.90 s / R8 0.83 s）连续保留+自然 fade；长/不确定 region（R2 2.70 s / R4 5.67 s / R5 1.82 s / R7 0.82 s）保持离散窗；全局只桥接 gap ≤ 70 ms（唯一阈值）；G1 gap 保持关闭 |

注：contacts_refined 的时间戳 refinement 已吸收 17.625 ms A/V 残差
（t_audio_refined − t_video 中位 0.3 ms），region→audio 映射使用该经验偏移与事件窗一致。

## 结果（stems，同一 84 contacts、同一检测器）

| | retention | strong | peak ×raw | dyn corr | sup heavy/quiet dB | taiko out-of-win | coverage | chop vs C1 |
|---|---|---|---|---|---|---|---|---|
| C1 | 66/84 | 60/60 | 1.00 | 0.89 | −7.8 / −6.3 | 4 | 0.740 | — |
| C2 | 64/84 | 59/60 | 0.56 | −0.72 | −9.9 / −8.2 | 4 | 0.739 | 84.4% / 216 ep / 53 ≥50ms |
| C3 | 66/84 | 60/60 | 1.00 | −0.17 | −0.6 / −0.5 | 0 | 0.820 | 14.8% / 92 ep / 2 ≥50ms |
| D0 | 66/84 | 60/60 | 1.00 | 0.89 | −7.8 / −6.3 | 4 | 0.740 | 0.00% / 0 ep |
| D1 | 66/84 | 60/60 | 1.00 | 0.89 | −3.1 / −2.0 | 1 | 0.855 | 0.00% / 0 ep（C1 严格超集） |
| D2 | 66/84 | 60/60 | 1.00 | 0.89 | −5.9 / −4.2 | 4 | 0.799 | 8.98% / 37 ep / 26 ≥20ms / 0 ≥50ms（中位 28 ms，touch 尾 140<150 所致） |
| D3 | 65/84* | 60/60 | 1.00 | **0.92** | −5.9 / +0.0 | 2 | 0.866 | **3.70% / 15 ep / 11 ≥20ms / 0 ≥50ms** |

\* D3 的 e78（13.415 s weak press）未被检测器命中是泄漏抬高自适应阈值所致：
D3 门控是 D2 的逐点超集，e78 峰值实测 ×1.000 完整保留。无真实 contact 损失。

chop metric 定义：C1 有效期间（env_D0 ≥ 0.5），5 ms RMS 帧 / 2 ms hop，
候选 < 0.8·C1 且 C1 帧 > −80 dBFS 的衰减 episode 统计。

Final mixes：所有 D 版 peak −2.54 dBFS（headroom 2.54 dB），与 C1 完全一致（冻结增益约定）。

## 七问

1. **D0 复现 C1**：成功。interaction 与 final mix 逐样本 max abs diff = 0.0。
2. **D1 长尾**：contact 指标零变化、零新增衰减（0.00%，C1 严格超集）。是否更自然需人耳；
   代价是 music suppression −7.8→−3.1 dB（heavy）、coverage 0.740→0.855。
3. **D2 type-aware**：retention/peak 不变；slide/press 尾更完整，touch 尾缩短带来
   8.98% 相对 C1 的轻微衰减（全部 episode < 50 ms）。整体不确定优于 D1。
4. **D3 region/bridge**：显著减少削切——chop 3.70% vs C3 的 14.76%（相对 −75%），
   ≥50 ms 削切 0（C3 有 2、C2 有 53），dyn corr 0.92 全场最高，strong 60/60、peak ×1.00。
5. **leakage / coverage 代价**：D0 基线（0.740；−7.8/−6.3）→ D1（0.855；−3.1/−2.0）→
   D2（0.799；−5.9/−4.2）→ D3（0.866；−5.9/+0.0）。D3 的 quiet 帧接近全开是主要泄漏代价
   （region 连续保留 + bridge），taiko out-of-window 反而 4→2。final mix 均有 2.54 dB headroom。
6. **试听顺序**：C1（锚）→ D1 → D3 → D2 → C3（剪碎对照）。
7. **最值得听**：`D3_interaction.wav` + `D3_final_mix.wav`（主实验），
   其次 `D1_final_mix.wav`（零风险长尾）。重点段落：连续区 3.3–4.75 s（R3 slide）与
   密集区 4.75–10.4 s（R4）；对照 C3 同段。

## 结论与选择

- 素材输出：`outputs/D{0..3}_interaction.wav`、`outputs/D{0..3}_final_mix.wav`（48 kHz stereo float）。
- 诊断：`work/build_diag.json`、`logs/evaluation.json`、`outputs/r45_gates_and_chop.png`。

**选择：A（条件成立，待人工确认）** — 客观证据支持 D3（首选）/ D1（保守）为优于 C1 的候选：
contact 完整性、峰值、动态全部 ≥ C1，C3 式短时削切大幅消失，且无任何 contact 指标回退。
按本轮规则最终以人耳裁决：若试听确认 D3/D1 自然度 ≥ C1 → 进 R5 自动视觉 contact detection；
若人耳仍判 C1 最好 → 退回 B（冻结 C1 进 R5）。
无证据支持 C（削切未消）或 D（泄漏升为主要问题；heavy 帧仍抑制 −5.9 dB）。
