# RhythmAlign Phase R5 — Automatic Contact Detection Report

**从手元视频自动检测玩家 contact，驱动已冻结的 C1 reconstruction（dev 调参 → 冻结 → blind holdout 验证）**

日期：2026-09-09 ｜ 素材：R1 Golden Sample 15 s（dev）+ 共感怪物AP.mp4 84.0–102.0 s 18 s（holdout）
约束遵守：生产代码未动 ✓ 未 commit ✓ C1/refinement 参数未改 ✓ 未重做 source separation / FIR / HPSS ✓ 未训练模型 ✓ 未用 heavy vision model ✓

---

## 0. RESULTS FIRST

- 检测器：**4 个简单视觉 feature**（judgment 文本、judgment 区变化、glass 命中特效闪光、手部运动）→ 加权融合 → 上升沿峰值检测。全部为可解释、低成本信号，无任何学习模型。
- **Dev（golden，允许调参）：event F1@±50ms = 0.605，strong recall 0.842**。关键发现：**在 oracle 连续区域之外，检测器 0 FP / 0 FN（完美）**；全部错误集中在密集连续段内部——那里 oracle 本身声明无法逐 event 分解（R4 notes）。
- 检测器于 02:35:00 冻结（`detector_config_frozen.json`）。**Holdout oracle 为盲注**（03:08:44 写入，先于任何 holdout 检测器运行，文件内有时间戳记录）。
- **Holdout（18 s，未调参）：event F1@±50ms = 0.488，strong recall 0.455 —— 明显低于理想门槛（0.85 / 0.90）**。
- 失败模式单一且可归因：**密集爆发段内 score 平台化，上升沿规则把爆发内的多个 contact 合并成爆发起点的一个 candidate**（FN 中 46/47 在连续区域内部；FP 中位距离最近 oracle 仅 99 ms——是"少报"而非"幻听"）。离散段（tap 之间有空隙的段落）依然近乎完美。
- 音频后果（holdout）：auto 门控只覆盖 **66.3%** 的 oracle contact（strong 72.3%），auto stem 里音乐泄漏 onset 41 个 vs oracle 14 个，auto final mix 与 oracle final mix RMS 差 **−9.8 dB**（dev 为 −15.0 dB）。**自动混音与 oracle 混音的差距在 holdout 上 audible**。
- 头部空间正常（−2.4 dBFS），refinement 在有 candidate 处全部锁定到同一 transient（配对 |err| 中位数 0.0 ms）——音频侧管线本身可复用。

**判定：B —— 基本可行，但视觉 detector 还需一轮改进（爆发内事件分离），不建议直接进入 R6 产品化。**

---

## 1. Detector

| Feature | 定义 | 为什么 |
|---|---|---|
| jz_text | judgment 文本区（两块 ROI）内 warm+white 像素计数 | maimai 每次有效判定都在判定点附近渲染 CRITICAL PERFECT / FAST+SLOW 文本——这是"发生了有效操作"的直接视觉证据（R4 已用同一逻辑人肉核验 89 候选） |
| jz_diff | judgment ROI 逐帧灰度差均值 | 文本出生是 1–2 帧内的突变，是所有通道里与 contact 对齐最锐的信号 |
| glass_bright | glass 椭圆内 gray>200 像素计数 | 命中特效（星形爆闪/白圈）在命中点闪光——不依赖文本，判定文本被手臂遮挡时仍有效 |
| hand_diff | y≥330 带内逐帧灰度差 | 手/臂运动能量：区分"画面里有特效"与"玩家真的在做动作"，也是 G1 悬空段全静默的来源 |

关键工程事实（Phase 2 audit 发现）：
- **判定文本位置不固定**：它渲染在判定点附近，实测散布 x∈[240,900]、y∈[170,300]（80px 分箱直通全带），near-side（S）按键时文本+按键 LED 出现在 y≈430–480 → 判定区必须用"上带 ∪ 下核心"两块 ROI。
- **warm 掩膜必须抗皮肤色**：玩家手臂经过判定区时 r−g 中位数 67（文本只有 1），`|r−g|<45` 项把手臂误报从 ~14467 px 压到 ~150 px。黄色 slide 缎带颜色上无法与文本区分（r−g≈10），只能靠时间规则抑制。
- 融合：novelty = 3 帧平滑序列的 2 帧正上升，按各自 p99.5 归一（clip 2.5），权重 1.8/1.5/1.0/0.3 相加；**时间规则唯一**：score ≥ 0.40 且 2 帧 rise ≥ 0.15 时输出 candidate，candidate 间隔 ≥ 3 帧。
- 调参过程（全部只用 dev）：5 点 threshold 扫描 → threshold×slope 3×3 → 一轮权重修订。没有大规模 grid search。

## 2. Development result（golden 15 s，oracle=82 contacts / 7 non-contact / 8 regions）

| 指标 | 值 |
|---|---|
| candidates | 113 |
| P / R / F1 @±50ms | 0.522 / 0.720 / **0.605** |
| P / R / F1 @±33ms | 0.354 / 0.488 / 0.410 |
| 匹配 timing error | median 26.1 ms / p95 44.3 ms |
| strong recall（oracle conf=high, n=19） | **0.842**（16/19；e14 差 53 ms，e30/e72 为 bottom press 各偏 83/120 ms） |
| region coverage | **1.000**（每个连续区域内部无 >0.3 s 空洞） |
| false-open duration | 0.000 s（oracle contact spans 覆盖全片） |
| FP / FN | 50 / 23 —— **全部位于连续区域内部；区域之外 0 FP / 0 FN** |

区域内部的错配必须如实解读：R4 oracle 明确声明密集段内嵌 sub-tap 无法在 60 fps 分解（用 region 包络表达）。因此区域内部的事件级 P/R 同时测量了"检测器错误"和"oracle 不完整"。区域内部 FP 到最近 oracle 的距离 median 81 ms——多数是 association echo（真 sub-tap 或相邻事件），部分（p75=125 ms，max 353 ms）是 slide 缎带/特效引发的真 FP。

## 3. Frozen configuration

`outputs/detector_config_frozen.json`，frozen_at = **2026-09-09 02:35:00**，在 holdout 选择（其后）、holdout 盲注（03:08:44）与 holdout 检测器运行（03:09 之后）之前。holdout 上没有做任何参数修改。

## 4. Holdout（18 s = handcam-native 84.0–102.0 s，blind oracle）

- **选段**：音频 onset 密度剖面（全曲 2 s 分箱）+ 选择期 contact sheet 目检。84–85 密集、86–95 稀疏、96–102 渐密（黄色 slide），无菜单/结算，与 golden [22.5, 37.5] 无重叠。`outputs/holdout_selection.json`。
- **盲注方法**：R4 同款三遍法——10 fps 全览 7 张 → 30 fps 19 张 → 95 个音频候选（R4 族 comb 包络，med+4MAD，55 ms 合并）逐个在 60 fps 三帧放大条上目检。92 contact + 3 rest/rejected，8→4 个 region（R1/R2/G1/R3）。**annotation_completed 03:08:44，先于 detector 在 holdout 上的任何运行**；检测器冻结时间戳更早。
- 对齐基础设施：ref 对齐用 R1 同款方法（全波形 40–800 Hz 相关，两个 12 s 窗独立测量一致到 0.1 ms）→ 固定 d = −11.367038 s；AV offset 先验 0 ms（依据 R4 refined 事件实测 t_video+0.2 ms median；±40 ms 搜索窗口不变）。

| 指标 | 值 |
|---|---|
| candidates | 80 |
| P / R / F1 @±50ms | 0.525 / 0.457 / **0.488** |
| P / R / F1 @±33ms | 0.438 / 0.380 / 0.407 |
| 匹配 timing error | median 21.6 ms / p95 43.2 ms |
| strong recall（n=22） | **0.455**（10/22） |
| region coverage | 0.889（R1 0.81 / R2 0.84 / R3 0.99） |
| false-open | 0.498 s |
| FP / FN | 36 / 47 |

refinement 后：62/80 candidates 有可信 transient；与 oracle refined 一对一匹配 48/92（80 ms）；gate coverage of oracle contacts **66.3%**（strong 72.3%）。

## 5. Error taxonomy

- **FN（47）**：46/47 在连续区域内部（12 个 high-confidence）。类型：press 36 / slide 11。根因是**爆发内 score 平台**：密集段 score 持续高于阈值且不再"上升"，上升沿规则在爆发内只触发一次（往往在爆发头），后续 100–200 ms 间距的 contact 合并丢失。离散段只有 1 个 FN（h61，在 G1 边缘）。
- **FP（36）**：34 在区域内部，2 在 G1 悬空段（11.26 / 11.34 s，过渡手势）。FP 到最近 oracle 距离 median 99 ms——绝大多数是爆发内 re-fire/echo，不是安静段的凭空幻报。分类归属：**B（continuous contact 表达不足）为主 + E（爆发内弱事件漏检）**；A（ROI 不稳）不成立（同一机位，ROI 直接过）；D（特效误报）只在 1.4–1.6 s 出现一小簇（手臂挥动无判定）；F（refinement 出错）不成立（有 candidate 必锁定）。
- dev→holdout 衰减（F1 0.605→0.488）说明检测器确实对 golden 的爆发形态有适应性，但主要问题是规则本身对长爆发的结构性缺陷，不是过拟合金样本。

## 6. Audio result（冻结 C1：pre 15 / post 150 ms，attack 10 / release 60，merge by max，final = 0.5 pristine + 0.5 stem）

| 指标 | Dev auto | Dev oracle | Holdout auto | Holdout oracle |
|---|---|---|---|---|
| stem contact 保留 | 56/84 (67%) | 66/84 (79%,=raw 上限) | 60/92 (65%)* | 56/92 (61%,=raw 上限) |
| stem strong 保留 | 49/60 | 60/60 | 35/47 | 40/47 |
| gate coverage | 0.638 | 0.647 | 0.389 | 0.562 |
| 假窗（无 oracle contact ±80 ms） | 48 个 / 3.21 s | — | 63 个 / 1.79 s | — |
| stem 内音乐 onset 匹配 | — | — | **41** | **14** |
| final mix 峰值 | −2.54 dBFS | −2.54 dBFS | −2.4 dBFS | −2.4 dBFS |
| auto vs oracle final RMS 差 | **−15.0 dB** | — | **−9.8 dB** | — |

\* holdout auto 的保留数被假窗泄漏抬高（音乐 onset 落进假窗被计为命中）；更真实的口径是 gate coverage 66.3%。

- 音频侧管线本身可复用：oracle 重建逐指标复现 R4（golden coverage 0.647、保留 66/84、strong 60/60 全对上）；refinement 配对误差 median 0.0 ms（同一 comb 峰）。
- **最值得人工试听**：
  1. `outputs/holdout_auto_C1_final_mix.wav` vs `outputs/holdout_oracle_C1_final_mix.wav` —— 直接听自动化的代价（~1/3 命中缺失 + 音乐泄漏）。
  2. `outputs/dev_auto_C1_final_mix.wav` vs `outputs/dev_oracle_C1_final_mix.wav` —— dev 上更接近（−15 dB），密集段几乎一致，可听出"爆发内少了几下"。
  3. `outputs/holdout_auto_C1_interaction.wav` —— 检查 63 个假窗的音乐泄漏（尤其 10.7–12.0 s 过渡段）。

## 7. Generalization verdict

冻结检测器在从未调参的 holdout 上：身份判定（"这里是否有玩家操作"）在离散段可靠、无幻报；但**爆发内事件计数失败**（R 0.457 / strong 0.455），音频后果可闻（gate coverage 66%、泄漏 ×3、mix 差 −9.8 dB）。理想门槛（F1 ≥ 0.85、strong ≥ 0.90、p95 ≤ 30–40 ms）未达到——p95 达标（43 ms），其余未达。

**最终选择：B —— 基本可行，但视觉 detector 还需一轮改进。**

理由：不属于 C（简单视觉特征信息不足）——trace 上逐事件信号确实存在（jz_diff/glass_bright 在爆发内每个事件都有尖峰），失败在融合/时间规则（平台期无上升沿），是规则层可修的：候选方向包括（a）平台期内用通道局部极小值/子峰分裂 re-trigger，（b）每爆发自适应阈值（hysteresis 下降臂），（c）以 jz_diff 尖峰为原子事件做爆发内细分。也不属于 A（选 D 放弃自动化）——离散段近乎完美 + 音频管线已验证，自动化上限仍高。

改进一轮后必须重新走"冻结 → 新 holdout → 盲注"流程；本轮 holdout 已被本 detector"污染"，不能再用于下一轮调参验证。

---

## 交付物

```
experiments/r5_auto_contact_detection/
├── outputs/
│   ├── detector_config.json / detector_config_frozen.json
│   ├── dev_contacts_auto_visual.json / dev_contacts_auto_refined.json
│   ├── dev_auto_C1_interaction.wav / dev_auto_C1_final_mix.wav
│   ├── dev_oracle_C1_interaction.wav / dev_oracle_C1_final_mix.wav
│   ├── holdout_video.mp4 / holdout_raw.wav / holdout_selection.json
│   ├── holdout_oracle_blind.json            (盲注, 03:08:44, 先于检测)
│   ├── holdout_contacts_auto_visual.json / holdout_contacts_auto_refined.json
│   ├── holdout_oracle_refined.json
│   ├── holdout_auto_C1_interaction.wav / holdout_auto_C1_final_mix.wav
│   ├── holdout_oracle_C1_interaction.wav / holdout_oracle_C1_final_mix.wav
│   ├── dev_timeline.png / holdout_timeline.png / pr_dev_vs_holdout.png
│   ├── holdout_fp_fn_strips.jpg / holdout_mix_compare.png
│   ├── dev_feature_traces.png / dev_refine_comparison.png
│   └── strips_dev/ (dev FP/FN 目检条)
├── work/  (features npz, comb, oracle 标注材料 7+19 sheets + 24 strips, 对齐)
├── logs/  (所有中间指标 json)
└── scripts/ (10 audit / 20-25 detector+freeze / 30,35,36 dev audio /
              50 holdout extract+align / 60,62 blind oracle / 70,75,76 holdout run / 80 plots)
```

源文件只读 ✓ ｜ 生产代码未动 ✓ ｜ 未 commit ✓ ｜ 无合成音频（全部样本取自 holdout_raw / golden_raw）✓
