# S1: paired supervised adaptation with an aligned-reference ablation

**Status: proposed and ready to implement after acquisition; no S1 training or listening results exist.**

Run one bounded experiment: fine-tune the same pretrained CLAPSep separator twice, with equal capacity and training budget, once using an aligned pristine nuisance reference and once with that reference replaced by a null condition. Compare both with frozen and corrected historical baselines on new, independently held-out real source takes and real recordings.

This experiment answers two separate questions:

1. Does a modern learned extractor supplied with trustworthy domain supervision materially improve target preservation versus nuisance suppression?
2. Does the exact aligned music reference add value beyond that supervision and model capacity?

A positive answer to the first and a negative answer to the second is a useful outcome. Neither answer depends on saving the old gate or publishing a new model.

## 1. Why this is the first experiment

The local CLAPSep checkpoint and code already exist. Its zero-shot trial used contaminated in-recording exemplars, so it did not test supervised adaptation with clean source-image targets. A small decoder adaptation isolates that missing evidence at lower cost than training a universal model. The local head is a bounded magnitude mask with mixture phase: representation limits must be diagnosed, not mistaken for a limit on all neural extraction. [CLAPSep](https://arxiv.org/abs/2402.17455), [official implementation](https://github.com/Aisaka0v0/CLAPSep).

Include a modern broad pretrained comparator because the field has moved beyond AudioSep/CLAPSep. SAM Audio already offers text/visual/span-conditioned separation; its public checkpoint access is gated and is an execution dependency. If access is unavailable, use a verified available SoloAudio or FlowSep checkpoint and record the missing comparison. A missing checkpoint is not a failed quality result. [SAM Audio](https://github.com/facebookresearch/sam-audio), [SoloAudio](https://github.com/WangHelin1997/SoloAudio), [FlowSep](https://github.com/Audio-AGI/FlowSep).

## 2. Acquisition prerequisite and split

Follow [the data plan](D:/Code/RhythmAlign/experiments/s0_research_reframing/DATA_AND_GROUND_TRUTH_PLAN.md). Acquire ten independent sessions, nominally eight scored minutes each: four minutes quiet actual interaction at phone position, one minute target-free playback, three minutes ordinary noisy gameplay. Capture close/contact sensors and video for independent evidence. Ten sessions yield about 40 minutes quiet target material, 10 minutes playback-only and 30 minutes real noisy recordings.

Split **four train / two development / four test sessions before segmentation**. The clean training budget is about 16 minutes, not thousands of independently observed targets created by remixing. Seek at least three players, two devices and two cabinets/days if feasible. Seal at least one player/device combination from training, and report any unavailable crossing. Record nuisance outside these blocks as needed, with source pools partitioned by recording identity. Aim for 30–60 minutes of distinct nuisance across the specified families; reusable licensed real recordings can supplement new capture.

Do not proceed to supervised training if quiet target quality is inadequate or only contaminated ordinary gameplay exists. In that case, S1a has found an acquisition problem; redesign capture. Do not use R4/R5 audio-assisted onset lists or separator outputs as clean waveform targets.

Keep the historical golden/holdout excerpts as **external legacy demonstrations** after tuning. They are not the new test population and have no exact target waveform. Existing visual labels remain proxies.

## 3. Training mixtures and sealed evaluation panels

### Training

Generate five-second crops online from train-only source pools, with 60% mixed target+nuisance, 20% target-only and 20% target-absent examples. Half of target-present crops should include weak contacts, friction or decays rather than only sharp strong taps. Within mixed examples balance music, speech/announcer, similar impacts and composite nuisance. Draw target-to-total-nuisance ratio uniformly over −20 to +10 dB in active target intervals. Record random seeds and mixture recipes.

Use real music-only acoustic recordings paired with pristine references where available; augment with source-wise RIR/EQ/loudspeaker transforms. Keep post-mixture clipping, codec and joint-AGC stresses in a labeled secondary panel because their target convention differs. Preserve native source waveforms; controlled construction is allowed training/evaluation data, never a replacement output for the user.

The reference-conditioned arm uses correct reference 75% of the time, null reference 10%, wrong song 10%, and a clearly misaligned reference 5%. Include correct references with small alignment uncertainty within the 75%. When context is deliberately wrong, the desired waveform remains the factual target; do not penalize the network for ignoring wrong context. The paired null arm sees the same mixture, target, query and augmentation schedule, but all reference tokens are null. Reserve reference-free target-only and target-absent scenarios too.

Use a fixed positive target description plus a small bank of clean different-take train-only audio exemplars, with query modality choice paired between arms. A generic interaction query selects the defined player–machine target set; instance-specific exemplars are labeled separately. Do not feed the exact isolated test target as its own query. Few-shot enrollment from test-session support is a separate secondary condition with support clips excluded from scoring.

### Exact constructed primary panel

Select 24 nonoverlapping five-second target takes, six per test session, before any model output is viewed. Stratify them across weak/sparse contacts, dense contacts and continuous friction/decays. Mix each with four independently chosen held-out nuisance recipes at four ratios: −20, −10, 0 and +10 dB. This yields **384 target-present constructed clips**, 32 minutes of output per method. These are repeated measures of 24 target takes, not 384 independent subjects.

Add **48 target-absent five-second clips**, twelve per test session, balanced over playback-only, voices, other-instance impacts and quiet/rest conditions. For near-silent input, score absolute false-output energy separately from unstable relative attenuation. Annotate at least 12 constructed counterfactual pairs with a missing expected contact and an added offbeat contact; these can reuse selected target takes with recipes fixed in advance, and are reported as a distinct control panel.

The 384-clip primary panel uses correctly aligned pristine references for the reference-containing recipes; non-music recipes use a null reference. Report the music-present subset separately. Apply null, wrong-song and shifted-reference evaluation to a preregistered 96-clip subset, balanced across source takes and ratios. Add wrong-query and no-query controls on a smaller balanced subset with an explicitly defined requested target; absence claims require a query whose requested source is independently known to be absent.

### Real and stress panels

Select **24 five-second real excerpts**, six per held-out session: sparse, dense, weak, contact+nuisance overlap, unrelated similar impact and rest/target-absent where independently supported. Include all available categories or explicitly record missing strata. Use sensor/video annotations and blind listening, not fictitious exact target SI-SDR.

Use an additional small labeled T3 stress set for changing path, nonlinear playback, phone AGC/compression and clipping. Report it separately, without mixing its approximate truth into the exact-panel score. No extra training campaign is authorized by a stress failure in S1.

## 4. Sanity checks before spending the training budget

1. Verify float mixture reconstruction, sample-rate/gain conventions, source/split hashes and event labels. A stored component sum must match its T2 mixture to numerical precision before model evaluation.
2. Validate output against identity/zero fixtures and inspect quiet target timbre/decays. Confirm no accidental per-output loudness normalization, channel swap or train/test source reuse.
3. Assess a ground-truth-informed bounded-mask representation diagnostic. For each STFT bin the complex-error minimizing real mask is `clip(Re(S*conj(Y))/(|Y|²+epsilon),0,1)`. Invert with the model's actual window/hop and compare with truth. This binwise solution is **not a certified global waveform upper bound**, because STFT redundancy/consistency and overlap-add matter. If necessary, briefly optimize bounded masks directly against waveform loss to obtain a stronger attained oracle result. Never call a poor attained oracle a mathematical proof that every mask fails.
4. Compare mixture phase versus target complex spectrum diagnostics on development data. If even optimized bounded masks lose decisive weak events or timbre, stop the CLAPSep-head experiment and carry that diagnosed representation issue to a compact complex-output pilot. Do not conclude that neural separation is impossible.
5. Run a 100-step training profile, verify finite nonzero decoder/adapter gradients and a tiny training-batch fit. This is implementation validation; it is not evidence of generalization.

The current local `inference_from_data` wrapper calls `eval()` and uses `torch.no_grad()`. Implement a separate training forward, preserve frozen encoder behavior intentionally, and train the separation layers through an actual differentiable path. Load existing checkpoint tensors strictly; document any new adapter keys. Do not silently ignore missing weights or change the backbone configuration to obtain a load.

## 5. Two adapted arms and fixed budget

| Item | Specification |
|---|---|
| Backbone | Existing local CLAPSep checkpoint and original configured separator dimensions |
| Frozen components | Text/query encoders, pretrained audio feature encoder and encoder normalization state; same in both arms |
| Trainable components | Existing separation decoder/output mask, small reference/null adapter, small presence head |
| Audio | Common 32-kHz mono for primary comparison; preserve original high-band material for secondary reporting |
| Model interface | Five seconds of valid supervised content padded to the pretrained ten-second interface; mask padding out of losses/presence labels |
| Reference adapter | Complex/log-magnitude reference STFT → 128-dimensional 25-Hz tokens; four attention heads of width 32; mixture-frame queries; relative-time bias within ±0.5 s; null token; zero-initialized output projection into existing pre-mask feature width |
| Null arm | Identical adapter modules/parameter count, reference replaced by null; any unused signal-encoder capacity disclosed |
| Output | Original sigmoid magnitude mask × complex mixture STFT; target waveform and residual `y-target` |
| Presence | 25-Hz probabilistic head, independently labeled/soft uncertainty intervals; logged, never a hard gate in S1 |
| Optimizer | AdamW; decoder LR `1e-4`, new adapter LR `3e-4`, presence LR `1e-4`; weight decay `1e-4`; clip gradient norm 5 |
| Warm-up | First 200 updates adapter/head only; then decoder+adapter/head; identical schedule in both arms |
| Budget | Two seeds per arm; 5,000 optimizer updates per run, including warm-up; four runs total |
| Batch / memory | Microbatch 1, accumulate 4 microbatches, AMP where numerically stable; checkpoint decoder activations if required |
| Checkpoint choice | Fixed final update is primary; development-selected earlier checkpoint may be reported as a declared secondary result |

Local code integration may require matching feature width/time interpolation to inspected tensor shapes. Preserve the design and record actual dimensions; no post-result architecture sweep. If the null adapter contains permanently unused reference-encoder parameters, report trainable **and active** counts and the small computation difference; the comparison still controls the shared separator and extra fusion capacity.

### Losses

For target-present valid samples, use `L = L_wave + 0.5*L_MRSTFT + 0.1*L_presence`. `L_wave` is mean absolute waveform error divided by target RMS with a fixed floor derived from train data. `L_MRSTFT` is the mean of spectral convergence and log-magnitude L1 at FFT sizes 256, 1024 and 2048, Hann windows and quarter-window hops at 32 kHz. Epsilon/floor and reduction conventions are fixed in the training manifest.

For target-absent examples use `L_absent = mean(s_hat²)/(mean(y²)+floor²)` with weight 1 plus `0.1*L_presence`; do not divide by zero target RMS or compute target SI-SDR. Train presence from independent audible-event labels/uncertainty, including friction duration and decays; physical contact of unknown audibility is excluded or softly weighted under a fixed rule. These are **proposed starting weights**, fixed across both arms and not literature-optimal constants.

No informative mixture-sum loss exists when residual is defined as `y-s_hat`; report exact bookkeeping without treating it as an authenticity objective. Do not add perceptual text-similarity reward or generative sound completion to this first comparison. SI-SDR is an evaluation measure; scale-dependent waveform loss prevents using arbitrary output attenuation as an easy solution.

## 6. Baselines and what each comparison establishes

| Baseline | Implementation / allowed information | Interpretation |
|---|---|---|
| Raw/current | Original mono input and current user-facing mixture/remix separately | Identity preservation, no suppression; existing playback preference |
| Corrected temporal gate | R5-FIX envelope construction; real detector and independently annotated timing variants labeled separately | Exposure/timing benefit; oracle timing is privileged diagnostic, not an automatic model |
| Shifted / coverage-matched gate | Same envelope support/ramps/coverage shifted under fixed rules | Separates timing information from generic attenuation |
| Reference cancellation | Frozen best documented R1 method plus linear reference/path fit using allowed playback-only calibration | Tested subtraction baseline; access to calibration equalized and labeled |
| AudioSep zero-shot | Fixed documented prompt selected on development data only | Legacy text-conditioned extraction |
| CLAPSep zero-shot | Fixed positive/negative query recipe; clean different-take exemplars permitted consistently | Effect of domain adaptation beyond a fair query baseline |
| Modern broad separator | SAM Audio if available, otherwise verified available SoloAudio/FlowSep checkpoint | Whether an existing stronger system is already sufficient |
| Adapted null-reference arm | Shared supervised training and capacity | Learned extraction hypothesis |
| Adapted reference arm | Same training plus aligned nuisance content | Incremental pristine-reference hypothesis |

Run a lightweight learned AEC such as NKF on a small development/diagnostic subset if its verified implementation is straightforward; it is an informative neighboring baseline, not an extra model-training program. Its released 16-kHz bandwidth must be labeled and scored fairly. A public AEC checkpoint that removes playback but also loses contact does not settle full-band target extraction. [NKF](https://github.com/fjiang9/NKF-AEC)

All test operating points are chosen using development data. For a common preservation–suppression sweep, also evaluate `s_alpha = alpha*s_hat + (1-alpha)*y`, `alpha∈{0,.25,.5,.75,1}`. This disclosed conservative mixing knob measures a frontier; it is not an independent separator. Use the same knob for eligible baselines and never select alpha by test-truth performance. For gates, retain nominal coverage and gain-threshold coverage at 0, 0.1 and 0.5, with valid boundary exclusions fixed before comparison.

## 7. Primary metrics and preregistered decision margins

Use the definitions and caveats in [the evaluation plan](D:/Code/RhythmAlign/experiments/s0_research_reframing/DATA_AND_GROUND_TRUTH_PLAN.md). Report per-session, per-nuisance and per-SNR results, including every adverse stratum. Scores below are **proposed practical feasibility margins**, not universal definitions of authenticity or outcomes already observed.

### Learned-path gate

Choose the strongest corrected/frozen baseline's eligible operating point on development data. On sealed test, an adapted arm must meet all of the following to count as a clear S1 go:

1. **Preservation:** event recall decreases by no more than 5 percentage points relative to that baseline and raw evidence labels; median target projection gain is within ±1 dB of the recorded target on well-conditioned cases. Target-only gain and friction/decay spectral error must not materially regress. Report recall and waveform/timbre separately; meeting recall alone is insufficient.
2. **Suppression:** at that preservation level, median combined nuisance projection attenuation improves by at least 3 dB over the baseline on the well-conditioned exact panel; direct fixed-gain reconstruction SNR improves by at least 2 dB. Both improvements must occur in at least three of four held-out sessions, not only pooled remixes. Report projection exclusions and the remaining panel's direct waveform metrics.
3. **Absence/evidence:** on nonquiet target-absent inputs, at least 95% of clips have output energy at least 30 dB below input; no more than one of the 48 absent controls contains a clearly recognizable unsupported target event after independent review. Report exact counts and uncertainty. Separate entirely invented events from retained real nuisance impacts: the latter are source misassignment/leakage, not proof of sound fabrication. Missing-hit/extra-hit controls must follow the actual target, with no systematic expected-hit completion.
4. **Perception:** finalist wins at least 60% of non-tied paired judgments against the strongest corrected/frozen baseline on real excerpts, while the median separate target-completeness and timbre ratings do not decline on the five-point anchored scale. Show listener/session uncertainty; if the confidence interval is broad, label this preliminary corroboration rather than significant superiority.

The pilot may be inconclusive despite promising point estimates. Do not change margins after seeing test results. If some criteria pass and others fail, record a conditional result with the precise failure and the next diagnostic, rather than an overall invented authenticity score. A model passing constructed truth but failing real listening does not pass the real-world feasibility gate. Independently tag cases whose requested instance is not identifiable from the supplied context before evaluation; report them as an ambiguity/abstention stratum as well as in the full panel. A hidden dataset identity label is not information available to the model, and a failure there does not establish model inadequacy on identifiable cases.

### Reference-value gate

Compare the two adapted arms using paired seeds, target takes and mixture recipes. On the music-present subset, correct-reference input should add at least **1.5 dB median nuisance attenuation** at the same preservation constraints, with benefit in at least three of four test sessions. Wrong/null reference should not produce a preservation failure greater than the 5-point recall/1-dB gain margin. Also compare the reference-trained model's correct/null/wrong inference conditions; this within-model test is not a substitute for the separately trained null arm.

If learned extraction passes but the reference gate fails, advance the adapted null model and record “reference advantage unestablished.” If a frozen modern model passes and fine-tuning adds little, prefer that existing system. The scientific recommendation follows measured evidence rather than the architecture selected in S0.

### Statistical treatment

Use paired session summaries as primary units, show all four test-session values, and bootstrap hierarchically by session/source take as an exploratory interval. Four test sessions cannot justify narrow population confidence or precise cross-domain claims. The two seeds diagnose optimization instability; do not select the better seed on test. Record both and their aggregate. The same source mixed at 16 settings is a repeated-measures factor, not a 16-fold independent sample gain.

## 8. Blind listening execution

Use 12 listeners and 24 held-out excerpts: 12 constructed and 12 real, chosen before outputs. Each listener judges 12 pairs plus separate completeness, fidelity, nuisance and artifact attributes. The balanced design distributes 144 judgments across 24 excerpts × three pair types, two judgments per cell. Pair types cover finalist/raw, finalist/strongest corrected-or-frozen baseline, and adapted reference/null. The finalist and comparator are fixed from development data.

Include hidden target anchors for constructed excerpts and repeated trials for QC. Do not present a close/contact track as perfect truth for a real excerpt. Randomize anonymous filenames and order, preserve fixed input-derived playback gain, and keep stems primary. Remixes receive identical pristine music gain and alignment in a separate block. A person familiar with the labels must not selectively curate pleasant output examples after unblinding.

## 9. Compute, runtime and stop rule

The verified local GPU is an RTX 4060 Laptop with 8,188 MiB VRAM. Historical inference allocation is not a training benchmark. Target a peak below 7.5 GB; run a 100-step profile that includes four-microbatch accumulation and optimizer state before committing to the full schedule.

Four runs × 5,000 optimizer updates = 20,000 updates. At an **assumed** 1.5–6 seconds per accumulated update, training is roughly 8–33 GPU hours; allow 8–36 hours as the provisional plan. Real throughput may be slower, especially under laptop thermal limits. Frozen baselines/evaluation may add 2–8 local GPU hours and a short cloud inference job. Measure peak allocated/reserved VRAM, wall time, GPU power mode and seconds of output per compute second.

Try at most three documented memory configurations: frozen encoders/AMP; decoder checkpointing and encoder offload; then reduced valid crop length with identical changes to both arms. If gradients still cannot fit or profiling forecasts over 48 aggregate training hours, use one 24-GB cloud GPU for the same bounded experiment or report a compute dependency. Do not degrade the target bandwidth or silently omit difficult cases to claim 8-GB feasibility. A modern foundation comparator may need a 24–48-GB inference allowance; actual selected model requirements must be measured.

Budget 30–100 GB for pilot media, generated evaluation outputs, checkpoints and manifests. On-the-fly training mixtures avoid storing every remix. Expected human/engineering effort is roughly 2–4 capture/curation days plus 3–7 implementation/evaluation days, with listener scheduling additional; these are planning estimates, not completion promises.

**Hard stop:** after the four fixed runs or 48 aggregate training GPU hours, whichever is reached first, freeze S1 and report. No open-ended prompt sweep, extra seeds or architecture search to rescue a test result. If the cap prevents completion, label the experiment incomplete and report the profile rather than a negative learning result. Reopening a hypothesis requires a new explicitly scoped stage and a fresh held-out evaluation if test information influenced design.

## 10. Interpret every possible outcome

| Outcome | Scientific interpretation / next action |
|---|---|
| Existing modern model passes; adapted models add little | Existing extraction may already be sufficient; focus on calibration, data and product integration. |
| Adaptation passes; reference adds value | Advance Route A; expand independent data and diagnose transfer/generalization. |
| Adaptation passes; reference does not | Learned path exists; use the null model, improve/abandon reference branch based on diagnostics. |
| Good bounded oracle, poor training fit | Implementation/optimization/data issue, not evidence against extraction. |
| Good training fit, poor held-out exact mixtures | Diversity/conditioning/generalization failure; inspect source splits and hard negatives. |
| Exact panel improves, real panel fails | Domain/target-truth mismatch; improve real capture/path modeling before scaling. |
| Bounded representation diagnostics fail, complex target succeeds | Test a phase-capable compact head in the next stage; no “all AI failed” conclusion. |
| Strong suppression, damaged target or absent-target failures | Violates the product contract despite pleasing demos; fidelity/abstention remains unresolved. |
| All learned systems fail under verified truth and meaningful diagnostics | Narrow conditions or reconsider route; do not infer general impossibility from this small domain pilot. |

## 11. Required S1 output bundle

Produce source/split manifests and hashes, capture QC, exact mixture recipes, model/checkpoint hashes, profile and training logs, outputs at fixed gain, per-source metrics with two axes, control results, anonymous listening pack and sealed key, listener data/QC, failure examples, and a decision report applying these fixed rules. Keep original recordings and disclose truth tier on every figure/table. This document specifies the next experiment; it does not claim that these future artifacts already exist.
