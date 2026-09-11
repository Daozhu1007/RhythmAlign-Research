# Data, ground truth and evaluation plan

**RhythmAlign S0 · 2026-09-10 · Proposed protocol; no new capture or listening study has been performed.**

The highest-value action is a quiet, muted-playback cabinet recording at the **actual handcam microphone position**, with synchronized close air microphone, contact microphone and video. Capture real taps, palm hits, button impacts, slide friction, decays and genuine rests. These recordings supply a trustworthy seed that the existing mixed excerpts do not provide.

The target is the acoustic image at the intended microphone, including the chosen room response, rather than an idealized dry impact or a contact-sensor waveform. See the formal target and nonlinear-device distinction in [the main study](D:/Code/RhythmAlign/experiments/s0_research_reframing/S0_RESEARCH_STUDY.md).

## 1. Truth tiers: what each dataset can establish

| Tier | Construction / observation | Legitimate claims | Claims it cannot support |
|---|---|---|---|
| T0: sensor/event evidence | Video, piezo/contact channel, close microphone, action log | Contact/action existence, approximate timing, selected identity, sensor agreement | Exact phone-domain waveform; every contact being audible |
| T1: isolated acoustic target take | Quiet scene, playback muted, phone-position air recording | Fidelity to this recorded source image, including measured noise contamination | Perfect noise-free truth if ventilation/handling/room noise remains |
| T2: exact constructed additive mixture | Stored real target `s`, recorded/path-transformed music `m`, independent nuisance `u`; `y=s+m+u` in float PCM | Exact waveform bookkeeping and target error for this constructed task | Exact realism of simultaneous loudspeaker/phone nonlinearities |
| T3: acoustic replay / device stress | Replay isolated target/nuisance stems through speakers and record real devices; optionally synchronized linear reference mic | Transfer, clipping/AGC stress, realism gap; approximate paired target at a reference mic | Exact phone target from separately recorded passes when gain/path/state change |
| T4: simultaneous real scene | Actual interactions plus actual arcade playback, independent sensor/video channels | Real-world perceptual usefulness, event correspondence, proxy agreement | SI-SDR against a close/contact track as if it were exact ground truth |

T1 target noise is not automatically removed by calling it truth. Measure rest floors; flag or reject contaminated clips using **input/source evidence before separator evaluation**. Do not clean targets using the model being assessed. Preserve raw takes and quality flags. T2 labels are exact relative to their stored stems; if a target stem contains hiss, disclose that the label contains hiss.

For post-mixture device distortion `y=D(s+m+u)`, do not label `D(s)` the additive target unless the intended task explicitly uses that counterfactual. For a gain-only simulation, record the factual gain trajectory `g_y` and use `g_y*s`, `g_y*m`, `g_y*u`; their sum is exact. For saturation/codec distortion, either retain a separate distortion residual with a declared pre-distortion target convention or evaluate as T3. Report clipped/erased intervals separately rather than forcing one allocation to be physically unique.

## 2. Pilot acquisition hardware and procedure

### Channels

| Channel | Purpose | Main limitation |
|---|---|---|
| Ordinary handcam phone/camera | Product-domain input, video, operational DSP behavior | AGC, clipping, lossy codec and unknown clock processing |
| Small air microphone at phone position into a recorder | Near-linear target source-image reference and reproducible level | A few centimeters of displacement can change high-frequency phase; not interchangeable with phone audio |
| Close air microphone near interaction region | Better target-to-room ratio, airborne timbre evidence | Music, voices and nearby impacts leak in; position alters timbre and relative contact levels |
| Safely attached contact/piezo sensor | Independent mechanical-contact timing | Speaker vibration, structural resonances, handling and missing airborne friction components |
| Optional directional air microphone | Additional target SNR / interference evidence | Off-axis coloration, reflections and imperfect spatial rejection |
| Pristine digital music + alignment manifest | Exact nuisance identity and source clock | Acoustic playback transfer and live machine effects remain unknown |

Use an existing recorder/interface where possible; exact equipment purchase is not necessary to define this study. Record uncompressed PCM at 48 kHz/24 bit when the device permits, plus the ordinary phone recording. Lock recorder gain, leave headroom and document phone settings. Preserve the native stereo input even though the primary S1 task is mono. Do not invent multichannel information from stereo tracks that are duplicated or heavily coupled.

No sensor attachment, volume change or service-mode action should occur without the cabinet operator's agreement. Arrange a quiet session and a safe placement method; this is an acquisition dependency, not a reason to postpone the rest of S0.

### Synchronization

1. Start every recorder before the take. Record a shared audible sync pattern visible on video at start and end, outside scored intervals. An LED driven by the same trigger is useful when available; an ordinary visible clap/contact is a lower-precision fallback.
2. Fit a clock map `t_phone = a + b*t_recorder` using multiple anchors. Add piecewise drift only when residuals justify it; retain the fit and residuals. Do not align each predicted output to truth independently.
3. Correct **clock offset/drift** separately from physical acoustic propagation. A contact sensor precedes air arrival and must not be shifted to match arbitrary mixture peaks. Document microphone geometry and sensor latency.
4. For paired waveform scoring, use a shared-clock multichannel recorder whenever possible. Proposed QC target: sub-sample residual alignment for those recorded channels; separately synchronized phone audio cannot be assumed phase accurate. For video/event labels, record an uncertainty interval; a nominal 60-fps frame provides about 16.7 ms sampling, not sample-level contact truth.
5. Validate start/end sync and a withheld intermediate anchor. Keep original timestamps and resampling filters. The pristine reference gets its own music clock map, with uncertainty; this map is not an interaction annotation.

### Per-session block, nominal eight minutes

| Block | Duration | Content |
|---|---:|---|
| Quiet target | 4 min | Playback muted; actual contacts, sparse/weak taps, dense contacts, palm hits, varied slide/friction, decays, deliberate rests |
| Playback only | 1 min | Player inactive; selected pristine music, at documented volume/path; captures target-absent reference behavior |
| Real mixture | 3 min | Ordinary gameplay with environment; synchronized auxiliary sensors and video |

Sync/calibration overhead is additional. A session means an independently set-up recording with documented player/device/position/day, not a camera restart. Aim for ten sessions: four train, two development, four test. This gives 80 minutes of scored capture and 16 minutes of quiet training material, before segmentation and QC. Cross at least three players, two devices and, if access permits, two cabinets/days; keep a device/player combination outside training. Ten sessions are a feasibility pilot, not a population-representative benchmark.

Muted interaction actions may differ from ordinary rhythm-guided playing. Include visual chart-following and natural free interaction, and document this shift. Do not replace the quiet acquisition with ordinary loud gameplay and infer clean labels from its music-correlated peaks. A later carefully arranged synchronized cue can improve realism, but earphone bleed and safety must be measured rather than assumed absent.

Record missing expected contacts and extra offbeat contacts, alongside normal patterns. Retain target-free cabinet animation/music and genuine contact motion that is inaudible at the phone. These cases distinguish waveform evidence from action/chart prediction.

## 3. Quantify proxy leakage instead of declaring it negligible

Before each setup, capture short blocks of: quiet rest; playback with no interaction; speech/noise with no interaction; isolated interaction; and simultaneous activity. For each sensor, report RMS/spectrum relative to phone-position target levels, clipping rate and coherence with pristine playback after alignment. Contact channels with substantial speaker vibration cannot establish “target present” from energy alone.

A useful diagnostic is target-to-leakage ratio measured on isolated calibration blocks at fixed gains. It estimates sensor selectivity in those conditions; it is not exact instantaneous SNR during double talk. Repeat after moving a microphone or changing cabinet volume. Keep leakage traces and calibration timestamps in the manifest.

Use proxy channels for corroboration, temporal annotation and uncertainty weighting. If training from proxies later, learn an explicit acoustic-domain mapping and report its error on quiet paired takes. Do not optimize phone output to reproduce piezo resonances. Do not choose the cleanest-looking proxy output after seeing model results.

## 4. Constructed mixtures that retain relevant difficulty

Generate mixtures from **recorded real target takes** and independently sourced real nuisance. Preserve every component before summing. Record music-only acoustic playback at the intended position paired with its pristine file; this gives a real nuisance path for additive construction. Supplement it with measured RIR/EQ paths and nonlinear loudspeaker transforms applied to music alone. Such component-wise transformations preserve additive labels, while jointly acting phone DSP belongs to the separately labeled stress set.

Nuisance strata must include cabinet music, neighboring music, NPC/announcer speech, ordinary speech, unrelated impacts from a similar object, sustained arcade ambience and mixed combinations. A single broad “noise” pool conceals the most important instance-selection failure. Similar impacts are particularly valuable negatives: a model can preserve impact timbre yet retain the wrong impact.

Use target-to-total-nuisance ratios of −20, −10, 0 and +10 dB, defined using active target intervals with a documented floor, not whole-clip silence. Report nominal and actual local ratios. Include simultaneous onset overlap, partial overlap, tails and dense friction; random mixing alone under-samples adversarial synchrony. Do not time-stretch test targets to line up with nuisance. Align nuisances instead and preserve target sample identity.

Training includes both independent placement and rhythm-correlated placement. Reserve songs, target takes, nuisance recordings, acoustic paths and exemplars by source identity **before** segmenting or remixing. Deduplicate near-identical media and derived assets across splits. No train clip may share its underlying target waveform or nuisance excerpt with a test clip, even at a new gain or offset.

Maintain one “perfectly aligned reference” constructed condition to isolate separation, and one operational condition using RhythmAlign's estimated map to include alignment errors. Include within-tolerance errors and large wrong-reference controls. Do not conflate an oracle alignment with a deployable aligner.

## 5. Dataset manifest and rights

For every asset retain: recording/session/source IDs; split; native sample rate/channels/codec; timestamps and clock maps; gain/DSP settings; device and geometry; player/cabinet IDs; target policy; pristine-track identity/hash; component hashes; mixture recipe; truth tier; event/sensor annotations and uncertainty; clipping/noise/leakage QC; source license/consent and permitted redistribution; processing version.

Store float PCM stems for evaluation and untouched originals. Suggested layout is `raw/`, `calibration/`, `source_stems/`, `mixtures/`, `annotations/`, `manifests/`, `splits/`, and `sealed_test/`. This is a future dataset layout, not a claim those data exist now. Music and private event recordings may prevent public redistribution; release recipes, permissible stems, checksums and evaluation scripts where raw media cannot be shared. Real-world participant capture requires appropriate consent and controlled access.

Two annotators independently label a stratified subset of events from separate evidence streams before resolving disagreements. Labels include audible contact, physical contact with uncertain audibility, nuisance impact, friction interval, source identity and unknown. Keep disagreement instead of forcing all labels to binary truth. Audio-visible event agreement is evidence; it is not proof of source isolation.

## 6. Self/weak supervision: objectives worth testing after S1

| Signal | Proposed objective | What it learns / safeguard |
|---|---|---|
| Target-free playback + pristine reference | `L_ref = norm(m_hat(y,r)-y)` on independently verified no-target blocks; path smoothness | Acoustic nuisance transfer. Exclude blocks using sensors/manual QC, not the current separator's belief. |
| Same real target, different independent nuisance | `L_pair = norm(F(s+n1,C1)-F(s+n2,C2))` plus supervised seed fidelity | Nuisance invariance. Pair consistency alone admits zero or a memorized target. |
| Target-only and absent-target blocks | Identity reconstruction `F(s,C)≈s`; energy penalty `F(n,C)≈0` | Scale/timbre and absence behavior. Retain other-instance same-class negatives. |
| Cross-modal event evidence | Soft interval presence BCE; audio-video correspondence/contrastive loss | Timing/identity representation. Video labels neither force waveform zero outside contact nor specify contact timbre. |
| Isolated high-confidence events | Quality-weighted waveform examples and different-take enrollment | Seed expansion. “High confidence” requires independent sensor/quiet evidence; loud mixture peaks are not clean stems. |
| Unlabeled mixture-of-mixtures | MixIT grouping/reconstruction | Universal decomposition. Needs a query/seed anchor to identify desired grouping; can split one source or merge target+nuisance. |
| Teacher remix / pseudo-labels | Cross-fitted teacher estimates remixed with independent residual; student consistency | Domain adaptation. Keep teacher uncertainty and real held-out audit; correlated teacher errors do not cancel automatically. |
| Masked audio / contrastive pretraining | Frozen or lightly adapted representations with supervised separation head | Useful features, not waveform labels. Semantic success cannot substitute for source fidelity. |
| Personal enrollment / test-time calibration | Fixed small adaptation budget, support reconstruction and seed replay | Capture/identity adaptation. Never update on sealed test targets or use residual-energy minimization alone. |

Primary precedents: [MixIT](https://arxiv.org/abs/2006.12701), [RemixIT](https://arxiv.org/abs/2202.08862), [SURF](https://arxiv.org/abs/2606.04921) and [AudioScope](https://audioscope.github.io/). These establish training mechanisms, not guarantees that pseudo-targets are authentic.

Repeated gameplay or repeated songs are **not repeated identical impact waveforms**. A model trained to predict expected taps from the reference could score well on timing while inventing missed contacts. Multiple takes help with identities and distributions; averaging them cannot recover the factual waveform of one obscured contact. Hold out deliberate missing hits and novel offbeat hits, and require their outcomes to follow the audio evidence. Weak supervision is a later ablation, not part of the first paired fine-tuning comparison.

## 7. Objective evaluation: preserve two axes

All primary waveform measures use the original fixed gain and common 32-kHz bandwidth for S1. Report native-rate/high-band results separately. Never normalize each output independently or choose a per-output alignment that conceals transient shifts. Register any common fixed latency correction on development data.

### Target preservation

For exact target `s` and estimate `s_hat`, with fixed epsilon/floor:

- **Normalized scale-dependent error:** `NMSE = ||s_hat-s||² / (||s||²+epsilon)`; report fixed-gain reconstruction SNR as `-10 log10(NMSE)` without projection/rescaling. This contains both distortion and leakage and must not be relabeled pure preservation. Use this explicit formula rather than an unspecified SDR implementation with different scaling/filter allowances.
- **SI-SDR:** project onto `s`, then report target-to-error ratio with the explicit standard formula. Its scale invariance can reward an over-attenuated output; always pair it with gain and event outcomes. [Le Roux et al.](https://arxiv.org/abs/1811.02508)
- **Target projection gain:** `a = <s_hat,s>/<s,s>` for isolated/target-only cases; report signed gain and `20log10(|a|+epsilon)`. In mixed outputs use the joint diagnostic below, where its conditioning is valid.
- **Spectral/timbre error:** multi-resolution STFT convergence and log-magnitude distance at fixed gain; add high-band energy ratios, transient rise/decay errors and modulation/envelope distance. Report metrics over annotated friction/tails, not just onset neighborhoods.
- **Event completeness:** optimal one-to-one matching to independently annotated target events, with 20-ms and 50-ms tolerances plus per-event energy/timbre checks. A coincident nuisance onset is not automatically a recovered target. Report physical contacts of uncertain audibility separately.
- **Energy retention:** target-only integrated energy ratio and event-local target projection; total output energy in mixtures is not retained-target energy. Include weak events and continuous contact coverage.

Keep SI-SDR and target-normalized reconstruction SNR undefined for target-absent truth; use absence measures below. Report failures, not favorable infinities caused by epsilon choices.

### Nuisance suppression and distortion attribution

With exact independent sources form `X=[s,m,u1,...]` over an appropriate clip/interval, solve the diagnostic least-squares projection `b=X^+ s_hat`, and let `q=s_hat-Xb`. Report target coefficient, combined nuisance projection energy and unexplained error energy separately. Fix the allowed projection to scalar source coefficients; long adaptive filters can forgive damaging distortion. Record rank/condition number and exclude ill-conditioned projections from component claims while retaining direct waveform errors. A preregistered normalized-Gram condition-number threshold of `1e4` is a QC flag, not a universal physical law.

This projection is a **diagnostic**, not a causal decomposition of a nonlinear network. Correlated sources and errors aligned with a source can bias attribution; coefficient subtraction does not manufacture exact per-source network outputs. Use independently varied nuisances and same-target paired mixtures to check whether output changes with nuisance. Processing `F(n,C)` alone tests absent-target behavior, but generally `F(s+n,C) != F(s,C)+F(n,C)`.

For an accepted projection diagnostic, define combined nuisance attenuation as `10log10((norm(m+sum(u_j))²+epsilon)/(norm(b_m*m+sum(b_j*u_j))²+epsilon))`, using the same scored interval and input gain. Report attenuation for each nuisance family; nuisance-only output/input energy ratio; residual source leakage; band-specific suppression; and listening-based nuisance annoyance. SI-SDR improvement alone cannot establish preservation and suppression separately. Perfect cancellation and nearly zero projected nuisance require floor/censoring counts rather than silently infinite scores.

### Authenticity, consistency and selective risk

Distinguish three failures: **invented event** (no corresponding recorded source event), **source misassignment** (a real nuisance event is presented as the target), and **acoustic distortion** (a real target's waveform/timbre is altered). All violate some aspect of the target contract, but only the first is event fabrication. Unknown faint-event audibility must remain unknown. Same-class cases with no available identity cue form an ambiguity/abstention stratum; they do not establish a physically identifiable extraction problem merely because the dataset has hidden labels.

- **Target-absent false-output energy:** `10log10((||F(n,C)||²+epsilon)/(||n||²+epsilon))`, plus unsupported target-event count/minute and human verification. Quiet numerical silence uses an absolute floor referenced to full scale; relative ratios with nearly zero inputs are unstable.
- **Existence errors:** event precision/recall and presence calibration on independent labels; score missing-hit and wrong-identity clips. A presence head can be right while waveform content is wrong.
- **Mixture residual error:** `||y-s_hat-r_hat||/||y||`, and STFT consistency if applicable. If `r_hat=y-s_hat`, report it as constructional bookkeeping, not empirical evidence of fidelity. Check source-head consistency **before** projection for models that estimate multiple heads. [Consistency constraints](https://research.google/pubs/differentiable-consistency-constraints-for-improved-deep-speech-enhancement/)
- **Selective risk:** fix a development-set acceptance threshold, then report waveform/event/timbre failure rate against accepted duration and target-event coverage. Report rejected targets and all output intervals; absence and inability-to-recover are different labels.
- **Counterfactual evidence tests:** hold context fixed while removing/inserting a target in exact mixtures; hold target fixed while changing nuisance or reference. Expected output follows actual target changes, not music/chart expectations. These tests provide behavioral provenance evidence, not proof of a unique inverse.

S1's small sample cannot certify very low hallucination rates. Zero observed errors still needs a confidence interval; clips sharing a session are not independent trials. At paper scale, define a specific failure event, report cluster-aware uncertainty and evaluate shifted domains separately.

## 8. Blind listening design

Use both T2 exact mixtures with a valid hidden target anchor and T4 real recordings with **no purported perfect anchor**. Ask listeners to score target completeness, apparent source/timbre fidelity, nuisance annoyance, artifacts and overall preference separately. On T2, listeners can compare to the true recorded target; on T4, call fidelity a perceptual judgment corroborated by sensor/video evidence, not waveform verification.

Randomize method labels/order and use a balanced subset when there are too many methods. Keep decoding, headphones and playback gains identical; choose a gain from the input/reference policy before seeing outputs. Include raw, a deliberately over-suppressed anchor, and hidden duplicate trials. Rate stems first. Evaluate remixes separately with identical pristine-music alignment and gain across methods; clean music must not mask poor stem preservation in the primary test.

For S1, recruit 12 listeners, preferably half familiar with rhythm-game interaction sounds and half general audio listeners. Use 24 held-out excerpts (12 constructed, 12 real; balanced sparse/dense/overlap/weak/rest) and 12 paired comparisons per listener, approximately 15–25 minutes including separate attribute ratings. A balanced incomplete design covers the finalist versus raw, strongest corrected/frozen baseline, and paired adapted reference/no-reference comparison. Assign 144 judgments across 72 excerpt-comparison cells, two judgments per cell; this is a **pilot with limited precision**, not a powered perceptual superiority trial.

Freeze the candidate and operating point using development data before generating anonymous listening files. An independent script generates the key; the researcher should not consult it during annotation. Analyze paired preference and attribute differences using listener and session effects or a cluster bootstrap; show wide intervals and per-session results. No listener, clip or confusing failure is dropped after unblinding except through preregistered QC, with counts disclosed.

## 9. Required controls and why they matter

| Control | What it can falsify |
|---|---|
| Rest / target-absent music / target-free similar impacts | Hallucinated target events, impact-class leakage, confusion between expected and audible action |
| Weak isolated and overlapped target | Preservation hidden by average SNR or loud events |
| Dense contacts, continuous slide, decays | Onset-only success concealing lost target duration/timbre |
| Correct vs shifted and coverage-matched gates | Timing benefit vs mere exposure reduction |
| Correct vs wrong pristine track; shuffled timing | Reference identity/path utility vs rhythm prediction or generic capacity |
| Correct vs wrong query/enrollment | Target selectivity vs universal impact enhancement |
| Reference/no-reference with equal trainable capacity | Conditional information benefit rather than extra parameters/data |
| No-context and later vision/no-vision | Incremental context benefit and missing-modality robustness |
| Missing expected hit / added offbeat hit | Prior-driven event completion or omission |
| Other-instance same-class source overlap | Instance binding rather than class extraction |
| Target-only input and nonmatching reference | Destructive nuisance branch behavior |
| Common-rate and native-rate reporting | Bandwidth/codec confounds, particularly high-frequency friction |

The corrected temporal gate remains a useful baseline, with exactly recorded active coverage and gain transitions. Do not compare only full-coverage raw to heavily gated output and attribute all attenuation to semantic separation.

## 10. Broader domains: collect later, define now

| Domain | Paired/proxy acquisition | Exact-control construction | Critical target policy / hard overlap |
|---|---|---|---|
| Wedding | Consented couple/MC lavaliers, isolated mixer channels, audience-position recorder/video | Room-source speech takes plus separate guests/music, including simultaneous same-gender voices | Which identities and nonverbal sounds count; target voice also reproduced through PA |
| Concert | Soundcheck multitracks, soundboard and audience-position microphones | Same-room isolated performer and crowd playback/source images | Chosen performer vs ensemble; studio track differs from factual performance; crowd unison singing |
| Interview/speech | Separate close microphones and far-field devices, aligned video/enrollment | Exact speech/noise mixtures with known room paths | Breath/laughter and interruptions; similar simultaneous speakers, wind/clipping |
| Sport | Controlled drills, object/contact sensors and distant camera microphones | Real ball/foot/equipment sources plus crowd/PA | Selected athlete/action vs another simultaneous similar impact |
| Live events/vlogs/industry/first-person | Isolated available feeds, local sensors and selected-person/object tracking | Staged representative sources in real paths | Moving geometry, intermittent offscreen targets, target relation changing over time |

Collect the arcade pilot now. Collect broader datasets only after S1 establishes a preservation gain and S2 stabilizes truth quality. At S5, prioritize one speech-identity domain and one non-speech object/action domain to test whether the common abstraction survives. A soundboard/lavalier channel is a useful proxy, not automatically the room-microphone target; the same acoustic-image definition must travel across domains.

## 11. Scale and decision gates

Pilot: ten sessions, about 80 minutes scored multichannel/video capture; roughly 30–100 GB working storage. Expansion: 5–30 hours of distinct target material and 50–300 hours of nuisance, across people/devices/venues, with 0.2–1 TB working storage. Cross-domain campaigns may require 50–300 hours of curated paired/proxy material and 1–5 TB. These are acquisition/planning ranges, not claims that a particular amount guarantees generalization.

Do not expand pseudo-label volume until the quiet-target pipeline passes alignment, leakage, rest and identity checks. If only close/contact proxies can be collected, continue perceptual/event research but explicitly narrow waveform-recovery claims. Better truth is more valuable than another architecture sweep on the same ambiguous excerpts.
