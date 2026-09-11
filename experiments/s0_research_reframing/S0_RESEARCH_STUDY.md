# RhythmAlign S0 research reframing and architecture study

**Pursue context-conditioned target sound extraction with an explicit fidelity contract.** The ambition is technically credible as a system that succeeds on supported cases, exposes uncertainty, and declines unsupported recovery. Universal exact restoration from arbitrary single-channel recordings is not a credible promise. The next investment should be a small target-recording pilot and supervised adaptation of a pretrained separator, with an aligned-reference ablation.

The recommended scientific name is **context-conditioned target sound extraction**, further qualified as **source-image-preserving extraction with selective prediction** when discussing the proposed evaluation. “Authentic” is a product value to operationalize, not a claim that the algorithm can prove every estimated sample originated from one source. The program is general in its target/context interface; its first validated operating domain should be physical interactions under known playback and unknown interference.

This is an S0 study, not an implemented extractor or a new listening experiment. Evidence was assessed through 10 September 2026. Architecture rankings, resource ranges and success thresholds are research judgments or proposed protocol choices, not measured S1 performance.

## Study artifacts and decision scope

| Artifact | Purpose |
|---|---|
| [Literature map](D:/Code/RhythmAlign/experiments/s0_research_reframing/LITERATURE_MAP.md) | Primary sources, neighboring families, release status and ten priority readings |
| [Architecture options](D:/Code/RhythmAlign/experiments/s0_research_reframing/ARCHITECTURE_OPTIONS.md) | Four implementable routes, conditioning, losses, failure modes and ranking |
| [Data and ground truth plan](D:/Code/RhythmAlign/experiments/s0_research_reframing/DATA_AND_GROUND_TRUTH_PLAN.md) | Target definitions, recording protocols, supervision and evaluation |
| [Minimal decisive experiment](D:/Code/RhythmAlign/experiments/s0_research_reframing/MINIMAL_DECISIVE_EXPERIMENT.md) | One preregisterable experiment, exact arms, thresholds and stop rules |
| [Paper direction memo](D:/Code/RhythmAlign/experiments/s0_research_reframing/PAPER_DIRECTION_MEMO.md) | Competing theses and evidence required before novelty claims |
| [Decision JSON](D:/Code/RhythmAlign/experiments/s0_research_reframing/S0_DECISION.json) | Compact machine-readable next-step decision |
| [Workspace evidence checks](D:/Code/RhythmAlign/experiments/s0_research_reframing/WORKSPACE_EVIDENCE_CHECKS.json) | Independent spot checks and hashes of inspected local evidence |

## Part 1 — Formal problem and output convention

Let physical source signals be \(x_j\), target membership be \(a_j(C)\in\{0,1\}\), and the room/microphone source images be

\[
u_j(t)=\int h_j(t,\tau)x_j(t-\tau)d\tau,\qquad z(t)=\sum_j u_j(t).
\]

A single delivered phone track is more realistically

\[
y[n]=Q_{\kappa}\{D_{\psi}[z;\xi]\}[n]+\eta[n].
\]

Here \(h_j\) may vary with motion; \(D_\psi\) includes filtering, dynamic gain, clipping, denoising and other device processing with internal state \(\xi\); \(Q_\kappa\) includes sampling, quantization and codec effects. Even if acoustic pressures add before the microphone, the delivered samples need not admit a unique physical additive decomposition after joint nonlinear processing.

Partition sources into target \(T\), known/partially known nuisance \(K\), and unknown nuisance \(U\). Available context is

\[
C=(q_{text},V,r_{1:L}^{aligned},e_{target},I_{target},\mathcal T,\mu,\rho_C),
\]

where \(\rho_C\) records confidence, timing uncertainty, availability and provenance of conditions. A **positive exemplar** says what to keep; an **aligned negative reference** explains a particular nuisance realization. They are not interchangeable. A transcript or visual contact can indicate identity/existence without containing the event's waveform.

The preferred estimand is the **target acoustic source image at the recording microphone**, retaining target room response, timing and incidental timbral variation. It is not the dry source at the object and not a contact microphone's vibration signal. Under a linear or known frozen-gain capture approximation \(A_{\xi_y}\), define

\[
s_T=A_{\xi_y}\left(\sum_{j\in T}u_j\right),\qquad
y=s_T+n_K+n_U+d.
\]

The capture gain/state is frozen to the factual recording; \(d\) contains approximation and joint-distortion errors. This avoids letting a quieter imagined room change the phone's AGC and thus redefine target loudness. When nonlinear coupling is substantial, this source-image convention is a **model-based target**, not a uniquely observable additive stem. Another valid estimand is a target-only counterfactual \(D[\sum_Tu_j]\); it has different device state and must be evaluated as a different task. S0 does not silently equate them.

The extractor returns

\[
F_\theta(y,C)=(\hat s_T,\hat n_K,\hat r,p_{exist},p_{recover},u,\mathcal A),
\]

where \(\hat r\) includes unknown nuisance and unallocated distortion, \(u\) is uncertainty, and \(\mathcal A\) contains accepted/uncertain/abstained intervals. Existence and recoverability are distinct: a visible tap may certainly have happened while its acoustic detail remains unresolvable.

| Category | Meaning here |
|---|---|
| Source separation | Estimate components of a mixture, possibly all sources, with permutation/grouping ambiguity. |
| Denoising | Remove content treated as noise relative to a predetermined signal class; often ignores which person/object is intended. |
| Enhancement | Improve perceived quality or task utility; may change reverberation, bandwidth, timbre or content. |
| Target sound extraction | Estimate only sound selected by a query, including unions of sources. Best established task category. |
| Source reconstruction | Estimate missing/latent signal detail using a source/acoustic model; can occur inside separation or restoration. Its fidelity depends on evidence. |
| Conditional source extraction | The broader computational formulation covering text, reference, exemplar, identity and multimodal TSE. |

Wedding “bride/groom/MC” is a time-varying union of identities. Arcade “player–machine physical interaction” is a **relation**, not every sound of the visible cabinet. Concert “intended performer” may include their PA reproduction and room response; that choice must be explicit. The broad interface has prior art in [SoundBeam](https://arxiv.org/abs/2204.03895), [OmniSep](https://arxiv.org/abs/2410.21269) and [SAM Audio](https://arxiv.org/abs/2512.18099).

## Part 2 — Identifiability and the role of AI

For the ideal additive mono observation \(y=s+n\), every pair \((s+\delta,n-\delta)\) yields the same observation unless additional constraints exclude \(\delta\). This is an underdetermined inverse problem, but not a claim that useful extraction is impossible. Long-term spectral structure, different source statistics, repeated identity and observed context can narrow the admissible set considerably.

Define \(\mathcal F(y,C)\) as physical/source-model configurations compatible with the observed recording, context uncertainty and a specified measurement tolerance. Exact identification requires its target projection to contain only one signal. Operational identification at tolerance \(\epsilon\) requires the diameter of that projection under a specified target-fidelity metric to be at most \(\epsilon\). A posterior estimate instead depends on the learned prior as well as the likelihood. A sharp but misspecified posterior can still be wrong.

| Level | What it means | Permissible claim |
|---|---|---|
| A. Mathematically unrecoverable exact waveform | Multiple materially different target waveforms fit all measurements/assumptions. | Cannot uniquely recover the exact waveform under those assumptions. |
| B. Statistically inferable signal | Training priors and scene structure favor some solutions. | An estimate with measured error/generalization, not recovered missing information. |
| C. Perceptually plausible reconstruction | Output resembles what such a source could sound like. | Restoration/synthesis quality; insufficient for authentic extraction. |
| D. Evidence-supported extraction | Event identity, timing and measured detail are retained within validated error bounds, with unsupported cases rejected or disclosed. | Selective extraction under a tested fidelity contract. |

A quiet tap under an NPC announcement need not disappear from the sampled waveform: it may remain encoded as a small perturbation that a good model can exploit. Auditory masking is not the same as digital deletion. But quantization, bandwidth loss, a codec discard or clipping can make different tap waveforms map to the same delivered samples. Perfect cancellation can also erase components. Visual certainty that a hand contacted glass cannot restore the lost phase, fine impact spectrum or friction realization.

For a simple known-source model \(y=Ax\), identifiability depends on the nullspace of \(A\) restricted to the source model, not just the number of scalar samples. A sparsity or trained manifold prior can make recovery possible on a restricted class, yet failure is expected where sources are indistinguishable or the class is wrong. Information processing can exploit available information; it cannot create measurement-specific information absent from \((y,C)\). Additional compute improves inference, not sensor information.

| Extra information | What changes | What remains unresolved |
|---|---|---|
| Exact pristine nuisance waveform | Reduces uncertainty about one source; with known transfer, its recorded source image can be subtracted exactly in the additive model. Inverting that transfer is unnecessary when the source reference is already known. | Transfer/device uncertainty and all unrelated nuisance; correlated target/reference. |
| Video | Target identity, causal action cues, temporal support, possibly spatial grouping. | Hidden contact, silent action, A/V offset and subframe waveform detail. |
| Clean target exemplar | Better timbre/identity prior; rejects different classes/instances. | Does not measure this take's exact event; exemplar contamination and same-class ambiguity. |
| Multiple microphones | More independent equations and spatial cues; rank can increase. | Co-located sources, insufficient aperture, synchronization, reverberation and channel rank. |
| Source/acoustic model | Constrains feasible waveforms and paths. | Misspecification; a likely waveform is not automatically the historical one. |
| Training distribution | Learns regularities that exploit weak evidence. | OOD failures and learned bias; repeated songs can encourage missing-hit completion. |
| Spatial information | Separates different directions when measured with a suitable array. | A mono source cannot acquire reliable missing interchannel phase from a direction label alone. |

The credible frontier is **recover more of the real target at a controlled distortion and unsupported-event rate**, particularly in partial overlap. It is not to promise every microscopic acoustic detail after arbitrary clipping or complete source ambiguity.

## Part 3 — Operational authenticity

Use a layered contract rather than a binary “generative/non-generative” label.

1. **Signal bookkeeping:** retain input sample clock, gain convention and output/residual files; measure reconstruction error before and after any projection. Enforce a mixture sum in the chosen additive observation domain.
2. **Event and identity fidelity:** measure missing target events, false target events, confusion with similar nuisance instances, timing, decay and continuous-contact preservation against independent evidence.
3. **Acoustic fidelity:** constrain target gain, phase/timbre error and variation across real takes; assess scale-dependent error as well as perceptual ratings.
4. **Selective reliability:** calibrate event existence and recoverability separately on held-out data; report accepted coverage and error among accepted outputs, including OOD stress tests.
5. **Traceability:** log input/context/checkpoint/configuration hashes, timing maps, source conventions and abstention intervals. This proves the processing lineage, not physical source identity by itself.

Strict sample gating \(g(t)y(t)\) preserves samples at gain one, but retains every coincident nuisance and can introduce modulation artifacts. A real STFT mask \(0\le M\le1\) restricts output magnitude and uses mixture phase; it can delete faint targets, retain the wrong source and reshape nuisance into apparent transients. It cannot represent every true source where destructive interference makes \(|S|>|Y|\).

Complex masks, learned filters and direct RI/waveform regression are admissible. Their extra freedom requires measurement, not prohibition. A fixed deterministic model can invent a stereotyped impact; a generative posterior model can accurately estimate an identifiable source. The distinction is empirical faithfulness and uncertainty management.

Mixture consistency is useful but insufficient. For arbitrary fabricated \(h\), \(\hat s=h\) and \(\hat r=y-h\) have **zero mixture error**. Joint target/residual supervision and negative examples discourage this solution; an algebraic projection alone does not. Differentiable consistency layers are established prior art. [Wisdom et al., 2019](https://research.google/pubs/differentiable-consistency-constraints-for-improved-deep-speech-enhancement/)

Energy is not generally conserved as a sum of source energies:

\[
\|s+n\|^2=\|s\|^2+\|n\|^2+2\langle s,n\rangle.
\]

Thus neither \(\|\hat s\|\le\|y\|\) nor \(\sum_k\|\hat s_k\|^2=\|y\|^2\) is a universal law for overlapping sources. An energy ceiling may be a conservative engineering policy, but must be tested for target loss and identified as such.

**Product definition:** “An estimate of the requested sound from this recording, evaluated for event completeness and timbre preservation. Uncertain intervals are identified; no replacement sound library is used.” Do not say “every original sample” or “verified authentic stem.” If abstaining, offer the raw interval as an explicitly unprocessed fallback, or leave it unexported; silence must not masquerade as verified absence.

**Paper definition:** For a preregistered distribution, tolerances and accepted set, report a vector of target distortion, event false discovery/miss rates, nuisance leakage, calibration and coverage. “Evidence-supported” is a tested operational property on that set, not a universal certificate. Generative refinement is eligible only if it improves that vector or its Pareto frontier on held-out data; semantic similarity or listener plausibility alone cannot authorize it.

## Part 4 — Prior art and the research gap

The [literature map](D:/Code/RhythmAlign/experiments/s0_research_reframing/LITERATURE_MAP.md) covers query extraction, multimodal foundations, reference cancellation, discriminative/complex/waveform backbones, spatial processing, weak supervision and evaluation.

**The broad task already exists.** SAM Audio is especially close in scope and explicitly discusses fidelity to recording attributes. OmniSep and USE further weaken a generic “one model, many conditions” novelty claim. Personalized AEC already combines known playback with target-speaker information. Even clean-music cancellation followed by remixed real steps/voice appears in a [2011 dance-teaching system](https://perso.telecom-paristech.fr/grichard/Publications/2011-final_ACM_MM.pdf).

The defensible unresolved question is narrower and demanding: **How much can an aligned nuisance realization improve selective preservation of faint, real, instance-specific target events under severe overlap and uncertain capture processing?** A credible answer needs synchronized target evidence, absent/missing-event controls and preservation–suppression curves. It may yield a general method; a successful benchmark/data study is also possible. Novelty is not established at S0.

## Part 5 — Historical evidence that survives

The [independent review](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/INDEPENDENT_REVIEW.md) and [R5-FIX report](D:/Code/RhythmAlign/experiments/r5_fix_validity_repair/REPORT.md) are the primary local synthesis. S0 additionally inspected cancellation code, local separator code/configuration, corrected construction and metric artifacts, and recomputed selected audio/gain quantities. The [receipt](D:/Code/RhythmAlign/experiments/s0_research_reframing/WORKSPACE_EVIDENCE_CHECKS.json) distinguishes recomputation from saved evaluator values. There was no independent listening in S0.

| Historical evidence | Surviving inference | Rejected inference |
|---|---|---|
| R1 total stereo energy reduction: A1 0.0535, A2 0.4735, A3 0.2933 dB, independently recomputed. | These tested fits were weak on the selected recording. Alignment code and reference assets remain valuable. | Total-mixture energy is not source-specific music attenuation; low scalar coherence is not a universal cancellation ceiling or proof of phone nonlinearity. |
| AudioSep/CLAPSep probes on one golden window; contaminated exemplars, text-trained AudioSep queried with audio. | Tested configurations did not demonstrate the desired isolation; R3 proxies did show some selectivity. | No class-wide learned-separation failure, no 8-GB ecosystem ceiling, no rejection of supervised adaptation. |
| R5-FIX corrects reversed ramps and wrong HPSS orientation. | Corrected C1 is a valid exposure baseline; implementation defects materially contaminated earlier comparisons. | Previous naturalness/product-readiness claims, source-specific Taiko suppression and old onset counts as authentic target counts. |
| Saved corrected golden metrics: 64/84 onset matches; unmatched outputs 24→3 after fade repair; shifted gate 50/84. | Timing has information beyond coverage; boundary construction affects detector counts. | These are not independently measured true-source recall or proof every removed onset was an invented physical event. |
| Recomputed E0 rest exposure: 1.4247/2 seconds, −1.8075 dB energy relative to raw; corrected oracle rest is digital zero. | Detector false exposure survives envelope repair. Correct support passes the rest construction check. | A smooth gate solves source identity or suppresses nuisance during contact. |
| Same-recording temporal holdouts and audio-assisted annotations. | Useful stress cases and weak labels. | Independent recording generalization or complete physical contact ground truth. |

The inspected R1 code tested scalar/stereo spectral fits, including a per-frequency ridge fit with a 256-ms STFT window; it did not exhaust learned adaptive or nonlinear transfer models. The identified music-only pool was only about 1.18 seconds, and the drift sweep selected its −100-ppm boundary. Short contaminated calibration, alignment/path uncertainty and limited model class all weaken a universal negative conclusion. Phone processing remains a plausible contributor, not a measured explanation that excludes other causes.

Correct timing helps because target energy is unevenly distributed. Gating is structurally limited because the same scalar gain acts on target and nuisance at each time; dense contact drives near-continuous exposure. Better vision can improve identification or condition spectrotemporal separation even when a gate saturates. Conversely, absent visual evidence is not evidence of acoustic absence.

| Disposition | Components |
|---|---|
| **KEEP** | Alignment/drift infrastructure; original media and pristine references; corrected C1 implementation; explicit gain/sample-clock conventions; corrected matching and baseline artifacts. |
| **REPURPOSE** | Visual events as uncertain auxiliary labels; E0 as a frozen legacy control; reference residual as one feature; original windows as development/stress examples; listening pack as a template. |
| **ARCHIVE** | R1 cancellation tuning, R2/R3 outputs, R4–R5.6 heuristic experiments and frozen configs as exploratory evidence. Preserve files unchanged. |
| **DISCARD from active claims/design** | Universal negative conclusions, alleged heuristic ceiling, hard visual/no-transient veto, Taiko-specific proxy interpretation, old faulty metrics and production readiness. This does not mean deleting history. |

## Part 6 — The pristine reference as a first-class condition

The reference's strongest value is its **realization-level temporal structure**. A global “music” embedding identifies a category; an aligned waveform can constrain which melody, drum transient and envelope the cabinet could have emitted at each instant. Therefore pooled negative-query CLAPSep is not an adequate test of the aligned-reference hypothesis.

Three levels should be distinguished. A transfer estimator predicts how pristine playback maps to the microphone. A latent reference matcher identifies explainable spectral/temporal structure despite phase mismatch. A joint extractor uses both nuisance explanation and target evidence to estimate the target directly. Only the first necessarily subtracts a modeled waveform.

For linear transfer, \(\hat m=\hat h*r\) is explicit and interpretable. For loudspeaker nonlinearity before propagation, a limited Hammerstein-style model \(\hat m=\sum_p h_p*\phi_p(r)\) can explain harmonics. Phone compression **after mixing** is different: it couples target and nuisance, so no reference-only transfer function can fully model it. Learning an unconstrained \(\hat m(y,r)\) can also absorb correlated target hits. Restrict capacity, supervise nuisance stems where known and validate on missing/extra contacts.

A practical temporal adapter uses mixture queries, aligned-reference keys/values, relative-time bias, and local attention over alignment uncertainty. Signal encoders retain complex or high-resolution spectral information; semantic embeddings alone are insufficient for phase-coherent removal. Train with reference dropout, wrong references, drift/path changes and explicit target-free playback. Model mismatch should reduce reliance on the reference, not force removal.

**Judgment:** this is a major *experimental advantage* and a plausible model advantage, especially when cabinet music dominates. It is not a solution to neighboring speech or unrelated similar impacts, and the benefit could be small if little recorded nuisance is explainable by the reference. Its value must be the causal difference between equal-capacity, equally trained correct-reference and no/wrong-reference arms. [Personalized AEC](https://arxiv.org/abs/2205.15195), [NKF](https://github.com/fjiang9/NKF-AEC) and [Meta-AF](https://arxiv.org/abs/2204.11942) establish relevant mechanisms, not arcade performance.

## Part 7 — Vision reconsidered

| Role | Expected value | Main risk / decision |
|---|---|---|
| A. No vision | Simplest waveform feasibility test; applies to audio-only archives. | Same-instance impacts may remain ambiguous. **S1 default.** |
| B. Presence probability | Soft cue for likely contacts, overlapping speaker activity and confidence. | Silent motion, occlusion and timing errors; never multiply audio by unvalidated binary contacts. |
| C. Target embedding | Bind sound to selected player/object/person. | Appearance alone may not distinguish acoustically identical sources. |
| D. Condition separator | Cross-attention/fusion supplies identity/action context while audio estimates waveform. | Needs real AV training and missing-modality robustness. **Preferred S5 vision route.** |
| E. Resolve similar impacts | Spatial/action correspondence may separate player's tap from nearby impact. | Perfectly synchronous similar sources can remain ambiguous; global cabinet ROI includes music. |
| F. Identity/tracking | Useful for designated people/performers and target sets over time. | Tracking switch, occlusion, offscreen speech and PA reproduction. |
| G. Training-only weak labels | Lower inference cost, audio-only deployment; frame/event supervision and contrastive learning. | Visual labels reveal contact, not necessarily an audible waveform; teacher leakage. |

Use hand/object motion and identity if they add information; the old judgment-text detector has no inherited architectural privilege. A camera can miss continuous friction and acoustically relevant offscreen action. Physical contact annotations should include uncertainty intervals. Weddings may work well with enrollment plus speaker identity and no vision; vision is an optional disambiguator, not a universal prerequisite. [AudioScope](https://audioscope.github.io/) provides an established weakly supervised AV precedent.

## Part 8 — Serious architecture candidates

Four complete designs and an explicit ranking appear in [Architecture options](D:/Code/RhythmAlign/experiments/s0_research_reframing/ARCHITECTURE_OPTIONS.md).

| Route | Technical choice | S0 judgment |
|---|---|---|
| **A — Reference-conditioned pretrained discriminative extractor** | Adapt local CLAPSep decoder with temporal reference features, target/residual bookkeeping and independent presence/recoverability heads. Move to complex mapping only if the representation is limiting. | **Best next route:** lowest-cost direct test of domain supervision and reference benefit. |
| B — Multimodal foundation separator with selective acceptance | Evaluate SAM Audio; later adapt multimodal latent separation, with reconstruction/fidelity constraints and calibrated acceptance. | Strong general baseline; generative fidelity and access/compute need validation. |
| C — Learned adaptive cancellation plus target extractor | NKF-style or constrained nonlinear transfer estimator; raw/residual/reference inputs feed a target separator. | Interpretable alternative; irreversible first-stage damage must be avoided through raw bypass. |
| D — Frozen foundation encoder plus compact personalized separator | M2D/CLAP query features and high-resolution mixture path; conditional TCN/GridNet decoder, optional few-shot adaptation. | Good contingency for data/compute; representation transfer and unseen-instance fidelity uncertain. |

## Part 9 — Do we need a new model?

**B. Fine-tuning an existing architecture is the best next step.** This is a decision about the next investment, not a claim that unchanged pretrained models cannot suffice.

First, strong existing systems might already achieve acceptable quality in a subset of conditions; include one modern broad baseline. Second, clean domain supervision may resolve much of the apparent difficulty seen in R2/R3. Third, a small temporal nuisance adapter is a justified hypothesis test, but adding cross-attention does not automatically justify a new-method paper.

A genuinely new conditioning architecture/objective becomes justified only if equal-budget adapted baselines fail and a diagnosed issue remains: phase representation, preserving near-synchronous correlated targets, unreliable evidence allocation, or missing-modality calibration. If an existing system passes the fidelity/coverage gate, prefer it and focus the contribution on evaluation/data/product evidence. If the no-reference tuned model wins, do not force the reference into the deployed route.

## Part 10 — Ground truth priority

**Record actual interactions at the intended phone position during a quiet, muted-playback cabinet session, synchronized with a close air microphone, contact microphone and video.** This is the single highest-value acquisition action. Obtain owner/operator permission and safe nonintrusive sensor placement; do not assume machine audio can be muted without arranging the session.

The phone-position air recording best matches the intended acoustic target. Close/contact tracks provide independent event evidence and leakage diagnostics. They are **proxies**, not perfect phone-domain target stems. Contact microphones omit airborne timbre and may contain cabinet speaker vibration; close microphones admit music, speech and other hits. Calibration and quiet rest recordings quantify these limits.

Build exact digital mixtures from real recorded target takes and separately recorded nuisance/path material. They give exact truth for that constructed mixture. Separately test acoustic re-recordings and simultaneous real arcade recordings, where the paired waveform target is approximate or absent. Never assert \(D(s+n)=D(s)+D(n)\) for a nonlinear phone pipeline. The [data plan](D:/Code/RhythmAlign/experiments/s0_research_reframing/DATA_AND_GROUND_TRUTH_PLAN.md) defines these truth tiers and wider domains.

## Part 11 — Self/weak supervision

Use self-supervision to expand a trusted seed, not to create truth by renaming separator outputs.

Reference-only intervals can supervise transfer modeling. Different nuisance remixes of the same recorded target can supervise invariant extraction. Target-only and absent-target examples teach identity and zero output. Independently confirmed events can provide weak temporal labels without imposing a hard support mask. Clean exemplars from different takes supervise identity rather than memorized waveform matching.

MixIT/RemixIT/SURF-style objectives can adapt to real mixture statistics, but mixture reconstruction leaves source grouping ambiguous. Repeated gameplay is especially dangerous because contacts and music are correlated. Pseudo-labels that miss friction will systematically teach its deletion. Split by recording/session and underlying source assets before mixing; use cross-fitted teachers, label quality flags and independently audited difficult cases. Neither contrastive embeddings nor masked prediction alone trains accurate waveform extraction. Relevant primary methods: [MixIT](https://arxiv.org/abs/2006.12701), [RemixIT](https://arxiv.org/abs/2202.08862), [SURF](https://arxiv.org/abs/2606.04921).

## Part 12 — One minimal decisive experiment

Run **S1: paired supervised adaptation with a randomized aligned-reference ablation**. Use a small multi-session pilot of genuine impacts and friction, exact constructed mixtures and independent real recordings. Compare frozen baselines with equal-budget CLAPSep adaptation without reference and with a small temporal reference adapter. Reserve target-absent, omitted-hit, added-offbeat, wrong-reference and same-class overlap cases before training.

The success question is whether learned extraction moves the preservation–suppression frontier beyond corrected gating and zero-shot baselines on held-out source takes, with corroborating blind listening on real recordings. The reference hypothesis is a separate paired comparison. The [experiment specification](D:/Code/RhythmAlign/experiments/s0_research_reframing/MINIMAL_DECISIVE_EXPERIMENT.md) fixes data counts, budget, success margins, failure interpretation and stop rule. It is designed for the 8-GB laptop; cloud inference of a modern comparator is useful if local access or memory fails.

## Part 13 — Evaluation principle

**Preservation and suppression are separate primary axes.** Report the Pareto frontier and named operating points, then preference; do not average them into one authenticity score.

With exact targets, report SI-SDR alongside fixed-gain normalized waveform error, target gain, spectral/timbre distance, per-event completeness and full friction/decay coverage. With independently known nuisance, estimate leakage while accounting for target distortion. For nonlinear networks, processing nuisance alone does not equal its contribution to the mixed-input output; include counterfactual nuisance perturbations and clearly labeled projection diagnostics.

Perceptual ratings separately ask target completeness, source/timbre fidelity, nuisance annoyance, artifacts and preference. Use hidden target anchors where valid, stems before fixed-gain remixes, randomized ordering and equal input-derived gain. A clean music track shared by both mixes cannot validate stem quality. SI-SDR's gain invariance is why an independent scale-dependent measure is essential. [Le Roux et al.](https://arxiv.org/abs/1811.02508)

Controls include quiet/rest, target-free music, target-free similar impacts, weak target, dense friction, exact overlap, shifted/coverage-matched gates, wrong query, wrong reference, reference/no-reference and later vision/no-vision. Report unknown annotations as unknown, source-level/session-level uncertainty and all abstentions. The [data/evaluation plan](D:/Code/RhythmAlign/experiments/s0_research_reframing/DATA_AND_GROUND_TRUTH_PLAN.md) specifies formulas and listening design.

## Part 14 — Generalization beyond games

| Domain | Target / nuisance | Context | Ground truth strategy | Hardest overlap |
|---|---|---|---|---|
| Wedding | Selected couple/MC, including specified laughter/breath; guests, music, room disturbance. | Enrollments, selected identities, video; optional PA feed. | Lavaliers/isolated mixer channels as proxies; controlled room speech source images. | Similar voices simultaneously speaking, PA duplication, intended person occluded. |
| Concert | Chosen performer or ensemble and explicitly chosen PA/room image; nearby shouting/crowd. | Performer track/video, soundboard feed; studio version only partial reference. | Multitrack/soundboard plus audience-position mics; soundcheck replay validation. | Shout overlapping a vocal note, performer/crowd singing together, clipping. |
| Interview | Selected speakers and desired nonverbal sounds; passersby, traffic, others. | Enrollment, face tracking, turns, optional interviewer channels. | Isolated lavaliers and far-field capture; measured leakage. | Similar simultaneous speakers, wind-induced clipping, offscreen answer. |
| Sport | Selected athlete/action, or commentator if requested; crowd/PA/unrelated impacts. | Object/person tracking, location, microphone metadata. | Instrumented controlled drills and close/contact microphones; real event proxies. | Multiple simultaneous ball/foot/contact sounds with occlusion. |
| Arcade | Player–machine contacts including friction; cabinet/neighbor music, voices, impacts. | Exact aligned music, exemplar, optional video/player identity. | Quiet phone-position targets + synchronized sensors; exact constructed and real mixtures. | Weak slide under music/NPC and acoustically similar adjacent impacts. |

A shared backbone and common context/uncertainty interface are realistic. **Domain-specific target conditioning, adaptation data and fidelity tests remain necessary.** Speech identity, performer grouping and object contact are different inference problems. Claims should progress from same domain/new recording to new device/venue, then to genuinely new target relations and domains. Vlogs, live events, industry and first-person recordings fit the interface, but are not validated merely by naming them.

## Part 15 — Product implications

An offline pipeline can parse a target description, accept a selected person/object/exemplar, align any exact nuisance reference, estimate target and residual, and provide uncertainty-marked previews. “Preserve interaction sounds,” “preserve this person,” and “remove everything except X” should resolve to a saved target specification. Ambiguous identity should be clarified rather than silently guessed.

Offer original/estimate/residual A/B previews and a fixed-gain music remix only after stem inspection. Uncertain segments can use a disclosed original-audio fallback or request a different target cue. Low extraction confidence must not trigger fabricated completion or quietly remove weak events. Preserve the original recording and timing map. Seconds or minutes per clip are acceptable; latency optimization follows demonstrated quality.

## Part 16 — Paper outlook

Several theses remain viable: reference-conditioned fidelity under correlated nuisance; selective multimodal extraction with measured evidence risk; a target-source-image dataset and evaluation protocol; few-shot adaptation to interaction sounds; or a careful application study. The [paper memo](D:/Code/RhythmAlign/experiments/s0_research_reframing/PAPER_DIRECTION_MEMO.md) specifies contributions, novelty risks, experiments and reviewer objections.

The strongest potential case combines **new trustworthy data, a difficult identifiable conditioning distinction, and a validated fidelity improvement**. The present evidence supports a promising application-led seed. It does not yet support a general algorithm claim or a likely top-conference outcome. A corrected gate alone has limited method novelty.

## Part 17 — Staged research program

| Stage | Question | Deliverable | Decision gate | Stop / change condition |
|---|---|---|---|---|
| S0 | Is there a credible problem/route beyond gating? | This study, literature/architecture map, acquisition and S1 specification. | Select one route with measurable falsification. | Stop broad architecture speculation after the route and evidence gaps are clear. |
| S1a | Can target evidence be captured and scored? | Small synchronized pilot, split/label manifest, sanity/oracle bounds. | Phone-position targets have useful SNR; timing and leakage are quantified. | If capture fails, redesign acquisition before training. |
| S1b | Does learned extraction materially improve the frontier? | One bounded adaptation experiment and blind listening. | Meets preregistered fidelity/suppression margins; correct-reference increment separately assessed. | No improvement after budget: diagnose data, representation or task ambiguity; no endless prompt sweep. |
| S2 | Does pilot truth scale reliably? | Expanded ground-truth/proxy dataset, independent annotations and capture documentation. | Leakage/timing quality stable across sessions/devices; sealed test partition. | If proxies cannot support claims, narrow evaluation or improve capture. |
| S3 | Which conditioning/objective explains improvement? | First trained extractor with full ablations, calibrated confidence and residuals. | Beats equal-capacity adapted baselines; no missing-hit completion. | Remove ineffective modules; no new model solely for branding. |
| S4 | Does it generalize across recordings? | Multi-player/song/device/venue evaluation and failure taxonomy. | Independent-session listening and fidelity/coverage meet target profile. | Confine product scope if failures cluster by capture/domain. |
| S5 | Does multimodal/identity conditioning generalize? | Vision/no-vision and at least two extra target domains. | Context adds causal benefit under contradictory/missing cues. | Keep shared backbone with adapters if one general model regresses. |
| S6 | Is the contribution paper-grade? | Reproducible benchmark, released permissible assets, ablations and powered listening. | Closest current prior art covered; claims match data and statistics. | Publish narrower dataset/application result if method novelty is weak. |

S1a is intentionally before substantial training; S2 is the larger dataset expansion rather than a prerequisite for the first feasibility test. No stage inherits the old R5 production decision.

## Part 18 — Resources and engineering

The available GPU was independently queried: **RTX 4060 Laptop, 8,188 MiB**. R3 documented approximately 1.4 GiB CLAPSep inference allocation; this is historical evidence and does not predict training peak memory. Laptop power/thermal limits and software versions can substantially change throughput.

| Scale | VRAM / compute planning range | Data / storage | Engineering and duration judgment |
|---|---|---|---|
| S1 signal checks and frozen baselines | CPU plus 2–8 GB chunked model inference | Roughly 80 minutes pilot capture; 30–100 GB including video, outputs and checkpoints | 2–4 days capture/curation; 3–7 engineering days; no new foundation training. |
| S1 decoder/adapter adaptation | Target 4–7.5 GB with batch 1, AMP, frozen encoders; **must profile** | About 16 minutes clean training source across independent sessions plus remixes; larger than its effective diversity is not claimed | 4 runs × 5,000 updates; provisional 8–36 aggregate GPU hours, hard 48-hour cap. |
| Phase-capable/domain model | 16–24 GB convenient; 8 GB compact prototype possible | 5–30 h distinct target takes, 50–300 h nuisance; 0.2–1 TB with video | Several days per training campaign; 2–6 weeks engineering/data work. |
| Cross-domain research | 1–4 × 24–80 GB GPUs; approximately 100–1,000 GPU hours per campaign | 50–300 h curated target/reference data, more unlabeled mixtures; 1–5 TB | Months of curation and controlled iteration; ranges are budgets, not benchmarks. |
| Broad foundation training from scratch | Potentially tens/hundreds of accelerators and much larger corpora | Multi-terabyte to much larger licensed assets | Not warranted by S0; use pretrained representations/separators first. |

For an accessible modern generative comparator, provision a **24–48 GB inference GPU as a planning allowance**, test 5–10-second windows, and measure actual use. Do not promise that every SAM variant fits there or on 8 GB. If S1 local gradients do not fit after the specified reduction, one 24-GB cloud GPU for roughly 10–50 hours is a bounded contingency. Full research should not be artificially capped at laptop memory.

At 48 kHz float32, one mono audio hour is about 0.69 GB; six channels are about 4.15 GB/hour before compression. Video often dominates. Store source stems once and create mixtures on the fly. The main scarce resource is reliable target evidence and careful evaluation, not the number of generated training mixtures. Monetary cloud quotes are deliberately not supplied as GPU-hour estimates are more portable; obtain a current provider quote only when a measured job size exists.

## Sources and evidentiary limits

The linked [literature map](D:/Code/RhythmAlign/experiments/s0_research_reframing/LITERATURE_MAP.md) is the full primary-source inventory. Historical source selection and independent spot-check scope are documented in [Evidence and scope](D:/Code/RhythmAlign/experiments/s0_research_reframing/EVIDENCE_AND_SCOPE.md). No previous holdout is reused as an independent S1 test, no clean stem is presumed to exist, and no S0 recommendation is represented as a training or listening result.

## 1. Feasibility verdict

**A. The grand goal is technically worth pursuing.** Pursue evidence-supported, selective extraction across contexts; exact recovery from every arbitrary recording remains outside the promise.

## 2. Best next technical route

**Route A: supervised adaptation of a pretrained discriminative target extractor with a temporally aligned pristine-nuisance reference adapter.** Start from the local CLAPSep checkpoint and compare against an equally trained no-reference arm.

## 3. Need for new model

**B. Fine-tuning an existing architecture is the best next step.** Test a small conditioning extension before investing in a new backbone or foundation model.

## 4. Ground-truth priority

Record quiet, muted-playback real cabinet interactions at the handcam microphone position, synchronized with close/contact microphones and video, to obtain real target source images and independently verified events.

## 5. Minimal decisive experiment

Run the single S1 paired adaptation experiment: real interaction takes plus recorded nuisance/path mixtures, held-out sessions, frozen baselines, equal-budget no-reference/reference fine-tuning, absence/missing-hit controls and blinded preservation-versus-suppression listening.

## 6. Paper outlook

**B. Promising but application-first.** General research potential depends on reproducible fidelity gains, credible target evidence and a contribution beyond existing prompted extraction and personalized AEC.

## 7. Top five actions

1. Acquire the synchronized quiet-interaction pilot and document target/proxy leakage and timing.
2. Freeze source/session splits, truth tiers, evaluation formulas and S1 decision thresholds before model selection.
3. Run frozen baselines, a modern broad separator where accessible, and representation-oracle checks.
4. Run equal-budget CLAPSep adaptation with and without temporal pristine-reference conditioning, including wrong-reference and absent-target controls.
5. Conduct blind stem/remix listening and decide whether to expand data, revise the representation, use an existing model, or stop this specific route.
