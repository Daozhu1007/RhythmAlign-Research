# Paper direction memo

**RhythmAlign S0 · 2026-09-10 · No novelty claim or publication outcome is established.**

The strongest current direction is an application-led study that can grow into a general contribution: **faithful target extraction when context includes an aligned nuisance reference and target activity is correlated with that nuisance**. First establish useful extraction with clean domain supervision. Then determine whether reference conditioning, evidence-sensitive evaluation or the dataset contains the contribution.

Broad text/visual/span-conditioned source separation already exists. SAM Audio explicitly discusses faithful extraction; AudioSep, CLAPSep and multimodal systems cover neighboring query formulations. Personalized acoustic echo cancellation already combines a microphone mixture, playback reference and target identity. A 2011 dance-teaching system canceled known music to recover real steps/voice and remix them. Neither “extract anything from context,” “use reference plus target query,” nor “preserve real impacts and add clean music” is safe novelty by itself. [SAM Audio](https://arxiv.org/abs/2512.18099), [Personalized AEC](https://arxiv.org/abs/2205.15195), [dance-teaching assistant](https://perso.telecom-paristech.fr/grichard/Publications/2011-final_ACM_MM.pdf).

## 1. Candidate theses

### Thesis A — Reference-conditioned extraction under target–nuisance correlation

**Scientific question.** Can aligned nuisance content improve recovery of a distinct physical target when its events correlate with the nuisance rhythm, without subtracting correlated target evidence or completing expected but absent events?

**Potential contribution.** A well-defined reference/target role distinction; a reference-conditioned adapted separator with safe handling of wrong/missing reference; data and counterfactual controls isolating causal reference value under acoustic path mismatch. A useful result could be a small method that measurably improves the preservation–suppression frontier rather than an entirely new backbone.

**Novelty risk.** High for architecture alone. AEC, personalized AEC, informed separation, negative exemplars and attention conditioning already supply much of the structure. Cross-attention to a music reference is not automatically a scientific advance. Need a current nearest-neighbor search before submission, including full-band/non-speech AEC and target-correlated interference.

**Experiments.** S1 reference/null paired training; correct/wrong/shuffled reference; pooled vs temporal reference; learned cancellation cascade vs direct conditioning; equal-capacity adaptation; absent/missed/offbeat events; unseen song, path, device and venue; at least one non-arcade reference-containing domain. Quantify failure when correlation is deliberately broken or increased.

**Likely publication strength, conditionally.** A convincing domain result with controls could support an audio application or workshop paper; a reproducible cross-domain gain with a new explanation/objective could support a stronger audio/signal-processing methods submission. No venue tier is justified before results.

**Biggest reviewer objection.** “This is personalized AEC or standard conditioned separation applied to a new dataset.” Answer with equal-budget closest baselines, a real conditioning failure they do not handle, and evidence across domains; otherwise accept a narrower application contribution.

### Thesis B — Selective extraction with measured evidence risk

**Scientific question.** Can a separator detect when the recording does not sufficiently support a faithful target estimate and trade coverage for a calibrated reduction in unsupported target events/timbre errors?

**Potential contribution.** An operational failure definition, held-out calibration protocol, risk–coverage evaluation and useful acceptance mechanism combining mixture evidence and context reliability. Evaluate deterministic and generative separators under the same contract. Product abstention must distinguish target absence from unrecoverable target presence.

**Novelty risk.** High unless beyond confidence heads, generic selective prediction and mixture projection. SAM Audio already includes output judging; mixture consistency and confidence are not new guarantees. A prior-trained judge may share the separator's errors. Do not name a “provenance constraint” without an observable test it changes.

**Experiments.** Exact and real/proxy tiers; source removal/insertion interventions; wrong/missing context; severe weak overlap and clipping; OOD captures; calibration by session/domain; coverage-matched gate and rejection baselines; estimated versus actual error correlation; sample diversity for generative systems. Any finite-sample guarantee must state its exchangeability and failure-label assumptions and not extend it automatically to new domains.

**Likely publication strength, conditionally.** Potentially a general methodology contribution if risk is defined well and improves across several model families/domains. Weak if it only suppresses uncertain audio or assigns cosmetic confidence scores.

**Biggest reviewer objection.** “Your authenticity label is subjective, and abstaining on everything makes the metric easy.” Answer with exact-truth behavioral endpoints, accepted target-event/duration coverage, useful operating points and independent real listening.

### Thesis C — A source-image dataset and evaluation protocol for real target extraction

**Scientific question.** How do current extraction systems trade preservation, nuisance reduction and unsupported content when targets include weak impacts, friction and identity-specific sources in actual scenes?

**Potential contribution.** Multi-tier synchronized data, transparent proxy leakage, exact constructed controls from real recordings, real-scene evaluation, calibrated event labels and a two-axis protocol. An evaluation contribution may be valuable even if an existing model wins.

**Novelty risk.** Existing DCASE language-queried separation, FUSS, spatial-semantic challenges and multimodal benchmarks are substantial prior art. “A new arcade dataset” alone may be small; the value must come from hard source-image truth, natural overlaps, correlated nuisance and measured unsupported-event behavior. [DCASE 2024 Task 9](https://dcase.community/challenge2024/task-language-queried-audio-source-separation), [DCASE 2025 Task 4](https://dcase.community/challenge2025/task-spatial-semantic-segmentation-of-sound-scenes).

**Experiments.** Reproducible acquisition and synchrony; inter-annotator uncertainty; proxy leakage audit; broad discriminative/generative/AEC baselines; repeated sessions/devices/venues; model ranking sensitivity to exact versus proxy truth; powered blind listening; released permissible source assets and reproducible recipes.

**Likely publication strength, conditionally.** Credible dataset/evaluation paper if scale, access and protocol enable reuse. A rigorous small pilot is useful groundwork, but is not yet a broad benchmark.

**Biggest reviewer objection.** “Your clean labels are synthetic or do not correspond to the real phone waveform.” Answer by clearly separating exact construction from real proxy evaluation, publishing acquisition errors, and avoiding fake paired-waveform claims on nonlinear phone outputs.

### Thesis D — Few-shot target adaptation beyond semantic class labels

**Scientific question.** How much clean enrollment or supervised source-image data is needed to preserve a new object's acoustic identity, including continuous friction, while rejecting similar non-target objects?

**Potential contribution.** Controlled data-efficiency curves, different-take enrollment, compact adaptation and robust same-class hard-negative training. A shared encoder plus small domain adaptation could support practical general extraction without a giant retrain.

**Novelty risk.** SoundFilter, SoundBeam, CLAPSep, SoundBeam–M2D and target-speaker adaptation already cover much of this idea. Need an instance/physical-source question beyond improved average separation from extra data. [SoundFilter](https://arxiv.org/abs/2011.02421), [SoundBeam](https://arxiv.org/abs/2204.03895), [SoundBeam meets M2D](https://arxiv.org/abs/2409.12528).

**Experiments.** Zero/one/five examples and increasing clean-data budgets; independent source takes and unseen instances; corrupted enrollment; frozen encoder vs adapter vs full tuning; target-only fidelity; same-class overlaps; multiple objects and one speech-identity setting. Separate support/query/test assets and report transductive adaptation explicitly.

**Likely publication strength, conditionally.** Competitive if an efficient method produces consistent novel-instance gains and explains the conditioning failure; otherwise a useful engineering adaptation study.

**Biggest reviewer objection.** “The model memorizes a cabinet timbre or the test exemplar.” Answer with strict source-instance/session splits, different-take queries, held-out devices and target-absent other-instance controls.

### Thesis E — Domain study of authentic interaction remixing

**Scientific question.** Can an offline pipeline provide a meaningful listening improvement for real rhythm-game recordings while preserving actual player interactions?

**Potential contribution.** An integrated alignment/extraction workflow, transparent failure policy, reproducible user evaluation and carefully bounded application evidence. The workflow can be useful even without method novelty.

**Novelty risk.** High for a methods paper: known-reference dance audio cancellation and standard extraction pipelines are close. A product-quality demonstration is not proof of general source recovery.

**Experiments.** Multiple players/songs/captures; fixed-gain stems and remixes; field user study; target completeness/fidelity/annoyance; robustness and fallback; comparison with existing models and corrected gate. Demonstrate actual value, not just a spectacular selected excerpt.

**Likely publication strength, conditionally.** Promising application/demo or practitioner contribution; a stronger human-centered paper requires its own user/system research question and study.

**Biggest reviewer objection.** “It is an assembly of existing components.” If that is the measured outcome, position it honestly and avoid unnecessary model invention for branding.

## 2. Recommended thesis order

Start with **A as the technical hypothesis and C as the evidence foundation**. This is one route: the paired reference-conditioned adaptation experiment in S1. C is not a parallel large dataset program; it supplies the small trustworthy pilot required to test A.

Pursue B only after real error labels and useful quality estimates exist. Pursue D if data-efficient adaptation or instance binding emerges as the decisive issue. Retain E as a legitimate application outcome if existing methods prove sufficient. Do not commit to a paper title that presupposes a novel architecture or guaranteed authenticity.

Possible working titles, explicitly provisional:

- “When the Interference Is Known: Evaluating Reference-Conditioned Target Audio Extraction”
- “Preservation Before Plausibility: Measuring Target Fidelity in Real-Scene Audio Extraction”
- “Real Interaction Sounds under Music and Crowd Interference: A Multi-Tier Evaluation”

## 3. Claims ladder and evidence required

| Claim | Minimum supporting evidence | Present status |
|---|---|---|
| Correct timing helps exposure control | Corrected envelopes and matched-coverage controls | Supported on historical proxy metrics; not source isolation |
| Neural extraction is useful for this target family | S1 held-out exact gain plus real listening and absence controls | Unmeasured |
| Pristine reference adds useful information | Paired null/correct/wrong reference, equal capacity/budget | Plausible, unmeasured |
| Model preserves authentic events better | Exact source-image, counterfactual and timbre tests with uncertainty | Unmeasured |
| Shared framework generalizes | New identities/instances and multiple independently evaluated domains | Unmeasured |
| Method is novel | Closest-primary-literature comparison and new mechanism/result | Not established |
| Exact recovery is guaranteed in arbitrary mixtures | Would require identifiability assumptions unavailable here | Not an intended claim |

## 4. Reviewer-facing analysis to prepare

Make the target definition explicit: which source identities/actions and room image count, what the phone distortion convention is, and whether the task accepts a set of sources. Provide the same target specification to all methods. Separate class selection from instance selection; retaining every impact is insufficient when the intended source is one player's interaction.

Keep all comparisons fair in sample rate, query quality, trainable capacity, data access, alignment information and postprocessing gain. Report inaccessible checkpoints transparently and use a strong available comparator. Recheck literature at submission, particularly 2026 open-vocabulary, multimodal and reference-aware systems; S0's literature cutoff is 2026-09-10.

Show failures: weak friction disappearance, same-class leakage, incorrect reference subtraction, clipped overlap and uncertain abstention. A useful scientific story identifies the boundary and improves a measured region of it. Do not imply that input masking proves source provenance or that a diffusion output is disqualified merely by its architecture family.

## 5. Stage-specific paper gates

S1 decides whether there is a credible learned path and whether the reference increment is real. S2 expands trustworthy data only after that signal. S3 tests the mechanism against equal-budget adapted alternatives. S4 establishes cross-recording robustness. S5 evaluates two additional domains and optional vision/identity conditioning. S6 adds powered listening, full reproducibility and the final novelty audit.

If only application evidence survives, publish that scope. If the reference increment fails but a modern model works, the product may still succeed. If a rigorous benchmark reveals systematic fidelity errors across strong systems, a dataset/evaluation thesis may be stronger than a weak architectural modification. The S0 paper outlook is **B. Promising but application-first**.
