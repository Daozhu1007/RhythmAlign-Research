# RhythmAlign R0–R5.6 independent research freeze review

Review date: 2026-09-09. Scope: the repository evidence available before R6. No production integration, research-code edits, model training, or new separation experiments were performed.

**The current freeze should not enter production integration.** E0 is the more defensible detector to retain for further validation, but the actual frozen C1 implementation has reversed attack/release ramps. Several metrics used to declare product readiness do not measure the claimed quantities. The evidence supports a limited engineering result—visual timing can control exposure of recorded audio—but does not establish authentic interaction-source recovery or acceptable perceptual quality.

## Evidence discovery and inspection

The evidence index was written before this review: [experiments/independent_review_r0_r56/EVIDENCE_INDEX.md](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/EVIDENCE_INDEX.md). It lists phase, directory, report, configs, evaluation artifacts, outputs, and actual inspection status. [experiments/independent_review_r0_r56/evidence_manifest.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/evidence_manifest.json) records 107 selected paths, sizes, and review-time hashes. These hashes establish this review's snapshot, not historical freeze integrity.

Recursive searches found all eight research directories from R1 through R5.6. **R0's original landscape/formulation artifact was not found. A standalone R1 narrative report was also not found.** R1 was reviewed directly through its audit JSON, alignment diagnostics, scripts, plots, and WAVs. Neither missing report was reconstructed from later summaries.

I read the seven available main reports, inspected the consequential evaluation and construction code, examined the frozen configurations and annotation provenance, viewed selected diagnostics and a frame strip, and inspected headers of 91 output WAVs. Numerical checks of existing audio are saved in [experiments/independent_review_r0_r56/artifact_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/artifact_checks.json); matching checks are in [experiments/independent_review_r0_r56/matching_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/matching_checks.json). These are verification calculations on existing artifacts, not continuation of the experimental program.

No auditory-inspection tool was available. I therefore make **no claim of independently hearing** naturalness, masking, clicks, or musical artifacts. The repository contains listening recommendations and some narrative listening judgments, but I found no separate blinded listening records, ratings, participant information, or formal C1-versus-D1/D3 adjudication.

## Part 1 — Reconstructed evidence chain

The labels below distinguish observations from interpretations and decisions. “Recorded” means present in an artifact; it does not certify the metric's validity.

### R0 — Unavailable primary evidence

**OBSERVATION:** No original R0 directory/report was found. **HYPOTHESIS:** The supplied review brief describes the intended task, but cannot establish what R0 researched or concluded. **INFERENCE:** Landscape completeness and the original rationale for route selection remain unauditable. **PRODUCT DECISION:** Do not cite R0 as completed evidentiary support until its original artifact is located.

### R1 — Reference cancellation, drift, and stereo

**HYPOTHESIS:** Aligning pristine music and fitting its acoustic transfer into the handcam could remove cabinet playback while preserving impacts; drift compensation or stereo cues might improve this.

**OBSERVATION:** A selected 15-second window at native time 22.5–37.5 seconds was processed using global gain fitting, a two-input STFT ridge transfer estimate, and an affine-warp variant. Verification of the saved residuals gives total stereo energy reductions of **0.053 dB for A1, 0.473 dB for A2, and 0.293 dB for A3**. Recorded band-averaged coherence ranges from 0.139 at 40–150 Hz to approximately 0.006 above 4.8 kHz. The nominal music-only pool contains only 1.181 seconds. The logged held-out path check gives 0.533 dB training reduction versus 0.242 dB testing reduction, with differing stereo/mono aggregation.

**INFERENCE:** These particular cancellation methods delivered little total-mixture reduction on this recording. Deprioritizing further tuning of this implementation was reasonable. Low coherence does not identify its cause, quantify all cabinet-music energy, or prove that all linear, adaptive, nonlinear, or spatial methods are ineffective.

**PRODUCT DECISION:** Preserve alignment infrastructure; do not integrate these cancellation variants as a demonstrated enhancement. Later R4/R5 processing still uses a scalar reference residual for onset analysis, so reference information was not actually abandoned wholesale.

Evidence: [experiments/r1_golden_sample/scripts/06_cancel.py](D:/Code/RhythmAlign/experiments/r1_golden_sample/scripts/06_cancel.py); [experiments/r1_golden_sample/scripts/07_audit.py](D:/Code/RhythmAlign/experiments/r1_golden_sample/scripts/07_audit.py); [experiments/r1_golden_sample/scripts/08_stereo_and_gate.py](D:/Code/RhythmAlign/experiments/r1_golden_sample/scripts/08_stereo_and_gate.py); [experiments/r1_golden_sample/work/audit_metrics.json](D:/Code/RhythmAlign/experiments/r1_golden_sample/work/audit_metrics.json); [experiments/r1_golden_sample/work/gate_diagnostics.json](D:/Code/RhythmAlign/experiments/r1_golden_sample/work/gate_diagnostics.json); [experiments/r1_golden_sample/outputs/alignment_diagnostics.json](D:/Code/RhythmAlign/experiments/r1_golden_sample/outputs/alignment_diagnostics.json); [experiments/independent_review_r0_r56/artifact_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/artifact_checks.json).

### R2 — Text-conditioned AudioSep

**HYPOTHESIS:** A pretrained text-conditioned separator could distinguish player impacts from music and arcade contamination.

**OBSERVATION:** AudioSep ran two selected prompts on raw and A2 inputs, producing four 32-kHz mono results. SAM-Audio was blocked and was not tested. The raw-p2 output retains 49/53 legacy click candidates, against 49/53 in the input itself; its overall RMS changes from −20.35 to −24.76 dBFS. The spectrum retains substantial mixture structure. These outputs do not provide a clean target reference or independently labeled interference.

**INFERENCE:** The tested checkpoint/prompts did not demonstrate satisfactory instance-specific isolation. “No meaningful separation at all” is too absolute: R3's same-evaluator comparison later measures positive selectivity for this output. Conversely, onset retention alone cannot establish preserved authentic impacts.

**PRODUCT DECISION:** Do not ship these outputs as an isolated interaction stem. Trying a bounded audio-query comparison was reasonable. Neither this experiment nor the blocked model supports rejection of learned separation generally.

Evidence: [experiments/r2_target_separation/REPORT.md](D:/Code/RhythmAlign/experiments/r2_target_separation/REPORT.md); [experiments/r2_target_separation/logs/audiosep_runs.json](D:/Code/RhythmAlign/experiments/r2_target_separation/logs/audiosep_runs.json); [experiments/r2_target_separation/logs/evaluation.json](D:/Code/RhythmAlign/experiments/r2_target_separation/logs/evaluation.json); [experiments/r2_target_separation/scripts/11_evaluate.py](D:/Code/RhythmAlign/experiments/r2_target_separation/scripts/11_evaluate.py); [experiments/r2_target_separation/outputs/spec_audiosep_raw_p2.png](D:/Code/RhythmAlign/experiments/r2_target_separation/outputs/spec_audiosep_raw_p2.png).

### R3 — Audio-query and CLAPSep

**HYPOTHESIS:** A real impact exemplar, optionally combined with a negative music query, could resolve the target identity better than text.

**OBSERVATION:** Two AudioSep audio-query probes, two CLAPSep positive queries, and one CLAPSep negative-music condition ran on the same golden window. AudioSep's released checkpoint was text-conditioned in training; its audio-query path was explicitly off-distribution. CLAPSep was a relevant multimodal-query test. Its best negative-query result records 47/53 click matches, −8.879 dB music-heavy-frame attenuation, and +6.400 dB selectivity. The R2 text baseline records 49/53, −9.842 dB, and +6.942 dB under the same R3 evaluator.

**INFERENCE:** These audio queries did not beat the tested text baseline on these proxies. The general “8-GB separator ecosystem ceiling” claim is unsupported: two related model families and contaminated, same-recording exemplars are not an exhaustive class test. R3 also weakens R2's literal uniform-attenuation interpretation: positive selectivity is inconsistent with a pure constant-gain copy.

**PRODUCT DECISION:** Deprioritize further prompt/exemplar tweaking without better target evidence. A temporal-conditioning feasibility experiment was justified. Revisit learned separation only with materially new information—cleaner exemplars, domain supervision, or independently evaluated visual conditioning.

Evidence: [experiments/r3_audio_query/REPORT.md](D:/Code/RhythmAlign/experiments/r3_audio_query/REPORT.md); [experiments/r3_audio_query/logs/00_env_audit.md](D:/Code/RhythmAlign/experiments/r3_audio_query/logs/00_env_audit.md); [experiments/r3_audio_query/logs/04_clapsep_runs.json](D:/Code/RhythmAlign/experiments/r3_audio_query/logs/04_clapsep_runs.json); [experiments/r3_audio_query/logs/06_evaluation.json](D:/Code/RhythmAlign/experiments/r3_audio_query/logs/06_evaluation.json); [experiments/r3_audio_query/scripts/06_evaluate.py](D:/Code/RhythmAlign/experiments/r3_audio_query/scripts/06_evaluate.py).

### R4 — Oracle-conditioned reconstruction

**HYPOTHESIS:** Correct contact timing would make conservative sample gating sufficient for the intended perceptual result.

**OBSERVATION:** Audio-assisted visual annotation produced 89 candidates. Refinement splits one event and carries 87 events, of which 84 open windows. C1 and C3 record 66/84 onset matches versus raw's 65/84; all 60 audio-defined strong events match. C1's matched-event median peak ratio is approximately one. Its envelope mean is 0.647, while nonzero waveform coverage is approximately 0.740. C3 opens more audio and has weaker music-frame attenuation. The saved C1 WAV exactly implements the reversed-ramp construction identified below.

**INFERENCE:** At selected times, gated mixture samples can retain peaks present in the mixture and zero other samples. This does not establish source identity, complete contact preservation, clean in-window audio, naturalness, or satisfactory final mixes. The “oracle succeeded” gate was conditional on subjective confirmation in the report itself.

**PRODUCT DECISION:** An automatic-timing experiment was a reasonable research continuation, not proof of audio-side product readiness. R4.5's fragmentation concern already weakened the strong perceptual interpretation; this audit further invalidates the smooth-envelope and HPSS comparison claims.

Evidence: [experiments/r4_contact_reconstruction/REPORT.md](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/REPORT.md); [experiments/r4_contact_reconstruction/outputs/contacts_oracle.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/outputs/contacts_oracle.json); [experiments/r4_contact_reconstruction/outputs/contacts_refined.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/outputs/contacts_refined.json); [experiments/r4_contact_reconstruction/scripts/20_build.py](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/scripts/20_build.py); [experiments/r4_contact_reconstruction/logs/evaluation.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/logs/evaluation.json).

### R4.5 — Conservative windows

**HYPOTHESIS:** Longer, typed, or region-bridged windows could reduce fragmentation without losing authentic impacts.

**OBSERVATION:** D0 reproduces C1 exactly. D1 extends the post-window to 210 ms; D2 uses event types; D3 adds selected oracle regions and short bridges. The saved D1 envelope is a pointwise superset of D0. Recorded nonzero coverage rises from 0.740 for D0 to 0.855 for D1 and 0.866 for D3. The report leaves the final choice to listening. All these builders inherit the reversed ramps.

**INFERENCE:** The experiment characterizes an exposure-versus-attenuation tradeoff relative to C1. A chop metric defined relative to C1 gives C1 zero chop by construction; it cannot show that C1 itself sounds continuous. D3's claimed best dynamic correlation is invalid as a dynamics-fidelity measure, as explained below.

**PRODUCT DECISION:** C1 is a simpler reference policy, but its perceptual superiority was not established by a documented adjudication. Longer/region policies remain unvalidated alternatives, not proven improvements or conclusively rejected methods.

Evidence: [experiments/r45_conservative_windows/REPORT.md](D:/Code/RhythmAlign/experiments/r45_conservative_windows/REPORT.md); [experiments/r45_conservative_windows/scripts/10_build.py](D:/Code/RhythmAlign/experiments/r45_conservative_windows/scripts/10_build.py); [experiments/r45_conservative_windows/logs/evaluation.json](D:/Code/RhythmAlign/experiments/r45_conservative_windows/logs/evaluation.json); [experiments/r45_conservative_windows/logs/build_run.log](D:/Code/RhythmAlign/experiments/r45_conservative_windows/logs/build_run.log); [experiments/r45_conservative_windows/work/env_D0.npy](D:/Code/RhythmAlign/experiments/r45_conservative_windows/work/env_D0.npy); [experiments/r45_conservative_windows/work/env_D1.npy](D:/Code/RhythmAlign/experiments/r45_conservative_windows/work/env_D1.npy).

### R5 — First automatic visual detector

**HYPOTHESIS:** Four inexpensive visual features plus a rising-edge rule could provide usable contact timing.

**OBSERVATION:** Golden-window development was followed by frozen evaluation on 84–102 seconds of the same recording. Recorded F1@50 ms falls from 0.605 to 0.488; visually high-confidence recall falls from 16/19 to 10/22. The report initially attributes dense misses principally to a score plateau. R5.5's trace audit subsequently finds low score at most missed events, weakening that explanation.

**INFERENCE:** The detector detects some useful game-associated events, but fails substantially on another temporal segment. Calling this failure “not overfitting” is unjustified: a structural rule defect and adaptation to one development window can coexist. Claims of near-perfect sparse performance have almost no positive sparse-event sample support under the later region classification.

**PRODUCT DECISION:** Deferring integration was correct. The next trace audit was more valuable than blindly adding retrigger rules.

Evidence: [experiments/r5_auto_contact_detection/REPORT.md](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/REPORT.md); [experiments/r5_auto_contact_detection/outputs/detector_config_frozen.json](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/outputs/detector_config_frozen.json); [experiments/r5_auto_contact_detection/outputs/holdout_oracle_blind.json](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/outputs/holdout_oracle_blind.json); [experiments/r5_auto_contact_detection/logs/holdout_metrics.json](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/logs/holdout_metrics.json); [experiments/r55_burst_event_splitting/logs/phase1_audit_summary.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/logs/phase1_audit_summary.json).

### R5.5 — Burst-aware detection

**HYPOTHESIS:** Burst state plus local-peak retriggering could recover missed dense contacts.

**OBSERVATION:** The old holdout was explicitly promoted to development data. On new Holdout-C, R5.5 improves F1@50 from 0.367 to 0.417, recall from 0.485 to 0.632, and visually high-confidence recall from 9/21 to 17/21. Yet music-onset matches rise from the reported same-window baseline's 19 to 33, and final-mix discrepancy worsens from −9.3 to −8.8 dB. Holdout-C's event universe comprises 68 audio-selected candidates, all accepted as contacts, within one full-window continuous region.

**INFERENCE:** Local retriggering improved recovery of the annotated events within this recording. It did not establish cleaner mixtures or comprehensive contact recovery. Low score at missed events supports a limitation of the extracted features; it does not prove physical invisibility or an information-theoretic ceiling. The approximately 0.59 rule-search result is only a finite sweep result, and its implementation has additional weaknesses.

**PRODUCT DECISION:** Retain R5.5 as a reference detector; do not treat it as a validated production baseline. A final bounded cleanup experiment was defensible, especially because its assumptions were audited first.

Evidence: [experiments/r55_burst_event_splitting/REPORT.md](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/REPORT.md); [experiments/r55_burst_event_splitting/outputs/detector_config_r55_frozen.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/outputs/detector_config_r55_frozen.json); [experiments/r55_burst_event_splitting/outputs/holdoutC_oracle_blind.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/outputs/holdoutC_oracle_blind.json); [experiments/r55_burst_event_splitting/logs/holdoutC_metrics.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/logs/holdoutC_metrics.json); [experiments/r55_burst_event_splitting/logs/holdoutC_audio_comparison.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/logs/holdoutC_audio_comparison.json); [experiments/r55_burst_event_splitting/scripts/12_rule_ceiling.py](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/scripts/12_rule_ceiling.py).

### R5.6 — Cleanup and audio completion

**HYPOTHESIS:** Weak-candidate filtering and audio completion inside visual bursts could reduce false exposure without sacrificing real contacts.

**OBSERVATION:** Development proxies improve, but Holdout-D provides no convincing incremental advantage over E0. The recorded F1 values are 0.534 versus 0.533; both recover 18/21 visually high-confidence events. E2 has lower audio-strong gate coverage and two added events inside the annotated rest. The saved burst interval is **[0.000, 18.005] seconds: the whole clip**. The “Taiko guard” outcome is not an external-Taiko measurement, and both methods actually have zero on that proxy in the JSON.

**INFERENCE:** The incremental E2 hypothesis failed its strongest available test. This supports stopping that heuristic-tuning sequence. It does not establish statistical equivalence, product readiness, or that a learned detector will solve the problem.

**PRODUCT DECISION:** Prefer E0 for subsequent controlled validation, but reject immediate integration of either frozen C1 pipeline. The repository's R5.6 report actually selects E2 for integration and permits E0 as a conservative alternative; the supplied CTO preference for E0 is a separate decision, not the report's primary frozen selection.

Evidence: [experiments/r56_precision_cleanup/REPORT.md](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/REPORT.md); [experiments/r56_precision_cleanup/logs/dev_eval_E0_E1_E2.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/logs/dev_eval_E0_E1_E2.json); [experiments/r56_precision_cleanup/logs/holdoutD_metrics.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/logs/holdoutD_metrics.json); [experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json); [experiments/r56_precision_cleanup/outputs/detector_config_r56_frozen.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/detector_config_r56_frozen.json).

## Part 2 — Audit of the negative results

### Reference cancellation: sensible deprioritization, overstated ceiling

The magnitude-squared coherence calculation in R1 is recognizable Welch/CSD estimation on aligned mono mid signals: |Sxy|²/(Sxx Syy), using 8192-sample segments and 50% overlap. The matching spectral settings are reasonable for measuring **this scalar stationary relationship**. The plot corroborates low broad-band coherence, with some stronger low-frequency peaks. It is not a measurement of the full two-reference stereo Wiener ceiling used by A2, nor a causal diagnosis of phone processing. Evidence: [experiments/r1_golden_sample/scripts/08_stereo_and_gate.py](D:/Code/RhythmAlign/experiments/r1_golden_sample/scripts/08_stereo_and_gate.py); [experiments/r1_golden_sample/work/msc.png](D:/Code/RhythmAlign/experiments/r1_golden_sample/work/msc.png).

The recorded coherent share is 0.05937, with ceiling −10 log10(1−share) = 0.266 dB. That transformation is meaningful only as a modeled reduction of **total mixture energy** by the estimated scalar coherent component. If unrelated impacts/interference dominate the denominator, strong removal of a smaller music component can still produce little total ERLE. Nonstationary transfer, residual timing errors, reference-version mismatch and finite-sample estimation further limit interpretation. No generating script for the standalone ceiling JSON was found. A fresh calculation using the stored stereo-mean WAVs and the visible CSD settings gives 0.07091 and 0.319 dB; the qualitative conclusion agrees, but the exact logged ceiling is not reproduced. Evidence: [experiments/r1_golden_sample/work/cancellation_ceiling.json](D:/Code/RhythmAlign/experiments/r1_golden_sample/work/cancellation_ceiling.json); [experiments/independent_review_r0_r56/artifact_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/artifact_checks.json).

A2 is a per-frequency, two-input STFT transfer fit applied to the same context used for fitting, with regularization selected on that material. It is not a comprehensive time-varying echo canceller. The separate path check is useful but does not use identical channel aggregation for training and testing. Therefore its limited performance is credible evidence about this implementation, while claims of a general linear impossibility are not. Evidence: [experiments/r1_golden_sample/scripts/06_cancel.py](D:/Code/RhythmAlign/experiments/r1_golden_sample/scripts/06_cancel.py); [experiments/r1_golden_sample/scripts/07_audit.py](D:/Code/RhythmAlign/experiments/r1_golden_sample/scripts/07_audit.py).

The drift sweep selects **−100 ppm at its search boundary**, with only a small objective benefit. The cancellation script's comment says epsilon is zero, but the actual alignment JSON does not. The defensible conclusion is that the evaluated local drift treatment did not materially improve this window, not that clock drift was absent or ruled out across the song. Evidence: [experiments/r1_golden_sample/outputs/alignment_diagnostics.json](D:/Code/RhythmAlign/experiments/r1_golden_sample/outputs/alignment_diagnostics.json); [experiments/r1_golden_sample/scripts/06_cancel.py](D:/Code/RhythmAlign/experiments/r1_golden_sample/scripts/06_cancel.py).

The handcam metadata identifies Xiaomi AI-audio processing. That supports a processing hypothesis, not a measured AGC/nonlinearity mechanism. The gain/loudness correlation is only +0.182. Decoded peaks above full scale occur in both handcam AAC and reference MP3; lossy decoding overshoots cannot establish clipping at acquisition. Stereo similarity and overlapping event features justify deprioritizing simple L/R discrimination on this recording, not rejecting calibrated spatial recording or array methods generally. Evidence: [experiments/r1_golden_sample/work/audit_report.json](D:/Code/RhythmAlign/experiments/r1_golden_sample/work/audit_report.json); [experiments/r1_golden_sample/work/audit_metrics.json](D:/Code/RhythmAlign/experiments/r1_golden_sample/work/audit_metrics.json); [experiments/r1_golden_sample/work/gate_diagnostics.json](D:/Code/RhythmAlign/experiments/r1_golden_sample/work/gate_diagnostics.json).

### Separators: unsuccessful accessible baselines, not a class-level refutation

AudioSep and CLAPSep are relevant baseline families; AudioSep's audio-query probe deserves less inferential weight because it changes conditioning modality relative to training. Their published scope also differs from identifying one particular player's impacts among acoustically similar sources. See the original [AudioSep paper](https://arxiv.org/abs/2308.05037) and [CLAPSep paper](https://arxiv.org/abs/2402.17455).

Three qualifications matter. First, the query clips contain music and are drawn from the tested recording, so this is not clean exemplar-conditioned generalization. Second, music-heavy frames and click-centered windows are still mixtures: positive selectivity is a useful proxy, not source SDR. Third, unmatched onset counts indicate detector disagreements, not necessarily synthesized events; zero such counts also does not prove absence of artifacts. The reports properly flag subjective uncertainty in several places but then sometimes make stronger perceptual statements than their evidence permits. Evidence: [experiments/r3_audio_query/REPORT.md](D:/Code/RhythmAlign/experiments/r3_audio_query/REPORT.md); [experiments/r3_audio_query/scripts/06_evaluate.py](D:/Code/RhythmAlign/experiments/r3_audio_query/scripts/06_evaluate.py); [experiments/r2_target_separation/scripts/11_evaluate.py](D:/Code/RhythmAlign/experiments/r2_target_separation/scripts/11_evaluate.py).

Do not revive prompt tuning merely because other settings might exist. Revisit a separation route only after obtaining isolated impact references or a representative evaluation set capable of detecting its benefit. A corrected HPSS control is worth revisiting for a different reason: the supposed negative HPSS experiment did not implement the stated control.

## Part 3 — R4/R4.5: what was actually established?

### Release-blocking construction defect: reversed fades

In `envelope_from_spans`, the beginning uses `ramp_dn` and the end uses `ramp_up`. The gain jumps from zero to one at the start, declines to zero over the nominal attack, jumps back to one, jumps to zero at the release start, rises to one, then abruptly becomes zero outside the span. This occurs in R4, R4.5, and the common C1 code inherited by R5–R5.6.

This is not a comment-only issue. Reconstructing the existing R4 C1 with that function matches `C1_interaction.wav` **exactly at float32 precision**. At 0.065479 seconds, the left-channel output jumps from zero to 0.049748; at 0.530188 seconds it drops from 0.118822 to zero. Such discontinuities create a concrete risk of boundary transients and corrupt comparisons intended to assess smooth reconstruction. Their audibility requires listening; the implementation error does not.

Evidence: [experiments/r4_contact_reconstruction/scripts/20_build.py](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/scripts/20_build.py); [experiments/r45_conservative_windows/scripts/10_build.py](D:/Code/RhythmAlign/experiments/r45_conservative_windows/scripts/10_build.py); [experiments/r5_auto_contact_detection/scripts/r5_common.py](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/scripts/r5_common.py); [experiments/r4_contact_reconstruction/outputs/C1_interaction.wav](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/outputs/C1_interaction.wav); [experiments/independent_review_r0_r56/artifact_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/artifact_checks.json).

![Measured gain from the saved C1 envelope](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/C1_envelope_audit.png)

Correcting the fades will create a new baseline requiring reevaluation. The old outputs must remain available as historical evidence. This review does not silently repair them.

### The C2 negative control is mislabeled

The STFT has frequency-by-time dimensions. C2 names a frequency-axis median `Zh`, names a time-axis median `Zp`, then puts **the time-axis median in the numerator** of its supposed percussive mask. It preferentially keeps sustained horizontal harmonic structure. This is reversed relative to the usual harmonic/percussive interpretation, documented in [librosa's HPSS tutorial](https://librosa.org/doc/latest/auto_tutorials/03-advanced/plot_hprss.html).

The saved C2 waveform agrees with this implementation to numerical precision. Its 0.56 median peak ratio and extra attenuation describe this actual mask; they do not demonstrate that correctly implemented percussive emphasis necessarily destroys impacts. Do not infer that corrected HPSS will succeed either. Evidence: [experiments/r4_contact_reconstruction/scripts/20_build.py](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/scripts/20_build.py); [experiments/r4_contact_reconstruction/logs/evaluation.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/logs/evaluation.json); [experiments/independent_review_r0_r56/artifact_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/artifact_checks.json).

### Oracle quality and timing

R4's annotation universe was built from **89 audio-onset candidates verified visually**, after overview passes. It was not an exhaustive independent physical-contact census. It records 82 press/touch/slide events, plus release, rest and rejected events. Refinement splits e68 into two; its 84 audio-positive build events include two releases and two entries labeled rest. Those categories are not automatically invalid audible interactions, but they cannot silently become the same denominator as 82 visual contacts. Identity annotation also changed during the supposedly timing-only refinement stage. Evidence: [experiments/r4_contact_reconstruction/outputs/contacts_oracle.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/outputs/contacts_oracle.json); [experiments/r4_contact_reconstruction/outputs/contacts_refined.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/outputs/contacts_refined.json); [experiments/r4_contact_reconstruction/scripts/10_refine.py](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/scripts/10_refine.py).

Judgment text provides evidence of game-recognized actions, not a calibrated measurement of physical impact time, impact force, or acoustic source identity. Persistent text, slide displays, occlusion, simultaneous hands and non-scoring physical taps complicate interpretation. Sixty-fps acquisition does not guarantee ±8.3-ms annotation error: text can lag contact by 0–2 frames, some inspection is at 30 fps, and A/V mapping is uncertain. R4's own metadata acknowledges roughly ±25-ms mapping uncertainty.

The ±40-ms refinement chooses the largest local multiband peak. It may select a neighboring hit, music, or another transient. R1's weak reference cancellation means the variable called a music-removed residual still contains music. Auto and oracle refinements landing on the same peak explain zero median paired timing error; that agreement is not independent evidence of the peak's identity. Evidence: [experiments/r4_contact_reconstruction/outputs/contacts_oracle.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/outputs/contacts_oracle.json); [experiments/r5_auto_contact_detection/scripts/r5_common.py](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/scripts/r5_common.py).

### The headline preservation metrics are inadequate

The oracle is audio-assisted, “strong” is defined by the refinement envelope, and preservation is evaluated by another related onset detector on the same mixture-derived signal. This is an **audio-selection and evaluator-dependence problem**, not fully independent validation. Broad windows and proximity matching allow music or a neighboring event to count as contact preservation. R4's music-only C0 even “preserves” 19/84 contacts and 6/60 audio-strong events, demonstrating the ambiguity directly. Evidence: [experiments/r4_contact_reconstruction/logs/evaluation.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/logs/evaluation.json); [experiments/r4_contact_reconstruction/scripts/30_evaluate.py](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/scripts/30_evaluate.py).

Additional defects in that evaluator require correction before its claims are reused:

- “Dynamics correlation” is `corr(peak_ratio, raw_peak)`, rather than correlation of output and reference amplitudes. For identity processing the ratio is essentially constant, so this correlation is undefined or dominated by tiny numerical variation. Raw's reported 0.92 does not validate the measure. The C1/C2/C3 dynamics ranking cannot support fidelity claims.
- Contact matching is nearest-neighbor, not one-to-one. More than one contact can share an output onset.
- The `false_onsets` comprehension zips output onsets with `m_ct`, which is indexed by oracle contacts. These arrays represent different entities and may differ in length. That field is not a valid unmatched-output-onset count.
- Peak/electrical-level statistics are conditioned on matched events. Their good medians do not account for missing contacts, full decay tails, or continuous rubbing/slide audio.

Evidence: [experiments/r4_contact_reconstruction/scripts/30_evaluate.py](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/scripts/30_evaluate.py).

### Gating is selective exposure, not demonstrated source separation

For mixture y = interaction + cabinet music + other interference, C1 produces g·y. At gain one, all components pass; at gain zero, all disappear. Correct onset timing does not separate coincident sources. “Unchanged samples” applies to the gain-one interior; fades amplitude-modulate samples. Sample provenance excludes replacement foley, but does not guarantee artifact-free output or exclusively authentic player sound.

The reported 31/39 Taiko coincidences are coincidences with **low-band candidates**, inherited from R1—not independently annotated neighboring Taiko strikes. At high contact density, many unrelated events will lie within ±80 ms of a contact by chance. Without a time-shift/chance baseline and valid source labels, neither rhythmic coupling nor inevitability is established. Coincident-source ambiguity is structural for scalar gating; it is not an excuse to declare the desired suppression achieved. Evidence: [experiments/r1_golden_sample/scripts/08_stereo_and_gate.py](D:/Code/RhythmAlign/experiments/r1_golden_sample/scripts/08_stereo_and_gate.py); [experiments/r4_contact_reconstruction/logs/evaluation.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/logs/evaluation.json).

“Outside windows = zero” is a construction property, equally satisfied by deleting everything. It says nothing about whether the retained windows contain the desired source or whether discarded samples contain authentic weak interactions. Near-continuous dense-region exposure weakens noise suppression precisely where many contacts occur. The reported rolling C3 envelope coverage near 0.96 is therefore a material limitation, not a passed criterion. Evidence: [experiments/r4_contact_reconstruction/REPORT.md](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/REPORT.md); [experiments/r4_contact_reconstruction/scripts/20_build.py](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/scripts/20_build.py).

Finally, the approximately −20-dB “music share” is a scalar reference-projection share. R1 already shows much music may be poorly linearly coherent; unprojected music does not become non-music. Spectral-ripple similarity and mono delayed correlation are not perceptual comb-filter or pumping tests. The strong “masking holds” and “no pumping” conclusions are unverified. Evidence: [experiments/r4_contact_reconstruction/scripts/30_evaluate.py](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/scripts/30_evaluate.py); [experiments/r4_contact_reconstruction/REPORT.md](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/REPORT.md).

**What R4 proved:** a particular annotation/refinement/gating construction can preserve many selected mixture peaks and suppress samples outside its support. **What it did not prove:** complete authentic-contact preservation, correct isolation within windows, acceptable continuous-contact sound, absence of boundary artifacts, or the counterfactual sound of a quiet arcade.

## Part 4 — Automatic detection and freeze discipline

### Development and holdout protocol

| Round | Development material | New temporal holdout | Freeze → annotation completion, as recorded |
|---|---|---|---|
| R5 | A: 22.5–37.5 s | B: 84–102 s | 02:35:00 → 03:08:44 |
| R5.5 | A + B | C: 40–58 s | 03:51:51 → 04:06:48 |
| R5.6 | A + B + C | D: 60–78 s | 13:13:37 → 13:39:22 |

The explicit promotion of old holdouts to development data, distinct new temporal windows, frozen parameter records, and reported annotation-before-prediction ordering are methodological strengths. Falsifying the plateau explanation and rejecting echo-ratio suppression also show useful hypothesis correction.

However, these are four windows totaling **69 seconds of one recording**, not independent recording-level tests. The original full-track density profiles and visual selection passes influenced window choice. They support purposive challenge tests, not random or representative sampling. I found no evidence that detector scores selected favorable holdouts; nevertheless, selection probabilities and predeclared population criteria are absent.

Historical ordering is documented, not independently certified. The experiment directories were already untracked; timestamps are ordinary strings, and `annotation_completed_utc` is written with local `time.strftime`. R5.6's runner uses `E1_CFG`/`E2_CFG` constants rather than loading those values from its frozen JSON. The inspected constants agree with the JSON, so this is a reproducibility weakness, not evidence of actual post-holdout tuning. Evidence: the three frozen config files in the index; [experiments/r56_precision_cleanup/scripts/common56.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/common56.py); [experiments/r56_precision_cleanup/scripts/64_write_oracle_d.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/64_write_oracle_d.py); [experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py).

### The annotation improvement is real but overstated

Holdout-D starts with video inspection and uses audio only for later gap checks; this reduces audio-defined candidate-universe bias compared with A/B/C. But the report's claim that the program now has **four video-first oracles is false**: the reused A/B/C JSONs preserve their audio-assisted construction. Holdout-D itself explicitly admits possible missing visually silent subevents. It is the strongest available oracle, not demonstrated complete ground truth.

The claimed 60-fps verification strips also need qualification. `strips()` selects every second frame from a nominal 60-fps sequence, showing five frames spaced about 33 ms apart. Native acquisition and actual inspection cadence are different. Evidence: [experiments/r56_precision_cleanup/scripts/63_oracle_materials.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/63_oracle_materials.py); [experiments/r56_precision_cleanup/outputs/holdoutD_oracle_blind.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_oracle_blind.json); [experiments/r4_contact_reconstruction/outputs/contacts_oracle.json](D:/Code/RhythmAlign/experiments/r4_contact_reconstruction/outputs/contacts_oracle.json); [experiments/r5_auto_contact_detection/outputs/holdout_oracle_blind.json](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/outputs/holdout_oracle_blind.json); [experiments/r55_burst_event_splitting/outputs/holdoutC_oracle_blind.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/outputs/holdoutC_oracle_blind.json).

Holdout-C's completeness warning is warranted by its construction, but similar density profiles with different candidate counts do not mathematically prove a specific number of missed contacts. Unmatched predictions may include missing labels, duplicate responses and actual false detections. They cannot all be excused as missed oracle events.

### Corrected Holdout-D comparison

These are values from the primary metric JSON, with interpretations restricted to their actual definitions.

| Quantity | E0 | E2 | Interpretation |
|---|---:|---:|---|
| Candidate count | 150 | 154 | E1 preserves the visual list; E2 adds four audio events |
| Opening events | 123 | 116 | E1 alone has 112 opening flags |
| Event precision / recall @50 ms | .420 / .733 | .416 / .744 | Annotation-relative detection, not source purity |
| Event F1 @50 ms | .534 | .533 | No useful observed increment |
| Event F1 @33 ms | .339 | .342 | Strong tolerance sensitivity |
| Visually high-confidence recall | 18/21 | 18/21 | Not loudness-defined “strong impact” recall |
| Matched-event timing p95 @50 ms | 44 ms | 44 ms | Conditional on passing the 50-ms match gate |
| Audio-positive oracle-point coverage | .931 | .917 | Evaluated on 72/86 events after 14 no-transient exclusions |
| Audio-strong oracle-point coverage | 32/35 | 31/35 | Different denominator from 18/21 |
| Reported false-window duration | 5.22 s | 5.07 s | Flawed center-of-waveform-span proxy |
| Reference-onset coincidences | 40 | 38 | Not measured music energy |
| Bass-dominated reference-onset coincidences | **0** | **0** | Not neighboring Taiko leakage |
| Nonzero mono waveform coverage | **.830** | .809 | Report incorrectly gives .740 for E0 |
| Useful-window ratio, high/medium labels | .743 | .712 | Annotation-neighborhood overlap, not useful-source purity |
| Final-mix relative RMS discrepancy | −13.8 dB | −13.6 dB | Same pristine track in both mixes |

Evidence: [experiments/r56_precision_cleanup/logs/holdoutD_metrics.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/logs/holdoutD_metrics.json); [experiments/r56_precision_cleanup/outputs/holdoutD_oracle_refined.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_oracle_refined.json); [experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json); [experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py).

The matching verification reproduces 63/86 and 64/86 matches at 50 ms. Maximum-cardinality matching produces the same counts at 33/50 ms, so the negligible E2 increment is not caused by greedy matching there. At 80 ms, greedy matching gives 78 matches versus a maximum of 80 for both; metric policy matters when tolerance widens. Evidence: [experiments/independent_review_r0_r56/matching_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/matching_checks.json).

“Statistically tied” should be replaced by **no demonstrated incremental benefit**. There is no equivalence test, margin, or independent-recording uncertainty estimate. Even an optimistic independent-event Wilson interval for 18/21 is approximately 0.654–0.950; correlated events from one clip provide still less evidence about new-recording reliability. Merely exceeding 0.85 by one rounded point estimate is not a product guarantee.

### Important discrepancies and metric failures

**1. Rest contamination is broader than the report's explanation.** The stored burst covers the whole 18-second window. Thus burst-constrained completion is effectively unconstrained in time on this clip, including its annotated G1 rest [6,8]. It cannot be credited with detecting that rest. The report's brief-hold-delay explanation is insufficient. Direct waveform measurements show E0 passes nonzero audio during 1.424 seconds of that rest and E2 during 1.746 seconds. Their rest-interval energies are only 1.77 and 0.85 dB below raw, respectively; the oracle output is digital zero. The exact E2 addition times are 6.102667 and 6.431833 seconds. This failure is relevant to the user's actual objective even if some general false-window proxy improves.

Evidence: [experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json); [experiments/r56_precision_cleanup/outputs/holdoutD_oracle_blind.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_oracle_blind.json); [experiments/r56_precision_cleanup/outputs/holdoutD_e0_C1_interaction.wav](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_e0_C1_interaction.wav); [experiments/r56_precision_cleanup/outputs/holdoutD_auto_C1_interaction.wav](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_auto_C1_interaction.wav); [experiments/independent_review_r0_r56/artifact_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/artifact_checks.json).

**2. The Taiko metric measures the wrong source.** The evaluator detects onsets in pristine music and labels some by reference-band dominance. A neighboring machine cannot be established by that test. Additionally, the report's E0→E2 Taiko improvement 1→0 contradicts the saved JSON's 0→0. The residual-band guard may reject some bass-heavy candidates; it has not demonstrated source-specific Taiko rejection. Evidence: [experiments/r56_precision_cleanup/scripts/common56.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/common56.py); [experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py); [experiments/r56_precision_cleanup/logs/holdoutD_metrics.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/logs/holdoutD_metrics.json).

**3. False-window duration is not false exposure duration.** Open spans are recovered from `abs(mono_stem)>1e-6`, so waveform zero crossings and the erroneous envelope notches fragment them. An entire span is called false if its midpoint lacks a nearby oracle onset. A long span can contain true events and fail that test, or contain substantial false exposure and pass it. Count, duration and thresholds depend on waveform amplitude and stereo cancellation. Compute gate support directly from the saved gain function; intersect it with explicit allowed-contact support and true rest intervals. Keep annotation uncertainty separate. Evidence: [experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py); [experiments/r56_precision_cleanup/scripts/common56.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/common56.py).

**4. Strong recall and strong coverage are not the same construct.** Visual `confidence=high` denotes annotation certainty. Audio `refined_confidence=strong` denotes local envelope contrast. Neither independently measures physical force or perceptual importance. Moreover, visual strong recall uses nearest-prediction existence, not exclusive matching, and E1 does not remove candidates from that list when closing their windows. Stable E1 event recall is partly a consequence of metric construction, not proof that audio preservation stayed unchanged. Evidence: [experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py); [experiments/r56_precision_cleanup/scripts/common56.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/common56.py).

**5. Region recall can be trivial.** All A contacts are classed dense, B has only two sparse contacts, and C has one full-window continuous region. Covering such regions with frequent predictions does not validate precise contacts or low rest exposure. “Sparse near-perfect” is unsupported as a general claim. Evidence: [experiments/r55_burst_event_splitting/logs/phase1_audit_summary.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/logs/phase1_audit_summary.json); [experiments/r55_burst_event_splitting/outputs/holdoutC_oracle_blind.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/outputs/holdoutC_oracle_blind.json).

**6. The numerical heuristic ceiling is not a ceiling.** A finite sweep's best result is an achieved score, not an upper bound over a feature family. The same repository reports later dense F1 values 0.609/0.611 and a single-channel 0.607. In the ceiling script, some normalized prominences are compared against thresholds scaled again by raw-channel magnitude; channel-vote accumulation is not strictly one vote per channel. These implementation details further weaken the attempted exhaustive interpretation. Evidence: [experiments/r55_burst_event_splitting/scripts/12_rule_ceiling.py](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/scripts/12_rule_ceiling.py); [experiments/r55_burst_event_splitting/logs/phase1c_rule_ceiling.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/logs/phase1c_rule_ceiling.json); [experiments/r55_burst_event_splitting/logs/phase1_audit_summary.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/logs/phase1_audit_summary.json); [experiments/r55_burst_event_splitting/outputs/detector_config_r55_frozen.json](D:/Code/RhythmAlign/experiments/r55_burst_event_splitting/outputs/detector_config_r55_frozen.json).

**7. Final-mix RMS is not perceptual closeness to the target.** Both mixes contain the identical 0.5 pristine reference, which cancels from their difference while contributing heavily to the normalization denominator. Mono averaging can hide channel-specific discrepancies. Increasing pristine level could improve the normalized score without improving the stem. The oracle stem itself retains contaminated samples. Compare stem errors, rest contamination, and listening outcomes separately. Evidence: [experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/scripts/70_holdoutD_run.py).

**8. Several narrative details are stale or incorrect.** The reported missed IDs d42/d56/d71 correspond in the actual oracle to **10.00/12.34/15.30 seconds**, not 4.27/8.21/14.30. The report says E1 opens nine more windows than E0, whereas the saved flags give 112 versus 123. R5's headline 42 matches out of 80/92 implies 38 unmatched predictions and 50 unmatched annotations; its narrative taxonomy's 36/47 uses a different counting basis. These do not erase all results, but causal explanations must be regenerated from the actual pairings. Evidence: [experiments/r56_precision_cleanup/REPORT.md](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/REPORT.md); [experiments/r56_precision_cleanup/outputs/holdoutD_oracle_blind.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_oracle_blind.json); [experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json); [experiments/r5_auto_contact_detection/REPORT.md](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/REPORT.md); [experiments/r5_auto_contact_detection/logs/holdout_metrics.json](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/logs/holdout_metrics.json).

## Part 5 — Ranked threats to validity

| Severity | Threat and consequence | Experiment or audit that resolves it |
|---|---|---|
| **CRITICAL** | Frozen reconstruction differs from its smooth-envelope specification; HPSS control is reversed. Perceptual comparisons and release readiness are compromised. | Preserve old outputs, independently correct and version both implementations, verify envelope endpoints/continuity and mask orientation, regenerate affected comparisons, then blind-listen. |
| **CRITICAL** | No clean target/interference reference; timing coincidences are treated as source recovery and Taiko suppression. | Record synchronized close/contact-mic targets and separately observed interference; include controlled real-impact mixtures and quiet-cabinet reference sessions. Validate source labels and score missed authentic energy plus retained interference. |
| **CRITICAL** | One recording, song, player, arcade, phone, layout and recording chain. Temporal holdouts cannot establish product generalization. | Collect crossed players/songs/devices/layouts and multiple venues/sessions. Split by original recording, with unseen sessions/devices/players where feasible. Hold back a final test set before development. |
| **MAJOR** | Audio-assisted, incomplete oracle; visual game effects stand in for physical events; strong means certainty or contrast. | Two independent annotators start video-first, retain uncertain intervals, adjudicate disagreements, and use synchronized physical/acoustic evidence to distinguish contact, sound onset, audibility and force. Audit audio-added events separately. |
| **MAJOR** | Related onset logic selects labels, refines predictions and evaluates preservation; gate/RMS/dynamics metrics are misleading or defective. | Correct evaluator definitions; score gain-support overlap, known-rest energy and target-waveform retention; use independent labels/detectors and preservation-versus-interference curves. Include random/time-shifted and coverage-matched controls. |
| **MAJOR** | No documented blinded listening study; expectations and common pristine music can conceal target losses. | Randomize and blind comparisons, evaluate stems and fixed-gain final mixes separately, include raw/current mix, music-only and oracle anchors, and obtain ratings for missing impacts, authenticity, naturalness, interference and preference. |
| **MAJOR** | Reused windows share rhythm and noise structure; held-out segment selection is purposive; thresholds and success criteria evolve. | Preregister primary outcomes and stop rules on a new recording-level test set. Use recording/session-level uncertainty, not event-level pseudo-replication. Report all selected windows and selection reasons. |
| **MAJOR** | Burst activity includes true rest; fixed ROIs and full-window normalization are uncalibrated quality signals. | Test true rest, display-only activity, missed notes, occlusion, camera shifts and dense play. Measure abstention error and preservation under fallback. Evaluate full-length inputs and different chunk boundaries. |
| **MAJOR** | Freeze/config/annotation/report relationships are not immutable; runtime constants can bypass frozen JSON. | Build a versioned manifest of actual runtime configuration, code, inputs, annotations and outputs; validate report tables against machine-readable results. Prospectively log freeze-before-test provenance. |
| **MODERATE** | ±33/50/80-ms tolerance changes conclusions; p95 only considers accepted matches; 30-fps inspection and frame-index clocks add uncertainty. | Report tolerance curves, signed timing errors, missed-event rates, annotation intervals and real frame PTS/A-V calibration. |
| **MODERATE** | Models use 32-kHz mono while gates use 48-kHz stereo; gains and source-level normalization affect comparisons. | Compare at matched bandwidth/channel conditions for objective metrics; retain native outputs for listening and report both. |
| **MINOR** | File naming, local/UTC labels and missing R0/R1 narrative artifacts impair traceability. | Resolve provenance and missing-document status explicitly without inventing historical content. |

These rankings follow the inspected implementation and artifact chain above. The implementation defects are immediate blockers; the single-recording and target-ground-truth limitations remain even after every defect is repaired.

## Part 6 — Production review

**1. Is E0 genuinely better?** It is the better *default candidate for further testing*: simpler, no extra audio-only completion, slightly better annotated audio-strong coverage and mix discrepancy, and materially less exposure during the known rest. It is not statistically proven superior across recordings. E2's development advantage did not survive the stronger holdout in a useful way.

**2. Is C1 sufficiently justified?** The general idea is a reasonable conservative baseline. The actual frozen implementation is not acceptable because of the fade defect; the intended smooth C1 also lacks sufficient perceptual and cross-recording validation. Both E0+C1 and E2+C1 lose to **not shipping this feature yet** under the present evidence.

**3–4. Fallback and refusal.** With the current implementation, keep processing disabled. After correction, an experimental version should abstain when timing/decoding is unreliable, the cabinet/hand geometry is unsupported, visibility is insufficient, or quality calibration indicates unacceptable authentic-event loss or excessive false exposure. Dense near-continuous gating should trigger a low-benefit assessment, not aggressive pruning merely to obtain silence. Explicit rest must not be inferred solely from the current burst state.

The safe fallback for this product objective is the existing aligned mix that retains original recorded audio. Falling back silently to pristine music alone deletes the very physical interactions the user wants. Preserve the source and provide an unprocessed comparison. Pure-music output is appropriate only as a separately chosen behavior.

**5. Internal quality signals.** Track real frame timestamps, A/V alignment uncertainty, ROI validity and visibility, contact-probability calibration, burst duration, explicit rest conflicts, refinement shift/ambiguity, audio-only additions, direct envelope coverage, longest open spans, sampled gain at annotated contacts in validation, clipping/headroom, and fallback decisions. Audit chunk sensitivity because p99.5/p99.9 normalization currently depends on the whole evaluation window. These signals are not probabilities until calibrated against independent recordings.

**6. Initial exposure.** **Hidden research mode now.** After the implementation/evaluation blockers and initial new-recording/listening checks pass, an explicitly opted-in, default-off experimental beta is defensible. Default-on is not supported.

**7. Defensible promise after validation.** “Uses visible gameplay activity to retain selected portions of the recorded sound and reduce audio between them. Some background sound may remain, and some interactions may be missed.” Do not promise isolated operation sounds, complete preservation, zero Taiko, or the sound of every other machine becoming quiet.

Evidence for this decision: [experiments/r5_auto_contact_detection/scripts/r5_common.py](D:/Code/RhythmAlign/experiments/r5_auto_contact_detection/scripts/r5_common.py); [experiments/r56_precision_cleanup/logs/holdoutD_metrics.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/logs/holdoutD_metrics.json); [experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json](D:/Code/RhythmAlign/experiments/r56_precision_cleanup/outputs/holdoutD_contacts_auto_visual.json); [experiments/independent_review_r0_r56/artifact_checks.json](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/artifact_checks.json).

## Part 7 — Paper Viability Memo

There is a research seed, but the current evidence is not a validated separation contribution. A targeted primary-literature check already establishes substantial neighboring work: [Visually Indicated Sounds](https://openaccess.thecvf.com/content_cvpr_2016/html/Owens_Visually_Indicated_Sounds_CVPR_2016_paper.html) studies sound prediction from physical interactions using synthesis; [The Sound of Pixels](https://openaccess.thecvf.com/content_ECCV_2018/html/Hang_Zhao_The_Sound_of_ECCV_2018_paper.html) studies visually grounded separation; [AudioSep](https://arxiv.org/abs/2308.05037) and [CLAPSep](https://arxiv.org/abs/2402.17455) cover query-conditioned extraction. Thus neither “vision helps with sounds” nor “query-conditioned separation” is a defensible novelty claim. This limited check is not an exhaustive novelty search, and the missing R0 artifact cannot fill that gap.

### A. Rhythm-game handcam enhancement — viable application study

**Question:** Can authentic recorded interaction sounds remain perceptually useful while non-target arcade audio is reduced? **Possible contribution:** A reproducible task definition, dataset/protocol, conservative enhancement baseline and documented failure tradeoffs. **Novelty risk:** An application-specific gate alone is weak novelty. **Existing evidence:** Stored comparisons and same-recording temporal detector tests. **Missing:** Cross-recording results and valid perceptual/source evaluation.

**Baselines/ablations:** Current raw-plus-pristine mix, pristine-only anchor, energy/onset gate, coverage-matched gate, corrected oracle/automatic C1, generic separation, and visual/refinement ablations. **Data/listening:** Multiple players, songs, camera layouts, devices and arcades; blinded ratings of authentic-impact completeness and interference. **Must not claim:** General-purpose source separation or product-level preservation.

### B. Vision-conditioned reconstruction of authentic human–object sounds — too broad for current data

**Question:** Can vision condition extraction from the original recording while preserving real interaction acoustics? **Possible contribution:** Explicit source-provenance constraints and evaluation of lost interaction detail. **Novelty risk:** Both visually indicated sound modeling and visual separation have long precedents; forbidding foley does not by itself establish novelty. **Existing evidence:** Verified sample-origin gating. **Missing:** Multiple objects/actions, source references, and a mechanism beyond fixed gameplay overlays.

**Baselines/ablations:** Audio-visual separation, supervised time-frequency masks, temporal gates, hand-only versus display-only cues, oracle timing and no-reference variants. **Data/listening:** Real impacts and continuous contacts across objects and recording chains, with separate target measurements; authenticity and timbre assessment. **Must not claim:** General human–object reconstruction based on this one cabinet/player, or equivalence to physically isolated sound.

### C. Sparse interaction recovery with a pristine reference — conditional framing

**Question:** How much does a known reference help preserve target interactions while suppressing playback contamination? **Possible contribution:** Measuring when reference alignment improves event localization even when waveform cancellation is weak. **Novelty risk:** Reference cancellation and conditioned extraction are established families; the reference's incremental role must be demonstrated. **Existing evidence:** Alignment artifacts, failed cancellation variants, and reference-assisted onset analysis. **Missing:** Reference ablations and valid source-specific outcomes. Most studied windows are dense, making “sparse recovery” an inaccurate global description.

**Baselines/ablations:** No-reference gate, aligned-reference gate, shifted/wrong-reference controls, linear/adaptive cancellation, learned reference-conditioned extraction; sparse-to-dense and interference-overlap sweeps. **Data/listening:** Real recorded isolated impacts combined with measured playback/interference paths, plus real arcade validation; score weak-hit preservation and coincident contamination. **Must not claim:** That the reference solves in-window interference or that low coherence proves it useless.

### D. Audio-visual contact localization for preservation — promising only with stronger labels

**Question:** Which visual/acoustic cues predict audible physical interactions under occlusion and dense gameplay? **Possible contribution:** A contact dataset with uncertainty, localization baselines and downstream preservation evaluation. **Novelty risk:** Current features mainly read game effects; a learned temporal classifier over four signals may be incremental. **Existing evidence:** The plateau hypothesis was challenged, and burst splitting improves annotation-relative recall. **Missing:** Independent physical-contact labels, synchronized acoustic onset references and unseen-video evaluation.

**Baselines/ablations:** Display-only, hand-only, audio-only, fused heuristic and learned temporal models; fusion, burst, refinement, completion, and frame-rate ablations. **Data/listening:** New recordings with multiple players and non-scoring contacts, including blocked views and no-contact display activity; determine whether timing improvements preserve audible events. **Must not claim:** Four clean video-first training datasets, a measured 0.59 feature ceiling, or universally observable physical contacts at 60 fps.

### E. Strongest near-term framing: preservation–interference tradeoffs and evaluation validity

**Question:** When does contact-conditioned gating reduce interference without deleting authentic interactions, and which familiar proxies fail to measure that tradeoff? **Possible contribution:** A calibrated evaluation protocol, a benchmark with known source overlap, and evidence linking event timing, window duty cycle, source leakage and listening judgments. **Novelty risk:** A collection of implementation bugs is not a contribution; the measurement failure modes must persist after correction and be compared with prior evaluation work. **Existing evidence:** Oracle/automatic gaps, tolerance sensitivity, near-continuous exposure, shared-reference RMS ambiguity, and rest contamination. **Missing:** Corrected implementations and replication across recordings/methods.

**Baselines/ablations:** Always-open, music-only, energy/onset gating, random/time-shifted gates matched for coverage, oracle/automatic gates and learned masks; vary event density, overlap, boundary policy and reference level. **Data/listening:** Known real-impact sources with controllable interference plus independently held-out real sessions; test whether proposed measurements predict missing-impact and interference ratings. **Must not claim:** Novel source separation or general impossibility of competing routes.

This framing has the best current alignment with the evidence. A task-specific engineering report is already possible after correcting factual errors; a credible research paper requires substantial additional validation.

## Part 8 — Two independent roadmaps

### Roadmap A — Product, ordered by information value

1. **Freeze the historical evidence, not the defective release candidate.** Preserve E0, E2, C1 and all original outputs/annotations with versioned hashes. Record the exact intended versus implemented envelope and mask definitions. Correct the report/JSON discrepancies.
2. **Repair the smallest decisive foundation.** Correct fades and evaluator definitions in a new experimental version. Treat a corrected HPSS control as a bounded validity repair. Reevaluate source-preservation, known-rest exposure, and audio boundary behavior before choosing a production audio policy. No new heuristic search is needed to establish these basics.
3. **Run a compact independent-recording gate.** Use several genuinely new recordings spanning player/camera/density changes, including explicit rest and weak/continuous interactions. Add blinded listening. If authentic detail is lost or interference remains unacceptable even with reliable timing, changing only the detector will not solve the product problem.
4. **Then integrate only the validated conservative path.** E0 is the initial detector candidate; use the newly validated smooth reconstruction, fixed-gain comparisons, sample provenance, runtime diagnostics and preservation-oriented fallback to the existing mix. Revalidate full-track/chunk behavior, actual PTS and export headroom. Do not label corrected C1 as the old measured baseline.
5. **Expose a default-off beta only after that gate passes.** Keep E2 completion, new learned detectors and unsupported layouts research-only until they independently improve the preservation/interference tradeoff. Broaden device/venue testing before considering default-on behavior.

### Roadmap B — Research/paper, ordered by information value

1. **Repair measurement and identity first.** Define the target acoustic event separately from game judgment, motion, audible onset and annotation confidence. Correct evaluators and add contact/rest/uncertainty intervals. Reannotate at least a subset independently to estimate label omissions and inter-annotator disagreement.
2. **Acquire ground-truth leverage.** Capture synchronized close/contact microphones and camera audio, reference playback and identifiable interference. Use quiet or controlled sessions to measure genuine attack/tail preservation. Controlled mixtures may combine real recorded impacts and interference for evaluation; they must not be presented as synthetic replacement output. Include real mixtures because phone processing may violate simple additive mixing assumptions.
3. **Collect independent crossed recordings before training.** A practical pilot could use 12–20 new sessions covering multiple players and songs, at least several camera/device setups and two or more venues; this is a planning scale, not a sufficiency theorem. Use the pilot to estimate variance and plan the larger test. Reserve unseen recording/session groups; avoid confounding each player with a unique device or venue.
4. **Establish simple, diagnostic baselines.** Include current mix, pure-music anchor, always-open/raw, coverage-matched temporal controls, audio-only gate, corrected oracle gate, E0, E1/E2 and a representative separator. Test visual-only, display-only, hand-only, reference-free, refinement-free, burst-free and timing-shifted variants. Plot authentic-target retention against retained interference, not one aggregate score.
5. **Only then test a learned temporal detector.** Start with a modest temporal model and compare it against E0 on fixed recording-level splits. Keep display-only leakage and annotation bias controls. If oracle gating itself fails the perceptual/source target in dense overlap, prioritize extraction or capture changes rather than claiming detector learning will solve it.
6. **Validate perception and uncertainty.** Preregister primary preservation/interference outcomes, use blinded randomized listening with stems and fixed-gain final mixes, record participant expertise and equipment, and analyze at the recording/session level with listener dependence accounted for. Choose sample sizes from pilot effect sizes and variance. Use genuine equivalence/noninferiority margins only when the scientific question and sample size support them.

## 1. Research verdict

**C. Major methodological issue requires new experiments before trusting the result.**

The limited sample-gating observation survives. The broader claim that oracle reconstruction and automatic strong-contact preservation are product-ready does not: the frozen reconstruction is defective, the HPSS control is reversed, and source-specific/perceptual success is not measured reliably.

## 2. Production verdict

**D. Do not integrate yet.**

Retain E0 as the preferred detector for the next validation round. Repair and reevaluate the audio/evaluation foundation before R6; a default-off beta may follow successful independent-recording and listening checks.

## 3. Paper verdict

**B. Promising research seed, substantial validation still required.**

A preservation-versus-interference evaluation study is the strongest current direction. General authentic-source recovery, separator-class failure, and a fixed heuristic ceiling must not be claimed.

## 4. Top five actions

1. Preserve the historical freeze, then correct and independently verify C1 fades, C2 mask orientation and defective evaluation fields in a new version.
2. Establish independent target/interference evidence and physical-contact/rest annotations; separate confidence, audibility and force.
3. Collect genuinely new recordings and reserve recording/session-level tests spanning players, songs, devices, layouts and arcades.
4. Run blinded preservation-focused listening and coverage-matched baselines, including explicit rest, weak contacts, continuous interaction and dense overlap.
5. Choose between a conservative E0 beta, learned detection, or a different extraction/capture route from those results—not from the current product-readiness claims.
