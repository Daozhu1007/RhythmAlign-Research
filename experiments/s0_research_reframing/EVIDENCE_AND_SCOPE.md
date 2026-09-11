# Evidence, scope and claim traceability

**Research date / cutoff: 2026-09-10.** This is a research study and execution specification. It does not contain a newly trained separator, new recording session, new blinded listening result or deployment change.

## Workspace work performed

Discovered the R1–R5-FIX experiment tree, read the independent review and validity-repair report, inspected selected corrected evaluation records and source code, inspected local model configuration/implementation and checkpoint availability, and queried the local GPU. The historical program was used as exploratory evidence. This was a bounded independent check, not another comprehensive R0–R5.6 re-audit.

The key report called “R5-FIX REPORT.md” in the request is stored as [REPORT.md](D:/Code/RhythmAlign/experiments/r5_fix_validity_repair/REPORT.md). The other primary history source is [INDEPENDENT_REVIEW.md](D:/Code/RhythmAlign/experiments/independent_review_r0_r56/INDEPENDENT_REVIEW.md). Existing reports, measurements and annotations are not treated as perfect truth. Selected claims were checked against code or decoded waveform arrays. No blinded listening key was inspected and no new subjective sound-quality judgment is reported.

The reproducible script [check_workspace_evidence.py](D:/Code/RhythmAlign/experiments/s0_research_reframing/scripts/check_workspace_evidence.py) writes [WORKSPACE_EVIDENCE_CHECKS.json](D:/Code/RhythmAlign/experiments/s0_research_reframing/WORKSPACE_EVIDENCE_CHECKS.json). It records hashes of eleven historical files and selected numerical checks; it reads historical artifacts and writes only its S0 receipt.

| Claim | Evidence level / source | Limit |
|---|---|---|
| R1 A1/A2/A3 total energy reductions approximately 0.0535/0.4735/0.2933 dB | Recomputed from saved stereo WAV arrays | Total mixture energy reduction, not isolated music attenuation |
| Corrected Holdout D E0 rest exposes 1.424729 s of the 2-s interval at gain > 0 | Recomputed stored envelope and waveform | Rest interval annotation is inherited; exposure does not identify content |
| E0 rest energy about −1.80753 dB relative to raw | Recomputed WAV energy | Mixed rest content, not a target-separation score |
| Corrected envelope/WAV agree within about 3e−8 amplitude | Recomputed multiplication consistency | Verifies saved gate implementation, not target authenticity |
| Corrected golden/holdout onset and rank counts | Read saved repaired metrics; not independently re-annotated | Audio/visual proxies, with known source ambiguity |
| Local CLAPSep uses a bounded mask and an inference-only no-grad wrapper | Inspected local source/config | Describes this checked implementation, not every CLAPSep version or neural separator |
| RTX 4060 Laptop, 8,188 MiB | Queried local `nvidia-smi` | No S1 training memory or throughput measured |

The receipt explicitly distinguishes recomputed quantities from `selected_saved_metrics_not_reestimated`. It is not an endorsement of every historical conclusion. Keep/archive/repurpose decisions in the study refer to research use; no historical files were deleted or rewritten.

## Literature method

Searched primary papers, author/project repositories, conference/journal pages and challenge documentation. Search scope covered target sound/speaker extraction, open-vocabulary and multimodal separation, known-reference/AEC methods, generative restoration, consistency constraints, self-supervision, foundation representations and evaluation. The date range emphasizes 2020–2026, with earlier work included when it directly challenges novelty or defines a necessary concept.

The [literature map](D:/Code/RhythmAlign/experiments/s0_research_reframing/LITERATURE_MAP.md) records task, conditioning, model/output, data, generative/consistency status, relevance, limitations and code availability. Ten recommended deep reads prioritize decisions, not citation counts. Recent 2026 preprints are labeled as such when publication was not verified. Screened work is not presented as fully reproduced.

“Code available” means a primary repository was found; it does not imply successful installation, permissive deployment rights or complete training reproduction. “Checkpoint available” reflects a visible release/link, not a downloaded and validated weight. “Not verified” is an evidence gap rather than a claim of nonexistence. Gated access and repository-specific licenses need checking at implementation time.

Published results are not transferred numerically to arcade audio. Resource ranges, architecture modules, loss weights, capture counts and success margins in the S0 plans are **proposals/engineering estimates**, explicitly distinct from literature measurements. No broad novelty claim follows merely because the search did not find an identical system.

## Reading order and deliverable boundaries

1. [Main study](D:/Code/RhythmAlign/experiments/s0_research_reframing/S0_RESEARCH_STUDY.md): formulation, identifiability, history, overall decisions and roadmap.
2. [Literature map](D:/Code/RhythmAlign/experiments/s0_research_reframing/LITERATURE_MAP.md): neighboring solutions and ten priority reads.
3. [Architecture options](D:/Code/RhythmAlign/experiments/s0_research_reframing/ARCHITECTURE_OPTIONS.md): four concrete routes and tradeoffs.
4. [Data and ground-truth plan](D:/Code/RhythmAlign/experiments/s0_research_reframing/DATA_AND_GROUND_TRUTH_PLAN.md): acquisition, truth tiers, weak supervision, evaluation and general domains.
5. [Minimal decisive experiment](D:/Code/RhythmAlign/experiments/s0_research_reframing/MINIMAL_DECISIVE_EXPERIMENT.md): one bounded next experiment, exact budget and stop rules.
6. [Paper memo](D:/Code/RhythmAlign/experiments/s0_research_reframing/PAPER_DIRECTION_MEMO.md): conditional contribution options and reviewer objections.
7. [Machine-readable decision](D:/Code/RhythmAlign/experiments/s0_research_reframing/S0_DECISION.json): compact selected route and next actions.

All work for this request is contained in `experiments/s0_research_reframing/`. The next stage is specified rather than silently launched: collecting new real-world audio and training models were not necessary to complete this S0 study.
