# RTA-0 — Claim Correction Map

**Date:** 2026-09-15
**Companion to:** `RTA0_INTEGRITY_AUDIT.md` (evidence) and
`SENIOR_REVIEW_2026-09-15.md` (review record).

This maps every public wording that the audit found overstated to the exact
correction applied. Repair principle: **append, never erase** — dated
POST-HOC INTEGRITY NOTE banners; JSON records gain an amendment field;
historical sentences are left as written so provenance stays readable.

| ID | Location | Original wording (problem) | Correction applied |
|---|---|---|---|
| C1 | root `README.md` stage table, R1 row | "cancellation alone is dead" | Scope note appended in the row + banner in this repo's review docs: tested variants produced weak total-mixture reduction **on the selected recording**; no universal ceiling established |
| C2 | `s1e_existing_corpus_feasibility/RESULTS.md` §4 | "**no subtraction-style use of the reference can work**" | Banner: supported only for the tested waveform-cancellation configurations on the evaluated phone recording; not a general result |
| C3 | `s1e_existing_corpus_feasibility/README.md` | "reference *waveform* cancellation is proven dead on this phone recording"; "Provisional verdict: B" framing carried into later summaries as a class-level negative | Banner: scoped to the tested configurations and this recording; "the pretrained configurations successfully evaluated in this study did not meet the product requirement" is the supported form |
| C4 | `s1e_existing_corpus_feasibility/S1E_DECISION.json` | `"reference_cancellation_dead"` key (class-level phrasing) | Amendment field added; historical key untouched |
| C5 | `s1w_existing_corpus_adaptation/README.md` | "Root cause: proxy/synthetic-distribution overfit plus chunk/context sensitivity" (stated as uniquely proven cause) | Banner: distribution mismatch and context sensitivity are **supported explanations**; individual causal contributions were not isolated; the transfer failure itself is unchanged |
| C6 | `s1r_real_mixture_adaptation/S1R_REPORT.md` §3 heading + body | "a student cannot outlearn a teacher it only copies … now demonstrated" (general-law phrasing) | Banner: valid for **this strongly anchored adaptation recipe**; not a general scientific claim |
| C7 | `s1r_real_mixture_adaptation/S1R_REPORT.md` §intro, `s1r_real_mixture_adaptation/README.md` | "perceptually indistinguishable" (from one 18-item / 0-of-6 pass) | Banner: replaced by "one owner listening pass found no clear overall preference over zero-shot"; no equivalence claim is made |
| C8 | `s1r_real_mixture_adaptation/S1R_REPORT.md` §1 + `README.md` | training signal described as anti-collapse hinge + transient-band floor (2–9 kHz) | Banner: the transient-band component indexed the wrong STFT axis and was not implemented as intended (audit §3.3); the total-RMS hinge and all published evaluation metrics (which use a different, correct implementation) are unaffected |
| C9 | `s2a_reference_conditioned_real_mixture/README.md`, `S2A_REPORT.md`, `S2A_DECISION.json`, `S2A_STATUS.json` | headline "REFERENCE CONDITIONING DOES NOT HELP" | Current-status label: "**NO USEFUL REFERENCE-CONTENT EFFECT WAS DEMONSTRATED BY THE RECORDED S2A IMPLEMENTATION**" with the five caveats (adapter geometry; non-isolated S1R comparator; very small metric differences; no equivalence test or human listening; does not decide whether a correctly implemented architecture can help) |
| C10 | `s2a_reference_conditioned_real_mixture/ARCHITECTURE.md` | adapter presented as aligned local TF fusion; init-parity presented as validation | Banner: stride-2 transposed conv + crop breaks one-to-one TF geometry; init parity remains a valid **no-op** check only and does not validate learned TF alignment |
| C11 | `s2a_reference_conditioned_real_mixture/REAL_TEST_RESULTS.md`, `REFERENCE_ABLATION_RESULTS.md`, `GENERALIZATION_RESULTS.md`, `S2A_DECISION.json` | "S1R (frozen base)" / "advantage over S1R" | Banner: the comparator was the **post-training S2A base model** (its mask head trained with the adapter), not an immutable original-S1R instance; those deltas are not a valid frozen-original-S1R comparison; within-model correct/zero/wrong ablations remain valid |
| C12 | `s2a_reference_conditioned_real_mixture/S2A_DECISION.json` (`why_C_and_not_D`) | "The experiment is scientifically valid" written pre-audit | Amendment field: validity statement superseded by the audit findings (geometry + comparator); the within-model null effect stands as descriptive evidence |
| C13 | `s2a_reference_conditioned_real_mixture/S2A_REPORT.md` §2 table | "+0.013 dB median advantage over S1R" treated as a frozen-teacher comparison | Covered by C11 banner; numbers preserved as recorded |
| C14 | root `README.md` (S2A row + current conclusion) | headline + "best advantage over S1R" framing; no training-pause status | Row amended with scope label; TRAINING STATUS: PAUSED PENDING TRUSTWORTHY REAL-TARGET EVIDENCE added; next step = acquisition/evaluation, not another learner |
| C15 | `RESEARCH_WORKFLOW.md` | no authorized-sequence gate | Authorized sequence now starts with real-target acquisition → truth QC → frozen baselines → mask-oracle headroom → decision; **NO NEW MODEL TRAINING BEFORE THIS GATE** |

Not corrected (checked and found already scoped): S0 documents and
`independent_review_r0_r56` already phrase the cancellation result with
explicit scope limits; `s1e_existing_corpus_feasibility/FAILURE_CASES.md`
already says "on this recording".
