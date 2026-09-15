# RTA-0 — Research Integrity Audit

**Date:** 2026-09-15
**Reviewed commit:** `3e6fcf3` ("research(s2a): evaluate aligned-reference conditioning"), HEAD of `main` at audit time.
**Trigger:** Independent senior research review supplied by the owner.
**Method:** Every code-level claim in the review was independently re-verified
against the actual implementation before any public record was changed. Review
claims were treated as hypotheses, not facts. All findings below carry the
verifying evidence (file, line, and where relevant a numerical check).

---

## 1. Files inspected (exact list)

Code (read in full):

- `experiments/s2a_reference_conditioned_real_mixture/scripts/s2a_model.py`
- `experiments/s2a_reference_conditioned_real_mixture/scripts/s2a_deveval.py`
- `experiments/s2a_reference_conditioned_real_mixture/scripts/s2a_losses.py`
- `experiments/s2a_reference_conditioned_real_mixture/scripts/s2a_common.py`
- `experiments/s2a_reference_conditioned_real_mixture/scripts/s2a_augment.py`
- `experiments/s2a_reference_conditioned_real_mixture/scripts/03_data_gate.py`
- `experiments/s2a_reference_conditioned_real_mixture/scripts/04_parity_check.py`
- `experiments/s2a_reference_conditioned_real_mixture/scripts/06_train.py`
- `experiments/s2a_reference_conditioned_real_mixture/scripts/09_real_test.py`
- `experiments/s1r_real_mixture_adaptation/scripts/s1r_losses.py`
- `experiments/s1r_real_mixture_adaptation/scripts/s1r_model.py`
- `experiments/s1r_real_mixture_adaptation/scripts/s1r_common.py`
- `experiments/s1r_real_mixture_adaptation/scripts/03_train.py`

Local data (split verification only; not published):

- `s1w_existing_corpus_adaptation/work/private/DATA_SPLIT.private.json`
- `s2a_reference_conditioned_real_mixture/work/private/S2A_SPLIT.private.json`
- one teacher anchor `.npy` (shape check)

Public record (wording audit):

- root `README.md`, `RESEARCH_WORKFLOW.md`
- `experiments/s2a_reference_conditioned_real_mixture/`: `README.md`,
  `S2A_REPORT.md`, `S2A_DECISION.json`, `S2A_STATUS.json`, `ARCHITECTURE.md`,
  `REAL_TEST_RESULTS.md`, `REFERENCE_ABLATION_RESULTS.md`,
  `GENERALIZATION_RESULTS.md`, `FAILURE_CASES.md`
- `experiments/s1r_real_mixture_adaptation/`: `README.md`, `S1R_REPORT.md`
- `experiments/s1w_existing_corpus_adaptation/`: `README.md`
- `experiments/s1e_existing_corpus_feasibility/`: `README.md`, `RESULTS.md`,
  `FAILURE_CASES.md`, `S1E_DECISION.json`

Numerical verification performed:

- Teacher anchor arrays fed to the S1R/S2A losses are **1-D** `(192000,)`
  float32 (verified on `work/audio/teacher/recording_0001/w111.59_canonical.npy`).
- `torch.stft` (torch 2.14, CPU) on a 1-D tensor of 192 000 samples,
  `n_fft=1024, hop=320` returns shape **(513, 601)** — i.e. `(freq, time)`;
  `S.size(1) == 601` is the **time** axis. The `linspace(0, 16000, 601)` mask
  with the `(>=2000) & (<=9000)` condition selects **263 slices, indices
  75–337** — a contiguous middle block of **time frames**, not frequency bins.

---

## 2. Review-claim verification results

| # | Review claim | Verdict | Evidence |
|---|---|---|---|
| 1 | S2A adapter ConvTranspose2d stride=2 expands the representation, then crops back to T×F, breaking one-to-one aligned TF geometry | **CONFIRMED** | `s2a_model.py:47` `ConvTranspose2d(ch, mid, kernel_size=4, stride=2, padding=1)` doubles both spatial axes; `s2a_model.py:56-57` computes `d` on the ~2T×2F grid then returns `d[:, :, :T, :F]` |
| 2a | `base_model.decoder_model.mask_net` was trainable during S2A | **CONFIRMED** | `s2a_model.py:27-30` `TRAINABLE_PREFIXES` includes `"base_model.decoder_model.mask_net."`; `06_train.py:184-200` puts all `requires_grad` params (adapter + base mask head) in the AdamW optimizer; `04_parity_check.py:154` explicitly verifies the mask head receives gradient |
| 2b | The reported "S1R baseline" in S2A DEV/TEST reused the post-training S2A `base_model` instead of an immutable original S1R model | **CONFIRMED** | `s2a_deveval.py:138` `s1r_aug = _s1r_forward(model.base_model, …)`; `09_real_test.py:141` and `:156` `y_s1r = si.ola_separate(s2a.base_model, …)` / `s1r_aug = si.ola_separate(s2a.base_model, xa_full, …)`. `s2a.base_model` is the same in-memory model whose `mask_net` was updated by S2A training. No post-training evaluation path reloads `s1r_selected.ckpt` (a fresh S1R load exists only in the pre-training `04_parity_check.py`) |
| 3 | The transient-protection loss indexes the wrong STFT axis; the documented 2–9 kHz guard was not implemented as intended | **CONFIRMED** | `s1r_losses.py:52-58`: inputs are 1-D waveforms, so `torch.stft` returns `(freq, time)`; `S.size(1)` is time (601 frames), and the "2–9 kHz" boolean mask selects time frames 75–337 (a middle ~44 % time block), with the sum then leaving per-frequency-bin values. The subsequent top-k therefore selects bins, not transient frames. See numerical check above |
| 4a | S2A DEV contains recording identities previously used in S1R TRAIN | **CONFIRMED (documented in-stage)** | `03_data_gate.py:23-26` promotes `recording_0027`, `recording_0052` from the S1W/S1R TRAIN split to S2A DEV; both IDs verified present in `DATA_SPLIT.private.json` `train` |
| 4b | S2A DEV songs overlap S2A TRAIN songs | **CONFIRMED (documented in-stage)** | DEV songs {song_10, song_21} both present in TRAIN songs; the stage documented this ("song stays in TRAIN via sibling recordings") |
| 4c | DEV wrong-reference selection uses TEST song references | **CONFIRMED** | `s2a_deveval.py:36-38` `wrong_pool = [p for p in pairs if p["s2a_split"] == "TEST"]` |
| 5 | Public wording overstates several conclusions | **CONFIRMED** | Specific locations and replacement wordings: see `RTA0_CLAIM_CORRECTION_MAP.md` |

No review claim was found to be NOT_CONFIRMED. One nuance was added by this
audit (§3.3): the evaluation-side transient metrics used a *different*,
correct implementation, so measured eval numbers are unaffected by the loss
axis bug.

---

## 3. Confirmed implementation defects

### 3.1 S2A adapter geometry is not one-to-one TF-aligned (interpretation defect)

`RefAdapter.forward` (`s2a_model.py:52-57`) runs a stride-2 transposed
convolution that doubles **both** time and frequency axes, applies the output
head on the ~2T×2F grid, and crops back to T×F. The delta mask added to the S1R
mask logits is therefore *sampled* from an internally upsampled coordinate
frame, not computed bin-for-bin in the mask's coordinate frame. Consequences:

- The stage's framing of "aligned local reference fusion" (module docstring)
  is not structurally guaranteed by the implementation.
- **Initialization parity remains a valid no-op check** (the zero-init output
  head forces delta ≡ 0 regardless of geometry; `04_parity_check.py` verified
  max|Δ| = 0.0). Parity validates "the adapter starts as a no-op", NOT "learned
  TF alignment is correct".
- S2A cannot be treated as a clean test of aligned local reference fusion.
- **The causal contribution of this geometry to the null result was not
  measured** and must not be quantitatively claimed. A sufficiently flexible
  conv stack can still learn useful functions; the defect compromises
  *interpretation*, not necessarily expressivity.

### 3.2 S2A "S1R baseline" was not an immutable original-S1R comparator

S2A training updated the base model's final mask head (`TRAINABLE_PREFIXES`
includes `base_model.decoder_model.mask_net.`), and every post-training
"S1R" row — DEV `resid_s1r_rms_db` / `suppression_adv_vs_s1r_db`
(`s2a_deveval.py:138`) and held-out `y_s1r` / `s1r_aug`
(`09_real_test.py:141,156`) — was computed from that same, now-updated
`base_model` object, not from a freshly loaded `s1r_selected.ckpt`.
Consequences:

- The reported "suppression advantage over S1R" (+0.0074 dB DEV-best,
  +0.013 dB held-out median, +0.027 dB max) is **S2A-vs-its-own-trained-base,
  not a frozen original-S1R comparison**. The label "S1R (frozen base)" in
  `REAL_TEST_RESULTS.md` is inaccurate.
- **Correct-vs-zero / correct-vs-wrong comparisons inside the same S2A model
  remain valid descriptive evidence** about reference-condition sensitivity of
  the recorded implementation — all three modes run the identical weights, so
  the 0.000 dB correct-vs-wrong gap is unaffected by this defect.
- Mitigating context (recorded, not excusing): the L2-SP weight anchor pulls
  the mask head toward its S1R initial values, and the within-model ablations
  show the adapter contributes ≈0 dB, so the *quantitative* distortion of the
  headline numbers is plausibly small — but it was not measured, and the
  comparison as reported is not the valid frozen comparison.

### 3.3 Transient-protection loss term indexed the wrong STFT axis

`frame_band_energy` (`s1r_losses.py:52-58`) documents "per-frame 2–9 kHz band
energy". For the actual 1-D inputs, `torch.stft` returns `(freq, time)`, so
`S.size(1)` is time. The "2–9 kHz" mask selects a middle block of **time
frames** (indices 75–337 of 601) and the reduction leaves per-**bin** values;
the following top-k then selects the top 20 % of those bins. The implemented
quantity is a per-frequency-bin spectral floor over the mid-window time region
— **not** the documented 2–9 kHz frame-localized transient guard.

Scope of use (verified by grep; no other users exist):

- **S1R training** (`03_train.py:113-115`, 3 calls/step).
- **S2A training** (`06_train.py:101-102`, via `s2a_losses.py` re-import,
  2 calls/step).
- **No evaluation path.** `04_dev_eval.py` imports `s1r_losses` but never calls
  it; all published transient metrics (`click_band_db` in `s1r_common.py`,
  `band_db` in `09_real_test.py`) use librosa with an explicit, correct
  frequency-axis mapping.

Validity notes:

- The **total-RMS energy hinge** in `anti_collapse` is a separate, correctly
  implemented term and remains valid.
- The defective term still acted as a (different, weaker) spectral floor; it
  did not reward louder output and co-existed with anchors. No result is
  invalidated by this defect alone; but no report may credit the S1R/S2A
  training signal with "a 2–9 kHz transient guard", and the transient
  protection actually in effect during training was weaker and differently
  shaped than documented.

### 3.4 Split / reference exposure (classification, not a training leak)

Verified facts: S2A DEV includes two ex-S1R-TRAIN recording identities
(documented in-stage as a split adaptation); S2A DEV songs overlap S2A TRAIN
songs (documented in-stage); DEV wrong-reference selection drew from the five
TEST songs' references.

Correct classification: **test-side reference-content exposure during model
selection** — plus a documented DEV/TRAIN song-overlap relaxation. It is
**NOT gradient leakage**: no TEST mixture, and no TEST content of any kind,
entered any training gradient (training used only S2A-TRAIN windows and their
own recordings' references; wrong references were never used in training at
all). Mitigating facts: section-25 checkpoint selection failed anyway (no
checkpoint passed), and the held-out test paired each window with a wrong
reference from a *different* TEST song than the one evaluated
(`09_real_test.py:126`). The exposure is recorded as a methodological risk
that did not demonstrably change the outcome.

---

## 4. Conclusions affected / not affected

### Classification legend

- **INVALID** — the defect makes the conclusion unusable.
- **COMPROMISED** — the conclusion must be re-scope/relabelled; the underlying
  measurement exists but the comparison or interpretation as published is not
  valid as stated.
- **LIMITED** — the evidence supports the conclusion only in a narrower scope
  than the wording implies.
- **UNCHANGED** — the defect does not reach the conclusion.

### Per-stage status

| Stage / conclusion | Status | Reason |
|---|---|---|
| R1 "cancellation alone is dead" | **LIMITED** | Measured on the selected recording; scalar-coherence ceiling does not establish a universal cancellation ceiling (already argued in S0/INDEPENDENT_REVIEW; wording in root README/S1E still overreaches) |
| S1E "no subtraction-style use of the reference can work" | **LIMITED** | Supported only for the tested waveform-cancellation configurations on the evaluated phone recording |
| S1E "current pretrained frontier insufficient" | **LIMITED** | True only of the configurations successfully evaluated in the study |
| S1W "adaptation fails on real transfer" (the result itself) | **UNCHANGED** | Not touched by any defect found here (S1W used its own loss library) |
| S1W "root cause: synthetic-proxy distribution overfit + context sensitivity" | **LIMITED** | Supported explanation; individual causal contributions were not isolated |
| S1R stage result: real-mixture adaptation stable, no clear product gain | **UNCHANGED** | Based on evaluation-side metrics (correct-axis implementations), sealed RMS comparisons, and the blind listening pass — none used the defective loss term |
| S1R training-signal description "anti-collapse hinge + transient-band floor protects 2–9 kHz transients" | **COMPROMISED** | The transient-band component was not implemented as intended (§3.3); the total-RMS hinge and the published eval metrics stand |
| S1R interpretation "a student cannot outlearn a teacher it only copies" (as a general claim) | **LIMITED** | Valid for this recipe; not a general scientific law; "perceptually indistinguishable" overstates one 6-item listening pass |
| S2A within-model reference ablations (correct ≈ zero ≈ wrong, 0.000 dB) | **UNCHANGED** (as descriptive evidence) | Same-weights comparisons; the strongest surviving evidence of the stage |
| S2A "suppression advantage over S1R" numbers (+0.007/+0.013/+0.027 dB) | **COMPROMISED** | Comparator was the post-training base model, not an immutable S1R instance (§3.2) |
| S2A headline "REFERENCE CONDITIONING DOES NOT HELP" | **LIMITED** → relabelled | Applies only to the recorded implementation; geometry (§3.1) + comparator (§3.2) compromise interpretation; no equivalence test or listening established perceptual identity; the experiment does not decide whether a correctly implemented reference-conditioned architecture can help |
| S2A "the experiment is scientifically valid" (`why_C_and_not_D`) | **COMPROMISED** | Written before the defects above were known; amended |

### Was any historical result invalidated?

**No.** No historical negative result is deleted or invalidated. Two S2A
result families are downgraded (comparator COMPROMISED; headline LIMITED), one
training-signal description in S1R/S2A is corrected (COMPROMISED), and several
wordings are scope-narrowed (LIMITED). The preserved core chain — S1W fails to
transfer, S1R is stable without product gain, S2A's recorded implementation
shows no reference-content effect — stands.

---

## 5. Proposed public wording changes

Executed in this repair (see `RTA0_CLAIM_CORRECTION_MAP.md` for the full
location-by-location map). Principle: **append, do not erase** — dated
"POST-HOC INTEGRITY NOTE" banners; historical text untouched; JSON decision
records extended with an amendment field, never rewritten.

Headline replacements:

- S2A: "REFERENCE CONDITIONING DOES NOT HELP" → current-status label
  "**NO USEFUL REFERENCE-CONTENT EFFECT WAS DEMONSTRATED BY THE RECORDED S2A
  IMPLEMENTATION**", with the five mandatory caveats (geometry, comparator,
  tiny metric differences, no equivalence/listening evidence, does not decide
  the general question).
- R1/S1E cancellation: "dead" → "the tested cancellation variants produced
  weak total-mixture reduction on the selected recording; this does not
  establish a universal cancellation ceiling."
- S1E frontier: → "the pretrained configurations successfully evaluated in
  this study did not meet the product requirement."
- S1W root cause: → "distribution mismatch and context sensitivity are
  supported explanations; their individual causal contributions were not
  isolated."
- S1R: → "this strongly anchored adaptation recipe improved context
  consistency without demonstrating a clear product gain over the teacher";
  "perceptually indistinguishable" → "one owner listening pass found no clear
  overall preference over zero-shot."

---

## 6. Audit integrity notes

- The product repository (`D:\Code\RhythmAlign`) was not modified.
- No historical report was deleted or silently rewritten; every change is an
  appended, dated note, an added JSON field, or a root-README status update.
- The transient-loss defect is fixed **only** going forward, inside the RTA1
  pilot tooling (which carries explicit STFT-axis tests). The historical S1R /
  S2A training scripts are preserved as-is; they are records of what ran.
- No model training, no human listening, and no product changes were performed
  during this audit.
