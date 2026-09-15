# S2A — Reference-Conditioned Real-Mixture Extraction

> ## POST-HOC INTEGRITY NOTE (2026-09-15, RTA-0 audit)
>
> A post-hoc integrity audit (`docs/reviews/RTA0_INTEGRITY_AUDIT.md`) confirmed
> two implementation/comparator findings that **compromise the interpretation**
> of this stage's headline. The original text below is preserved unchanged.
>
> 1. **Adapter geometry:** the adapter's stride-2 transposed convolution doubles
>    both STFT axes before the output is cropped back to T×F
>    (`scripts/s2a_model.py`), so learned one-to-one TF alignment is NOT
>    structurally guaranteed. Initialization parity remains a valid no-op check
>    only; it does not validate learned TF alignment.
> 2. **S1R comparator not isolated:** the base model's final mask head was
>    trainable during S2A, and every post-training "S1R baseline" row was
>    computed from that same, updated base model — not from an immutable
>    original-S1R instance. The reported "advantage over S1R" numbers are
>    therefore not a valid frozen-original-S1R comparison. The within-model
>    CORRECT/ZERO/WRONG ablations remain valid descriptive evidence.
>
> **Current-status label (supersedes the headline below):**
> **NO USEFUL REFERENCE-CONTENT EFFECT WAS DEMONSTRATED BY THE RECORDED S2A
> IMPLEMENTATION.** Reference-condition metric differences were very small; no
> equivalence test or human listening established literal perceptual identity;
> and this experiment does NOT decide whether a correctly implemented
> reference-conditioned architecture can help.

**Stage question:** RhythmAlign natively operates with the time-aligned pristine
song the user provides — does that genuinely NEW information source let a model
move beyond the stable S1R baseline WITHOUT deleting authentic interaction,
collapsing, passing the mixture through, or memorizing songs?

**FINAL VERDICT (original 2026-09-15 wording): C — REFERENCE CONDITIONING DOES
NOT HELP.** The data gate
passed at preferred minimums (TRAIN 16 recs/14 songs, DEV 3/3, TEST 5/5
song-disjoint; 24/31 alignment pairs accepted conservatively), the adapter
was built and initialized exactly as designed (S1R parity exact, max|Δ| = 0.0),
training was safe at every checkpoint — and the causal reference effect simply
does not exist: correct, wrong, and zero references are behaviorally
indistinguishable (held-out median correct-vs-wrong gap 0.000 dB; best
suppression advantage over S1R 0.027 dB vs the 1.5 dB evidence bar). Section-26
stop: REFERENCE_ADAPTER_IGNORED; section-30 listening gate failed; **no human
listening was requested.** See [S2A_REPORT.md](S2A_REPORT.md),
[S2A_DECISION.json](S2A_DECISION.json).

## Method summary

1. **Reference audit + alignment** (read-only corpus): 22 in-tree pristine song
   references paired with raw handcams; production hybrid alignment + segment
   stability + song-identity controls; ambiguous pairs rejected, never forced.
2. **Minimal adapter** (protocol §15): 12k-param TF CNN on standardized
   mix+reference log-magnitude, zero-init output head acting as the residual
   gate, fused at the S1R mask logits. Frozen S1R base; trainable = adapter +
   final mask head (4.85%).
3. **Training** (protocol §20): S1R anchor + known-music injection invariance
   (x_aug = x + α·T(m), authentic mixture preserved) + cross-context consistency
   + anti-collapse + L2-SP. Two full runs (run 1 invalidated by a reference-
   position sign bug — recorded; run 2 corrected, identical hyperparameters).
4. **Evaluation:** DEV selection (§25) → causal reference-use gate (§26) →
   frozen held-out test with CORRECT/ZERO/WRONG ablations on 5 song-disjoint
   recordings (§27-29).

## Public deliverables (this directory)

| file | content |
|---|---|
| `REFERENCE_CORPUS_AUDIT.md` | reference availability + data gate (§8/11) |
| `REFERENCE_MANIFEST.public.json` | anonymized pairs, alignment statistics, promotions, rejections |
| `ALIGNMENT_AUDIT.md` | alignment method, acceptance criteria, failure modes |
| `ARCHITECTURE.md` | adapter design + initialization-parity evidence |
| `TRAINING_PLAN.md` / `TRAINING_CONFIG.json` | declared-a-priori objective and schedule |
| `TRAINING_LOG.public.md` | both runs + selection outcome (numbers; details in logs/) |
| `CHECKPOINT_MANIFEST.public.json` | hashes only; weights stay local |
| `DEV_RESULTS.md` | DEV trajectory + section-25 selection failure |
| `REFERENCE_ABLATION_RESULTS.md` | the mandatory CORRECT/ZERO/WRONG causal triad |
| `GENERALIZATION_RESULTS.md` | held-out song-disjoint transfer of the null effect |
| `REAL_TEST_RESULTS.md` | held-out machine comparison vs RAW/ZERO-SHOT/S1R |
| `FAILURE_CASES.md` | sign-bug record, mechanism analysis, honest negatives |
| `S2A_REPORT.md` / `S2A_DECISION.json` / `S2A_STATUS.json` | final records |

## Privacy / scope guards

Raw media, pristine copyrighted references, private manifests with local paths,
checkpoints and any listening material stay local under ignored `work/`,
`checkpoints/` (repo .gitignore). Recording and song identities are published as
stable anonymized IDs only. The product repository D:\Code\RhythmAlign was never
modified. No copyrighted audio left the local corpus.

## Prior-stage records (read-only inputs)

- `../s1r_real_mixture_adaptation/` — frozen S1R baseline (verdict B), the base
  checkpoint, phase-0 window conventions
- `../s1w_existing_corpus_adaptation/` — frozen split, corpus audit/provenance
  (private manifests), raw32k extractions
- `../r3_audio_query/` — vendored CLAPSep (read-only), query Q1, checkpoints
