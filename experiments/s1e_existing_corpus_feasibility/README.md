# RhythmAlign S1E — Existing-Corpus AI Extraction Feasibility

**Stage ID:** S1E · **Date:** 2026-09-10 · **Status:** inference + diagnostics complete;
blind listening pack ready; decision **provisional** pending the owner's listening pass
(see `S1E_DECISION.json`).

S1E answers exactly one product question and nothing more:

> Given existing real handcam recordings only, can any currently available modern
> separator / extraction pipeline (a) strongly suppress obvious non-target content,
> (b) preserve authentic player interaction sounds, and (c) sound materially better
> than the current raw/aligned baseline?

This is **product-feasibility**, not paper-grade evidence: no exact target stem exists,
so there is **no SI-SDR, no source-attenuation claim, no fabricated ground truth**.
No vision, no chart gating, no Holdout-D oracle. Phase 1 is frozen pretrained
inference only — no fine-tuning, no training, no new field capture.

## One-paragraph answer (proxies only — listening pending)

Of the bounded frontier, only **CLAPSep** shows a real, repeatable separation signature
on this material: player transients kept at ~full level (−0.1…−1.1 dB) while music
frames drop ~9 dB relative, plus the stage's strongest single result — clear announcer
speech cut from 6.5 s to 1.3 s VAD-active on the end-of-video clip. But ~9 dB is
attenuation, not removal: music and the 2:08 taiko system prompt remain clearly present.
AudioSep replicates R2's "uniform attenuator" failure. SoloAudio (generative) produces
silence or invented content out-of-domain. A deterministic reference-informed specialist
pipeline is transparent but weak (~0.5 dB), and reference *waveform* cancellation is
proven dead on this phone recording (+0.4–0.8 dB ceiling, phase incoherence).
SAM-Audio (gated) and FlowSep/FlowSep 2 (Zenodo unreachable / unreleased) were recorded
BLOCKED/UNAVAILABLE, not judged. Provisional verdict: **B — learned extraction shows
real selectivity but adaptation is needed**; the listening pack decides A/C/D.

## Deliverables map

| File | Purpose |
|---|---|
| [RESULTS.md](RESULTS.md) | Full proxy diagnostics, named-case reporting, decision rules |
| [FAILURE_CASES.md](FAILURE_CASES.md) | Ten concrete negative results (with references) |
| [CHALLENGE_CLIPS.md](CHALLENGE_CLIPS.md) | Clip list, selection basis, honesty notes |
| [MODEL_MATRIX.md](MODEL_MATRIX.md) | Systems, access status, fixed conditions, disclosures |
| [RUN_MANIFEST.json](RUN_MANIFEST.json) | Environment, checkpoint hashes, run logs, inventory |
| [S1E_DECISION.json](S1E_DECISION.json) | Provisional verdict + listening gates |
| [listening_pack/LISTENING_INSTRUCTIONS.md](listening_pack/LISTENING_INSTRUCTIONS.md) | Blind listening protocol (36 items) |
| [listening_pack/listening_key.json](listening_pack/listening_key.json) | Sealed anonymization key — do not open before rating |
| `clips/`, `outputs/`, `diagnostics/`, `logs/`, `scripts/`, `work/` | Media, results, figures, logs, code, intermediates |

## What was run (bounded frontier, fixed conditions)

1. **CLAPSep** — official checkpoint (local via R3), audio query Q1 + zero negative,
   official 32 kHz / 10 s-chunk protocol.
2. **AudioSep** — official base checkpoint (local via R2), text prompt p2; historical
   baseline; upstream short-input bug handled by disclosed pad→infer→trim.
3. **SoloAudio** — official v2 public checkpoints, DDIM 50, seed 2024; generative,
   disclosed as such.
4. **S0-Specialist** — this stage's deterministic multi-stage pipeline
   (reference-informed magnitude Wiener + VAD soft gate), all parameters fixed a priori.
5. BLOCKED/UNAVAILABLE (recorded with evidence): **SAM-Audio**, **FlowSep**, **FlowSep 2**.

## Primary test recording

`D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4` (owner-provided; read-only).
Aligned pristine reference: `D:\Daozh\Videos\舞萌手元\13.2\共感觉\共感觉.mp3`
(48 kHz stereo; alignment inherited from R1: d* = −11.3747 s, drift negligible).

## Environment

Runs used `experiments/r2_target_separation/.venv` (Python 3.12.13, torch 2.14.0+cu126,
RTX 4060 Laptop 8 GB) — the experiment venv created in R2 — extended with S1E
dependencies. The RhythmAlign production `.venv` was not touched. Details and hashes:
`RUN_MANIFEST.json`.

## How to reproduce

```bash
cd experiments/s1e_existing_corpus_feasibility
PY=D:/Code/RhythmAlign/experiments/r2_target_separation/.venv/Scripts/python.exe
$PY scripts/01_extract_and_timeline.py   # decode sources + timeline diagnostics
$PY scripts/02_region_zoom.py            # zoom diagnostics + candidate scans
$PY scripts/04_cut_clips.py              # cut the 8 challenge clips
$PY scripts/05_run_clapsep.py            # CLAPSep (aq all clips; tx probes c1-c3)
$PY scripts/06_run_audiosep.py           # AudioSep baseline (c1,c2,c3,c6)
$PY scripts/07_run_soloaudio.py          # SoloAudio (aq all clips; tx probes c1-c3)
$PY scripts/08_specialist.py             # deterministic specialist pipeline
$PY scripts/09_diagnostics.py            # proxy metrics -> logs/09_diagnostics.json
$PY scripts/11_build_pack.py             # rebuild listening pack + key
$PY scripts/10_spectrograms.py           # comparison panels after pack exists
```

## Rules respected

- No historical experiment modified; R1/R2/R3 files used read-only.
- Nothing committed or pushed; large audio kept out of Git staging via `.gitignore`.
- Gain policy: method outputs unnormalized in `outputs/audio/`; listening pack peak
  normalization only, with per-item gains in the sealed key.
- No synthetic Foley added to any output.
- No per-output normalization that hides deletion (pack normalization is peak-only and
  documented; raw outputs ship unmodified).
