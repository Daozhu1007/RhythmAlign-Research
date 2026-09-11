# S1E — Model / Pipeline Matrix

**Rule applied:** bounded set (4 run candidates + 1 specialist), frozen pretrained
inference only, no fine-tuning, no vision, no chart gating. Access failure is recorded as
BLOCKED and never counted as model failure.

| # | System | Kind | Access status on this machine | Verdict this stage |
|---|---|---|---|---|
| 1 | **SAM Audio** (Meta, `facebook/sam-audio-small`) | unified audio segmentation/separation, text/span queries | **BLOCKED** — checkpoint gated `manual`; anonymous file resolve → HTTP 401 (re-verified 2026-09-10); no HF auth exists on this machine; gate not bypassed | not tested |
| 2 | **SoloAudio** (`westbrook/SoloAudio` v2; diffusion TSE) | generative, audio/text query ("enrollment") | **RAN** — public checkpoints, 592 MB + 549 MB, sha256 in RUN_MANIFEST | fails product bar (see below) |
| 3 | **FlowSep** (Audio-AGI) | generative LASS, flow matching | **BLOCKED** — checkpoint distributed only via Zenodo record 13869712; zenodo.org unreachable from this network (connection failure, not a model fault) | not tested |
| 4 | **FlowSep 2** | self-supervised flow matching LASS | **UNAVAILABLE** — demo page only; official README marks code "coming soon", no checkpoint exists | not released |
| 5 | **CLAPSep** (official multimodal-query checkpoint) | discriminative query separation (CLAP-conditioned masking) | **RAN** — official checkpoints already local from R3 (`best_model.ckpt`, `music_audioset_epoch_15_esc_90.14.pt`) | best candidate; real but modest selectivity |
| 6 | **AudioSep** (official base, 4M steps) | text-query separation (historical baseline, per brief) | **RAN** — checkpoint already local from R2 | replicates R2 failure mode |
| 7 | **S0-Specialist** (this stage; deterministic multi-stage DSP) | known-reference magnitude Wiener + VAD soft gate | **RAN** — fully local, deterministic | transparent but does not suppress enough |

Priority-2 of the brief ("strong modern generative/query separator such as SoloAudio /
FlowSep / FlowSep 2 where runnable") is satisfied by SoloAudio (runnable) with FlowSep /
FlowSep 2 recorded BLOCKED/UNAVAILABLE with evidence above.

## Fixed conditions (no per-run tuning)

- **Audio query (primary for CLAPSep and SoloAudio):** R3's `queries/query_q1.wav` —
  a single visually-confirmed player-contact event (0.113 s) selected in R3 by
  music-energy/transient ranking + frame-level visual confirmation. Reused unchanged.
- **Text query (AudioSep primary; CLAPSep/SoloAudio probes on c1–c3):** R2's best prompt
  "finger taps and button presses on a rhythm game cabinet" (p2).
- **CLAPSep negative condition:** zero embedding (official "empty" mechanism).
- **SoloAudio:** DDIM 50 steps, seed 2024 (deterministic), official v2 checkpoints,
  official inference path; output 24 kHz mono (model limitation).
- **S0-Specialist (all parameters fixed a priori):** per-clip local delay refine ±10 ms
  around R1's d* = −11.3747 s → sinc-warped stereo reference → STFT (2048/512, hann)
  magnitude-domain Wiener gain `G = max(−18 dB, sqrt(max(0, 1 − (|H|·|R|)²/|Y|²)))` with
  per-bin |H| = 5-bin-median-smoothed median |Y|/|R| over ref-active frames → Silero-VAD
  soft gate `1 − 0.7·p(t)` on a 32 ms grid. Rationale: measured waveform-cancellation
  ceiling on this recording is only ~0.4–0.8 dB (phone transfer phase-incoherent; R1
  measured 5.9 % coherent share), so magnitude-estimate suppression is used instead of
  subtraction. Not tuned against any listening result.

## Protocol notes / disclosed workarounds

- CLAPSep ran via the official Space protocol (32 kHz mono, 10 s chunks, 0.9-peak
  rescale undone after inference), using R3's vendored loader, read-only.
- AudioSep: official `chunk_inference` has an upstream boundary bug
  (`while current_idx + WINDOW < L` leaves the final partial window silent; a 5 s input
  produces all zeros). Disclosed workaround in S1E: pad 8 s of silence → infer → trim.
  Vendored code unmodified.
- SoloAudio outputs are **generative reconstructions** (disclosed on every result); they
  are judged by fidelity, not rejected for being generative.
- No output was gain-normalized in `outputs/audio/`. The listening pack alone is peak
  normalized to −3 dBFS for comparability, with per-item gains recorded in
  `listening_pack/listening_key.json`.
