# S1E — Results

**Stage:** existing-corpus AI extraction feasibility · **Date:** 2026-09-10
**Nature:** product-feasibility, NOT paper-grade evidence. No exact target stem exists;
no SI-SDR, no source-attenuation claims, no fake ground truth. All numbers below are
**proxies** with stated definitions. Human blind listening (`listening_pack/`) is the
primary decision evidence and is still **pending** — the decision in `S1E_DECISION.json`
is provisional with explicit listening gates.

## 1. Systems actually run

Four systems + one specialist pipeline ran (details, fixed conditions and disclosed
workarounds in `MODEL_MATRIX.md`): CLAPSep (audio query Q1, official checkpoint),
AudioSep (text p2, historical baseline), SoloAudio (generative, audio query Q1 + text
probe, seed 2024), S0-Specialist (deterministic reference-informed Wiener + VAD gate).
SAM-Audio BLOCKED (gated, HTTP 401 re-verified), FlowSep BLOCKED (Zenodo unreachable),
FlowSep 2 UNAVAILABLE (no code release).

## 2. Fixed-gain waveform + proxy diagnostics

Full data: `logs/09_diagnostics.json` (fields defined in `scripts/09_diagnostics.py`).
Abbreviations — `ret`: whole-output RMS change vs raw mixture (negative = energy
removed); `click`: click-band (2–9 kHz) level change — a **descriptive transient-level
proxy, NOT a contact-recall claim**; `rel_music`: music-frame level change minus whole
output change (relative music suppression on reference-active frames excluding ±150 ms
around transients); `rel_click`: click-band change minus whole change (positive =
transients relatively emphasized); `VAD s`: Silero speech-active seconds (nuisance proxy;
works only where speech is unmasked).

### c6_dense_golden (dense interaction, reference-covered — comparability with R2/R3)

| method | ret dB | click dB | rel_music dB | rel_click dB | VAD s |
|---|---|---|---|---|---|
| raw | 0.0 | 0.0 | 0.0 | 0.0 | 0.48 |
| CLAPSep | −3.7 | −0.3 | **−8.9** | **+3.4** | 0.00 |
| AudioSep | −4.8 | −1.1 | −19.8 | +3.7 | 0.00 |
| SoloAudio | −87.2 | −90.2 | (n/m) | (n/m) | 0.00 |
| specialist | −0.8 | −0.5 | −0.7 | +0.3 | 0.10 |

### All clips — relative suppression / preservation proxies (dB)

| clip | CLAPSep rel_music / rel_click | AudioSep rel_music / rel_click | specialist rel_music / rel_click |
|---|---|---|---|
| c1_speech_npc_a | +1.7 / +4.0 | −1.8 / −1.3 | +0.0 / +0.2 |
| c2_speech_npc_b | −9.3 / +3.7 | −5.8 / +3.2 | −0.7 / +0.3 |
| c3_taiko_prompt | −8.6 / +2.9 | −6.7 / +2.5 | −2.5 / +0.4 |
| c6_dense_golden | −8.9 / +3.4 | −19.8 / +3.7 | −0.7 / +0.3 |
| c7_weak_taps | −0.9 / +5.2 | — | −0.2 / +0.2 |
| c8_slide_friction | +2.7 / +3.8 | — | −0.5 / +0.2 |

(SoloAudio omitted — its outputs are −14 to −90 dB reconstructions; ratios are
meaningless. c1/c8 CLAPSep rel_music ≈ 0 reflects sparse eligible music frames there,
not music removal; see raw columns in the JSON.)

**Reading.** CLAPSep is the only system with a consistent signature across clips:
transients kept near full level (absolute click-band change −0.1…−1.1 dB) while music
frames drop ~9 dB relative on the strong cases (c2/c3/c6). AudioSep remains closer to a
uniform attenuator (whole-output ≈ −4.5 dB on every clip; R2's failure mode replicates),
with occasional larger music-frame drops of uncertain meaning. The specialist pipeline
is essentially transparent (≤0.8 dB overall change) — it suppresses far less than the
learned systems. No method produced onset-level hallucination evidence in the proxies;
SoloAudio's outputs are instead wholesale reconstructions (silence + invented blobs,
`diagnostics/panel_c3_taiko_prompt.png`, `panel_c6_dense_golden.png`).

## 3. Named challenge cases (per brief)

### 3.1 NPC / announcer speech region (owner-named 1:26–1:50; clips c1, c2)

- **Speech reduction:** Silero VAD is *blind* to this speech in the raw mixture
  (~0.00 s speech-active in both clips): it is heavily masked by music. VAD therefore
  cannot certify speech reduction here for any method. Speech-AM diagnostics and the
  spectrograms show no method removes the mid-band speech-formant content; CLAPSep's
  ~9 dB relative music suppression on c2 necessarily attenuates everything in
  music-overlapping bins, speech included, but this is attenuation, not speech removal.
  **Listening pack group c1/c2 is the only valid evidence.**
- **Interaction preservation:** CLAPSep keeps click-band level within −0.3 dB of raw on
  both clips (proxy — listening must confirm naturalness and weak-hit survival).
- **Artifacts:** none visible at proxy level for CLAPSep/specialist; AudioSep shows its
  usual uniform dulling; SoloAudio output on c1 is a faint unrelated reconstruction
  (corr with mixture 0.05, `logs/09_diagnostics.json`).
- Curious artifact flag: AudioSep's c1 output contains 0.26 s of VAD-detected
  speech-like content where the raw clip has none — either buried speech revealed by
  attenuation or a mild generative artifact; flagged for listening.

### 3.2 Loud Taiko system prompt (owner-named 2:08–2:09; clip c3)

- **Did any system materially suppress the prompt?** **No.** Best case (CLAPSep,
  AudioSep) the music-frame band around the prompt region drops ~9–11 dB relative to
  their whole-output change — audible attenuation, but the prompt remains clearly
  present in the spectrogram (`diagnostics/panel_c3_taiko_prompt.png`); nothing removes
  it as an event. SoloAudio replaces the clip with near-silence (−24 dB) plus an
  invented low-frequency blob at the clip tail — suppression by deletion, not
  extraction. The specialist's reference Wiener takes only ~3 dB (the prompt is a
  point-source interference case the magnitude estimate cannot isolate).
- **Did anything damage simultaneous interaction?** CLAPSep click-band −1.1 dB absolute
  (proxy: essentially intact); AudioSep −1.9 dB; specialist −0.4 dB; SoloAudio −43.5 dB
  (destroyed). Listening must confirm the perceptual side.

### 3.3 Clear announcer speech, no interaction (clip c5, negative control)

- CLAPSep reduces Silero speech-active time from **6.50 s (raw) to 1.34 s** — the only
  case in this stage where a pretrained system demonstrably suppresses speech as an
  event rather than attenuating uniformly. Cost: click-band −7.5 dB — but this clip
  contains no player interaction, so band suppression there is ambience removal, not
  target loss. Specialist VAD gate: 6.50 → 6.75 s (ineffective: a −5.5 dB level cut
  leaves VAD-confident speech VAD-confident). SoloAudio: 0.16 s — deletion, not
  extraction.

## 4. Reference-informed processing facts (bounds what any method can do here)

- Waveform cancellation of the aligned pristine music yields only **+0.4–0.8 dB**
  reduction on this phone recording (verified on three windows; R1 measured 5.9 %
  coherent share). Music reaching the phone mic is phase/magnitude-incoherent with the
  pristine file; **no subtraction-style use of the reference can work**, which is why
  the specialist uses a magnitude Wiener and why final mixes built from
  reference+residual (R1 A0 style) remain the only sane reference usage in production.
- Purely deterministic suppression (specialist) is far from sufficient on this material.

## 5. Listening pack (primary evidence — pending owner)

36 anonymized items (8 clips × {raw, CLAPSep, specialist, SoloAudio} + AudioSep on
c1/c2/c3/c6), level-normalized for comparability, key sealed in
`listening_pack/listening_key.json`. Instructions and rating sheet:
`listening_pack/LISTENING_INSTRUCTIONS.md`.

## Decision

See `S1E_DECISION.json`. Verdict **B (provisional)** — "learned extraction shows real
source selectivity, but the current pretrained frontier does not yet deliver a usable
real-world tradeoff; the failure mode here is *insufficient nuisance suppression*, not
target deletion." The listening pass upgrades this to **A** if the owner rates CLAPSep
items clearly better than raw across several groups with intact interaction, or
downgrades to **C** if no group shows a useful tradeoff. Rules:

- **→ A** if: on ≥ 3 clip groups the CLAPSep item is preferred overall AND interaction
  ratings ≥ raw's AND speech/prompt/ambience audibly reduced.
- **→ C** if: on ≥ 6 of 8 groups no method item is preferred over raw.
- **→ D** only if access/audio problems had prevented a fair test (not the case).
