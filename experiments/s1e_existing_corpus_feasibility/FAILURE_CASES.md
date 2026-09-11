# S1E — Failure Cases and Negative Results

Every item below is a concrete, reproducible observation from this stage (log/JSON
references in parentheses). Historical experiments were not modified.

## 1. Pretrained generative extraction (SoloAudio) fails out-of-domain

- Setup: official `westbrook/SoloAudio` v2 checkpoints, official inference path, DDIM 50,
  seed 2024 (deterministic), audio query = R3's visually-confirmed contact exemplar,
  text probe = R2's p2 prompt (`logs/07_soloaudio_runs.json`).
- On its own demo pair the same path produces a sane extraction (output/mixture corr
  0.48, plausible level) — so the failure is domain, not plumbing.
- On our real clips: outputs are −14 to −90 dB below the mixture; c6/c7 are digital
  silence; c1/c3 are faint unrelated reconstructions (corr 0.045/0.060) with tonal junk
  (flatness 0.003); c3 shows an invented low-frequency blob at the clip end
  (`diagnostics/panel_c3_taiko_prompt.png`). As a product output this is "delete
  everything, sometimes invent something" — the worst possible tradeoff, and the
  clearest demonstration that 2024-era generative TSE does not transfer to handcam
  rhythm-game recordings. Fairness note (per brief): rejected on fidelity, not for
  being generative.

## 2. SAM-Audio remains inaccessible (not a model failure)

- `facebook/sam-audio-small` is gated `manual`; anonymous file resolve returns HTTP 401
  (re-verified 2026-09-10; `RUN_MANIFEST.access_status`). The machine has no HF
  account/token; per stage rules the gate was not bypassed. Same blocker R2 recorded.

## 3. FlowSep / FlowSep 2 could not be tested

- FlowSep v1: checkpoint only via Zenodo 13869712; zenodo.org is unreachable from this
  network (connection failure after retries) — BLOCKED (access), not judged.
- FlowSep 2: official repo is a demo page whose README states code is "coming soon";
  no checkpoint exists — UNAVAILABLE.

## 4. AudioSep's official long-audio path silently drops the clip tail

- Upstream `chunk_inference` loops `while current_idx + WINDOW < L`, so the final
  partial window is never written: our 5 s taiko clip returned **pure digital silence**
  (−219.6 dB, first run, `logs/06_audiosep_runs.json`), and 8–15 s clips had trailing
  silence. Disclosed workaround: pad → infer → trim (vendored code unmodified).
  Anyone reusing r2/r3 numbers on short clips should re-check this.

## 5. AudioSep's separation remains a non-specific attenuator on new material

- Replicating R2 on S1E clips: whole-output energy ≈ −4.4…−4.9 dB regardless of clip
  content (uniform dulling), target ≈ attenuated copy. On c1 its output additionally
  contains 0.26 s of VAD-detectable speech-like content absent from the raw clip
  (revealed-by-attenuation or mild artifact — unresolved, flagged for listening).

## 6. CLAPSep: real selectivity, insufficient absolute suppression (with one caveat)

- Genuine: transients kept at −0.1…−1.1 dB absolute while music frames drop ~9 dB
  relative (c2/c3/c6), consistent with R3's +5…+6 dB selectivity; and on the one clip
  where speech is VAD-visible (c5) it cuts speech-active time 6.50 → 1.34 s — the
  single strongest positive result of this stage.
- Caveat: R3 already found 1–2 hallucinated onsets on the golden window; S1E's proxies
  cannot rule onset-level hallucination in or out — this is exactly what the listening
  pass must judge. Absolute music suppression (~9 dB) is below what R3 judged usable.

## 7. The pristine reference cannot be waveform-cancelled on this recording

- Measured again on three windows (golden, NPC region, taiko region): best local-delay
  FIR cancellation removes only +0.41…+0.82 dB full-band, consistent with R1's
  cancellation ceiling (5.9 % coherent share). Root cause: phone-mic transfer of the
  cabinet music (reverb + phone `ai_audio` DSP + speaker nonlinearity) is
  phase-incoherent with the pristine file. Consequence: any pipeline idea that assumes
  reference subtraction (or residual + clean mix as "target+music") is dead on this
  corpus; the reference is only usable as a magnitude/statistical prior or in final-mix
  styling.

## 8. Raw cross-correlation cannot find the alignment (periodicity trap)

- On this music, unweighted full-lag cross-correlation of the 40–300 Hz band locks onto
  spurious periodicity peaks (e.g. ncc 0.99 at a lag 3.25 s away from the true delay,
  which cancels nothing), while the true delay yields only ~0.2 raw ncc. Alignment must
  anchor on R1's cancellation-objective result (d* = −11.3747 s) and refine only within
  a small window. Recorded so future stages don't re-derive a wrong offset from
  correlation.

## 9. Silero VAD is blind to music-masked arcade speech

- VAD probability ≈ 0 across the owner-named NPC region (c1/c2) and the taiko prompt
  clip, while firing strongly on the clear end-of-video announcer (c5). Two
  consequences: (a) VAD-gated suppression cannot address the in-music NPC nuisance;
  (b) VAD cannot certify speech reduction for c1/c2 — the specialist's speech stage is
  limited by exactly this. Any future speech-supervision needs its own (weak) labels.

## 10. Deterministic reference-informed suppression is transparent but weak

- The S0-Specialist (magnitude Wiener toward the warped reference + VAD gate) changes
  clips by ≤0.8 dB overall (≤3.3 dB on music frames; c5 speech effectively untouched).
  Honest negative: with phase-incoherent reference and VAD-blind speech, classical
  fixed processing cannot solve this material; learned/adapted models are genuinely
  needed — which is the core of this stage's verdict.
