# S1a blind listening instructions

**Read this BEFORE opening any files in `listening_pack/`. Do NOT open
`listening_key.json` until every rating is recorded.**

## What this is

A small diagnostic listening set for the S1a representation question:

> "Even if the separator knew the correct answer, does this output representation
> preserve the real interaction sound well enough?"

You will hear anonymized 5-second clips derived from exact constructed mixtures
(T2). For each mixture, up to five variants exist, but **the mapping from files to
variants is hidden** (fixed-seed shuffle; key stored only in `listening_key.json`):

- one is the raw mixture (input as-is);
- up to three are oracle-processed versions of that mixture;
- one is the TRUE target alone (anchor).

This pack contains **no trained model outputs** — every processed version had access
to the exact answer. If even the best-sounding processed versions lose important
target content versus the anchor, that is a representation failure, not a model
failure.

## What to rate

For each file, rate 0–5 (5 = indistinguishable from a perfect recording) on each
axis, independently:

1. **Completeness** — are all contact sounds fully present (nothing faint or missing)?
2. **Timbre fidelity** — does it sound like the same object/surface, same brightness?
3. **Transient strength** — do taps/hits keep their attack and punch?
4. **Friction continuity** — are slides continuous, without pumping/chopping?
5. **Decay/tail** — do releases and ringing die away naturally?
6. **Artifacts** — any musical-noise, smearing, "chopping", or doubled sounds? (0 = many)
7. **Residual nuisance** — any leftover music/speech/ambience? (0 = a lot)

Optionally name which files you believe belong to the same mixture.

## Playback rules

- Same device/headphones for all files; no volume changes between files (files share
  a fixed-gain convention — differences you hear are real differences).
- Listen in any order; replay freely; take breaks.
- Rate stems only; there are no remixes in this pack.

## Files

`listening_pack/X01.wav … XNN.wav` — each is a 5 s mono 32 kHz clip.
Typical pack: 16–20 files. Write ratings into any spreadsheet/table with
`file, completeness, timbre, transient, friction, decay, artifacts, nuisance`.

After all ratings are recorded, open `listening_key.json` and compute per-variant
averages. Do not re-rate after unblinding; do not discard any rating.
