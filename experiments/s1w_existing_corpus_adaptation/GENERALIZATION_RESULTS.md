# S1W — GENERALIZATION_RESULTS (post-freeze; descriptive only)

**Date:** 2026-09-12 · 28 song-active windows (2 per recording) from
14 TEST_GENERALIZATION identities (14
recordings held out from ALL training/selection/proxy/nuisance material). Zero-shot vs
ADAPTED only; these results were NOT used to retrain or choose a checkpoint. Fixed
gain; descriptive proxies only — no ground truth exists for these recordings.

| statistic (mean over windows) | raw | zero-shot | adapted |
|---|---|---|---|
| 2–9 kHz click-band level vs raw (dB) | 0.0 | -1.25 | -27.72 |
| sustained-friction flatness fraction | 0.307 | 0.404 | 0.374 |

Reading: click-band retention vs raw is the transient-preservation signal on unseen
recordings; the flatness fraction tracks whether sustained-friction texture survives.
Both methods reduce mid content; whether the adapted tradeoff sounds better is left to
a future small listening check (protocol section 37 — not requested now).
Per-window numbers: `logs/12_generalization.json` (local, machine-readable).
