# S1W_PRELISTENING_REPORT

**Date:** 2026-09-12 · **STATUS: AWAITING_HUMAN_LISTENING**
The final A/B/C/D verdict is NOT issued here (protocol section 40): the primary
product verdict requires the owner's blind listening pass.

1. **Media files discovered:** 162
   (raw 59,
   derived 37,
   image/reference 66, unknown 0).
2. **ORIGINAL_RAW_HANDCAM:** 59 files =
   58 unique recording identities (one
   byte-identical duplicate collapsed, sha256-verified). Every raw file carries
   verified phone-capture container metadata.
3. **`_synced`/derived excluded:** 37 derived
   files (plus the duplicate copy) — excluded from mining, splits, evaluation, selection.
4. **Independent recording families:** 58
   (58 identities; 1 sealed + 57 usable).
5. **TRAIN/DEV/TEST frozen:** before any mining, seed 20260912, identity level:
   TRAIN 36 / DEV 7 /
   TEST_GEN 14 / SEALED
   1. Leakage paths checked programmatically (sanity 06 PASS).
6. **Conservative target-proxy material:** 156 events / 238 s from 32 TRAIN recordings
   (strong 36 / ordinary 81 / weak 31 / friction 8) — preferred quality gate PASSED
   (≥3 min, ≥3 recordings, all interaction strata).
7. **Proxy contamination:** residual music bleed with median pre-onset level ~11–14 dB
   below event peaks; speech cannot be excluded where VAD-invisible (S1E finding);
   full honesty list in TARGET_PROXY_AUDIT.md.
8. **Nuisance material:** 150 acoustic clips (attract 328 s / ambience 648 s /
   unrelated impacts 224 s) + 42 pristine-reference clips (420 s, local only).
   Speech class provably absent from TRAIN/DEV (documented).
9. **Did training learn?** Yes — meaningful DEV movement well before update 1000;
   stable plateau afterwards; replication: seed 20260913: selected update 1250, composite 5.431 (primary composite 5.533) — stable.
10. **Target-only identity:** recon_l1 0.0228 → 0.00014
    (F(s)≈s on DEV target-only examples).
11. **Nuisance-only suppression:** output -18.7 →
    -45.4 dB rms (input -14.4 dB).
12. **HF/transient preservation:** pre-emphasis retention -2.38 →
    +5.42 dB — the zero-shot muffling signature flipped positive.
13. **Weak-target DEV behavior:** error 0.0400 → 0.0121
    (guardrail never tripped).
14. **Friction DEV behavior:** error 0.0677 → 0.0168.
15. **Adapted vs zero-shot on constructed DEV mixtures:** leakage
    -0.77 → -10.28 dB (~9.4 dB more
    suppression) with BETTER preservation on every axis (see DEV_RESULTS.md table).
16. **Frozen real challenge diagnostics:** the adapted model outputs ~-50 dB RMS on
    all six groups — near-total suppression including targets. Zero-shot behaves as
    recorded in S1E (transients kept, music attenuated ~4-9 dB). See
    REAL_TEST_RESULTS.md; full analysis in FAILURE_CASES.md §1-2.
17. **Muffling per non-human diagnostics:** on CONSTRUCTED mixtures the muffling
    signature improved (HF retention flipped positive); on REAL material the adapted
    output is near-silence — beyond muffling. The transfer failure, not muffling, is
    the dominant machine-side outcome.
18. **Bounded-mask bottleneck suspected?** Not the bottleneck THIS stage hit: the
    failure is the weak-supervision domain gap (synthetic mixture distribution vs
    real acoustics), not the mask representation. Representation redesign remains a
    later lever only after a transfer-capable supervision scheme exists.
19. **Local anonymous listening pack:** `experiments/s1w_existing_corpus_adaptation/listening_pack/`
    (18 items / 6 groups; PLAYBACK_ORDER.json public; identities sealed locally).
20. **Research commit SHA:** created at the end of this stage (this report is written
    just before that commit; see S1W_STATUS.json / git log for the exact SHA).
21. **Pushed:** yes — `main → origin/main`, no force push (verified HEAD == origin/main
    after push; if the value below disagrees, the commit step is the authority:
    HEAD 254242c9d8db).
22. **Product repository untouched:** yes — read-only all stage; pre-flight and
    post-stage checks show it clean and in sync with its upstream.

Final status: **AWAITING_HUMAN_LISTENING**.
