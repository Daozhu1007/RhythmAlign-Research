# S1W — FAILURE_CASES and honest negatives

**Date:** 2026-09-12. Nothing here is hidden behind averages. The dominant finding of
this stage is NEGATIVE and is stated first.

## 1. THE ADAPTATION DOES NOT TRANSFER TO REAL RECORDINGS (primary finding)

On constructed DEV mixtures the adapted model looks excellent (~9.4 dB more nuisance
suppression than zero-shot with 3.3x better target preservation). On the frozen REAL
sealed clips it outputs ~-50 dB RMS on every group — it suppresses EVERYTHING, real
player interaction included (c6_dense_golden: click band 31.3 dB,
RMS -51.5 dBFS vs raw -20.3). Only the friction
window (c8) passes a little (RMS -31.1 dBFS).
The reloaded checkpoint reproduces the DEV numbers exactly (verified), so this is not a
save/load artifact: the model learned the SYNTHETIC mixture distribution
(canvas-reconstructed proxies vs pristine/attract/ambience nuisance) and not the
product target. Classic weak-supervision domain overfit. Machine-side, S1W therefore
fails its transfer goal; the listening pass will confirm what ears make of it.

## 2. The adapted masker is chunk-protocol sensitive

With the S1E clip protocol (zero-padding clips to 10-s multiples) the adapted model
PASSES REAL AUDIO THROUGH nearly unchanged (mask p95 = 1.0 on c7); with exact 10-s
chunks (the trained distribution, no padding) it near-totally suppresses. Zero-shot
CLAPSep is robust to both. All reported real numbers use exact-chunk overlap-add
inference applied IDENTICALLY to both models, but the sensitivity itself is a
robustness failure worth recording for any future masker work.

## 3. Additional honest negatives

1. **Speech nuisance is absent from TRAIN/DEV.** Best 8-s VAD speech fraction on any
   TRAIN/DEV recording is ~0.11; most have none. S1W could not train speech removal;
   c1/c2 outcomes are pure generalization.
2. **Friction proxies are scarce** (8 cuts, 7 recordings); friction is the only real
   class that partly survived adaptation (c8), consistent with its proxy cuts being
   the most real-like material in the bank.
3. **Proxy music bleed** (median pre-onset ~11-14 dB below event peaks) plausibly
   taught the model that quiet music belongs to the target — the opposite of the
   product need on real material.
4. **Bounded real mask unchanged** (deliberate): S1a's destructive-interference
   ceiling still applies to any future masker redesign.
5. **Two implementation failures occurred before the successful run** (a nuisance
   crossfade loop that hung one mixture build; a GPU-memory leak from the vendor
   model's persistent feature hooks that stalled one launch). Both fixed and
   documented; total GPU use ~1.1 h of the 24 h budget.
6. **A misleading training-log print** (accumulated sums labelled per-example) caused
   a long debugging detour; the run itself was healthy; published numbers are correct.
7. **No real-capture ground truth exists**, so no SI-SDR / recall / attenuation claims
   are made anywhere in S1W.
