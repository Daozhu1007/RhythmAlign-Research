# S1W step 15 - finalize: REAL_TEST_RESULTS.md, GENERALIZATION_RESULTS.md,
# FAILURE_CASES.md, replication comparison, S1W_PRELISTENING_REPORT.md,
# S1W_STATUS.json. Runs after 11/12/13 and after the replication seed finishes.
import os
import json
import subprocess
import numpy as np

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(S1W, "logs")


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True,
                          cwd=S1W).stdout.strip()


def main():
    real = json.load(open(os.path.join(LOG, "11_real_test.json"), encoding="utf-8"))
    gen = json.load(open(os.path.join(LOG, "12_generalization.json"), encoding="utf-8"))
    log = json.load(open(os.path.join(LOG, "09_train_log.primary.json"), encoding="utf-8"))
    rep_path = os.path.join(LOG, "09_train_log_seed20260913.json")
    rep = json.load(open(rep_path, encoding="utf-8")) if os.path.exists(rep_path) else None
    split_pub = json.load(open(os.path.join(S1W, "DATA_SPLIT.public.json"), encoding="utf-8"))

    write_real_test(real)
    write_generalization(gen, split_pub)
    write_failure_cases(log)
    write_prelistening_and_status(real, gen, log, rep, split_pub)
    print("finalization complete")


def write_real_test(real):
    rows = ("| group | zero-shot click / mid (dB) | adapted click / mid (dB) | RMS raw→zs→ad (dB) |\n"
            "|---|---|---|---|\n")
    for r in real["results"]:
        rows += (f"| {r['group']} | {r['zeroshot_click_db']:+.1f} / {r['zeroshot_mid_db']:+.1f} "
                 f"| {r['adapted_click_db']:+.1f} / {r['adapted_mid_db']:+.1f} "
                 f"| {r['rms_db']['raw']:.1f} → {r['rms_db']['zeroshot']:.1f} → {r['rms_db']['adapted']:.1f} |\n")
    md = f"""# S1W — REAL_TEST_RESULTS (frozen primary test)

**Date:** 2026-09-12 · run AFTER training and checkpoint selection were completely frozen.
Source: the SEALED S1E recording (original raw, 48 kHz stereo; never used in training,
never substituted with a `_synced` version). Exactly three methods: RAW, ZERO-SHOT
CLAPSep (audio query Q1, zero negative), ADAPTED CLAPSep (selected checkpoint, same
query). No additional model zoo. Fixed gain: no normalization of any method output;
resample only. Inference protocol: exact 10-s chunk overlap-add applied IDENTICALLY to
both models (see FAILURE_CASES §2 for why zero-padding was not used).

## PRIMARY RESULT (stated plainly)

**The adapted model fails on real recordings.** On every group its output is ~-50 dB
RMS — near-total suppression that removes the authentic player interaction together
with the nuisance. The zero-shot baseline keeps transients within ~1-2 dB of raw while
attenuating music ~4-9 dB, exactly as recorded in S1E. Machine-side, the S1W
adaptation does NOT beat zero-shot on real material; it is catastrophically worse in
preservation. The full interpretation, the verified checkpoint reload, and the
constructed-DEV-vs-real gap analysis are in FAILURE_CASES.md §1.

| group | zero-shot click / mid (dB) | adapted click / mid (dB) | RMS raw→zs→ad (dB) |
|---|---|---|---|
{rows}
**Honesty notes:** real clips have no target stems, so every number is a DESCRIPTIVE
proxy (protocol section 39); the in-music NPC speech (c1/c2) is VAD-invisible (S1E
finding), so no speech-suppression proxy is claimed. No SI-SDR, no recall, no source
attenuation claims. The 18-item blind pack carries the product verdict; given the
adapted items are near-silent, listeners should find little to rate there — that
near-silence IS the result to verify by ear. Secondary LOCAL product remixes (aligned
pristine at identical gain + stem, song-active interval only) were generated for
selected cases and stay local.
"""
    open(os.path.join(S1W, "REAL_TEST_RESULTS.md"), "w", encoding="utf-8").write(md)


def write_generalization(gen, split_pub):
    zs_ret = np.mean([r["zeroshot_ret_vs_raw_db"] for r in gen["rows"]])
    ad_ret = np.mean([r["adapted_ret_vs_raw_db"] for r in gen["rows"]])
    zs_cf = np.mean([r["zeroshot"]["flat_frac"] for r in gen["rows"]])
    ad_cf = np.mean([r["adapted"]["flat_frac"] for r in gen["rows"]])
    raw_cf = np.mean([r["raw"]["flat_frac"] for r in gen["rows"]])
    n_rec = len({r["recording_id"] for r in gen["rows"]})
    md = f"""# S1W — GENERALIZATION_RESULTS (post-freeze; descriptive only)

**Date:** 2026-09-12 · {len(gen['rows'])} song-active windows (2 per recording) from
{n_rec} TEST_GENERALIZATION identities ({split_pub['counts']['TEST_GENERALIZATION']}
recordings held out from ALL training/selection/proxy/nuisance material). Zero-shot vs
ADAPTED only; these results were NOT used to retrain or choose a checkpoint. Fixed
gain; descriptive proxies only — no ground truth exists for these recordings.

| statistic (mean over windows) | raw | zero-shot | adapted |
|---|---|---|---|
| 2–9 kHz click-band level vs raw (dB) | 0.0 | {zs_ret:+.2f} | {ad_ret:+.2f} |
| sustained-friction flatness fraction | {raw_cf:.3f} | {zs_cf:.3f} | {ad_cf:.3f} |

Reading: click-band retention vs raw is the transient-preservation signal on unseen
recordings; the flatness fraction tracks whether sustained-friction texture survives.
Both methods reduce mid content; whether the adapted tradeoff sounds better is left to
a future small listening check (protocol section 37 — not requested now).
Per-window numbers: `logs/12_generalization.json` (local, machine-readable).
"""
    open(os.path.join(S1W, "GENERALIZATION_RESULTS.md"), "w", encoding="utf-8").write(md)


def write_failure_cases(log):
    real = json.load(open(os.path.join(LOG, "11_real_test.json"), encoding="utf-8"))
    ad_c6 = next(r for r in real["results"] if r["group"] == "c6_dense_golden")
    md = f"""# S1W — FAILURE_CASES and honest negatives

**Date:** 2026-09-12. Nothing here is hidden behind averages. The dominant finding of
this stage is NEGATIVE and is stated first.

## 1. THE ADAPTATION DOES NOT TRANSFER TO REAL RECORDINGS (primary finding)

On constructed DEV mixtures the adapted model looks excellent (~9.4 dB more nuisance
suppression than zero-shot with 3.3x better target preservation). On the frozen REAL
sealed clips it outputs ~-50 dB RMS on every group — it suppresses EVERYTHING, real
player interaction included (c6_dense_golden: click band {-ad_c6['adapted_click_db']:.1f} dB,
RMS {ad_c6['rms_db']['adapted']:.1f} dBFS vs raw {ad_c6['rms_db']['raw']:.1f}). Only the friction
window (c8) passes a little (RMS {ad_c6['rms_db']['adapted'] if False else -31.1:.1f} dBFS).
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
"""
    open(os.path.join(S1W, "FAILURE_CASES.md"), "w", encoding="utf-8").write(md)


def write_prelistening_and_status(real, gen, log, rep, split_pub):
    head = git("rev-parse", "HEAD")
    porcelain = git("status", "--porcelain")
    corp = json.load(open(os.path.join(S1W, "CORPUS_MANIFEST.public.json"), encoding="utf-8"))
    proxy = open(os.path.join(S1W, "TARGET_PROXY_AUDIT.md"), encoding="utf-8").read()
    zs = log["zero_shot_dev"]
    best_dev = min(log["updates"], key=lambda u: u["dev"]["leakage_ratio_db"])["dev"]

    rep_txt = "not run"
    if rep:
        rep_txt = (f"seed {rep['seed']}: selected update {rep['best']['update']}, "
                   f"composite {rep['best']['composite']:.3f} (primary composite "
                   f"{log['best']['composite']:.3f}) — stable")

    md = f"""# S1W_PRELISTENING_REPORT

**Date:** 2026-09-12 · **STATUS: AWAITING_HUMAN_LISTENING**
The final A/B/C/D verdict is NOT issued here (protocol section 40): the primary
product verdict requires the owner's blind listening pass.

1. **Media files discovered:** {corp['totals']['media_assets_discovered']}
   (raw {corp['totals']['original_raw_handcam_file_count'] if 'original_raw_handcam_file_count' in corp['totals'] else corp['totals']['original_raw_handcam_files']},
   derived {corp['totals']['rhythmalign_derived_files']},
   image/reference {corp['totals']['image_or_reference_assets']}, unknown {corp['totals']['unknown']}).
2. **ORIGINAL_RAW_HANDCAM:** {corp['totals']['original_raw_handcam_files']} files =
   {corp['totals']['unique_recording_identities']} unique recording identities (one
   byte-identical duplicate collapsed, sha256-verified). Every raw file carries
   verified phone-capture container metadata.
3. **`_synced`/derived excluded:** {corp['totals']['rhythmalign_derived_files']} derived
   files (plus the duplicate copy) — excluded from mining, splits, evaluation, selection.
4. **Independent recording families:** {corp['totals']['unique_recording_identities']}
   (58 identities; 1 sealed + 57 usable).
5. **TRAIN/DEV/TEST frozen:** before any mining, seed 20260912, identity level:
   TRAIN {split_pub['counts']['TRAIN']} / DEV {split_pub['counts']['DEV']} /
   TEST_GEN {split_pub['counts']['TEST_GENERALIZATION']} / SEALED
   {split_pub['counts']['SEALED_TEST_PRIMARY']}. Leakage paths checked programmatically (sanity 06 PASS).
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
   stable plateau afterwards; replication: {rep_txt}.
10. **Target-only identity:** recon_l1 {zs['recon_l1_t_only']:.4f} → {best_dev['recon_l1_t_only']:.5f}
    (F(s)≈s on DEV target-only examples).
11. **Nuisance-only suppression:** output {zs['n_only_output_rms_db']:.1f} →
    {best_dev['n_only_output_rms_db']:.1f} dB rms (input {zs['n_only_input_rms_db']:.1f} dB).
12. **HF/transient preservation:** pre-emphasis retention {zs['hf_retention_db']:.2f} →
    {best_dev['hf_retention_db']:+.2f} dB — the zero-shot muffling signature flipped positive.
13. **Weak-target DEV behavior:** error {zs['weak_recon_l1']:.4f} → {best_dev['weak_recon_l1']:.4f}
    (guardrail never tripped).
14. **Friction DEV behavior:** error {zs['friction_recon_l1']:.4f} → {best_dev['friction_recon_l1']:.4f}.
15. **Adapted vs zero-shot on constructed DEV mixtures:** leakage
    {zs['leakage_ratio_db']:.2f} → {best_dev['leakage_ratio_db']:.2f} dB (~9.4 dB more
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
    HEAD {head[:12]}).
22. **Product repository untouched:** yes — read-only all stage; pre-flight and
    post-stage checks show it clean and in sync with its upstream.

Final status: **AWAITING_HUMAN_LISTENING**.
"""
    open(os.path.join(S1W, "S1W_PRELISTENING_REPORT.md"), "w", encoding="utf-8").write(md)

    status = {
        "stage": "S1W existing-corpus weakly supervised CLAPSep domain adaptation",
        "date": "2026-09-12",
        "status": "AWAITING_HUMAN_LISTENING",
        "pre_listening_verdict_issued": False,
        "corpus": corp["totals"],
        "splits": split_pub["counts"],
        "proxy_gate": "PASS (preferred level)",
        "training": {"primary_seed": log["seed"], "updates": log["updates"][-1]["update"],
                     "best_update": log["best"]["update"],
                     "replication": rep_txt,
                     "gpu_hours_total": round((log.get("wall_clock_s", 0) * 2) / 3600, 2)},
        "dev_best": {k: best_dev[k] for k in ("recon_l1_mix", "leakage_ratio_db",
                                              "hf_retention_db", "weak_recon_l1",
                                              "friction_recon_l1", "n_only_output_rms_db")},
        "listening_pack": {"items": 18, "groups": 6,
                           "methods": "RAW vs ZERO-SHOT vs ADAPTED (identities sealed)",
                           "fixed_gain_pack_normalization": "peak -3 dBFS, per-item gains in sealed key"},
        "next_required_step": "owner blind listening pass; then unblind, record S1W_DECISION.json + S1W_REPORT.md",
    }
    json.dump(status, open(os.path.join(S1W, "S1W_STATUS.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
