# RTA1 step 03 - truth QC gate.
#
# A take is NOT ground truth merely because it is quiet. This script measures
# clipping, noise floor, scored-active vs rest separation, duration/continuity
# and channel integrity, merges optional manual flags, and outputs per take:
#   PASS / REVIEW / FAIL  (+ TRUTH_QUALITY_INSUFFICIENT marker)
# with reasons. Low RMS alone never passes a take.
#
# Inputs:
#   ingest manifest (01)          - required
#   scored_regions.private.json   - optional; {"session/file": {"active": [[t0,t1],...],
#                                   "rest": [[t0,t1],...]}} seconds; when absent the
#                                   script derives provisional regions from the signal
#                                   (top-quartile frames = active, bottom-quintile = rest)
#                                   and marks the QC REVIEW accordingly
#   --manual-json                 - optional manual flags {"session/file": {"ok": bool, "notes": ...}}
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import rta1_lib as rl  # noqa: E402

FRAME_S = 0.5
CONTAMINATION_SEP_DB = 20.0     # required residual-contamination separation
CLIP_FAIL = 1e-3                # > 0.1 % clipped samples in any channel
CLIP_REVIEW = 1e-4
SILENT_DB = -80.0
DROPOUT_S = 2.0


def frame_rms_db(x, sr, frame_s=FRAME_S):
    n = int(frame_s * sr)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_frames = len(x) // n
    if n_frames == 0:
        return np.zeros((0, x.shape[1]))
    fr = x[: n_frames * n].reshape(n_frames, n, x.shape[1])
    return 20.0 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)


def qc_take(entry, path, regions, manual):
    x, sr = rl.decode_to_float32(path)
    reasons, flags = [], []
    res = {"session": entry["session"],
           "original_filename": entry["original_filename"],
           "sha256": entry["sha256"]}

    # clipping
    clip = rl.clipping_stats(x)
    res["clipping"] = clip
    if clip["clip_frac_max"] > CLIP_FAIL:
        reasons.append(f"clipping {clip['clip_frac_max']:.4%} > {CLIP_FAIL:.2%}")
    elif clip["clip_frac_max"] > CLIP_REVIEW:
        flags.append(f"clipping {clip['clip_frac_max']:.4%} (review level)")

    # decode/duration integrity
    dur_dec = len(x) / sr
    dur_meta = entry.get("duration_s") or 0.0
    res["duration_decoded_s"] = round(dur_dec, 2)
    res["duration_metadata_s"] = dur_meta
    if dur_meta and abs(dur_dec - dur_meta) / dur_meta > 0.05:
        reasons.append(f"duration mismatch {dur_dec:.1f}s vs {dur_meta:.1f}s")
    elif dur_meta and abs(dur_dec - dur_meta) / dur_meta > 0.02:
        flags.append("duration mismatch 2-5%")

    # channel integrity
    if x.shape[1] != (entry.get("channels") or x.shape[1]):
        reasons.append(f"channel count changed: decoded {x.shape[1]} "
                       f"vs metadata {entry.get('channels')}")
    per_ch_db = [round(rl.rms_db(x[:, c]), 2) for c in range(x.shape[1])]
    res["channel_rms_db"] = per_ch_db
    silent = [c for c, v in enumerate(per_ch_db) if v < SILENT_DB]
    if silent:
        reasons.append(f"silent channels {silent}")
    drop = []
    n_drop = int(DROPOUT_S * sr)
    for c in range(x.shape[1]):
        runs = np.abs(x[:, c]) < 1e-4
        if runs.sum() > n_drop:
            # contiguous silent runs longer than the dropout threshold
            idx = np.where(np.convolve(runs.astype(int), np.ones(n_drop, int),
                                       mode="valid") == n_drop)[0]
            if len(idx):
                drop.append({"channel": c, "n_positions": int(len(idx))})
    if drop:
        flags.append(f"possible dropouts {drop}")

    # noise floor + scored-active vs rest separation
    fr = frame_rms_db(x, sr)
    mono_fr = fr.max(axis=1)  # a frame is as loud as its loudest channel
    if regions and regions.get("active") and regions.get("rest"):
        act_mask = np.zeros(len(mono_fr), bool)
        rest_mask = np.zeros(len(mono_fr), bool)
        for t0, t1 in regions["active"]:
            act_mask[int(t0 / FRAME_S):int(t1 / FRAME_S)] = True
        for t0, t1 in regions["rest"]:
            rest_mask[int(t0 / FRAME_S):int(t1 / FRAME_S)] = True
        region_source = "declared"
    else:
        q_hi, q_lo = np.quantile(mono_fr, [0.75, 0.20])
        act_mask, rest_mask = mono_fr >= q_hi, mono_fr <= q_lo
        region_source = "derived-provisional"
    if act_mask.any() and rest_mask.any():
        if region_source == "declared":
            # declared spans contain quiet gaps between events; compare the
            # loud tail of active frames against the quiet tail of rest frames
            act_db = float(np.quantile(mono_fr[act_mask], 0.75))
            rest_db = float(np.quantile(mono_fr[rest_mask], 0.20))
            stat = "p75(active)/p20(rest)"
        else:
            act_db = float(np.median(mono_fr[act_mask]))
            rest_db = float(np.median(mono_fr[rest_mask]))
            stat = "median (already tail-selected)"
    else:
        act_db = rest_db = None
        stat = "n/a"
    floor_db = float(np.percentile(mono_fr, 5))
    res.update({"noise_floor_db": round(floor_db, 2),
                "scored_active_db": None if act_db is None else round(act_db, 2),
                "rest_db": None if rest_db is None else round(rest_db, 2),
                "region_source": region_source, "separation_stat": stat})
    if act_db is None or rest_db is None or not (act_mask.any() and rest_mask.any()):
        reasons.append("no usable active/rest regions to establish separation")
        res["contamination_separation_db"] = "NOT_ESTABLISHABLE"
    else:
        sep = act_db - rest_db
        res["contamination_separation_db"] = round(sep, 2)
        if sep < CONTAMINATION_SEP_DB:
            flags.append(f"contamination separation {sep:.1f} dB < "
                         f"{CONTAMINATION_SEP_DB:.0f} dB requirement")
            res["truth_quality"] = "TRUTH_QUALITY_INSUFFICIENT"
        if region_source == "derived-provisional":
            flags.append("regions were derived, not declared - manual review needed")

    if manual is not None:
        res["manual"] = manual
        if manual.get("ok") is False:
            reasons.append(f"manual review failed: {manual.get('notes', '')}")

    if reasons:
        res["verdict"] = "FAIL"
    elif flags:
        res["verdict"] = "REVIEW"
    else:
        res["verdict"] = "PASS"
    res["reasons"] = reasons
    res["review_flags"] = flags
    res.setdefault("truth_quality", "OK")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture-dir", default=None)
    ap.add_argument("--manual-json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.fixture_dir
    manifest_dir = os.path.join(root, "_manifests") if root else rl.PRIVATE
    inbox_root = root or rl.INBOX
    manifest = rl.load_json(os.path.join(manifest_dir,
                                         "ingest_manifest.private.json"))
    regions_path = os.path.join(manifest_dir, "scored_regions.private.json")
    regions_all = rl.load_json(regions_path) if os.path.exists(regions_path) else {}
    manual_all = rl.load_json(args.manual_json) if args.manual_json else {}

    if args.dry_run:
        for e in manifest["files"]:
            print(f"DRY-RUN would QC [{e['session']}] {e['original_filename']}")
        return 0

    rows = []
    for e in manifest["files"]:
        path = os.path.join(inbox_root, e["session"], e["original_filename"])
        key = f'{e["session"]}/{e["original_filename"]}'
        print(f"QC [{key}] ...", flush=True)
        rows.append(qc_take(e, path, regions_all.get(key), manual_all.get(key)))
        r = rows[-1]
        print(f"  -> {r['verdict']}"
              f"{' (' + r['truth_quality'] + ')' if r['truth_quality'] != 'OK' else ''}")

    out_dir = manifest_dir
    rl.save_json(os.path.join(out_dir, "truth_qc.private.json"),
                 {"fixture": bool(root), "takes": rows})
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    insufficient = sum(1 for r in rows
                       if r.get("truth_quality") == "TRUTH_QUALITY_INSUFFICIENT")
    print("verdicts:", counts,
          f"| truth-quality-insufficient: {insufficient}/{len(rows)}")
    if insufficient * 2 >= len(rows) and rows:
        print("TRUTH_QUALITY_INSUFFICIENT on half or more of the material - "
              "the pilot would end at DECISION_RULES Outcome D if this holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
