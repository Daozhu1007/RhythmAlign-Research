# S1R step 01 - PHASE 0: REAL-MIXTURE TEACHER VIABILITY AUDIT
# (protocol sections 10-16).
#
# Frozen zero-shot CLAPSep teacher on AUTHENTIC REAL handcam mixtures from the
# frozen S1W TRAIN + DEV identities ONLY. For the same central ~6 s real region
# the teacher runs under three 10-s context views (early/canonical/late) and
# under +-3 dB global gain (corrected before comparison). Multi-metric
# consistency diagnostics decide HIGH_CONFIDENCE_TEACHER_ANCHOR membership and
# the teacher gate. No test identities. No synthetic remixing anywhere.
import os
import sys
import json
import time
import argparse
import numpy as np

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402


def teacher_chunk(model, chunk, emb, zeros, device):
    import torch
    import clapsep_lib

    with torch.no_grad():
        return clapsep_lib.separate(model, chunk, emb, zeros, device)


def audit_window(model, x, w0, kinds, emb, zeros, device):
    """Run all teacher views for one candidate window; compute consistency metrics."""
    sr = rc.SR
    span0 = int(round(w0 * sr))
    span = x[span0:span0 + int(rc.WINDOW_SPAN_S * sr)]
    need = int(rc.WINDOW_SPAN_S * sr)
    if len(span) < need:
        span = np.pad(span, (0, need - len(span)))
    c0, c1 = int(4 * sr), int(10 * sr)          # central 6 s inside the 14 s span
    central = span[c0:c1]

    views = {}
    for name, margin_s in zip(("early", "canonical", "late"), rc.LEFT_MARGINS_S):
        s = int((4 - margin_s) * sr)
        views[name] = span[s:s + rc.CHUNK]
    gain_views = {}
    for g_db in (-3.0, 3.0):
        g = 10.0 ** (g_db / 20.0)
        gain_views[g_db] = views["canonical"] * g

    outs = {k: teacher_chunk(model, v, emb, zeros, device) for k, v in views.items()}
    outs_gain = {g: teacher_chunk(model, v, emb, zeros, device) for g, v in gain_views.items()}

    # aligned central crops: the central 6 s sits at span offset 4 s, so within a
    # view that starts at span offset (4 - margin_s) the crop begins at margin_s
    view_names = ("early", "canonical", "late")
    oc = {}
    for name, margin_s in zip(view_names, rc.LEFT_MARGINS_S):
        cs = int(margin_s * sr)
        oc[name] = outs[name][cs:cs + rc.CENTRAL]
    for k in oc:
        assert len(oc[k]) == rc.CENTRAL, (k, len(oc[k]))
    og = {}
    for g, out in outs_gain.items():
        corrected = out / (10.0 ** (g / 20.0))
        og[g] = corrected[int(2 * sr):int(2 * sr) + rc.CENTRAL]

    names = view_names
    pair_corr, pair_spec, pair_mrstft = [], [], []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = oc[names[i]], oc[names[j]]
            pair_corr.append(rc.wf_corr(a, b))
            pair_spec.append(rc.spec_logmag_corr(a, b))
            pair_mrstft.append(rc.mrstft_logmag_dist(a, b))

    view_list = [oc[n] for n in names] + [og[-3.0], og[3.0]]
    view_rms = np.array([np.sqrt(np.mean(v ** 2)) for v in view_list])
    spread_db = float(20 * np.log10(view_rms.max() / (view_rms.min() + 1e-12) + 1e-12))
    canon_db = 20 * np.log10(view_rms[1] + 1e-12)
    view_db = 20 * np.log10(view_rms + 1e-12)
    collapse_db = float((view_db - canon_db).min())
    amplify_db = float((view_db - canon_db).max())

    env_canon = rc.onset_env(oc["canonical"])
    lags, corrs = [], []
    for n in ("early", "late"):
        lag, cor = rc.onset_env_align_lag_corr(rc.onset_env(oc[n]), env_canon)
        lags.append(lag)
        corrs.append(cor)

    raw_rms_db = rc.rms_db(central)
    out_rms_db = rc.rms_db(oc["canonical"])
    out_raw_db = out_rms_db - raw_rms_db

    rate, level = rc.onset_rate_and_level(central)
    rec = {
        "w0_s": round(float(w0), 2),
        "tags": sorted(set(rc.window_tags(kinds, w0, w0 + rc.WINDOW_SPAN_S))),
        "wf_corr_min": float(np.min(pair_corr)),
        "wf_corr_pairs": [round(float(c), 5) for c in pair_corr],
        "spec_corr_min": float(np.min(pair_spec)),
        "mrstft_max": float(np.max(pair_mrstft)),
        "rms_spread_db": spread_db,
        "collapse_db": collapse_db,
        "amplify_db": amplify_db,
        "onset_lag_max_ms": float(np.max(np.abs(lags))),
        "onset_corr_min": float(np.min(corrs)),
        "out_raw_db": float(out_raw_db),
        "raw_rms_db": raw_rms_db,
        "teacher_rms_db": out_rms_db,
        "click_retention_db": float(rc.click_band_db(oc["canonical"]) - rc.click_band_db(central)),
        "flat_frac": rc.flat_frac(central),
        "onset_rate_per_s": float(rate),
        "onset_median_level_db": float(level),
    }
    rec["accepted"] = bool(
        rec["wf_corr_min"] >= rc.CORR_MIN
        and rec["spec_corr_min"] >= 0.90
        and rec["mrstft_max"] <= rc.MRSTFT_MAX
        and rec["rms_spread_db"] <= rc.RMS_SPREAD_MAX_DB
        and rec["collapse_db"] >= rc.COLLAPSE_MAX_DB
        and rec["amplify_db"] <= rc.AMPLIFY_MAX_DB
        and rec["onset_lag_max_ms"] <= rc.ONSET_LAG_MAX_MS
        and rec["onset_corr_min"] >= rc.ONSET_CORR_MIN
        and rc.OUT_RAW_MIN_DB <= rec["out_raw_db"] <= rc.OUT_RAW_MAX_DB
    )
    if not rec["accepted"]:
        why = []
        if rec["wf_corr_min"] < rc.CORR_MIN:
            why.append("waveform_corr")
        if rec["spec_corr_min"] < 0.90:
            why.append("spectral_corr")
        if rec["mrstft_max"] > rc.MRSTFT_MAX:
            why.append("mrstft_distance")
        if rec["rms_spread_db"] > rc.RMS_SPREAD_MAX_DB:
            why.append("rms_spread")
        if rec["collapse_db"] < rc.COLLAPSE_MAX_DB:
            why.append("context_collapse")
        if rec["amplify_db"] > rc.AMPLIFY_MAX_DB:
            why.append("amplification")
        if rec["onset_lag_max_ms"] > rc.ONSET_LAG_MAX_MS:
            why.append("timing_shift")
        if rec["onset_corr_min"] < rc.ONSET_CORR_MIN:
            why.append("transient_shape")
        if not (rc.OUT_RAW_MIN_DB <= rec["out_raw_db"] <= rc.OUT_RAW_MAX_DB):
            why.append("out_of_band_output_level")
        rec["reject_reasons"] = why
    # interaction-type label for coverage accounting (may overlap)
    tags = set(rec["tags"])
    labels = set()
    if tags & {"strong_impact", "impact"} or rec["onset_rate_per_s"] >= 4.0:
        labels.add("transient_rich")
    if len([t for t in rec["tags"] if t in ("impact", "strong_impact")]) >= 3 or \
       rec["onset_rate_per_s"] >= 5.0:
        labels.add("dense")
    if "weak_contact" in tags or (rec["onset_median_level_db"] < 0.0
                                  and rec["onset_rate_per_s"] > 0.5):
        labels.add("weak")
    if "friction" in tags or rec["flat_frac"] >= 0.30:
        labels.add("friction")
    rec["interaction_labels"] = sorted(labels)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="TRAIN,DEV")
    ap.add_argument("--limit-records", type=int, default=0, help="smoke-test only")
    args = ap.parse_args()
    splits = tuple(args.splits.split(","))

    import torch  # noqa: E402
    import clapsep_lib  # noqa: E402

    t00 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = clapsep_lib.load_clapsep(device)
    emb = clapsep_lib.embed_audio_query(model, rc.Q1_WAV)
    zeros = np.zeros((1, 512), dtype=np.float32)

    split = rc.load_split()
    bank = rc.load_proxy_bank()
    windows = rc.build_window_bank(split, bank, splits=splits)
    if args.limit_records:
        windows = windows[:args.limit_records]
    print(f"auditing {len(windows)} recordings x <=7 windows on {device}")

    results = []
    for n_rec, wd in enumerate(windows):
        x = rc.rec_audio(wd)
        for w0 in wd["windows"]:
            try:
                rec = audit_window(model, x, w0,
                                   rc.event_tags_for(wd["recording_id"], bank),
                                   emb, zeros, device)
            except Exception as e:  # noqa: BLE001 - record and continue
                rec = {"w0_s": round(float(w0), 2), "error": str(e)[:200], "accepted": False}
            rec.update({"recording_id": wd["recording_id"], "split": wd["split"]})
            results.append(rec)
        del x
        print(f"[{n_rec + 1}/{len(windows)}] {wd['recording_id']} done "
              f"({time.time() - t00:.0f}s)", flush=True)

    # ------- aggregate -------
    acc = [r for r in results if r.get("accepted")]
    train_acc = [r for r in acc if r["split"] == "TRAIN"]

    def gate_stats(anchors, min_s, min_recs, label):
        if not anchors:
            return {"gate": label, "pass": False, "seconds": 0.0, "recordings": 0}
        per_rec = {}
        for r in anchors:
            per_rec.setdefault(r["recording_id"], 0.0)
            per_rec[r["recording_id"]] += rc.CENTRAL / rc.SR
        total_s = sum(per_rec.values())
        n_recs = len(per_rec)
        max_share = max(per_rec.values()) / total_s if total_s else 1.0
        labels = {}
        for r in anchors:
            for lb in r["interaction_labels"]:
                labels[lb] = labels.get(lb, 0) + 1
        songs = len({r["recording_id"] for r in anchors})
        return {
            "gate": label, "seconds": total_s, "recordings": n_recs,
            "max_recording_share": round(max_share, 4),
            "interaction_coverage": labels,
            "thresholds": {"min_seconds": min_s, "min_recordings": min_recs,
                           "max_recording_share": rc.GATE_MAX_REC_SHARE},
        }

    gate = gate_stats(train_acc, rc.GATE_MIN_SECONDS, rc.GATE_MIN_RECORDINGS, "preferred")
    g = gate
    gate_pass = (g["seconds"] >= rc.GATE_MIN_SECONDS and g["recordings"] >= rc.GATE_MIN_RECORDINGS
                 and g["max_recording_share"] <= rc.GATE_MAX_REC_SHARE
                 and all(g["interaction_coverage"].get(k, 0) >= 2
                         for k in ("transient_rich", "weak", "dense"))
                 and g["interaction_coverage"].get("friction", 0) >= 1)

    all_corr = [r["wf_corr_min"] for r in results if "wf_corr_min" in r]
    robust_strong = bool(len(all_corr) and np.median(all_corr) >= 0.95 and
                         len(acc) >= 0.6 * len(results))
    pilot = gate_stats(train_acc, rc.PILOT_MIN_SECONDS, rc.PILOT_MIN_RECORDINGS, "reduced_pilot")
    pilot_pass = (not gate_pass and robust_strong
                  and pilot["seconds"] >= rc.PILOT_MIN_SECONDS
                  and pilot["recordings"] >= rc.PILOT_MIN_RECORDINGS
                  and pilot["max_recording_share"] <= rc.GATE_MAX_REC_SHARE)

    summary = {
        "stage": "S1R Phase 0 - real-mixture teacher viability audit",
        "date": "2026-09-14",
        "splits_audited": list(splits),
        "n_recordings": len(windows),
        "n_windows": len(results),
        "n_windows_errored": sum(1 for r in results if "error" in r),
        "n_accepted": len(acc),
        "n_accepted_train": len(train_acc),
        "acceptance_rate": round(len(acc) / max(len(results), 1), 4),
        "metric_distributions": {
            k: {
                "p10": round(float(np.percentile(vals, 10)), 4),
                "median": round(float(np.median(vals)), 4),
                "p90": round(float(np.percentile(vals, 90)), 4),
            }
            for k, vals in {
                "wf_corr_min": [r["wf_corr_min"] for r in results if "wf_corr_min" in r],
                "spec_corr_min": [r["spec_corr_min"] for r in results if "spec_corr_min" in r],
                "mrstft_max": [r["mrstft_max"] for r in results if "mrstft_max" in r],
                "rms_spread_db": [r["rms_spread_db"] for r in results if "rms_spread_db" in r],
                "collapse_db": [r["collapse_db"] for r in results if "collapse_db" in r],
                "amplify_db": [r["amplify_db"] for r in results if "amplify_db" in r],
                "out_raw_db": [r["out_raw_db"] for r in results if "out_raw_db" in r],
                "click_retention_db": [r["click_retention_db"] for r in results
                                       if "click_retention_db" in r],
            }.items() if vals
        },
        "reject_reason_counts": {},
        "gate_preferred": gate,
        "gate_preferred_pass": bool(gate_pass),
        "teacher_robustness_clearly_strong": robust_strong,
        "gate_reduced_pilot": pilot,
        "gate_reduced_pilot_pass": bool(pilot_pass),
        "teacher_gate": "PASS" if (gate_pass or pilot_pass) else "FAIL",
        "gate_mode": ("preferred" if gate_pass else ("reduced_pilot" if pilot_pass else "FAIL")),
        "wall_clock_s": round(time.time() - t00, 1),
    }
    for r in results:
        for why in r.get("reject_reasons", []):
            summary["reject_reason_counts"][why] = summary["reject_reason_counts"].get(why, 0) + 1

    rc.save_json(os.path.join(S1R, "work", "private", "phase0_windows.private.json"), results)
    rc.save_json(os.path.join(S1R, "logs", "01_phase0_summary.json"), summary)

    # public manifest: anonymized IDs only, no local paths
    public_anchors = [{
        "recording_id": r["recording_id"],
        "split": r["split"],
        "window_start_s": r["w0_s"],
        "anchor_seconds": round(rc.CENTRAL / rc.SR, 2),
        "tags": r.get("tags", []),
        "interaction_labels": r.get("interaction_labels", []),
        "consistency": {k: round(r[k], 4) for k in
                        ("wf_corr_min", "spec_corr_min", "mrstft_max", "rms_spread_db",
                         "collapse_db", "amplify_db", "onset_lag_max_ms", "onset_corr_min",
                         "out_raw_db")},
    } for r in acc]
    rc.save_json(os.path.join(S1R, "TEACHER_ANCHOR_MANIFEST.public.json"), {
        "stage": "S1R high-confidence teacher anchors (public manifest)",
        "note": "recording_ids are stable anonymized corpus identifiers; absolute source "
                "paths remain local under work/private/ (protocol sections 44/46)",
        "anchor_definition": {
            "central_crop_s": rc.CENTRAL / rc.SR,
            "context_views": list(rc.LEFT_MARGINS_S),
            "gain_views_db": [-3.0, 3.0],
            "criteria": {
                "wf_corr_min": rc.CORR_MIN, "spec_corr_min": 0.90,
                "mrstft_max": rc.MRSTFT_MAX, "rms_spread_db_max": rc.RMS_SPREAD_MAX_DB,
                "collapse_db_min": rc.COLLAPSE_MAX_DB, "amplify_db_max": rc.AMPLIFY_MAX_DB,
                "onset_lag_max_ms": rc.ONSET_LAG_MAX_MS, "onset_corr_min": rc.ONSET_CORR_MIN,
                "out_raw_db_range": [rc.OUT_RAW_MIN_DB, rc.OUT_RAW_MAX_DB],
            },
        },
        "gate": summary["gate_preferred"],
        "gate_pass": summary["teacher_gate"],
        "gate_mode": summary["gate_mode"],
        "n_anchors": len(public_anchors),
        "anchors": public_anchors,
    })
    print(json.dumps({k: summary[k] for k in
                      ("n_windows", "n_accepted", "acceptance_rate", "teacher_gate",
                       "gate_mode", "reject_reason_counts")}, indent=1))


if __name__ == "__main__":
    main()
