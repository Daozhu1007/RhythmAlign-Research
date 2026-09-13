# S1R step 06 - TEST_GENERALIZATION (protocol section 35).
#
# Runs AFTER the checkpoint is fully frozen. The exact 28 windows S1W used (2 per
# recording across the 14 frozen TEST_GENERALIZATION identities) so results are
# directly comparable with the S1W record. Methods: RAW / ZERO-SHOT / S1R.
# Machine diagnostics remain DESCRIPTIVE (no true stems exist). No tuning from
# these results.
import os
import sys
import json
import numpy as np
import torch

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402
import s1r_model as sm  # noqa: E402
import s1r_infer as si  # noqa: E402

S1W_GEN = os.path.join(rc.S1W, "logs", "12_generalization.json")
CKPT = os.path.join(S1R, "checkpoints", "s1r_selected.ckpt")


def main():
    import clapsep_lib  # noqa: E402

    with open(S1W_GEN, encoding="utf-8") as f:
        s1w = json.load(f)
    rows_in = s1w["rows"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = sm.load_student_from_zero_shot(device, CKPT)
    zeroshot = clapsep_lib.load_clapsep(device)
    emb = clapsep_lib.embed_audio_query(zeroshot, rc.Q1_WAV)
    zeros = np.zeros((1, 512), dtype=np.float32)

    split_all = rc.load_split()
    raws = {r["recording_id"]: (r.get("family"), r.get("work32k"))
            for r in split_all["recordings"]}

    out_rows = []
    for i, row in enumerate(rows_in):
        rid = row["recording_id"]
        t0, t1 = row["window"]
        fam, wk = raws[rid]
        x = rc.rec_audio({"family": fam, "work32k": wk})
        a, b = int(t0 * rc.SR), int(t1 * rc.SR)
        raw = x[a:b]
        zs = si.ola_separate(zeroshot, x, a, b, emb, zeros, device, student=False)
        st = si.ola_separate(student, x, a, b, emb, zeros, device, student=True)

        def feats(y):
            return {
                "click_band_db": round(rc.click_band_db(y), 3),
                "flat_frac": round(rc.flat_frac(y), 4),
                "rms_db": round(rc.rms_db(y), 3),
            }

        r = {
            "recording_id": rid, "window": [t0, t1],
            "raw": feats(raw), "zeroshot": feats(zs), "s1r": feats(st),
            "zeroshot_ret_vs_raw_db": round(feats(zs)["click_band_db"] - feats(raw)["click_band_db"], 3),
            "s1r_ret_vs_raw_db": round(feats(st)["click_band_db"] - feats(raw)["click_band_db"], 3),
            "s1r_divergence_from_zeroshot_mrstft": round(rc.mrstft_logmag_dist(st, zs), 4),
            "s1r_collapse_check_db": round(rc.rms_db(st) - rc.rms_db(zs), 3),
            "corr_s1r_raw": round(rc.wf_corr(st, raw), 4),
            "corr_zs_raw": round(rc.wf_corr(zs, raw), 4),
        }
        out_rows.append(r)
        print(f"[{i + 1}/{len(rows_in)}] {rid} {t0}-{t1}s "
              f"ret zs {r['zeroshot_ret_vs_raw_db']} s1r {r['s1r_ret_vs_raw_db']} "
              f"collapse {r['s1r_collapse_check_db']}", flush=True)
        del x

    def mean(rows, method, key):
        return round(float(np.mean([r[method][key] for r in rows])), 4)

    summary = {
        "note": "descriptive only; no ground truth; windows identical to S1W record",
        "n_windows": len(out_rows),
        "mean_click_ret_vs_raw_db": {
            "zeroshot": round(float(np.mean([r["zeroshot_ret_vs_raw_db"] for r in out_rows])), 3),
            "s1r": round(float(np.mean([r["s1r_ret_vs_raw_db"] for r in out_rows])), 3),
        },
        "mean_flat_frac": {m: mean(out_rows, m, "flat_frac") for m in ("raw", "zeroshot", "s1r")},
        "mean_rms_db": {m: mean(out_rows, m, "rms_db") for m in ("raw", "zeroshot", "s1r")},
        "mean_s1r_collapse_check_db": round(float(np.mean(
            [r["s1r_collapse_check_db"] for r in out_rows])), 3),
        "collapse_windows_lt_12db": int(sum(1 for r in out_rows
                                            if r["s1r_collapse_check_db"] < -12.0)),
        "passthrough_windows_corr_gt_09": int(sum(1 for r in out_rows
                                                  if r["corr_s1r_raw"] > 0.9)),
        "rows": out_rows,
    }
    rc.save_json(os.path.join(S1R, "logs", "06_generalization.json"), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
