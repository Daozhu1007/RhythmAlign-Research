# S1R step 05 - DEV checkpoint selection + machine-side anti-collapse gate
# (protocol sections 32-34).
#
# Re-evaluates every saved eval checkpoint on the real DEV anchor set, applies the
# section 32 validity checks (all must hold) and the section 33 anti-collapse gate,
# selects the best valid checkpoint by cross-context consistency improvement, and
# freezes it (weights identity recorded; CHECKPOINT_MANIFEST.public.json).
# No TEST/SEALED data is touched anywhere in this script.
import os
import sys
import json
import glob
import hashlib
import numpy as np
import torch

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s1r_dev_eval", os.path.join(S1R, "scripts", "04_dev_eval.py"))
    dev = importlib.util.module_from_spec(spec)
    sys.modules["s1r_dev_eval"] = dev
    spec.loader.exec_module(dev)
    import s1r_model as sm  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = sm.load_student_from_zero_shot(device)
    emb = sm.embed_q1(base)

    zs_summary, _ = dev.evaluate_model(base, emb, device)
    # full-passthrough divergence scale (median raw-vs-teacher L1 on DEV anchors)
    dev_anchors, _ = dev.load_anchor_tables()
    split_all = rc.load_split()
    raws = {r["recording_id"]: (r.get("family"), r.get("work32k"))
            for r in split_all["recordings"]}
    pt_divs, raw_cache = [], {}
    for a in dev_anchors:
        k = a["recording_id"]
        if k not in raw_cache:
            fam, wk = raws[k]
            raw_cache[k] = rc.rec_audio({"family": fam, "work32k": wk})
        v = rc.build_views(raw_cache[k], a["w0_s"])
        t = np.load(os.path.join(rc.S1R, "work", "audio", "teacher",
                                 k, a["files"]["canonical"]))
        pt_divs.append(float(np.abs(v["central"] - t).mean()))
    passthrough_div_l1 = float(np.median(pt_divs))

    ckpts = sorted(glob.glob(os.path.join(S1R, "checkpoints", "full_upd*.ckpt")))
    rows = []
    for path in ckpts:
        name = os.path.basename(path)
        if "zero_shot" in name:
            continue
        model = sm.load_student_from_zero_shot(device, path)
        summary, _ = dev.evaluate_model(model, emb, device)
        checks = dev.check_validity(summary, zs_summary, passthrough_div_l1)
        row = {"ckpt": name, "checks": checks,
               "consistency_combined": summary["consistency_combined"],
               "summary": summary}
        rows.append(row)
        print(json.dumps({"ckpt": name, "checks": checks,
                          "consistency_combined": row["consistency_combined"],
                          "collapse_rate_12db": summary["collapse_rate_12db"],
                          "ratio_median_db": summary["student_teacher_ratio_db"]["median"],
                          "anchor_div_l1": summary["anchor_divergence_l1"]["median"],
                          "passthrough_margin": summary["passthrough_margin_corr"]["median"],
                          "click_ret_db": summary["click_retention_db"]["median"]}),
              flush=True)
        del model
        torch.cuda.empty_cache()

    valid = [r for r in rows if r["checks"]["all"]]
    selected = min(valid, key=lambda r: r["consistency_combined"]) if valid else None

    # section 33 machine-side anti-collapse gate on the selected checkpoint
    gate = None
    if selected:
        s = selected["summary"]
        gate = {
            "collapse_rate_12db": s["collapse_rate_12db"],
            "median_ratio_db": s["student_teacher_ratio_db"]["median"],
            "max_suppression_db_on_anchors": s["student_teacher_ratio_db"]["min"],
            "passthrough_margin_median": s["passthrough_margin_corr"]["median"],
            "pass": bool(s["collapse_rate_12db"] == 0.0
                         and -3.0 <= s["student_teacher_ratio_db"]["median"] <= 3.0
                         and s["student_teacher_ratio_db"]["min"] > -12.0
                         and s["passthrough_margin_corr"]["median"] <= 0.05),
        }

    sel_path = None
    if selected:
        sel_path = os.path.join(S1R, "checkpoints", selected["ckpt"])
        frozen = os.path.join(S1R, "checkpoints", "s1r_selected.ckpt")
        import shutil
        shutil.copyfile(sel_path, frozen)
        sel_path = frozen

    out = {
        "zero_shot_dev": zs_summary,
        "passthrough_div_l1": passthrough_div_l1,
        "checkpoints": rows,
        "selected": selected["ckpt"] if selected else None,
        "selection_rule": "lowest DEV consistency_combined among checkpoints passing ALL "
                          "section-32 validity checks (never quiet-output-consistency)",
        "anti_collapse_gate": gate,
        "frozen_checkpoint": "checkpoints/s1r_selected.ckpt" if selected else None,
        "frozen_sha256": sha256(sel_path) if sel_path else None,
    }
    rc.save_json(os.path.join(S1R, "logs", "05_selection.json"), out)
    print("SELECTED:", out["selected"])
    print("ANTI-COLLAPSE GATE:", gate)


if __name__ == "__main__":
    main()
