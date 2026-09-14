# S2A step 8 - DEV checkpoint selection (protocol section 25).
#
# Validity is a CONJUNCTION (no single composite score). Among checkpoints that
# pass every section-25 check, the selected rung maximizes the median
# injection-suppression advantage over S1R subject to hard no-regression bounds
# (transient retention, context consistency, anchor divergence). The S1R
# reference row (identical DEV code, zero adapter) is computed here as well.
import os
import sys
import json
import glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402
import s2a_model as smod  # noqa: E402
import s2a_deveval as de  # noqa: E402


def main():
    device = torch.device("cuda")
    zeros = np.zeros((1, 512), dtype=np.float32)
    import s1r_model as s1m
    tmp = s1m.load_model(device)
    e_pos = s1m.embed_q1(tmp)
    del tmp
    torch.cuda.empty_cache()
    e_neg = zeros

    # S1R reference row: fresh model = frozen S1R + untouched (zero) adapter
    model = smod.build_model(device)
    pt_div = None
    dev_items = de.load_dev_windows()
    divs = []
    for it in dev_items:
        v = rc_build_views(it)
        t = np.load(os.path.join(sc.S2A, "work", "audio", "s1r_targets",
                                 it["recording_id"], it["files"]["canonical"]))
        divs.append(float(np.abs(v["central"] - t).mean()))
    pt_div = float(np.median(divs))
    smod.set_training_mode(model, training=False)
    s1r_summary, _ = de.evaluate_model(model, (e_pos, e_neg), device, pt_div)
    print("S1R reference row:", json.dumps(
        {k: s1r_summary[k] for k in ("student_s1r_ratio_db", "click_retention_db",
                                     "consistency_combined",
                                     "suppression_adv_vs_s1r_db")}, default=str))

    cands = sorted(glob.glob(os.path.join(sc.S2A, "checkpoints", "*_upd*.ckpt")))
    rows = []
    for c in cands:
        if "pilot_upd00002" in c:
            continue
        ck = torch.load(c, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True)
        smod.set_training_mode(model, training=False)
        summary, _ = de.evaluate_model(model, (e_pos, e_neg), device, pt_div)
        checks = de.check_validity(summary)
        rows.append({"ckpt": os.path.basename(c), "update": ck["update"],
                     "summary": summary, "checks": checks})
        print(f'{os.path.basename(c)}: checks={checks["all"]} '
              f'adv={summary["suppression_adv_vs_s1r_db"]["median"]} '
              f'gap0={summary["causal_gap_zero_db"]["median"]} '
              f'click={summary["click_retention_db"]["median"]} '
              f'cons={summary["consistency_combined"]}')

    # no-regression bounds vs the S1R reference row
    s1r_click = s1r_summary["click_retention_db"]["median"]
    s1r_cons = s1r_summary["consistency_combined"]
    s1r_div = s1r_summary["anchor_divergence_l1"]["median"]

    def no_regression(r):
        return (r["summary"]["click_retention_db"]["median"] >= s1r_click - 1.0
                and r["summary"]["consistency_combined"] <= s1r_cons * 1.05 + 0.001
                and r["summary"]["anchor_divergence_l1"]["median"] <= s1r_div + 0.01)

    eligible = [r for r in rows if r["checks"]["all"] and no_regression(r)]
    selected = None
    if eligible:
        selected = max(eligible,
                       key=lambda r: (r["summary"]["suppression_adv_vs_s1r_db"]["median"],
                                      -r["summary"]["anchor_divergence_l1"]["median"]))
    out = {
        "s1r_reference_row": s1r_summary,
        "passthrough_div_l1": pt_div,
        "candidates": [{"ckpt": r["ckpt"], "update": r["update"], "checks": r["checks"],
                        "dev": r["summary"]} for r in rows],
        "selected": selected["ckpt"] if selected else None,
        "selection_rule": "all section-25 checks AND no regression (click >= s1r-1dB, "
                          "consistency <= s1r*1.05+0.001, anchor div <= s1r+0.01); "
                          "max median suppression advantage vs S1R",
        "cause": None if selected else
                 "no checkpoint satisfied every section-25 check together with the "
                 "no-regression bounds",
    }
    sc.save_json(os.path.join(sc.S2A, "logs", "08_selection.json"), out)
    print("SELECTED:", out["selected"])


def rc_build_views(it):
    import s1r_common as rc
    return rc.build_views(it["x"], it["w0_s"])


if __name__ == "__main__":
    main()
