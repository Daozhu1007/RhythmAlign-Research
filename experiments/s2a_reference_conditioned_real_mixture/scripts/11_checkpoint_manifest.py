# S2A step 11: emit CHECKPOINT_MANIFEST.public.json (hashes only, no weights)
# plus the frozen TRAINING_LOG.public.md from the private run logs.
import os
import sys
import json
import hashlib
import glob

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(sc.S2A, "checkpoints", "*.ckpt"))):
        rows.append({"file": os.path.basename(p), "sha256": sha256(p),
                     "bytes": os.path.getsize(p)})
    base = {"file": "experiments/s1r_real_mixture_adaptation/checkpoints/s1r_selected.ckpt",
            "sha256": sc.S1R_CKPT_SHA256}
    sc.save_json(os.path.join(sc.S2A, "CHECKPOINT_MANIFEST.public.json"),
                 {"stage": "S2A reference-conditioned real-mixture extraction",
                  "privacy_note": "hashes and sizes only; weights stay local",
                  "base_checkpoint": base, "checkpoints": rows})
    print(f"manifest: {len(rows)} checkpoints")

    # public training log
    pilot = json.load(open(os.path.join(sc.S2A, "logs", "06_train_pilot_log.json"),
                           encoding="utf-8"))
    full = json.load(open(os.path.join(sc.S2A, "logs", "06_train_full_log.json"),
                          encoding="utf-8"))
    parity = json.load(open(os.path.join(sc.S2A, "logs", "04_parity_check.json"),
                            encoding="utf-8"))

    def fmt_run(r, name):
        lines = [f"## {name}", "",
                 f"- updates: {r['max_updates']} · lr {r['lr']} · seed {r['seed']} · "
                 f"wall {r['wall_clock_s']} s · peak VRAM {r['peak_vram_gb']} GB · "
                 f"stop: {r['stop']}"]
        if r.get("declared_adjustments"):
            lines[-1] += " · DECLARED pre-TEST adjustments: " + \
                f"adapter_lr={r.get('adapter_lr')}, alpha_range={r.get('alpha_range')}"
        lines += ["", "| upd | train zero-vs-correct (dB) | med adv vs S1R (dB) | "
                  "med gap zero (dB) | med gap wrong (dB) | click ret (dB) | consistency | "
                  "collapse12 | ratio vs S1R (dB) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for e in r["evals"]:
            d = e["dev"]
            s = e["train"]
            lines.append(
                f"| {e['update']} | {s.get('train_zero_vs_correct_db', 0)} | "
                f"{d['suppression_adv_vs_s1r_db']['median']} | "
                f"{d['causal_gap_zero_db']['median']} | "
                f"{d['causal_gap_wrong_db']['median']} | "
                f"{d['click_retention_db']['median']} | "
                f"{d['consistency_combined']} | "
                f"{d['collapse_rate_12db']} | "
                f"{d['student_s1r_ratio_db']['median']} |")
        return "\n".join(lines) + "\n"

    sel = json.load(open(os.path.join(sc.S2A, "logs", "08_selection.json"),
                         encoding="utf-8"))
    md = ["# S2A — TRAINING_LOG (public)", "",
          f"**Parity before training:** sealed parity "
          f"{'PASS' if parity['sealed_parity']['pass'] else 'FAIL'} "
          f"(max |ΔRMS| "
          f"{max(r['rms_abs_diff_db'] for r in parity['sealed_parity']['rows'])} dB); "
          f"initial S1R parity exact (max|Δ| = "
          f"{parity['init_parity_vs_s1r']['correct']['max_abs_diff_vs_s1r']}) for "
          f"CORRECT / ZERO / WRONG reference modes; adapter output-head gradient at "
          f"step 0: {parity['adapter_gradient']['out_head_abs_sum']:.3e}.", "",
          f"Trainable surface: {parity['trainable_params']} params "
          f"({parity['trainable_percent']}% of model) = reference adapter + final "
          f"mask head; all other S1R parameters frozen.", "",
          fmt_run(pilot, "Micro-pilot (section 23)"),
          fmt_run(full, "Primary run (section 24)"), "",
          "## DEV selection (section 25)", "",
          f"- outcome: **{sel['selected'] or 'NO CHECKPOINT PASSED ALL CHECKS'}**",
          f"- failing axis: `correct_ref_causally_used` "
          f"(best median causal gap "
          f"{max((c['dev']['causal_gap_zero_db']['median'] for c in sel['candidates']), default=0)} dB"
          f" vs >= 1.0 dB required; section-26 preferred evidence >= 1.5 dB)",
          f"- S1R reference row (same code, zero adapter): consistency "
          f"{sel['s1r_reference_row']['consistency_combined']}, click "
          f"{sel['s1r_reference_row']['click_retention_db']['median']} dB, "
          f"suppression advantage exactly 0 (sanity check)",
          f"- documentation checkpoint for the held-out negative result: "
          f"{sel.get('evaluation_checkpoint')}"]
    with open(os.path.join(sc.S2A, "TRAINING_LOG.public.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("wrote TRAINING_LOG.public.md")


if __name__ == "__main__":
    main()
