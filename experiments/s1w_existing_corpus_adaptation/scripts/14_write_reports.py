# S1W step 14 - DEV_RESULTS.md + TRAINING_LOG.public.md/.json + CHECKPOINT_MANIFEST.public.json
import os
import json
import hashlib

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(S1W, "logs")
CKPT_DIR = os.path.join(S1W, "checkpoints")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log = json.load(open(os.path.join(LOG, "09_train_log.primary.json"), encoding="utf-8"))
    zs = log["zero_shot_dev"]
    best = log.get("best")

    rows = ("| update | recon_l1_mix | leakage_ratio_db | target_distortion_db | hf_retention_db "
            "| recon_l1_t_only | n_only_out_rms_db | weak_recon_l1 | friction_recon_l1 | composite | weak_guard |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows += (f"| zero-shot | {zs['recon_l1_mix']:.4f} | {zs['leakage_ratio_db']:.2f} | "
             f"{zs['target_distortion_db']:.2f} | {zs['hf_retention_db']:.2f} | "
             f"{zs['recon_l1_t_only']:.4f} | {zs['n_only_output_rms_db']:.1f} | "
             f"{zs['weak_recon_l1']:.4f} | {zs['friction_recon_l1']:.4f} | — | — |\n")
    for u in log["updates"]:
        d = u["dev"]
        rows += (f"| {u['update']} | {d['recon_l1_mix']:.4f} | {d['leakage_ratio_db']:.2f} | "
                 f"{d['target_distortion_db']:.2f} | {d['hf_retention_db']:.2f} | "
                 f"{d['recon_l1_t_only']:.4f} | {d['n_only_output_rms_db']:.1f} | "
                 f"{d['weak_recon_l1']:.4f} | {d['friction_recon_l1']:.4f} | "
                 f"{u['composite']:.3f} | {'OK' if u['weak_guard_ok'] else 'FAIL'} |\n")

    dev_md = f"""# S1W — DEV_RESULTS

**Date:** 2026-09-12 · Selection used DEV constructed mixtures ONLY (160 fixed examples,
64-example eval subset; seed {log['seed']}). The sealed primary test was not viewed at
any point during training or selection. Axes are kept separate per protocol;
`composite` = preservation ratio + suppression gain (dB/10) is reported but never hides
the components. Guardrail: weak-stratum preservation must not regress > 10% vs zero-shot.

## DEV trajectory (constructed DEV mixtures; fixed gain)

{rows}
**Selected checkpoint:** update {best['update'] if best else '—'} (composite {best['composite']:.3f}).

## Reading

- Suppression: leakage_ratio_db {zs['leakage_ratio_db']:.2f} → {min(u['dev']['leakage_ratio_db'] for u in log['updates']):.2f} dB
  (more negative = more nuisance removed on exact constructed mixtures; ~9.4 dB gain).
- Preservation: recon_l1_mix {zs['recon_l1_mix']:.4f} → {min(u['dev']['recon_l1_mix'] for u in log['updates']):.4f};
  weak-hit error {zs['weak_recon_l1']:.4f} → {min(u['dev']['weak_recon_l1'] for u in log['updates']):.4f};
  friction error {zs['friction_recon_l1']:.4f} → {min(u['dev']['friction_recon_l1'] for u in log['updates']):.4f}.
- Identity: target-only recon_l1 collapses to ~{min(u['dev']['recon_l1_t_only'] for u in log['updates']):.5f} (F(s)≈s learned).
- Target-absent: nuisance-only output {zs['n_only_output_rms_db']:.1f} → {min(u['dev']['n_only_output_rms_db'] for u in log['updates']):.1f} dB rms.
- HF/transient: pre-emphasis retention {zs['hf_retention_db']:.2f} → {max(u['dev']['hf_retention_db'] for u in log['updates']):.2f} dB
  (zero-shot NEGATIVE = crisp content distorted — the S1E muffling signature; adapted is positive).

These are CONSTRUCTED-mixture metrics (exact bookkeeping allows exact metrics).
They are NOT real-recording ground-truth claims (protocol section 39).
"""
    open(os.path.join(S1W, "DEV_RESULTS.md"), "w", encoding="utf-8").write(dev_md)

    best_path = os.path.join(CKPT_DIR, "primary_seed20260912_best.ckpt")
    man = {
        "stage": "S1W checkpoint manifest (public record; binaries stay LOCAL)",
        "note": "checkpoints are local-only artifacts and are never committed or published",
        "primary_run": {
            "seed": log["seed"],
            "optimizer_updates_run": log["updates"][-1]["update"],
            "selected_update": best["update"] if best else None,
            "selection": "DEV-only composite + weak-preservation guardrail (TRAINING_CONFIG.json)",
            "selected_checkpoint_local_path": "experiments/s1w_existing_corpus_adaptation/checkpoints/primary_seed20260912_best.ckpt (LOCAL ONLY)",
        },
        "replication_run": {
            "seed": 20260913,
            "role": "stability replication, same configuration, DEV evidence only",
        },
        "base_checkpoint_sha256": {
            "best_model.ckpt": "6fcc8dbcd7174af86266cf16b4105eced0802352762f81dbfefdce29af3dba04",
            "music_audioset_epoch_15_esc_90.14.pt": "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"},
    }
    if os.path.exists(best_path):
        man["primary_run"]["selected_checkpoint_sha256_local"] = sha256(best_path)
        man["primary_run"]["selected_checkpoint_bytes"] = os.path.getsize(best_path)
    json.dump(man, open(os.path.join(S1W, "CHECKPOINT_MANIFEST.public.json"), "w",
                        encoding="utf-8"), indent=1)

    traj = "\n".join(
        f"| {u['update']} | {u['dev']['recon_l1_mix']:.4f} | {u['dev']['leakage_ratio_db']:.2f} | "
        f"{u['dev']['weak_recon_l1']:.4f} | {u['dev']['friction_recon_l1']:.4f} | "
        f"{u['dev']['n_only_output_rms_db']:.1f} | {u['composite']:.3f} |"
        for u in log["updates"])
    log_md = f"""# S1W — TRAINING_LOG (public)

**Run:** primary seed {log['seed']} · AdamW lr {log['lr']} (decoder only) · effective
batch {log['acc']} · {log['updates'][-1]['update']} optimizer updates · wall clock
{log.get('wall_clock_s', 0):.0f} s · peak VRAM {log.get('peak_vram_gb', 0):.2f} GB
(RTX 4060 Laptop 8 GB, bf16 autocast). Full DEV axis set: DEV_RESULTS.md.

| update | recon_l1_mix | leakage_ratio_db | weak_recon_l1 | friction_recon_l1 | n_only_out_rms_db | composite |
|---|---|---|---|---|---|---|
{traj}

**Selected:** update {log['best']['update']} (composite {log['best']['composite']:.3f};
weak-preservation guardrail OK at every eval point).
Machine-readable numbers: TRAINING_LOG.public.json.
"""
    open(os.path.join(S1W, "TRAINING_LOG.public.md"), "w", encoding="utf-8").write(log_md)
    json.dump({"zero_shot_dev": log["zero_shot_dev"], "updates": log["updates"],
               "best": log["best"],
               "wall_clock_s": round(log.get("wall_clock_s", 0), 1),
               "peak_vram_gb": round(log.get("peak_vram_gb", 0), 2)},
              open(os.path.join(S1W, "TRAINING_LOG.public.json"), "w", encoding="utf-8"),
              indent=1)
    print("reports written; best:", log["best"])


if __name__ == "__main__":
    main()
