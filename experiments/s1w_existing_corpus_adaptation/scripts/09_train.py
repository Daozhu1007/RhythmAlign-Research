# S1W step 09 - primary training run (protocol sections 24-32).
# Scope: decoder_model only (mask/output head); CLAP + audio_branch + LoRA + BN
# state frozen. Query policy: FIXED S1E audio query Q1, zero negative (no sweep).
# AdamW lr 1e-4 on decoder; microbatch 1, grad-accum 8; max 2000 optimizer updates;
# DEV eval every 250 updates; seed 20260912; fixed-gain convention, no normalization.
import os
import sys
import json
import time
import numpy as np
import torch

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1W, "scripts"))
import clapsep_train_lib as ctl  # noqa: E402
import s1w_objectives as obj  # noqa: E402
from s1w_data import load_recipes, MixtureExample  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20260912
LR = 1e-4
ACC = 8
MAX_UPDATES = 2000
EVAL_EVERY = 250
CKPT_DIR = os.path.join(S1W, "checkpoints")
LOG = os.path.join(S1W, "logs")

torch.manual_seed(SEED)
np.random.seed(SEED)


def model_forward_pred(model, y, emb, device, amp=True):
    zeros = np.zeros((1, 512), dtype=np.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        _mask, pred, _ = ctl.train_forward(model, y, emb, zeros)
    return pred


def evaluate_dev(model, emb, dev_examples, device, limit=64):
    """Fixed-gain DEV diagnostics; axes kept separate (protocol section 31)."""
    ctl.set_training_mode(model, training=False)
    metrics = {"mix": [], "t_only": [], "n_only": [], "weak": [], "friction": []}
    zeros = np.zeros((1, 512), dtype=np.float32)
    subset = dev_examples[:limit]
    with torch.no_grad():
        for ex in subset:
            y, s, n = ex.tensors(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                _mask, pred, _ = ctl.train_forward(model, y, emb, zeros)
            pred = pred.float()
            typ = ex.rec["type"]
            spans = ex.spans()
            if typ in ("MIX", "HARD", "T_ONLY"):
                m = obj.dev_metrics(pred, s, n if typ != "T_ONLY" else None,
                                    ex.rec.get("nuisance_gain"), spans)
                m["stratum"] = ex.rec.get("target", {}).get("stratum")
                key = "t_only" if typ == "T_ONLY" else "mix"
                metrics[key].append(m)
                if m["stratum"] == "weak":
                    metrics["weak"].append(m)
                if m["stratum"] == "friction":
                    metrics["friction"].append(m)
            else:
                m = obj.dev_metrics(pred, n, n, None, None)
                metrics["n_only"].append(m)
    def mean(key, field, alt=None):
        vals = [m[field] for m in metrics[key] if field in m]
        return float(np.mean(vals)) if vals else (alt if alt is not None else float("nan"))
    torch.cuda.empty_cache()
    summary = {
        "n_eval": len(subset),
        "recon_l1_mix": mean("mix", "recon_l1"),
        "target_distortion_db": mean("mix", "target_distortion_db"),
        "leakage_ratio_db": mean("mix", "leakage_ratio_db"),
        "hf_retention_db": mean("mix", "hf_retention_db"),
        "recon_l1_t_only": mean("t_only", "recon_l1"),
        "n_only_output_rms_db": mean("n_only", "output_rms_db"),
        "n_only_input_rms_db": mean("n_only", "input_rms_db"),
        "weak_recon_l1": mean("weak", "recon_l1"),
        "friction_recon_l1": mean("friction", "recon_l1"),
        "weak_hf_retention_db": mean("weak", "hf_retention_db"),
        "counts": {k: len(v) for k, v in metrics.items()},
    }
    return summary, metrics


def main():
    device = torch.device("cuda")
    # guard against stale-bytecode silent execution: the loaded loss MUST be the
    # log-magnitude MR-STFT revision (linear ratio version derails optimization)
    import inspect
    _src = inspect.getsource(obj.mr_stft_loss)
    assert "log10" in _src, "STALE s1w_objectives module loaded (no log10) — aborting"
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    model = ctl.load_model(device)
    ctl.set_trainable_scope(model)
    emb = ctl.embed_q1(model)
    zeros = np.zeros((1, 512), dtype=np.float32)

    params = ctl.trainable_parameters(model)
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-2)

    _, recipes = load_recipes()
    train_recs, dev_recs = recipes["train"], recipes["dev"]
    dev_examples = [MixtureExample(r) for r in dev_recs]
    print(f"train examples {len(train_recs)}, dev {len(dev_recs)}, trainable params {n_tr}")

    # --- zero-shot DEV reference (selection baseline) ---
    t0 = time.time()
    zs, _ = evaluate_dev(model, emb, dev_examples, device)
    print("zero-shot DEV:", json.dumps(zs, indent=1), f"({time.time()-t0:.0f}s)")
    log = {"seed": SEED, "lr": LR, "acc": ACC, "max_updates": MAX_UPDATES,
           "trainable_params": n_tr, "zero_shot_dev": zs, "updates": []}

    rng = np.random.default_rng(SEED)
    model.train()
    ctl.set_training_mode(model, training=True)
    opt.zero_grad(set_to_none=True)
    update = 0
    accum_loss, accum_stats = 0.0, {}
    best = None
    peak_mem = 0.0
    t00 = time.time()
    step = 0
    while update < MAX_UPDATES:
        order = rng.permutation(len(train_recs))
        for i in order:
            rec = train_recs[i]
            ex = MixtureExample(rec)
            y, s, n = ex.tensors(device)
            _mask, pred, _ = ctl.train_forward(model, y, emb, zeros)
            pred = pred.float()
            typ = rec["type"]
            if typ in ("MIX", "HARD", "T_ONLY"):
                loss, parts = obj.recon_loss(pred, s)
            else:
                loss, parts = obj.absent_loss(pred)
            (loss / ACC).backward()
            accum_loss += float(loss.detach()) / ACC
            for k, v in parts.items():
                accum_stats[k] = accum_stats.get(k, 0.0) + float(v) / ACC
            step += 1
            if step % ACC != 0:
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            update += 1
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 2 ** 30)
            if update % 25 == 0:
                torch.cuda.empty_cache()  # hygiene against allocator creep
                n_samples = update * ACC
                print(f"upd {update:4d} loss/sample {accum_loss/n_samples:.4f} "
                      + " ".join(f"{k}/sample {v/n_samples:.4f}" for k, v in accum_stats.items())
                      + f" gn {float(gn):.2f} mem {peak_mem:.2f}GB "
                      f"({time.time()-t00:.0f}s)", flush=True)
            if update % EVAL_EVERY == 0 or update == MAX_UPDATES:
                summ, _ = evaluate_dev(model, emb, dev_examples, device)
                # declared composite: preservation ratio + suppression gain (dB/10),
                # components always reported separately
                pres_now = summ["recon_l1_mix"] + summ["recon_l1_t_only"]
                pres_zero = zs["recon_l1_mix"] + zs["recon_l1_t_only"]
                supp_gain = (zs["leakage_ratio_db"] - summ["leakage_ratio_db"]) / 10.0
                composite = pres_zero / max(pres_now, 1e-9) + supp_gain
                guard_ok = summ["weak_recon_l1"] <= zs["weak_recon_l1"] * 1.10 + 1e-6
                print(f"DEV @{update}: {json.dumps(summ)}", flush=True)
                log["updates"].append({"update": update, "train_loss": accum_loss,
                                       "train_stats": accum_stats, "dev": summ,
                                       "composite": composite, "weak_guard_ok": guard_ok})
                ck = {"update": update, "model_state": model.state_dict(),
                      "dev_summary": summ, "composite": composite,
                      "weak_guard_ok": guard_ok}
                torch.save(ck, os.path.join(CKPT_DIR, f"upd{update:05d}.ckpt"))
                if guard_ok and (best is None or composite > best[0]):
                    best = (composite, update)
                    torch.save(ck, os.path.join(CKPT_DIR, "best.ckpt"))
                accum_loss, accum_stats = 0.0, {}
                model.train()
                ctl.set_training_mode(model, training=True)
            if update >= MAX_UPDATES:
                break
            if accum_loss and not np.isfinite(accum_loss):
                print("NON-FINITE LOSS — stopping")
                break
    log["best"] = {"composite": best[0], "update": best[1]} if best else None
    log["wall_clock_s"] = time.time() - t00
    log["peak_vram_gb"] = peak_mem
    with open(os.path.join(LOG, f"09_train_log_seed{SEED}.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print("TRAINING DONE. best:", log["best"])


if __name__ == "__main__":
    main()
