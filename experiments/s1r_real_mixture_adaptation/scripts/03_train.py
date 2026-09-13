# S1R step 03 - real-mixture adaptation training (protocol sections 17-30).
#
# Student initialized from the EXACT frozen zero-shot weights. Training signal:
#   1. teacher-anchor distillation on HIGH_CONFIDENCE_TEACHER_ANCHOR regions
#      (precomputed frozen-teacher central crops; pseudo-labels, not truth)
#   2. cross-context consistency across two real context views of the SAME region
#   3. scale equivariance under a known +-3 dB global gain
#   4. anti-collapse hinge guard + transient-band floor
#   5. L2-SP weight-space conservatism toward the zero-shot parameters
# L = 1.00*anchor + 0.75*consistency + 0.25*scale + 1.00*anti_collapse + 0.10*weight
#
# Modes: --pilot (micro-pilot, section 29) and --full (primary run, section 30).
# Hard in-loop stop conditions: non-finite loss; collapse; raw passthrough drift.
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402
import s1r_model as sm  # noqa: E402
import s1r_losses as sl  # noqa: E402

LR = 2e-5                # section 30: 1e-5..3e-5; S1W's 1e-4 explicitly not defaulted to
ACC = 8
WEIGHT_DECAY = 0.01
GRAD_CLIP = 5.0
EVAL_TRAIN_SUBSET = 10   # TRAIN anchors for quick in-training consistency tracking
SEED = rc.SEED


def official_forward(model, chunk, emb, zeros, device=None):
    """Grad-capable official-protocol forward (mirrors clapsep_lib.separate)."""
    if device is None:
        device = next(model.parameters()).device
    arr = np.asarray(chunk, dtype=np.float32)
    max_val = float(np.max(np.abs(arr)))
    scale_back = 1.0
    x = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
    if max_val > 1:
        x = x * (0.9 / max_val)
        scale_back = max_val / 0.9
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        _mask, pred = sm.train_forward(model, x, emb, zeros)
    return pred.float().squeeze(0) * scale_back


def central(pred, cs):
    return pred[..., cs:cs + rc.CENTRAL]


def load_train_data():
    with open(os.path.join(S1R, "work", "private", "phase0_windows.private.json"),
              encoding="utf-8") as f:
        results = json.load(f)
    with open(os.path.join(S1R, "work", "private", "teacher_precompute.private.json"),
              encoding="utf-8") as f:
        pre = json.load(f)
    pre_map = {(a["recording_id"], a["w0_s"]): a["files"] for a in pre["anchors"]}
    anchors = [r for r in results
               if r.get("accepted") and r["split"] == "TRAIN"
               and (r["recording_id"], r["w0_s"]) in pre_map]
    for a in anchors:
        a["files"] = pre_map[(a["recording_id"], a["w0_s"])]
    # cache spans
    split_all = rc.load_split()
    fam_of = {r["recording_id"]: r.get("family") for r in split_all["recordings"]}
    wk_of = {r["recording_id"]: r.get("work32k") for r in split_all["recordings"]}
    spans = {}
    for a in anchors:
        k = a["recording_id"]
        if k not in spans:
            spans[k] = rc.rec_audio({"family": fam_of[k], "work32k": wk_of[k]})
    return anchors, spans


def train_step(model, a, x, emb, zeros, theta0, device):
    v = rc.build_views(x, a["w0_s"])
    name_b = "early" if (a["_step"] % 2 == 0) else "late"
    g = a["_gain"]
    files = a["files"]

    predA = official_forward(model, v["views"]["canonical"], emb, zeros)
    predB = official_forward(model, v["views"][name_b], emb, zeros)
    gained = v["views"]["canonical"] * (10.0 ** (g / 20.0))
    predC = official_forward(model, gained, emb, zeros) / (10.0 ** (g / 20.0))

    csA = v["crop_starts"]["canonical"]
    csB = v["crop_starts"][name_b]
    pA, pB, pC = central(predA, csA), central(predB, csB), central(predC, csA)

    tA = torch.tensor(np.load(os.path.join(rc.S1R, "work", "audio", "teacher",
                                           a["recording_id"], files["canonical"])),
                      device=device)
    tB = torch.tensor(np.load(os.path.join(rc.S1R, "work", "audio", "teacher",
                                           a["recording_id"], files[name_b])),
                      device=device)
    tG = torch.tensor(np.load(os.path.join(rc.S1R, "work", "audio", "teacher",
                                           a["recording_id"], files[f"gain{int(g)}"])),
                      device=device)

    la, pa = sl.distill(pA, tA)
    lb, pb = sl.distill(pB, tB)
    lc, pc = sl.distill(pC, tG)
    anchor = (la + lb + lc) / 3.0

    consist, pcons = sl.consistency(pA, pB)
    scale, pscale = sl.consistency(pC, pA)
    collA, pcolA = sl.anti_collapse(pA, tA)
    collB, pcolB = sl.anti_collapse(pB, tB)
    collC, pcolC = sl.anti_collapse(pC, tG)
    collapse = (collA + collB + collC) / 3.0
    weight = sl.weight_anchor(model, theta0)

    loss = (sl.W_ANCHOR * anchor + sl.W_CONSIST * consist + sl.W_SCALE * scale
            + sl.W_COLLAPSE * collapse + sl.W_WEIGHT * weight)
    parts = {
        "anchor": float(anchor.detach()), "consist": float(consist.detach()),
        "scale": float(scale.detach()), "collapse_db": float(pcolA["ratio_db"]),
        "l_trans": float(pcolA["l_transient"]), "weight": float(weight),
    }
    return loss, parts


@torch.no_grad()
def quick_train_metrics(model, anchors, spans, emb, zeros, device, n=EVAL_TRAIN_SUBSET):
    """Quick TRAIN-anchor consistency + ratio (pilot evidence, section 29)."""
    sm.set_training_mode(model, training=False)
    cons, ratios = [], []
    for a in anchors[:n]:
        v = rc.build_views(spans[a["recording_id"]], a["w0_s"])
        predA = official_forward(model, v["views"]["canonical"], emb, zeros)
        predB = official_forward(model, v["views"]["late"], emb, zeros)
        pA = central(predA, v["crop_starts"]["canonical"]).detach().cpu().numpy()
        pB = central(predB, v["crop_starts"]["late"]).detach().cpu().numpy()
        tA = np.load(os.path.join(rc.S1R, "work", "audio", "teacher",
                                  a["recording_id"], a["files"]["canonical"]))
        cons.append(np.abs(pA - pB).mean())
        ratios.append(20 * np.log10(float(np.sqrt(np.mean(pA ** 2))) /
                                    (float(np.sqrt(np.mean(tA ** 2))) + 1e-12) + 1e-9))
    sm.set_training_mode(model, training=True)
    return {"train_consistency_l1": round(float(np.mean(cons)), 5),
            "train_median_ratio_db": round(float(np.median(ratios)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), required=True)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "pilot":
        max_updates = args.updates or 200
        eval_every = args.eval_every or 50
    else:
        max_updates = args.updates or 1500
        eval_every = args.eval_every or 250

    from importlib import import_module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s1r_dev_eval", os.path.join(S1R, "scripts", "04_dev_eval.py"))
    dev = importlib.util.module_from_spec(spec)
    sys.modules["s1r_dev_eval"] = dev
    spec.loader.exec_module(dev)

    t00 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    model = sm.load_student_from_zero_shot(device)
    sm.set_trainable_scope(model)
    emb = sm.embed_q1(model)
    zeros = np.zeros((1, 512), dtype=np.float32)
    params = sm.trainable_parameters(model)
    n_tr = sum(p.numel() for p in params)
    theta0 = sm.snapshot_zero_shot_state(model)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # zero-shot DEV reference + full-passthrough divergence scale:
    # median L1(raw central, teacher central) over DEV anchors = the divergence the
    # student would show if it degenerated to wholesale raw passthrough
    zs_summary, _ = dev.evaluate_model(model, emb, device)
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
    del raw_cache

    anchors, spans = load_train_data()
    for i, a in enumerate(anchors):
        a["_step"] = i
    print(f"TRAIN anchors {len(anchors)} | trainable params {n_tr} "
          f"({100.0 * n_tr / sum(p.numel() for p in model.parameters()):.2f}% of model) "
          f"| passthrough_div_l1 {passthrough_div_l1:.4f}", flush=True)
    print("zero-shot DEV:", json.dumps(zs_summary), flush=True)

    log = {"mode": args.mode, "seed": SEED, "lr": LR, "acc": ACC,
           "max_updates": max_updates, "trainable_params": n_tr,
           "passthrough_div_l1": passthrough_div_l1,
           "zero_shot_dev": zs_summary, "evals": [], "stop": None}

    rng = np.random.default_rng(SEED)
    sm.set_training_mode(model, training=True)
    opt.zero_grad(set_to_none=True)
    update, step = 0, 0
    accum_loss, accum_stats = 0.0, {}
    peak_mem = 0.0
    order = rng.permutation(len(anchors))
    abort = None

    while update < max_updates and abort is None:
        if step >= len(order):
            order = rng.permutation(len(anchors))
            step = 0
        a = anchors[order[step]]
        a["_step"] = int(order[step])
        a["_gain"] = float(rng.choice([-3.0, 3.0]))
        x = spans[a["recording_id"]]
        try:
            loss, parts = train_step(model, a, x, emb, zeros, theta0, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            abort = "CUDA_OOM"
            break
        if not torch.isfinite(loss):
            abort = "NON_FINITE_LOSS"
            break
        (loss / ACC).backward()
        accum_loss += float(loss.detach()) / ACC
        for k, val in parts.items():
            accum_stats[k] = accum_stats.get(k, 0.0) + val / ACC
        step += 1
        if step % ACC != 0:
            continue
        gn = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        if not torch.isfinite(gn):
            abort = "NON_FINITE_GRAD"
            break
        opt.step()
        opt.zero_grad(set_to_none=True)
        update += 1
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 2 ** 30)
        if update % 25 == 0:
            torch.cuda.empty_cache()
            n = update * ACC
            print(f"upd {update:4d} loss/ex {accum_loss / n:.4f} "
                  + " ".join(f"{k}/ex {v / n:.5f}" for k, v in accum_stats.items())
                  + f" gn {float(gn):.3f} mem {peak_mem:.2f}GB "
                  f"({time.time() - t00:.0f}s)", flush=True)
        if update % eval_every == 0 or update == max_updates:
            tm = quick_train_metrics(model, anchors, spans, emb, zeros, device)
            summary, per = dev.evaluate_model(model, emb, device)
            checks = dev.check_validity(summary, zs_summary, passthrough_div_l1)
            print(f"EVAL @{update}: train {json.dumps(tm)} dev "
                  f"{json.dumps({k: summary[k] for k in ('collapse_rate_12db','collapse_rate_6db','consistency_combined','student_teacher_ratio_db','anchor_divergence_l1','passthrough_margin_corr','click_retention_db')}, default=str)}",
                  flush=True)
            ck = {"update": update, "model_state": model.state_dict(),
                  "dev_summary": summary, "checks": checks,
                  "train_metrics": tm, "mode": args.mode}
            torch.save(ck, os.path.join(S1R, "checkpoints",
                                        f"{args.mode}_upd{update:05d}.ckpt"))
            log["evals"].append({"update": update, "train": tm, "dev": summary,
                                 "checks": checks})
            sm.set_training_mode(model, training=True)
            # section 29/48 in-loop stop conditions
            if summary["collapse_rate_12db"] > 0.3 or \
               summary["student_teacher_ratio_db"]["median"] < -8.0:
                abort = "COLLAPSE_GUARD"
            if summary["passthrough_margin_corr"]["median"] > 0.15:
                abort = "PASSTHROUGH_GUARD"
            if abort:
                break

    log["stop"] = abort or "COMPLETED"
    log["wall_clock_s"] = round(time.time() - t00, 1)
    log["peak_vram_gb"] = round(peak_mem, 3)
    rc.save_json(os.path.join(S1R, "logs", f"03_train_{args.mode}_log.json"), log)
    print("TRAINING", log["stop"], f"after {update} updates ({time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
