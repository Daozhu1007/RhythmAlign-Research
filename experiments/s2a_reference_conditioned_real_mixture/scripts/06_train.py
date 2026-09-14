# S2A step 6 - reference-adapter training (protocol sections 20/23/24).
#
# Per accumulation step, one window with its three real context views:
#   A. S1R anchor:            F(x_can, m_can)  vs frozen S1R(x_can)        (w 1.00)
#   B. injection invariance:  F(x_aug_can, m_can) vs S1R(x_can)           (w 0.75)
#   C. cross-context consist: F(x_aug_e/_l, m_e/_l) vs F(x_aug_can, m_can)(w 0.75)
#   D. anti-collapse hinge:   on all of the above vs the S1R target       (w 1.00)
#   + L2-SP weight anchor toward the frozen S1R trainable weights         (w 0.10)
#
# Modes: --pilot (micro-pilot, section 23) and --full (primary, section 24).
# In-loop stop: non-finite loss/grad, collapse guard, passthrough guard.
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402
import s2a_model as smod  # noqa: E402
import s2a_losses as sl  # noqa: E402
import s2a_augment as sa  # noqa: E402
import s2a_deveval as de  # noqa: E402
import s1r_common as rc  # noqa: E402

LR = 2e-5                # section 24 adapter/head range 2e-5..1e-4; conservative end
WEIGHT_DECAY = 0.01
GRAD_CLIP = 5.0
SEED = 20260915


def load_train_data():
    bank = json.load(open(os.path.join(sc.S2A_PRIVATE, "TRAIN_WINDOW_BANK.private.json"),
                          encoding="utf-8"))
    wins = [w for w in bank["windows"] if w["s2a_split"] == "TRAIN"]
    split = sc.load_split()
    rec_of = {r["recording_id"]: r for r in split["recordings"]}
    pairs = json.load(open(os.path.join(sc.S2A_PRIVATE, "S2A_SPLIT.private.json"),
                           encoding="utf-8"))["pairs"]
    pair_of = {p["recording_id"]: p for p in pairs}
    spans, refs = {}, {}
    for w in wins:
        rid = w["recording_id"]
        if rid not in spans:
            spans[rid] = rc.rec_audio(rec_of[rid])
            refs[rid] = (pair_of[rid], sc.ref_audio32k(pair_of[rid]))
    return wins, spans, refs


def view_ref(pair, ref32k, w0, margin):
    v0 = w0 + 4.0 - margin
    m, _cov = sc.ref_segment(pair, ref32k, v0, v0 + 10.0)
    return m[:rc.CHUNK]


def train_step(model, it, spans, refs, e_pos, e_neg, theta0, device, rng):
    x = spans[it["recording_id"]]
    pair, ref32k = refs[it["recording_id"]]
    w0 = it["w0_s"]
    v = rc.build_views(x, w0)
    m_can = view_ref(pair, ref32k, w0, 2.0)
    name_b = "early" if (rng.random() < 0.5) else "late"
    margin_b = 4.0 if name_b == "early" else 0.0
    m_b = view_ref(pair, ref32k, w0, margin_b)

    t = torch.tensor(np.load(os.path.join(sc.S2A, "work", "audio", "s1r_targets",
                                          it["recording_id"], it["files"]["canonical"])),
                     device=device)

    def fwd(chunk, ref, grad=True):
        arr = np.asarray(chunk, dtype=np.float32)
        mx = float(np.max(np.abs(arr)))
        scale_back = 1.0
        xt = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
        rt = torch.tensor(np.asarray(ref, dtype=np.float32), device=device).unsqueeze(0)
        if mx > 1:
            xt = xt * (0.9 / mx)
            scale_back = mx / 0.9
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            _mask, pred = smod.train_forward(model, xt, rt, e_pos, e_neg)
        return pred.float().squeeze(0) * scale_back

    cs_can, cs_b = rc.CHUNK // 2 - rc.CENTRAL // 2, None
    cs_can = int(2.0 * rc.SR)
    cs_b = int(margin_b * rc.SR)

    predA = fwd(v["views"]["canonical"], m_can)
    x_aug_can, alpha = sa.make_augmented(v["views"]["canonical"], m_can, rng)
    predB = fwd(x_aug_can, m_can)
    x_aug_b, _ = sa.make_augmented(v["views"][name_b], m_b, rng)
    predC = fwd(x_aug_b, m_b)
    pA = predA[cs_can:cs_can + rc.CENTRAL]
    pB = predB[cs_can:cs_can + rc.CENTRAL]
    pC = predC[cs_b:cs_b + rc.CENTRAL]

    l_anchor, p_anchor = sl.distill(pA, t)
    l_invar, p_invar = sl.distill(pB, t)
    l_cons, p_cons = sl.consistency(pB, pC)
    collA, pcA = sl.anti_collapse(pA, t)
    collB, pcB = sl.anti_collapse(pB, t)
    collapse = (collA + collB) / 2.0
    weight = sl.weight_anchor(model, theta0)

    loss = (sl.W_ANCHOR * l_anchor + sl.W_INVAR * l_invar + sl.W_CONSIST * l_cons
            + sl.W_COLLAPSE * collapse + sl.W_WEIGHT * weight)
    parts = {
        "anchor": float(l_anchor.detach()), "invar": float(l_invar.detach()),
        "cons": float(l_cons.detach()),
        "collapse_db": float(pcA["ratio_db"]), "l_trans": float(pcB["l_transient"]),
        "weight": float(weight.detach()), "alpha": float(alpha),
    }
    return loss, parts


@torch.no_grad()
def quick_train_metrics(model, wins, spans, refs, e_pos, e_neg, device, n=8):
    """Cheap TRAIN-side signal: injection residual with correct vs zero ref."""
    smod.set_training_mode(model, training=False)
    adv, gaps = [], []
    for it in wins[:n]:
        x = spans[it["recording_id"]]
        pair, ref32k = refs[it["recording_id"]]
        w0 = it["w0_s"]
        v = rc.build_views(x, w0)
        m_can = view_ref(pair, ref32k, w0, 2.0)
        t = np.load(os.path.join(sc.S2A, "work", "audio", "s1r_targets",
                                 it["recording_id"], it["files"]["canonical"]))
        rng = np.random.default_rng(12345)
        x_aug, _ = sa.make_augmented(v["views"]["canonical"], m_can, rng)
        y_c = de._forward(model, x_aug, m_can, e_pos, e_neg, device)
        y_z = de._forward(model, x_aug, np.zeros(rc.CHUNK, dtype=np.float32),
                          e_pos, e_neg, device)
        cs = int(2.0 * rc.SR)
        r_c = float(np.sqrt(np.mean((y_c[cs:cs + rc.CENTRAL] - t) ** 2)))
        r_z = float(np.sqrt(np.mean((y_z[cs:cs + rc.CENTRAL] - t) ** 2)))
        adv.append(20 * np.log10(r_z / (r_c + 1e-9) + 1e-9))
    smod.set_training_mode(model, training=True)
    return {"train_zero_vs_correct_db": round(float(np.median(adv)), 3)}


def compute_passthrough_div(dev_items):
    divs = []
    for it in dev_items:
        v = rc.build_views(it["x"], it["w0_s"])
        t = np.load(os.path.join(sc.S2A, "work", "audio", "s1r_targets",
                                 it["recording_id"], it["files"]["canonical"]))
        divs.append(float(np.abs(v["central"] - t).mean()))
    return float(np.median(divs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), required=True)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--adapter-lr", type=float, default=None,
                    help="optional higher LR for the adapter param group "
                         "(section 24 range 2e-5..1e-4); head stays at LR")
    ap.add_argument("--alpha-lo", type=float, default=sa.ALPHA_RANGE[0])
    ap.add_argument("--alpha-hi", type=float, default=sa.ALPHA_RANGE[1])
    args = ap.parse_args()
    if args.mode == "pilot":
        max_updates = args.updates or 200
        eval_every = args.eval_every or 50
    else:
        max_updates = args.updates or 1500
        eval_every = args.eval_every or 250
    if args.alpha_lo != sa.ALPHA_RANGE[0] or args.alpha_hi != sa.ALPHA_RANGE[1]:
        sa.ALPHA_RANGE = (args.alpha_lo, args.alpha_hi)
        print(f"DECLARED ADJUSTMENT (pre-TEST, TRAIN/DEV evidence): ALPHA_RANGE -> "
              f"{sa.ALPHA_RANGE}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    t00 = time.time()
    model = smod.build_model(device)
    e_pos = smod.embed_q1(model.base_model)
    e_neg = np.zeros((1, 512), dtype=np.float32)
    smod.set_trainable_scope(model)
    params = smod.trainable_parameters(model)
    n_tr = sum(p.numel() for p in params)
    theta0 = smod.snapshot_theta0(model)
    if args.adapter_lr:
        adapter_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and n.startswith("adapter.")]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("adapter.")]
        opt = torch.optim.AdamW([
            {"params": head_params, "lr": LR},
            {"params": adapter_params, "lr": args.adapter_lr},
        ], weight_decay=WEIGHT_DECAY)
        print(f"DECLARED ADJUSTMENT (pre-TEST, TRAIN/DEV evidence): adapter LR -> "
              f"{args.adapter_lr}", flush=True)
    else:
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    dev_items = de.load_dev_windows()
    pt_div = compute_passthrough_div(dev_items)
    wins, spans, refs = load_train_data()
    rng = np.random.default_rng(SEED)
    print(f"TRAIN windows {len(wins)} | trainable {n_tr} params "
          f"({100.0 * n_tr / sum(p.numel() for p in model.parameters()):.3f}%) | "
          f"passthrough_div_l1 {pt_div:.4f}", flush=True)

    log = {"mode": args.mode, "seed": SEED, "lr": LR, "max_updates": max_updates,
           "trainable_params": n_tr, "passthrough_div_l1": pt_div,
           "adapter_lr": args.adapter_lr or LR,
           "alpha_range": [args.alpha_lo, args.alpha_hi],
           "declared_adjustments": bool(args.adapter_lr or args.alpha_lo != 0.12
                                        or args.alpha_hi != 0.45),
           "evals": [], "stop": None}
    smod.set_training_mode(model, training=True)
    opt.zero_grad(set_to_none=True)
    update = step = 0
    accum_loss, accum_stats = 0.0, {}
    peak_mem = 0.0
    order = rng.permutation(len(wins))
    abort = None

    while update < max_updates and abort is None:
        if step >= len(order):
            order = rng.permutation(len(wins))
            step = 0
        it = wins[int(order[step])]
        try:
            loss, parts = train_step(model, it, spans, refs, e_pos, e_neg, theta0,
                                     device, rng)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            abort = "CUDA_OOM"
            break
        if not torch.isfinite(loss):
            abort = "NON_FINITE_LOSS"
            break
        (loss / 8.0).backward()
        accum_loss += float(loss.detach()) / 8.0
        for k, val in parts.items():
            accum_stats[k] = accum_stats.get(k, 0.0) + val / 8.0
        step += 1
        if step % 8 != 0:
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
            n = update * 8
            print(f"upd {update:4d} loss/ex {accum_loss / n:.4f} "
                  + " ".join(f"{k}/ex {v / n:.5f}" for k, v in accum_stats.items())
                  + f" gn {float(gn):.3f} mem {peak_mem:.2f}GB "
                  f"({time.time() - t00:.0f}s)", flush=True)
        if update % eval_every == 0 or update == max_updates:
            tm = quick_train_metrics(model, wins, spans, refs, e_pos, e_neg, device)
            summary, _per = de.evaluate_model(model, (e_pos, e_neg), device, pt_div)
            checks = de.check_validity(summary)
            keys = ("collapse_rate_12db", "student_s1r_ratio_db", "consistency_combined",
                    "anchor_divergence_l1", "passthrough_margin_corr",
                    "click_retention_db", "suppression_adv_vs_s1r_db",
                    "causal_gap_zero_db", "causal_gap_wrong_db", "wrong_clean_ratio_db")
            print(f"EVAL @{update}: train {json.dumps(tm)} dev "
                  f"{json.dumps({k: summary[k] for k in keys}, default=str)} "
                  f"checks {json.dumps(checks)}", flush=True)
            ck = {"update": update, "model_state": model.state_dict(),
                  "dev_summary": summary, "checks": checks, "train_metrics": tm,
                  "mode": args.mode}
            torch.save(ck, os.path.join(sc.S2A, "checkpoints",
                                        f"{args.mode}_upd{update:05d}.ckpt"))
            log["evals"].append({"update": update, "train": tm, "dev": summary,
                                 "checks": checks})
            smod.set_training_mode(model, training=True)
            if summary["collapse_rate_12db"] > 0.3 or \
               summary["student_s1r_ratio_db"]["median"] < -8.0:
                abort = "COLLAPSE_GUARD"
            if summary["passthrough_margin_corr"]["median"] > 0.15:
                abort = "PASSTHROUGH_GUARD"
            if abort:
                break

    log["stop"] = abort or "COMPLETED"
    log["wall_clock_s"] = round(time.time() - t00, 1)
    log["peak_vram_gb"] = round(peak_mem, 3)
    sc.save_json(os.path.join(sc.S2A, "logs", f"06_train_{args.mode}_log.json"), log)
    print("TRAINING", log["stop"], f"after {update} updates "
          f"({time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
