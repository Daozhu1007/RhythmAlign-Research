# R5.5 Phase 1 - TRACE AUDIT on Dev-A (golden) + Dev-B (old holdout).
# Evidence BEFORE any code change: per-contact channel behaviour around every
# TP / FN / FP (+/-300 ms), per-channel atomic-peak triggering power inside
# dense regions, and the R5 failure signature (score platform at FN times).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig

from common55 import (R55_OUT, R55_LOG, load_windows, save_json, load_json,
                      detect_candidates, R5_FROZEN_CFG, FPS)

CHANNELS = [("jz_text", "jz_warm", "jz_white"),
            ("jz_diff", "jz_diff", None),
            ("glass_bright", "glass_bright", None),
            ("hand_diff", "hand_diff", None)]
VALLEY_W = 9          # local-prominence valley half-window, ~150 ms
MATCH_MS = 50


def chan_series(series):
    return {name: series[a] + (series[b] if b else 0.0) for name, a, b in CHANNELS}


def local_prominence(x, pk_idx, valley_w=VALLEY_W):
    """prominence = height - max(nearest left valley, nearest right valley)."""
    prom = np.zeros(len(pk_idx))
    for i, p in enumerate(pk_idx):
        a, b = max(p - valley_w, 0), min(p + valley_w + 1, len(x))
        left = np.min(x[a:p + 1]) if p > a else x[p]
        right = np.min(x[p:b]) if b > p + 1 else x[p]
        prom[i] = x[p] - max(left, right)
    return prom


def find_atomic_peaks(x, h_norm, pr_norm, min_dist):
    scale = np.percentile(x, 99.5) + 1e-9
    pk, _ = sig.find_peaks(x, distance=min_dist)
    if len(pk) == 0:
        return pk, np.zeros(0)
    prom = local_prominence(x, pk)
    keep = (x[pk] / scale >= h_norm) & (prom / scale >= pr_norm)
    return pk[keep], prom[keep] / scale


def main():
    windows = load_windows()
    audit = {}
    cases_out = {}

    for w in windows:
        score, cands = detect_candidates(w.series, FPS, R5_FROZEN_CFG)
        tp_t = np.array([c["t_video"] for c in cands])
        tc = w.tc()
        # one-to-one @50ms: oracle index -> candidate index (or None)
        from common55 import match_one_to_one
        mc, mp, pairs = match_one_to_one(tp_t, tc, MATCH_MS / 1000.0)
        cand_of_contact = {k: j for j, k, _ in pairs}
        for e in w.contacts:
            i = e["idx"] = w.contacts.index(e)
        fns = [e for i, e in enumerate(w.contacts) if i not in mc]
        tps = [w.contacts[k] for k in mc]
        fps = [c for j, c in enumerate(cands) if j not in mp]

        ch = chan_series(w.series)
        dense_t = w.dense_mask

        # ---- per-contact channel audit (dense only for atomic questions) ----
        rows = []
        for e in w.contacts:
            k = int(round(e["t_video"] * FPS))
            row = {"id": e["id"], "t": e["t_video"], "dense": e["dense"],
                   "type": e["type"], "conf": e.get("confidence"),
                   "is_tp": e in tps, "is_fn": e in fns}
            for name, x in ch.items():
                seg = x[max(k - 3, 0):k + 4]
                row[f"{name}_at"] = float(x[k])
                row[f"{name}_near_max"] = float(seg.max()) if len(seg) else None
            # score platform signature: fused score level and max rise near contact
            row["score_at"] = float(score[k])
            rise = np.zeros(len(score))
            rise[2:] = score[2:] - score[:-2]
            row["score_rise_near_max"] = float(
                rise[max(k - 3, 0):k + 4].max()) if 0 <= k < len(score) else None
            rows.append(row)

        dense_rows = [r for r in rows if r["dense"]]

        # ---- per-channel atomic trigger sweep (dense only) ----
        sweeps = {}
        for name, x in ch.items():
            best = None
            grid = []
            for h in (0.10, 0.15, 0.20, 0.30, 0.40):
                for pr in (0.03, 0.06, 0.10, 0.15):
                    pk, prom = find_atomic_peaks(x, h, pr, max(4, int(0.08 * FPS)))
                    pk_dense = [p for p in pk if dense_t[p]]
                    pt = np.array(pk_dense) / FPS
                    matched = set()
                    for t in pt:
                        if len(tc) and np.min(np.abs(tc - t)) <= MATCH_MS / 1000.0:
                            # one-to-one greedy
                            order = np.argsort(np.abs(tc - t))
                            for k2 in order[:4]:
                                if abs(tc[k2] - t) <= MATCH_MS / 1000.0 and k2 not in matched:
                                    matched.add(k2)
                                    break
                    n_dc = sum(1 for e in w.contacts if e["dense"])
                    prec = len(matched) / max(len(pt), 1)
                    rec = len(matched) / max(n_dc, 1)
                    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                    grid.append({"h": h, "pr": pr, "peaks_dense": len(pt),
                                 "P": round(prec, 3), "R": round(rec, 3), "F1": round(f1, 3)})
                    if best is None or f1 > best["F1"]:
                        best = grid[-1]
            sweeps[name] = {"best": best, "grid": grid}

        # ---- complementarity: contacts missed by jz_diff peaks, caught by glass ----
        jd, gb, hd = ch["jz_diff"], ch["glass_bright"], ch["hand_diff"]
        scale_jd = np.percentile(jd, 99.5) + 1e-9
        scale_gb = np.percentile(gb, 99.5) + 1e-9
        pk_jd, _ = find_atomic_peaks(jd, 0.15, 0.06, max(4, int(0.08 * FPS)))
        pk_gb, _ = find_atomic_peaks(gb, 0.15, 0.06, max(4, int(0.08 * FPS)))
        t_jd, t_gb = np.array(pk_jd) / FPS, np.array(pk_gb) / FPS
        miss_jd = hit_gb = 0
        for e in w.contacts:
            if not e["dense"]:
                continue
            has_jd = len(t_jd) and np.min(np.abs(t_jd - e["t_video"])) <= MATCH_MS / 1000.0
            has_gb = len(t_gb) and np.min(np.abs(t_gb - e["t_video"])) <= MATCH_MS / 1000.0
            if not has_jd:
                miss_jd += 1
                hit_gb += int(has_gb)

        # ---- hand_diff support/veto check on R5 candidates ----
        scale_hd = np.percentile(hd, 99.5) + 1e-9
        fp_stats, tp_stats = [], []
        for c in fps:
            k = int(round(c["t_video"] * FPS))
            wmax = float(hd[max(k - 3, 0):k + 4].max()) if 0 <= k < len(hd) else 0
            fp_stats.append(wmax / scale_hd)
        for e in tps:
            k = int(round(e["t_video"] * FPS))
            wmax = float(hd[max(k - 3, 0):k + 4].max()) if 0 <= k < len(hd) else 0
            tp_stats.append(wmax / scale_hd)

        # ---- locked-average waveforms (persistence / sharpness, Q5) ----
        HALF = 18  # +/-300 ms at 60fps
        lock = {}
        dense_idx = [int(round(e["t_video"] * FPS)) for e in w.contacts if e["dense"]]
        for name, x in ch.items():
            acc = []
            for k in dense_idx:
                if k - HALF >= 0 and k + HALF < len(x):
                    v = x[k - HALF:k + HALF + 1].copy()
                    v /= (np.percentile(x, 99.5) + 1e-9)
                    acc.append(v)
            lock[name] = np.mean(acc, axis=0) if acc else np.zeros(2 * HALF + 1)
        # jz_text decay after single contacts: contacts with no neighbour within 500 ms
        iso_idx = []
        tt = w.tc()
        for k, e in enumerate(w.contacts):
            others = np.delete(tt, k)
            if len(others) and np.min(np.abs(others - e["t_video"])) > 0.5:
                iso_idx.append(int(round(e["t_video"] * FPS)))
        decay = {}
        x = ch["jz_text"]
        prof = []
        for k in iso_idx:
            if k + HALF < len(x):
                v = x[k - 2:k + HALF] / (np.percentile(x, 99.5) + 1e-9)
                prof.append(v)
        decay = np.median(np.array(prof), axis=0) if prof else None

        # ---- inter-contact intervals inside dense regions ----
        dt = []
        for e in w.contacts:
            if not e["dense"]:
                continue
            near = [o["t_video"] for o in w.contacts if o is not e]
            if near:
                d = float(np.min(np.abs(np.array(near) - e["t_video"])))
                if d < 0.6:
                    dt.append(d)
        dt = np.array(dt)

        key = w.key
        audit[key] = {
            "n_contacts": len(w.contacts),
            "n_dense": sum(e["dense"] for e in w.contacts),
            "n_sparse": sum(not e["dense"] for e in w.contacts),
            "r5_candidates": len(cands),
            "r5_TP": len(tps), "r5_FN": len(fns), "r5_FP": len(fps),
            "fn_ids": [e["id"] for e in fns],
            "fn_in_dense": sum(e["dense"] for e in fns),
            "fn_score_at_median": float(np.median([r["score_at"] for r in rows if r["is_fn"]])) if fns else None,
            "fn_score_above_thr_frac": float(np.mean([r["score_at"] >= R5_FROZEN_CFG["threshold"]
                                                      for r in rows if r["is_fn"]])) if fns else None,
            "tp_score_at_median": float(np.median([r["score_at"] for r in rows if r["is_tp"]])),
            "channel_atomic_sweep_best": {k: v["best"] for k, v in sweeps.items()},
            "jz_diff_missed_dense_contacts": miss_jd,
            "of_those_glass_bright_hits": hit_gb,
            "hand_diff_norm_at_fp_median": float(np.median(fp_stats)) if fp_stats else None,
            "hand_diff_norm_at_tp_median": float(np.median(tp_stats)) if tp_stats else None,
            "dense_intercontact_ms": {
                "n": len(dt), "p10": float(np.percentile(dt, 10) * 1000),
                "p25": float(np.percentile(dt, 25) * 1000),
                "median": float(np.median(dt) * 1000)},
        }
        cases_out[key] = {"rows": rows, "sweeps": sweeps, "lock": lock,
                          "decay": decay, "score": score, "cands": cands,
                          "tps": [e["id"] for e in tps],
                          "fns": [e["id"] for e in fns],
                          "fps": [round(c["t_video"], 3) for c in fps]}
        save_json(os.path.join(R55_LOG, f"phase1_audit_{key}.json"), audit[key])

    # ---- cross-window summary answers ----
    summary = {}
    for q in ("Q1_sharpest_dense_channel", "Q2_atomic_trigger", "Q3_glass_complement",
              "Q4_hand_support", "Q5_text_persistence"):
        summary[q] = None
    save_json(os.path.join(R55_LOG, "phase1_audit_summary.json"),
              {"windows": audit, "answers_placeholder": summary})

    # ---- trace figures: FN / TP-in-burst / FP cases ----
    for w in windows:
        d = cases_out[w.key]
        rows = d["rows"]
        ch = chan_series(w.series)
        fns = [r for r in rows if r["is_fn"] and r["dense"]]
        tps_b = [r for r in rows if r["is_tp"] and r["dense"]]
        fps_r = d["fps"]
        # choose up to 8 FN, 4 TP (dense), 4 FP
        sel = [("FN", r) for r in fns[:8]]
        sel += [("TP", r) for r in tps_b[:4]]
        ncol = 4
        nrow = int(np.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow * 4, ncol, figsize=(4.6 * ncol, 2.0 * 4 * nrow),
                                 sharex=False)
        tt = w.tc()
        for ci, (kind, r) in enumerate(sel):
            k0 = int(round(r["t"] * FPS))
            a, b = max(k0 - HALF, 0), min(k0 + HALF + 1, w.n)
            xs = (np.arange(a, b) - k0) / FPS * 1000
            for ri, (name, x) in enumerate(ch.items()):
                ax = axes[(ci // ncol) * 4 + ri][ci % ncol]
                ax.plot(xs, x[a:b], lw=0.9, color=["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"][ri])
                for t in tt:
                    tm = (t - r["t"]) * 1000
                    if -300 <= tm <= 300:
                        ax.axvline(tm, color="green", lw=0.7, alpha=0.6)
                for t in d["cands"]:
                    tm = (t["t_video"] - r["t"]) * 1000
                    if -300 <= tm <= 300:
                        ax.axvline(tm, color="orange", lw=0.7, alpha=0.5, ls="--")
                if ri == 0:
                    kind_mark = {"FN": "x FN", "TP": "o TP"}[kind]
                    ax.set_title(f"{kind_mark} {r['id']} t={r['t']:.2f}s "
                                 f"({r['type']}/{r['conf']})", fontsize=8)
                if ci % ncol == 0:
                    ax.set_ylabel(name, fontsize=7)
        for ci in range(len(sel), nrow * ncol):
            for ri in range(4):
                axes[(ci // ncol) * 4 + ri][ci % ncol].axis("off")
        fig.suptitle(f"{w.key}: per-contact traces (green=oracle, dashed orange=R5 cand)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(os.path.join(R55_OUT, f"phase1_traces_{w.key}.png"), dpi=110)
        plt.close(fig)

    # ---- locked-average figure ----
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    ms = np.arange(-HALF, HALF + 1) / FPS * 1000
    for wi, w in enumerate(windows):
        d = cases_out[w.key]
        for ni, name in enumerate(["jz_text", "jz_diff", "glass_bright", "hand_diff"]):
            ax = axes[wi][ni]
            ax.plot(ms, d["lock"][name], lw=1.2)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{w.key} locked avg: {name} (dense contacts)", fontsize=9)
        ax = axes[wi][4]
        if d["decay"] is not None:
            msd = np.arange(-2, len(d["decay"]) - 2) / FPS * 1000
            ax.plot(msd, d["decay"], lw=1.2, color="purple")
            ax.axhline(0.5, color="gray", lw=0.6, ls="--")
            half_ms = msd[np.argmax(d["decay"] < 0.5)] if (d["decay"] < 0.5).any() else None
            ax.set_title(f"{w.key} jz_text after isolated contact "
                         f"(half-decay ~{half_ms:.0f} ms)" if half_ms is not None
                         else f"{w.key} jz_text decay (no half-decay in 300ms)", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no isolated contacts", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(R55_OUT, "phase1_locked_avg.png"), dpi=110)
    plt.close(fig)

    # console summary
    for key, a in audit.items():
        print(f"\n=== {key}: {a['n_contacts']} contacts ({a['n_dense']} dense / "
              f"{a['n_sparse']} sparse) | R5: TP {a['r5_TP']} FN {a['r5_FN']} "
              f"({a['fn_in_dense']} dense) FP {a['r5_FP']}")
        print(f"  FN fused score at contact: median {a['fn_score_at_median']:.2f} "
              f"(thr {R5_FROZEN_CFG['threshold']}), above-thr {a['fn_score_above_thr_frac']:.0%}"
              f" | TP median {a['tp_score_at_median']:.2f}")
        for chname, b in a["channel_atomic_sweep_best"].items():
            print(f"  atomic [{chname}]: h={b['h']} pr={b['pr']} -> P {b['P']:.3f} "
                  f"R {b['R']:.3f} F1 {b['F1']:.3f} (peaks {b['peaks_dense']})")
        print(f"  jz_diff-missed dense contacts: {a['jz_diff_missed_dense_contacts']}, "
              f"of which glass_bright hits {a['of_those_glass_bright_hits']}")
        print(f"  hand_diff norm: TP median {a['hand_diff_norm_at_tp_median']:.3f} "
              f"FP median {a['hand_diff_norm_at_fp_median']:.3f}")
        di = a["dense_intercontact_ms"]
        print(f"  dense inter-contact: p10 {di['p10']:.0f} p25 {di['p25']:.0f} "
              f"median {di['median']:.0f} ms (n={di['n']})")


if __name__ == "__main__":
    main()
