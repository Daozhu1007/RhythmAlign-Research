# R5.5 plots: Holdout-C timeline (features + score + burst mask + candidates vs
# oracle), dev-vs-holdout metric summary, FP distance histogram.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common55 import (R55_OUT, R55_LOG, R55_WORK, load_json, save_json, FPS,
                      load_windows, match_one_to_one)
from detector55 import DEFAULT55, detect55

CFG = {**DEFAULT55, "burst_enter": 0.40, "burst_stay": 0.22,
       "burst_hold_frames": 30, "a_prom_frac": 0.03, "a_min_dist_frames": 3}


def timeline():
    d = np.load(os.path.join(R55_WORK, "holdoutC_features.npz"))
    series = {k: d[k] for k in d.files if k != "fps"}
    score, mask, ivs, env, cands = detect55(series, FPS, CFG, rule="A")
    oracle = load_json(os.path.join(R55_OUT, "holdoutC_oracle_blind.json"))
    contacts = [e for e in oracle["events"] if e["status"] == "ok"]
    tc = [e["t_video"] for e in contacts]
    tp = [c["t_video"] for c in cands]
    t = np.arange(len(score)) / FPS
    ch = {"jz_text": series["jz_warm"] + series["jz_white"],
          "jz_diff": series["jz_diff"],
          "glass_bright": series["glass_bright"],
          "hand_diff": series["hand_diff"]}
    fig, axes = plt.subplots(5, 1, figsize=(18, 11), sharex=True)
    for ax, (name, x) in zip(axes[:4], ch.items()):
        ax.plot(t, x, lw=0.7)
        ax.set_ylabel(name, fontsize=9)
        for g in tc:
            ax.axvline(g, color="green", lw=0.8, alpha=0.65)
    axes[4].plot(t, score, lw=0.8, color="crimson", label="fused score")
    axes[4].fill_between(t, 0, mask * np.max(score), color="orange", alpha=0.15,
                         label="burst state")
    axes[4].set_ylabel("fused score", fontsize=9)
    for g in tc:
        axes[4].axvline(g, color="green", lw=0.8, alpha=0.65)
    for c in cands:
        axes[4].axvline(c["t_video"], color="blue", lw=0.6, alpha=0.35, ls="--")
    axes[4].legend(fontsize=8, loc="upper right")
    axes[4].set_xlabel("holdout-C time (s)")
    fig.suptitle("Holdout-C: green = blind oracle, dashed blue = R5.5 frozen detector, "
                 "orange fill = LEVEL-1 burst state", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(R55_OUT, "holdoutC_timeline.png"), dpi=100)
    plt.close(fig)


def summary_bar():
    rows = [
        ("devA F1@50", 0.605, 0.619), ("devA strong R", 0.842, 0.947),
        ("devB F1@50", 0.488, 0.586), ("devB strong R", 0.455, 0.818),
        ("holdC F1@50", 0.367, 0.417), ("holdC strong R", 0.429, 0.810),
        ("holdC gate cov", 0.735, 0.765), ("holdC strong gate", 0.765, 0.794),
    ]
    x = np.arange(len(rows))
    r5v = [a[1] for a in rows]
    r55v = [a[2] for a in rows]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - 0.2, r5v, 0.4, label="R5 frozen (baseline)", color="gray")
    ax.bar(x + 0.2, r55v, 0.4, label="R5.5 frozen (rule A)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right", fontsize=9)
    ax.axhline(0.80, color="green", lw=0.8, ls=":", label="strong target 0.80")
    for xi, v in zip(x - 0.2, r5v):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    for xi, v in zip(x + 0.2, r55v):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.legend(fontsize=9)
    ax.set_title("R5 vs R5.5 (holdC baseline = R5 detector on the SAME window)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(R55_OUT, "r55_vs_r5_summary.png"), dpi=110)
    plt.close(fig)


def fp_hist():
    vis = load_json(os.path.join(R55_OUT, "holdoutC_contacts_auto_visual.json"))["candidates"]
    oracle = load_json(os.path.join(R55_OUT, "holdoutC_oracle_blind.json"))
    tc = np.array([e["t_video"] for e in oracle["events"] if e["status"] == "ok"])
    tp = np.array([c["t_video"] for c in vis])
    _, mp, _ = match_one_to_one(tp, tc, 0.050)
    fpd = [float(np.min(np.abs(tc - t))) * 1000 for j, t in enumerate(tp) if j not in mp]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(fpd, bins=24, color="salmon", edgecolor="k")
    ax.set_xlabel("FP distance to nearest oracle contact (ms)")
    ax.set_ylabel("count")
    ax.set_title(f"Holdout-C FP structure (n={len(fpd)}): burst-association echo dominant")
    fig.tight_layout()
    fig.savefig(os.path.join(R55_OUT, "holdoutC_fp_hist.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    timeline()
    summary_bar()
    fp_hist()
    print("plots written")
