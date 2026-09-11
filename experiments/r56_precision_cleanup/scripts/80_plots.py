# R5.6 plots: dev E0/E1/E2 summary + holdout D E0-vs-E2.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common56 as c

dev = c.load_json(os.path.join(c.R56_LOG, "dev_eval_E0_E1_E2.json"))
hd = c.load_json(os.path.join(c.R56_LOG, "holdoutD_metrics.json"))


def bars(ax, labels, series, title, ylabel):
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    for i, (name, vals) in enumerate(series):
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()


def main():
    wins = ["devA", "devB", "devC"]
    fig, axs = plt.subplots(2, 3, figsize=(17, 8))
    for ax, (title, get) in zip(axs.flat, [
            ("false-window duration (s, lower better)", lambda m: m["false_windows"]["total_s"]),
            ("music leakage (onsets, lower better)", lambda m: m["music_onset_matched"]),
            ("USEFUL WINDOW RATIO (hi+med, higher better)", lambda m: m["useful_window_ratio_highmed"]),
            ("strong gate coverage (higher better)", lambda m: m["gate_coverage_strong"]),
            ("event F1 @50ms", lambda m: m["events"]["tol50"]["F1"]),
            ("final mix RMS diff vs oracle (dB, lower better)", lambda m: m["auto_vs_oracle_final_rms_db"])]):
        series = [(v, [get(dev[w][v]) for w in wins]) for v in ("E0", "E1", "E2")]
        bars(ax, wins, series, title, "")
    fig.suptitle("R5.6 dev: E0 = R5.5 frozen | E1 = cleanup | E2 = cleanup + burst audio completion")
    fig.tight_layout()
    fig.savefig(os.path.join(c.R56_OUT, "dev_summary.png"), dpi=110)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    hdE0, hdE2 = hd["E0"], hd["E2"]
    for ax, (title, k0, k2, get) in zip(axs, [
            ("strong recall / gate / strong-gate", "strong_recall", "strong_recall",
             None), ("false windows (s)", "false_windows", "false_windows", "total_s"),
            ("music leak (count)", "music_onset_matched", "music_onset_matched", None)]):
        if isinstance(k0, str) and k0 in ("strong_recall",):
            vals0 = [hdE0["strong_recall"], hdE0["gate_coverage"], hdE0["gate_coverage_strong"]]
            vals2 = [hdE2["strong_recall"], hdE2["gate_coverage"], hdE2["gate_coverage_strong"]]
            bars(ax, ["strong-R", "gate", "strong-gate"], [("E0", vals0), ("E2", vals2)],
                 "Holdout D: recall/gate", "")
        elif get:
            bars(ax, ["holdoutD"], [("E0", [hdE0[k0][get]]), ("E2", [hdE2[k2][get]])], title, "")
        else:
            bars(ax, ["holdoutD"], [("E0", [hdE0[k0]]), ("E2", [hdE2[k2]])], title, "")
    fig.suptitle("R5.6 Holdout D (video-first blind oracle, 86 events): E0 = R5.5 vs E2 = R5.6 frozen")
    fig.tight_layout()
    fig.savefig(os.path.join(c.R56_OUT, "holdoutD_summary.png"), dpi=110)
    print("plots written")


if __name__ == "__main__":
    main()
