# R5-FIX PART 1b - diagnostic plots.
# corrected_C1_envelope_audit.png : corrected envelope on the real first golden
#   span (same region as the review's audit of the saved defective C1) + step sanity.
# old_vs_corrected_C1_envelope.png : full golden window old vs corrected + zoom.
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

W = fx.FIX_WORK
SR = fx.SR


def main():
    g_fixed = np.load(os.path.join(W, "env_golden_fixed.npy"))
    g_old = np.load(os.path.join(W, "env_golden_old.npy"))
    t = np.arange(len(g_fixed)) / SR * 1000.0   # ms

    # ---- audit plot (analogous to the review's C1_envelope_audit.png) ----
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.4]})
    ax = axes[0]
    msk = (t >= 30) & (t <= 580)
    ax.plot(t[msk], g_fixed[msk], lw=1.8, color="#1a7f37",
            label="corrected C1 gain (fixed implementation)")
    ax.axhline(1.0, color="#999", lw=0.6, ls=":")
    ax.axhline(0.0, color="#999", lw=0.6, ls=":")
    s0 = 2663 / SR * 1000
    e0 = 25449 / SR * 1000
    ax.annotate("entry: rises 0→1 over 10 ms", xy=(s0 + 5, 0.5), xytext=(s0 + 40, 0.42),
                arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
    ax.annotate("interior = 1", xy=(150, 1.0), xytext=(170, 1.06), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("exit: falls 1→0 over 60 ms", xy=(e0 - 30, 0.5), xytext=(e0 - 130, 0.55),
                arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
    ax.annotate("outside support = 0", xy=(572, 0.0), xytext=(480, 0.16),
                arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("C1 gain")
    ax.set_title("R5-FIX corrected C1 envelope audit — first golden span "
                 f"({s0:.1f}–{e0:.1f} ms; policy pre15/post150 unchanged)")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    step_fix = np.abs(np.diff(g_fixed))
    step_old = np.abs(np.diff(g_old))
    td = t[1:]
    mskd = msk[1:]
    ax.plot(td[mskd], step_old[mskd], lw=0.9, color="#b02a2a",
            label="|Δgain| OLD (defective) — jumps ≈ 1")
    ax.plot(td[mskd], step_fix[mskd], lw=0.9, color="#1a7f37",
            label="|Δgain| corrected — bounded ≤ 3.3e-3")
    ax.set_xlabel("golden-window time (ms)")
    ax.set_ylabel("|Δ gain| / sample")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(fx.FIX, "corrected_C1_envelope_audit.png"), dpi=150)
    plt.close(fig)

    # ---- old vs corrected, full window + zoom ----
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax = axes[0]
    ax.fill_between(t / 1000.0, g_old, step="mid", alpha=0.35, color="#b02a2a",
                    label="OLD defective envelope (historical, unchanged)")
    ax.plot(t / 1000.0, g_fixed, lw=0.4, color="#1a7f37", label="corrected envelope")
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("golden-window time (s)")
    ax.set_ylabel("C1 gain")
    ax.set_title("old vs corrected C1 envelope — golden oracle spans "
                 "(84 events, 40 merged spans, identical support)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    msk = (t >= 30) & (t <= 580)
    ax.plot(t[msk], g_old[msk], lw=1.6, color="#b02a2a",
            label="OLD: entry 1→0 fall, 0→1 jump; exit 1→0 jump, 0→1 rise, final drop")
    ax.plot(t[msk], g_fixed[msk], lw=1.6, color="#1a7f37",
            label="corrected: 0→1 rise, interior 1, 1→0 fall")
    for x, txt in ((2663 / SR * 1000, "OLD 0→1 jump at entry"),
                   (3143 / SR * 1000, "OLD 0→1 jump at attack end"),
                   (22569 / SR * 1000, "OLD 1→0 jump at release start"),
                   (25449 / SR * 1000, "OLD abrupt 1→0 at span end")):
        ax.axvline(x, color="#b02a2a", lw=0.7, ls="--", alpha=0.6)
        ax.text(x, 1.05, txt, rotation=90, fontsize=7.5, va="top", ha="right", color="#b02a2a")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("golden-window time (ms)")
    ax.set_ylabel("C1 gain")
    ax.set_title("zoom on the first golden span (55.5–530.2 ms): the four OLD discontinuities")
    ax.legend(fontsize=8.5, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fx.FIX, "old_vs_corrected_C1_envelope.png"), dpi=150)
    plt.close(fig)
    print("plots written")


if __name__ == "__main__":
    main()
