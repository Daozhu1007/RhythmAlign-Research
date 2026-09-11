# R5 - diagnostic plots: timelines, P/R summary, FP/FN strips, mix spectrograms.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from PIL import Image, ImageDraw

from r5_common import (OUT, WORK, LOG, R1_OUT, R4_OUT, SR, load_json, save_json,
                       detector_score, iter_frames)
from importlib import import_module
d20 = import_module("20_detector_dev")
FPS = 60.04

frozen = load_json(os.path.join(OUT, "detector_config_frozen.json"))
cfg = frozen["config"]


def timeline(features_npz, oracle_path, cands_path, title, out_png, dur):
    d = np.load(features_npz)
    series = {k: d[k] for k in d.files if k != "fps"}
    score, ns = detector_score(series, cfg)
    t = np.arange(len(score)) / FPS
    oracle = load_json(oracle_path)
    if "status" in oracle["events"][0]:
        contacts = [e for e in oracle["events"] if e.get("status") == "ok"]
        nonc = [e for e in oracle["events"] if e.get("status") != "ok"]
    else:
        contacts = [e for e in oracle["events"]
                    if e["type"] in ("press", "touch", "slide")
                    and e.get("status", "ok") != "rejected_noncontact"]
        nonc = [e for e in oracle["events"]
                if e["type"] in ("release", "rest", "none")
                or e.get("status") == "rejected_noncontact"]
    cands = load_json(cands_path)["candidates"]
    fig, axes = plt.subplots(6, 1, figsize=(20, 14), sharex=True)
    axes[0].plot(t, score, lw=0.7, color="k")
    axes[0].scatter([c["t_video"] for c in cands], [c["score"] for c in cands],
                    marker="v", color="r", s=25, zorder=5, label="candidates")
    axes[0].axhline(cfg["threshold"], color="b", lw=0.6, ls="--", label="threshold")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("score")
    for ax, nm in zip(axes[1:5], ["jz_text", "jz_diff", "glass_bright", "hand_diff"]):
        n = ns[nm]
        ax.plot(t, n / (np.percentile(n, 99.5) + 1e-9), lw=0.7)
        ax.set_ylabel(nm, fontsize=8)
    ax = axes[5]
    for rg in oracle["regions"]:
        c = "r" if rg["id"].startswith("G") else "b"
        ax.axvspan(rg["t0"], rg["t1"], alpha=0.2, color=c)
        ax.text((rg["t0"] + rg["t1"]) / 2, 0.5, rg["id"], ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for ax in axes:
        for e in contacts:
            ax.axvline(e["t_video"], color="g", lw=0.4, alpha=0.5)
        for e in nonc:
            ax.axvline(e["t_video"], color="orange", lw=1.0, alpha=0.8)
    axes[0].set_title(f"{title} | green=oracle contact, orange=rest/rejected, red v=candidates")
    axes[-1].set_xlim(0, dur)
    axes[-1].set_xlabel("s (window time)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=80)
    plt.close(fig)
    print("saved", out_png)


def pr_summary():
    dev = load_json(os.path.join(LOG, "dev_metrics_chosen.json"))
    ho = load_json(os.path.join(LOG, "holdout_metrics.json"))
    rows = [("Precision@50", dev["tol50"]["precision"], ho["tol50"]["precision"]),
            ("Recall@50", dev["tol50"]["recall"], ho["tol50"]["recall"]),
            ("F1@50", dev["tol50"]["f1"], ho["tol50"]["f1"]),
            ("F1@33", dev["tol33"]["f1"], ho["tol33"]["f1"]),
            ("Strong recall", dev["strong"]["recall_50ms"], ho["strong"]["recall_50ms"]),
            ("Region coverage", dev["region_coverage_overall"], ho["region_coverage_overall"])]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, label="dev (golden)")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.36, label="holdout (blind)")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("R5 detector: dev vs holdout")
    for xi, r in zip(x, rows):
        ax.text(xi - 0.18, r[1] + 0.02, f"{r[1]:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.18, r[2] + 0.02, f"{r[2]:.2f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pr_dev_vs_holdout.png"), dpi=110)
    plt.close(fig)
    print("saved pr_dev_vs_holdout.png")


def fpfn_strips():
    ho = load_json(os.path.join(LOG, "holdout_metrics.json"))
    frames = {}
    picks = []
    for x in ho["false_positives"][::9][:4]:
        picks.append((x["t"], f"FP s={x['score']}"))
    fn_ids = ho["false_negatives"][::12][:4]
    oracle = load_json(os.path.join(OUT, "holdout_oracle_blind.json"))
    ev = {e["id"]: e for e in oracle["events"]}
    for fid in fn_ids:
        picks.append((ev[fid]["t_video"], f"FN {fid} {ev[fid]['type']}/{ev[fid]['confidence']}"))
    want = set()
    for t, _ in picks:
        i0 = int(round(t * FPS))
        want.update(range(max(i0 - 1, 0), i0 + 2))
    for i, fr in iter_frames(os.path.join(OUT, "holdout_video.mp4")):
        if i in want:
            frames[i] = fr.astype(np.uint8)
        if len(frames) == len(want):
            break
    TW, TH = 384, 216
    sheet = Image.new("RGB", (TW * 3 + 8, len(picks) * (TH + 20)), (8, 8, 8))
    for j, (t, tag) in enumerate(picks):
        i0 = int(round(t * FPS))
        for k, off in enumerate((-1, 0, 1)):
            fi = min(max(i0 + off, 0), 1080)
            im = Image.fromarray(frames[fi]).resize((TW, TH))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 220, 16], fill=(0, 0, 0))
            dr.text((3, 2), f"{tag} t={t:.3f} {off:+d}f",
                    fill=(0, 255, 255) if off == 0 else (180, 180, 180))
            sheet.paste(im, (k * (TW + 2), j * (TH + 20)))
    sheet.save(os.path.join(OUT, "holdout_fp_fn_strips.jpg"), quality=88)
    print("saved holdout_fp_fn_strips.jpg")


def mix_spec():
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    nf, hop = 1536, 384
    rows = [("holdout_auto_C1_final_mix.wav", "AUTO final mix"),
            ("holdout_oracle_C1_final_mix.wav", "ORACLE final mix"),
            ("holdout_auto_C1_interaction.wav", "AUTO interaction stem"),
            ("holdout_oracle_C1_interaction.wav", "ORACLE interaction stem")]
    oracle = load_json(os.path.join(OUT, "holdout_oracle_blind.json"))
    tcts = [e["t_video"] for e in oracle["events"] if e["status"] == "ok"]
    for ax, (f, title) in zip(axes, rows):
        y, _ = librosa.load(os.path.join(OUT, f), sr=SR, mono=True)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y.astype(np.float32),
                                                        n_fft=nf, hop_length=hop)) + 1e-10)
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(y) / SR, 0, SR // 2], vmax=S.max(), vmin=S.max() - 90)
        for t in tcts:
            ax.axvline(t, color="cyan", lw=0.3, alpha=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("s (holdout window)")
    fig.suptitle("holdout: AUTO vs ORACLE (cyan = blind-oracle contacts)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "holdout_mix_compare.png"), dpi=100)
    plt.close(fig)
    print("saved holdout_mix_compare.png")


if __name__ == "__main__":
    timeline(os.path.join(WORK, "dev_features.npz"),
             os.path.join(R4_OUT, "contacts_oracle.json"),
             os.path.join(OUT, "dev_contacts_auto_visual.json"),
             "DEV (golden 15 s)", os.path.join(OUT, "dev_timeline.png"), 15.0)
    timeline(os.path.join(WORK, "holdout_features.npz"),
             os.path.join(OUT, "holdout_oracle_blind.json"),
             os.path.join(OUT, "holdout_contacts_auto_visual.json"),
             "HOLDOUT (18 s, frozen detector)", os.path.join(OUT, "holdout_timeline.png"), 18.0)
    pr_summary()
    fpfn_strips()
    mix_spec()
