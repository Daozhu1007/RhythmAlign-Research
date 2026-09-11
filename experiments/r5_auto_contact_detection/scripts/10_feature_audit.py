# R5 Phase 2 - visual feature audit on the R1 Golden Sample (development set).
# Extracts per-frame features (shared code in r5_common) and audits them against
# the R4 oracle: do candidate features fire at player contacts and stay quiet at
# known non-contacts (hover gap / rejected / rest)?
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from r5_common import (R1_OUT, R4_OUT, WORK, OUT, LOG, load_json, save_json,
                       extract_features, iter_frames, JZ_BAND as JZ)

VID = os.path.join(R1_OUT, "golden_video.mp4")
FPS = 60.04


def main():
    n, S = extract_features(VID)
    print("frames:", n, "features:", list(S))
    np.savez(os.path.join(WORK, "dev_features.npz"), fps=FPS, **S)

    oracle = load_json(os.path.join(R4_OUT, "contacts_oracle.json"))
    contacts = [e for e in oracle["events"]
                if e["type"] in ("press", "touch", "slide")
                and e.get("status") != "rejected_noncontact"]
    noncontacts = [e for e in oracle["events"]
                   if e["type"] in ("release", "rest", "none")
                   or e.get("status") == "rejected_noncontact"]
    print(f"oracle: {len(contacts)} contacts, {len(noncontacts)} non-contacts")
    strong = [e for e in contacts if e["confidence"] == "high"]

    t = np.arange(n) / FPS
    tc = np.array([e["t_video"] for e in contacts])
    tnc = np.array([e["t_video"] for e in noncontacts])
    tstr = np.array([e["t_video"] for e in strong])

    # --- per-event feature onset audit -------------------------------------
    # For each event: max positive 1-frame jump of each feature in [t-2, t+4] frames
    # (judgment text appears 0-2 frames AFTER physical contact, R4 notes).
    half_before, half_after = int(0.034 * FPS), int(0.067 * FPS)
    rows = []
    for e in contacts:
        i0 = int(round(e["t_video"] * FPS))
        a, b = max(i0 - half_before, 1), min(i0 + half_after + 1, n)
        row = {"id": e["id"], "t": e["t_video"], "type": e["type"],
               "conf": e["confidence"]}
        for k in S:
            d = np.diff(S[k])
            row[k] = float(d[a - 1:b].max())
        rows.append(row)
    for e in noncontacts:
        i0 = int(round(e["t_video"] * FPS))
        a, b = max(i0 - half_before, 1), min(i0 + half_after + 1, n)
        row = {"id": e["id"], "t": e["t_video"], "type": e["type"],
               "conf": e["confidence"], "noncontact": True}
        for k in S:
            d = np.diff(S[k])
            row[k] = float(d[a - 1:b].max())
        rows.append(row)
    save_json(os.path.join(LOG, "feature_audit_events.json"), rows)

    keys = ["jz_warm", "jz_white", "jz_diff", "glass_bright", "hand_diff"]
    print("\nfeature jump at contacts (median / p25) vs non-contacts (median / max):")
    for k in keys:
        cv = np.array([r[k] for r in rows if not r.get("noncontact")])
        nv = np.array([r[k] for r in rows if r.get("noncontact")])
        sv = np.array([r[k] for r in rows
                       if not r.get("noncontact") and r["conf"] == "high"])
        print(f"  {k:11s} contacts med {np.median(cv):7.1f} p25 {np.percentile(cv, 25):7.1f} | "
              f"strong med {np.median(sv):7.1f} | noncontact med {np.median(nv):6.1f} "
              f"max {nv.max():7.1f}")

    # --- separation check: threshold sweep per feature ----------------------
    print("\nsingle-feature threshold sweep (event recall @ FP count):")
    sweep = {}
    for k in keys:
        d = np.diff(S[k])
        for thr in ([50, 100, 200, 400, 800, 1600, 3200] if k in ("jz_warm", "jz_white", "glass_bright")
                    else [0.5, 1, 2, 4, 8, 16, 32]):
            # frame-level: is there a jump > thr within +/-3 frames of the event?
            hit = 0
            for e in contacts:
                i0 = int(round(e["t_video"] * FPS))
                a, b = max(i0 - 3, 1), min(i0 + 5, n)
                if (d[a - 1:b] > thr).any():
                    hit += 1
            fp = 0
            for e in noncontacts:
                i0 = int(round(e["t_video"] * FPS))
                a, b = max(i0 - 3, 1), min(i0 + 5, n)
                if (d[a - 1:b] > thr).any():
                    fp += 1
            sweep.setdefault(k, []).append({"thr": thr, "contact_recall": hit / len(contacts),
                                            "strong_recall": np.mean([
                                                (d[max(int(round(e['t_video']*FPS))-3, 1):
                                                   min(int(round(e['t_video']*FPS))+5, n)] > thr).any()
                                                for e in strong]), "noncontact_fp": fp})
    for k in keys:
        print(f"  {k}:")
        for row in sweep[k]:
            print(f"    thr {row['thr']:6.1f}: recall {row['contact_recall']:.2f} "
                  f"strong {row['strong_recall']:.2f} noncontact FP {row['noncontact_fp']}")
    save_json(os.path.join(LOG, "feature_audit_sweep.json"), sweep)

    # --- trace plots vs oracle ---------------------------------------------
    fig, axes = plt.subplots(len(keys) + 1, 1, figsize=(18, 12), sharex=True)
    for ax, k in zip(axes, keys):
        ax.plot(t, S[k], lw=0.6)
        for e in contacts:
            ax.axvline(e["t_video"], color="g", lw=0.5, alpha=0.6)
        for e in noncontacts:
            ax.axvline(e["t_video"], color="r", lw=1.2, alpha=0.8)
        ax.set_ylabel(k, fontsize=8)
    axes[0].set_title("dev features | green=oracle contact, red=non-contact "
                      "(release/rest/rejected)")
    # empty bottom axis: oracle regions
    ax = axes[-1]
    for rg in oracle["regions"]:
        c = "r" if rg["id"].startswith("G") else "b"
        ax.axvspan(rg["t0"], rg["t1"], alpha=0.25, color=c)
        ax.text((rg["t0"] + rg["t1"]) / 2, 0.5, rg["id"], ha="center")
    ax.set_xlim(0, 15)
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "dev_feature_traces.png"), dpi=80)
    print("saved outputs/dev_feature_traces.png")

    # --- judgment zone spot-check crops (ROI validity) ----------------------
    from PIL import Image
    picks = [0.4, 7.42, 11.9]  # quiet / strong touch / late tap
    tiles = []
    for tp in picks:
        i0 = min(int(round(tp * FPS)), n - 1)
        got = None
        for i, fr in iter_frames(VID):
            if i == i0:
                got = fr
                break
        x0, y0, x1, y1 = JZ
        im = Image.fromarray(got[y0:y1, x0:x1].astype(np.uint8))
        im = im.resize((im.width * 3, im.height * 3), Image.NEAREST)
        tiles.append((tp, im))
    sheet = Image.new("RGB", (sum(im.width for _, im in tiles) + 20, tiles[0][1].height),
                      (0, 0, 0))
    xoff = 0
    for tp, im in tiles:
        sheet.paste(im, (xoff, 0))
        xoff += im.width + 10
    sheet.save(os.path.join(OUT, "dev_jz_crop.png"))
    print("saved outputs/dev_jz_crop.png (t=0.4 quiet / 7.42 touch / 11.9 tap)")


if __name__ == "__main__":
    main()
