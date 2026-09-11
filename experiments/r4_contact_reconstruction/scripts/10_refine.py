# R4 Phase 2 - audio onset refinement of oracle contacts.
# VISION determines identity (contacts_oracle.json); AUDIO determines exact timing.
# t0_audio = t_video + 17.625 ms (known AAC residual offset); search +/-40 ms.
import json
import os

import numpy as np
from scipy import signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R4 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(R4, "work")
OUT = os.path.join(R4, "outputs")
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
SR = 48000
AV_OFF_MS = 17.625

def main():
    import librosa
    y, _ = librosa.load(os.path.join(R1_WORK, "golden_ctx.wav"), sr=SR, mono=True)
    g = y[int(3.0 * SR):int(18.0 * SR)]
    ref, _ = librosa.load(os.path.join(R1_WORK, "ref_warp_fixed.wav"), sr=SR, mono=True)
    r = ref[int(3.0 * SR):int(18.0 * SR)]
    gg, rr = g - g.mean(), r - r.mean()
    gain = float(np.dot(gg, rr) / np.dot(rr, rr))
    resid = g - gain * r

    def band_env(x, lo, hi):
        sosb = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
        xb = sig.sosfiltfilt(sosb, x)
        env = np.abs(xb)
        kern = sig.windows.hann(49)  # ~1 ms
        return sig.convolve(env, kern, mode="same")

    bass = band_env(g, 150, 500)          # body thump (raw, music shares band)
    mid = band_env(resid, 500, 2000)      # impact body (music removed)
    hi1 = band_env(resid, 2000, 6000)     # click
    hi2 = band_env(resid, 6000, 14000)    # attack edge

    def norm(e):
        p99 = np.percentile(e, 99.9)
        return e / (p99 + 1e-9)

    parts = [norm(mid), 1.15 * norm(hi1), 0.9 * norm(hi2), 0.45 * norm(bass)]
    comb = sum(parts)

    oracle = json.load(open(os.path.join(OUT, "contacts_oracle.json")))
    events = [e for e in oracle["events"] if e.get("status") != "rejected_noncontact"]
    # e68 split: vision saw two contacts (NE-E bezel press ~11.33 and S tap ~11.40)
    refined = []
    for e in events:
        targets = [(e["id"], e["t_video"], e)]
        if e["id"] == "e68":
            targets = [("e68a", 11.326, e), ("e68b", 11.40, {**e, "area": "S", "notes": "S-tap seen at 30fps ~11.40 (judgment text)"})]
        for eid, tv, ee in targets:
            t0 = tv + AV_OFF_MS / 1000.0
            i0 = int(round(t0 * SR))
            lo, hi = max(i0 - int(0.040 * SR), 0), min(i0 + int(0.040 * SR), len(comb))
            seg = comb[lo:hi]
            k = int(np.argmax(seg)) + lo
            t_peak = k / SR
            # local noise floor: median of comb in +/-0.25s excluding +/-0.06s
            a, b = max(k - int(0.25 * SR), 0), min(k + int(0.25 * SR), len(comb))
            excl = np.ones(len(comb), bool)
            excl[max(k - int(0.06 * SR), 0):k + int(0.06 * SR)] = False
            sel = np.arange(a, b)
            med = float(np.median(comb[sel[excl[sel]]]))
            ratio = float(seg.max() / (med + 1e-6))
            conf = "strong" if ratio > 4 and seg.max() > 0.35 else ("present" if ratio > 2.5 else ("weak" if ratio > 1.6 else "no_transient"))
            refined.append({
                "id": eid, "t_video": tv, "type": ee["type"], "area": ee["area"],
                "hand": ee.get("hand", "-"), "oracle_confidence": ee["confidence"],
                "t_audio_coarse": round(t0, 4), "t_audio_refined": round(t_peak, 4),
                "refinement_shift_ms": round((t_peak - t0) * 1000, 1),
                "peak": round(float(seg.max()), 3), "local_median": round(med, 3),
                "peak_over_noise": round(ratio, 1), "refined_confidence": conf,
                "evidence": ee["evidence"],
            })
    json.dump(refined, open(os.path.join(WORK, "contacts_refined_raw.json"), "w"), indent=1)

    shifts = np.array([r["refinement_shift_ms"] for r in refined if r["refined_confidence"] != "no_transient"])
    print("events refined:", len(refined), " with transient:", len(shifts))
    print("shift ms: mean %.1f median %.1f p5 %.1f p95 %.1f" %
          (shifts.mean(), np.median(shifts), np.percentile(shifts, 5), np.percentile(shifts, 95)))
    conf_ct = {}
    for r in refined:
        conf_ct[r["refined_confidence"]] = conf_ct.get(r["refined_confidence"], 0) + 1
    print("refined confidence:", conf_ct)

    # plot
    t = np.arange(len(comb)) / SR
    fig, ax = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    ax[0].plot(t, comb, lw=0.4)
    for r in refined:
        c = {"strong": "g", "present": "g", "weak": "orange", "no_transient": "r"}[r["refined_confidence"]]
        ax[0].axvline(r["t_audio_refined"], color=c, lw=0.8, alpha=0.8)
        ax[0].axvline(r["t_audio_coarse"], color="b", lw=0.4, alpha=0.35)
    ax[0].set_title("combined onset envelope (green/strong, orange/weak, blue=video-derived coarse)")
    ax[1].hist(shifts, bins=40)
    ax[1].set_title("refinement shift distribution (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(WORK, "refinement_plot.png"), dpi=70)

    # also save the envelopes for phase 3
    np.save(os.path.join(WORK, "comb_env.npy"), comb.astype(np.float32))
    np.save(os.path.join(WORK, "band_envs_refine.npy"),
            np.stack([norm(bass), norm(mid), norm(hi1), norm(hi2)]).astype(np.float32))
    json.dump({"gain": gain, "av_off_ms": AV_OFF_MS},
              open(os.path.join(WORK, "refine_meta.json"), "w"))

if __name__ == "__main__":
    import librosa  # noqa
    main()
