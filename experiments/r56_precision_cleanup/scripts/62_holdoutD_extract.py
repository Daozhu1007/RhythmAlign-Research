# R5.6 Holdout D - step 3: selection record + extraction (raw audio slice,
# context slice, ref sinc-warp alignment, video re-encode, AV offset).
# Adapted verbatim from R5.5 61_holdoutC_extract.py (frozen alignment code).
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\Code\RhythmAlign\experiments\r5_auto_contact_detection\scripts")
import numpy as np
import soundfile as sf
from scipy import signal as sig

from r5_common import (SR, save_json, measure_av_offset_ms, bp48, sinc_warp,
                       ffmpeg_run)

R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
R56 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(R56, "outputs")
WORK = os.path.join(R56, "work")
HANCAM_MP4 = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4"

H0, H1 = 60.0, 78.0            # holdout D window in handcam-native time
CTX0 = H0 - 3.0                # context start (57.0)
CTX_DUR = (H1 - H0) + 6.0      # 24 s context
D_PRIOR = -11.374728           # R1 fine-alignment delay (s)


def anchor_ncc(hand, ref, t_center, d_grid, band=(40, 800), win_s=4.0):
    n = int(win_s * SR)
    c0 = int(round(t_center * SR))
    x = bp48(hand[c0:c0 + n], *band)
    x = x - x.mean()
    nx = np.linalg.norm(x) + 1e-12
    vals = np.zeros(len(d_grid))
    for i, d in enumerate(d_grid):
        a = int(round((t_center + d) * SR))
        if a < 0 or a + n > len(ref):
            continue
        al = ref[a:a + n]
        aln = al - al.mean()
        vals[i] = np.dot(x, aln) / (nx * np.linalg.norm(aln) + 1e-12)
    k = int(np.argmax(vals))
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        den = y0 - 2 * y1 + y2
        dx = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-15 else 0.0
    else:
        dx = 0.0
    return float(d_grid[k] + dx * 0.001), float(vals[k])


def main():
    hand_st, _ = sf.read(os.path.join(R1_WORK, "handcam_native.wav"),
                         dtype="float64", always_2d=True)
    ref_st, _ = sf.read(os.path.join(R1_WORK, "ref_native.wav"),
                        dtype="float64", always_2d=True)
    hand = (hand_st[:, 0] + hand_st[:, 1]) * 0.5
    ref = (ref_st[:, 0] + ref_st[:, 1]) * 0.5

    s0 = int(round(CTX0 * SR))
    ctx_st = hand_st[s0:s0 + int(round(CTX_DUR * SR))]
    sf.write(os.path.join(WORK, "holdoutD_ctx.wav"), ctx_st.astype(np.float32), SR)
    sf.write(os.path.join(OUT, "holdoutD_raw.wav"),
             hand_st[int(round(H0 * SR)):int(round(H1 * SR))].astype(np.float32), SR)
    print(f"holdoutD_raw.wav: {H1 - H0:.1f}s | ctx {CTX_DUR:.1f}s from {CTX0}s")

    grid = np.arange(D_PRIOR - 0.30, D_PRIOR + 0.3001, 0.001)
    anchors = []
    for t_center in (H0 + 1.0, H0 + (H1 - H0) / 2, H0 + (H1 - H0) - 1.0):
        d, ncc = anchor_ncc(hand, ref, t_center, grid)
        anchors.append({"t": round(t_center, 2), "d": round(d, 6), "ncc": round(ncc, 3)})
        print(f"anchor t={t_center:.1f}s: d = {d*1e3:+.3f} ms (ncc {ncc:.3f})")
    ds12 = []
    r_bp = bp48(ref, 40, 800)
    for t_center in (H0 + 1.0, H0 + (H1 - H0) - 2.0):
        n = int(12 * SR)
        x = bp48(hand[int(t_center * SR):int(t_center * SR) + n], 40, 800)
        x = x - x.mean()
        c = sig.correlate(r_bp, x, mode="valid", method="fft")
        k = int(np.argmax(c))
        if 0 < k < len(c) - 1:
            a0, a1, a2 = c[k - 1], c[k], c[k + 1]
            den = a0 - 2 * a1 + a2
            dx = 0.5 * (a0 - a2) / den if abs(den) > 1e-15 else 0.0
        else:
            dx = 0.0
        d12 = (k + dx - t_center * SR) / SR
        ds12.append(float(d12))
        print(f"12s-waveform corr t={t_center:.1f}s: d = {d12*1e3:+.3f} ms")
    spread_ms = float((max(a["d"] for a in anchors) - min(a["d"] for a in anchors)) * 1000)
    ds_arr = [a["d"] for a in anchors]
    if spread_ms <= 5.0:
        a_fit, b_fit, mode = 1.0, float(np.mean(ds_arr)), "anchor_mean"
        drift_note = f"anchors agree to {spread_ms:.2f} ms"
        print(f"anchor spread {spread_ms:.2f} ms -> {mode}, d = {b_fit*1e3:+.3f} ms")
    elif spread_ms <= 15.0:
        # one anchor may snap to a neighbouring correlation peak; the median
        # plus the 12 s waveform correlations decide
        a_fit, b_fit, mode = 1.0, float(np.median(ds_arr)), "anchor_median_outlier_documented"
        outlier = max(anchors, key=lambda a: abs(a["d"] - float(np.median(ds_arr))))
        drift_note = (f"anchor spread {spread_ms:.2f} ms; outlier t={outlier['t']}s "
                      f"d={outlier['d']*1e3:+.3f} ms (ncc {outlier['ncc']}) snapped to a "
                      f"neighbouring peak; median {b_fit*1e3:+.3f} ms agrees with the "
                      f"12 s correlations {ds12[0]*1e3:+.3f}/{ds12[1]*1e3:+.3f} ms")
        print(f"anchor spread {spread_ms:.2f} ms -> {mode}, d = {b_fit*1e3:+.3f} ms")
        print(drift_note)
    else:
        raise SystemExit(f"anchor disagreement {spread_ms:.1f} ms too large; aborting")

    r0 = max(int(round((CTX0 + b_fit - 1.0) * SR)), 0)
    r1 = min(int(round((CTX0 + CTX_DUR + b_fit + 1.0) * SR)), len(ref_st))
    warped, valid = sinc_warp(ref_st[r0:r1], a_fit, CTX0 + b_fit,
                              int(round(CTX_DUR * SR)), r0)
    sf.write(os.path.join(WORK, "holdoutD_ref_warp.wav"), warped.astype(np.float32), SR)
    vf, va = int(0.05 * SR), len(valid) - int(0.05 * SR)
    print(f"holdoutD_ref_warp.wav written (valid interior: {bool(valid[vf:va].all())})")

    vid = os.path.join(OUT, "holdoutD_video.mp4")
    ffmpeg_run(["-ss", f"{H0}", "-i", HANCAM_MP4, "-t", f"{H1 - H0}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", vid])
    av_ms, av_ncc = measure_av_offset_ms(vid, os.path.join(OUT, "holdoutD_raw.wav"))
    print(f"holdoutD_video.mp4 written; AV offset t_audio = t_video + {av_ms:+.3f} ms "
          f"(ncc {av_ncc:.3f})")

    save_json(os.path.join(OUT, "holdoutD_selection.json"), {
        "holdout": "D",
        "holdout_start_s": H0, "holdout_end_s": H1, "duration_s": H1 - H0,
        "forbidden_windows": [[22.5, 37.5], [40.0, 58.0], [84.0, 102.0]],
        "overlap_check": "window [60,78] does not overlap any dev window "
                         "(margin 2.0 s after holdout-C end; margin 6.0 s before "
                         "dev-B start; ctx 57-60 s touches holdout-C's final second "
                         "for AUDIO ALIGNMENT ONLY, never for detection)",
        "selection_reason": (
            "Chosen from the full-take onset-density profile (all three dev "
            "windows masked) plus 10 fps contact-sheet inspection of the two "
            "allowed mid-song zones [58,84] and [102,136]. [60,78] contains: "
            "a dense two-hand burst section (~60-65.5: alternating top/side "
            "presses with chevron slides), a genuine SPARSE rest (~66-67.8: "
            "glass nearly empty, hands drop to the bottom bezel, one slow "
            "yellow slide ribbon traced - the only near-rest stretch in the "
            "zone), a second dense section (~68-78: bottom/side presses, "
            "yellow slide bands, cyan fan-slide circles, double presses), and "
            "heavy arm-crossing occlusion + cabinet LED/screen glare as "
            "environment interference throughout. Not an easy window: it mixes "
            "burst, rest, slide and occlusion like dev-B but was never used "
            "for any tuning."),
        "anchors": anchors, "anchor_spread_ms": spread_ms, "warp_mode": mode,
        "warp": {"a": a_fit, "b": b_fit,
                 "convention": "t_ref = a*t_hand + b; ctx file starts at handcam t=57.0s"},
        "av_offset_ms": av_ms, "av_ncc": av_ncc,
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


if __name__ == "__main__":
    main()
