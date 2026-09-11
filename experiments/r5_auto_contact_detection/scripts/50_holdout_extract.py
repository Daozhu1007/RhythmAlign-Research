# R5 Phase 7 - holdout extraction: raw audio slice, video re-encode, AV offset,
# and reference alignment using R1's validated method (anchor NCC + sinc warp).
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf
from scipy import signal as sig

from r5_common import (HANCAM_MP4, R1_WORK, WORK, OUT, LOG, SR, save_json,
                       load_json, measure_av_offset_ms, bp48, sinc_warp,
                       ffmpeg_run)

H0, H1 = 84.0, 102.0          # holdout window in handcam-native time
CTX0 = H0 - 3.0               # context start (81.0)
CTX_DUR = (H1 - H0) + 6.0     # 24 s context
D_PRIOR = -11.374728          # R1 fine-alignment delay at golden window (s)


def anchor_ncc(hand, ref, t_center, d_grid, band=(40, 800), win_s=4.0):
    """hand: FULL handcam mono array; t_center in handcam-native time."""
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

    # ---- anchors at holdout start / middle / end ----
    s0 = int(round(CTX0 * SR))
    s1 = int(round((CTX0 + CTX_DUR) * SR))
    ctx_st = hand_st[s0:s1]
    sf.write(os.path.join(WORK, "holdout_ctx.wav"), ctx_st.astype(np.float32), SR)
    n_win = H1 - H0
    sf.write(os.path.join(OUT, "holdout_raw.wav"),
             hand_st[int(round(H0 * SR)):int(round(H1 * SR))].astype(np.float32), SR)
    print(f"holdout_raw.wav: {n_win:.1f}s | holdout_ctx.wav: {CTX_DUR:.1f}s")

    grid = np.arange(D_PRIOR - 0.30, D_PRIOR + 0.3001, 0.001)
    hand = (hand_st[:, 0] + hand_st[:, 1]) * 0.5
    ref = (ref_st[:, 0] + ref_st[:, 1]) * 0.5
    anchors = []
    for t_center in (H0 + 1.0, H0 + n_win / 2, H0 + n_win - 1.0):
        d, ncc = anchor_ncc(hand, ref, t_center, grid)
        anchors.append({"t": round(t_center, 2), "d": round(d, 6), "ncc": round(ncc, 3)})
        print(f"anchor t={t_center:.1f}s: d = {d*1e3:+.3f} ms (ncc {ncc:.3f})")
    # robust delay: full-waveform bandpass correlation on two 12 s windows
    # (1 s anchor windows proved noisy; 12 s windows agreed to <0.1 ms in scan)
    ds12 = []
    for t_center in (H0 + 1.0, H0 + n_win - 2.0):
        n = int(12 * SR)
        x = bp48(hand[int(t_center * SR):int(t_center * SR) + n], 40, 800)
        x = x - x.mean()
        r_seg = bp48(ref, 40, 800)
        c = sig.correlate(r_seg, x, mode="valid", method="fft")
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
    spread_ms = float(abs(ds12[0] - ds12[1]) * 1000)
    if spread_ms <= 3.0:
        a_fit, b_fit, mode = 1.0, float(np.mean(ds12)), "fixed"
        print(f"drift {spread_ms:.2f} ms <= 3 ms -> fixed warp, d = {b_fit*1e3:+.3f} ms")
    else:
        raise SystemExit(f"anchor disagreement {spread_ms:.1f} ms too large; aborting")
    ds = np.array([a["d"] for a in anchors])
    d_at_mid = a_fit * (H0 + n_win / 2) + b_fit - (H0 + n_win / 2)
    print(f"anchor spread {spread_ms:.2f} ms -> {mode} warp (d@{H0+n_win/2:.1f}s = {d_at_mid*1e3:+.3f} ms)")

    # ---- sinc warp ref into holdout ctx timebase ----
    r0 = max(int(round((CTX0 + b_fit - 1.0) * SR)), 0)
    r1 = min(int(round((CTX0 + CTX_DUR + b_fit + 1.0) * SR)), len(ref_st))
    warped, valid = sinc_warp(ref_st[r0:r1], a_fit, CTX0 + b_fit,
                              int(round(CTX_DUR * SR)), r0)
    sf.write(os.path.join(WORK, "holdout_ref_warp.wav"), warped.astype(np.float32), SR)
    vf, va = int(0.05 * SR), len(valid) - int(0.05 * SR)
    print(f"holdout_ref_warp.wav written (valid interior: {bool(valid[vf:va].all())})")

    # ---- video extract + AV offset ----
    vid = os.path.join(OUT, "holdout_video.mp4")
    ffmpeg_run(["-ss", f"{H0}", "-i", HANCAM_MP4, "-t", f"{H1 - H0}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", vid])
    av_ms, av_ncc = measure_av_offset_ms(vid, os.path.join(OUT, "holdout_raw.wav"))
    print(f"holdout_video.mp4 written; AV offset t_audio = t_video + {av_ms:+.3f} ms "
          f"(ncc {av_ncc:.3f})")

    save_json(os.path.join(OUT, "holdout_selection.json"), {
        "holdout_start_s": H0, "holdout_end_s": H1, "duration_s": H1 - H0,
        "selection_reason": ("audio onset-density profile over the full take: 84-85 "
                             "dense burst, 86-95 sparse passage, 96-102 denser with "
                             "yellow slide traces; mid-song (no menu/result), no "
                             "overlap with golden [22.5, 37.5]; contact sheets "
                             "verify press/touch/slide mix with both hands"),
        "anchors": anchors, "anchor_spread_ms": spread_ms, "warp_mode": mode,
        "warp": {"a": a_fit, "b": b_fit, "convention": "t_ref = a*t_hand + b; "
                 "ctx file starts at handcam t=81.0s"},
        "av_offset_ms": av_ms, "av_ncc": av_ncc,
    })


if __name__ == "__main__":
    main()
