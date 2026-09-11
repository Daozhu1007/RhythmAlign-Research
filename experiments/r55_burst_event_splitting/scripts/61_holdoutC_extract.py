# R5.5 Phase 6 - Holdout-C extraction (adapted from R5 50_holdout_extract.py):
# raw audio slice, context slice, ref sinc-warp alignment, video re-encode,
# AV offset. Selection reason recorded here BEFORE any annotation.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"D:\Code\RhythmAlign\experiments\r5_auto_contact_detection\scripts")
import numpy as np
import soundfile as sf
from scipy import signal as sig

from r5_common import (SR, save_json, measure_av_offset_ms, bp48, sinc_warp,
                       ffmpeg_run)

R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"
R55 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(R55, "outputs")
WORK = os.path.join(R55, "work")
HANCAM_MP4 = r"D:\Daozh\Videos\舞萌手元\13.2\共感觉\AP\共感怪物AP.mp4"

H0, H1 = 40.0, 58.0           # holdout C window in handcam-native time
CTX0 = H0 - 3.0               # context start (37.0)
CTX_DUR = (H1 - H0) + 6.0     # 24 s context
D_PRIOR = -11.374728          # R1 fine-alignment delay (s)


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
    sf.write(os.path.join(WORK, "holdoutC_ctx.wav"), ctx_st.astype(np.float32), SR)
    sf.write(os.path.join(OUT, "holdoutC_raw.wav"),
             hand_st[int(round(H0 * SR)):int(round(H1 * SR))].astype(np.float32), SR)
    print(f"holdoutC_raw.wav: {H1 - H0:.1f}s | ctx {CTX_DUR:.1f}s from {CTX0}s")

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
    spread_ms = float(max(a["d"] for a in anchors) - min(a["d"] for a in anchors))
    drift_note = ""
    if spread_ms <= 3.0:
        # anchors at 41/49/57 s cover the holdout interior; if a 12 s window
        # disagrees, the mismatch comes from content OUTSIDE the window
        # (its span reaches up to 10 s past the holdout end), not from
        # in-window drift - anchor consensus wins, disagreement documented.
        a_fit, b_fit, mode = 1.0, float(np.mean([a["d"] for a in anchors])), "fixed_anchor_consensus"
        drift_note = (f"12s windows: {ds12[0]*1e3:+.3f} / {ds12[1]*1e3:+.3f} ms; "
                      f"second window disagrees by {abs(ds12[0]-ds12[1])*1e3:.1f} ms "
                      f"but covers [56,68] s which extends 10 s beyond the holdout "
                      f"end; anchors inside [40,58] agree to {spread_ms:.2f} ms")
        print(f"anchor spread {spread_ms:.2f} ms -> {mode}, d = {b_fit*1e3:+.3f} ms")
        print(drift_note)
    else:
        raise SystemExit(f"anchor disagreement {spread_ms:.1f} ms too large; aborting")

    r0 = max(int(round((CTX0 + b_fit - 1.0) * SR)), 0)
    r1 = min(int(round((CTX0 + CTX_DUR + b_fit + 1.0) * SR)), len(ref_st))
    warped, valid = sinc_warp(ref_st[r0:r1], a_fit, CTX0 + b_fit,
                              int(round(CTX_DUR * SR)), r0)
    sf.write(os.path.join(WORK, "holdoutC_ref_warp.wav"), warped.astype(np.float32), SR)
    vf, va = int(0.05 * SR), len(valid) - int(0.05 * SR)
    print(f"holdoutC_ref_warp.wav written (valid interior: {bool(valid[vf:va].all())})")

    vid = os.path.join(OUT, "holdoutC_video.mp4")
    ffmpeg_run(["-ss", f"{H0}", "-i", HANCAM_MP4, "-t", f"{H1 - H0}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", vid])
    av_ms, av_ncc = measure_av_offset_ms(vid, os.path.join(OUT, "holdoutC_raw.wav"))
    print(f"holdoutC_video.mp4 written; AV offset t_audio = t_video + {av_ms:+.3f} ms "
          f"(ncc {av_ncc:.3f})")

    save_json(os.path.join(OUT, "holdoutC_selection.json"), {
        "holdout": "C",
        "holdout_start_s": H0, "holdout_end_s": H1, "duration_s": H1 - H0,
        "forbidden_windows": [[22.5, 37.5], [84.0, 102.0]],
        "overlap_check": "window [40,58] does not overlap forbidden windows "
                         "(margin 2.5 s after dev-A end; ctx 37-40 used only for "
                         "audio alignment, never for detection)",
        "selection_reason": (
            "Chosen from the full-take audio onset-density profile plus 10 fps "
            "contact-sheet inspection of candidate zones [134,158] and [38,60]. "
            "[134,158] was REJECTED because the take's result screen starts ~143 s. "
            "[40,58] is fully mid-song and contains: dense two-hand press bursts "
            "(~48-51, ~56-58), a large fan slide + yellow slide ribbons (~44-45, "
            "~50.0), chevron-trace slides (~53.3), bottom and side presses "
            "throughout, medium-density stretches between bursts (46-48, 52-54, "
            "onset bins 73-80), strong CRITICAL PERFECT text events as well as "
            "faint ones, and heavy arm-occlusion of the judgment zone (elbow "
            "crossings ~48-49) acting as arcade-style interference. Density "
            "profile bins (onsets/2s): 40:77 42:89 44:96 46:80 48:73 50:93 52:77 "
            "54:69 56:96 58:88 - a mixed sparse/dense profile resembling the "
            "forbidden dev-B structure, not an easy pick."),
        "anchors": anchors, "anchor_spread_ms": spread_ms, "warp_mode": mode,
        "warp_note": drift_note,
        "warp": {"a": a_fit, "b": b_fit,
                 "convention": "t_ref = a*t_hand + b; ctx file starts at handcam t=37.0s"},
        "av_offset_ms": av_ms, "av_ncc": av_ncc,
        "extracted_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    })


if __name__ == "__main__":
    main()
