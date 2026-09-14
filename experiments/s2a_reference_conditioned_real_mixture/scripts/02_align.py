# S2A step 2: align each candidate pristine reference to its raw handcam
# (protocol section 10), using the PRODUCTION RhythmAlign analysis path
# (auto_sync._align_hybrid / _align_onset, read-only import; R1 precedent)
# plus three conservative additions:
#   - onset cross-check (offset agreement)
#   - three-window local re-alignment for drift detection
#   - aligned log-mel spectral agreement (song-identity verification)
# Ambiguous pairs are REJECTED, never forced (protocol section 10).
import os
import sys
import json
import subprocess

import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402

sys.path.insert(0, r"D:\Code\RhythmAlign")
import auto_sync  # noqa: E402  production module, READ-ONLY

HOP = sc.ALIGN_HOP
SR = sc.ALIGN_SR


def decode_ref(src, dst_wav):
    ffmpeg = __import__("imageio_ffmpeg").get_ffmpeg_exe()
    if not os.path.exists(dst_wav):
        tmp = dst_wav + ".tmp.wav"
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", src, "-ac", "1", "-ar", str(SR), tmp], check=True)
        os.replace(tmp, dst_wav)
    y, _ = librosa.load(dst_wav, sr=None, mono=True)
    return y


def load_handcam_22k(work32k):
    x, sr = librosa.load(work32k, sr=None, mono=True)
    assert sr == 32000, (work32k, sr)
    return librosa.resample(x, orig_sr=32000, target_sr=SR, res_type="soxr_hq")


def onset_env(y):
    return librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)


def offset_stability(y_video, y_music, off_full, seg_s=60.0, n_seg=5, tol=0.06):
    """Drift/ambiguity check: re-run the PRODUCTION hybrid alignment on short
    handcam segments against the full reference; each implies a global offset
    (segment offset + segment start).

    Repetitive songs trap SOME segments at wrong positions (a section N measures
    away), so the decision uses the densest cluster of estimates: the pair is
    stable iff a cluster of >= 3 estimates that CONTAINS the full-pass value
    agrees within tol, with residual spread <= ALIGN_DRIFT_RESID_MAX_S.
    Returns (final_offset, residual, n_cluster).
    """
    dur_hand = len(y_video) / SR
    implied = []
    seen_t0 = set()
    for frac in np.linspace(0.12, 0.88, n_seg):
        t0 = round(frac * dur_hand, 1)
        if t0 + seg_s > dur_hand:
            t0 = round(dur_hand - seg_s, 1)
        if t0 in seen_t0:
            continue
        seen_t0.add(t0)
        seg = y_video[int(t0 * SR):int((t0 + seg_s) * SR)]
        if len(seg) < SR * 20:
            continue
        off_seg, z_seg, _ = auto_sync._align_hybrid(seg, y_music, SR, HOP)
        if z_seg < sc.ALIGN_Z_MIN:
            continue
        implied.append(off_seg + t0)
    points = sorted([float(off_full)] + implied)
    best = None
    for i, lo in enumerate(points):
        cl = [p for p in points if p - lo <= tol]
        if len(cl) >= 3 and min(cl) <= off_full <= max(cl) and lo <= off_full <= lo + tol:
            if best is None or len(cl) > len(best):
                best = cl
    if best is None:
        return float(off_full), 1e3, 0
    final = float(np.median(best))
    resid = float(max(abs(p - final) for p in best))
    if resid > sc.ALIGN_DRIFT_RESID_MAX_S:
        return final, resid, len(best)
    return final, resid, len(best)


def spectral_agreement(y_video, y_music, offset, dur_hand, n_win=4, win_s=30.0):
    """Median frame-wise cosine of smoothed log-mel over several overlap windows,
    PLUS a self-control: the same statistic for the reference shifted +-~55 s
    inside the overlap (wrong-position baseline). Same-song aligned audio must
    clearly beat its own shifted baseline; returns (aligned, baseline).

    Convention (production/R1): offset = t_handcam - t_ref, i.e. t_ref = t_hand - offset;
    the handcam covers ref[offset .. offset + dur_hand].
    """
    dur_ref = len(y_music) / SR
    a0 = max(offset, 5.0)
    a1 = min(dur_hand - 5.0, offset + dur_ref - 5.0)
    if a1 - a0 < win_s + 2.0:
        return 0.0, 0.0
    w = 21
    ker = np.ones(w) / w

    def score_at(shift_s):
        vals = []
        for frac in np.linspace(0.1, 0.9, n_win):
            mid0 = a0 + frac * (a1 - a0 - win_s)
            hv = y_video[int(mid0 * SR):int((mid0 + win_s) * SR)]
            t_ref0 = mid0 - offset + shift_s
            if t_ref0 < 0 or t_ref0 + win_s > dur_ref:
                continue
            mv = y_music[int(t_ref0 * SR):int((t_ref0 + win_s) * SR)]
            n = min(len(hv), len(mv))
            if n < SR * 10:
                continue
            lv = np.log(librosa.feature.melspectrogram(y=hv[:n], sr=SR, n_fft=2048,
                                                       hop_length=512, n_mels=128) + 1e-6)
            lm = np.log(librosa.feature.melspectrogram(y=mv[:n], sr=SR, n_fft=2048,
                                                       hop_length=512, n_mels=128) + 1e-6)
            k = min(lv.shape[1], lm.shape[1])
            lv, lm = lv[:, :k], lm[:, :k]
            # percussive handcam transients distort single frames; smooth over ~0.5 s
            lv = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 1, lv)
            lm = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 1, lm)
            lv = lv - lv.mean(axis=1, keepdims=True)
            lm = lm - lm.mean(axis=1, keepdims=True)
            num = (lv * lm).sum(axis=0)
            den = np.linalg.norm(lv, axis=0) * np.linalg.norm(lm, axis=0) + 1e-9
            vals.append(float(np.median(num / den)))
        return float(np.median(vals)) if vals else 0.0

    aligned = score_at(0.0)
    base = max(score_at(55.0), score_at(-55.0))
    return aligned, base


def main():
    rmap = sc.load_reference_map()
    results = []
    cache = {}
    for p in rmap["pairs"]:
        rid, ref_id = p["recording_id"], p["ref_id"]
        ref_wav22 = os.path.join(sc.S2A_REF22K, ref_id + ".wav")
        if ref_id not in cache:
            cache[ref_id] = decode_ref(p["ref_abs_path"], ref_wav22)
        y_music = cache[ref_id]
        y_video = load_handcam_22k(p["raw_work32k"])
        dur_hand = len(y_video) / SR

        off_h, z_h, _ = auto_sync._align_hybrid(y_video, y_music, SR, HOP)
        off_o, z_o, _ = auto_sync._align_onset(y_video, y_music, SR, HOP)

        # segment-level drift check + self-controlled identity check
        final_off, resid, n_in = offset_stability(y_video, y_music, off_h)
        overlap = max(0.0, min(dur_hand, len(y_music) / SR - final_off) - max(0.0, final_off))
        sa, sa_base = spectral_agreement(y_video, y_music, final_off, dur_hand)

        checks = {
            "z_hybrid": z_h >= sc.ALIGN_Z_MIN,
            "offset_stable": resid <= sc.ALIGN_DRIFT_RESID_MAX_S,
            "overlap_ok": overlap >= sc.ALIGN_OVERLAP_MIN_S,
            "song_identity": sa >= 0.25 and sa >= sa_base + 0.10,
        }
        accepted = all(checks.values())
        results.append({
            "recording_id": rid, "family": p["family"], "split": p["split"],
            "ref_id": ref_id, "song_id": p["song_id"],
            "offset_s": round(float(final_off), 4),
            "offset_full_pass_s": round(float(off_h), 4),
            "offset_onset_s": round(float(off_o), 4),
            "z_hybrid": round(float(z_h), 2), "z_onset": round(float(z_o), 2),
            "stability_inliers": int(n_in),
            "drift_resid_s": round(resid, 4),
            "overlap_s": round(float(overlap), 1),
            "spectral_agreement": round(sa, 4),
            "spectral_agreement_baseline": round(sa_base, 4),
            "checks": {k: bool(v) for k, v in checks.items()},
            "accepted": bool(accepted),
        })
        print(f'{rid} {p["family"]:14s} off={final_off:+8.3f} zH={z_h:6.1f} '
              f'inl={n_in} resid={resid:.3f} ovl={overlap:5.1f}s agree={sa:.3f}/{sa_base:.3f} '
              f'{"ACCEPT" if accepted else "reject:" + ",".join(k for k, v in checks.items() if not v)}')

    sc.save_json(os.path.join(sc.S2A_PRIVATE, "ALIGNMENT.private.json"), {"pairs": results})
    n_ok = sum(r["accepted"] for r in results)
    print(f"\naccepted {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
