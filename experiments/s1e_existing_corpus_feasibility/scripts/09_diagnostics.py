# S1E step 09 - diagnostics per (method, clip). NO invented source metrics: these are
# fixed-gain waveform stats, pristine-reference coherence (music proxy), VAD speech
# proxies (nuisance proxy), and click-band level retention (descriptive proxy only,
# NOT a contact-recall claim).
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf
import resampy
from scipy import signal as sig
from scipy.ndimage import median_filter

from s1e_common import WORK_DIR, CLIP_DIR, OUT_DIR, LOG_DIR, SR, load_wav, save_json
from align_cancel import cancel_clip

NFFT, HOP = 2048, 512

manifest = json.load(open(os.path.join(CLIP_DIR, "clip_manifest.json"), encoding="utf-8"))

METHODS = {
    "raw":       (os.path.join(CLIP_DIR, "audio", "{cid}_48k_stereo.wav"), 48000),
    "clapsep":   ("clapsep_aq_{cid}_target.wav", 32000),
    "audiosep":  ("audiosep_p2_{cid}_target.wav", 32000),
    "soloaudio": ("soloaudio_aq_{cid}_target.wav", 24000),
    "specialist": ("specialist_{cid}_target.wav", 48000),
}
AVAILABLE = {"audiosep": {"c1_speech_npc_a", "c2_speech_npc_b", "c3_taiko_prompt",
                          "c6_dense_golden"}}


def db(x):
    return float(10 * np.log10(np.mean(np.asarray(x) ** 2) + 1e-12))


def load_method(path, sr_target):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = np.mean(x, axis=1)
    if sr != sr_target:
        x = resampy.resample(x.astype(np.float32), sr, sr_target)
    return np.asarray(x, dtype=np.float64)


def band_env_db(x, sr, lo, hi):
    sos = sig.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    xb = sig.sosfiltfilt(sos, x)
    hop = int(0.01 * sr)
    n = len(xb) // hop
    return np.sqrt(np.mean(xb[: n * hop].reshape(n, hop) ** 2, axis=1))


def align_len(out, n_ref):
    if len(out) >= n_ref:
        return out[:n_ref]
    return np.concatenate([out, np.zeros(n_ref - len(out))])


def music_proxy(x, sr, rw_ref, t0):
    """Coherence (0-1) between output and warped pristine ref + music-frame level."""
    n2_win = int(0.05 * sr)
    f, Cxy = sig.coherence(x, rw_ref, fs=sr, nperseg=n2_win)
    return float(np.mean(Cxy[(f > 100) & (f < 4000)]))


def music_frames_mask(x, sr, rw):
    """Frames where warped ref is active (music-present) and no click-band transient."""
    hop = int(0.01 * sr)
    e_ref = band_env_db(rw, sr, 60, 6000)
    e_clk = band_env_db(x, sr, 2000, 9000)
    ref_active = e_ref > np.percentile(e_ref, 50)
    from scipy.ndimage import binary_dilation
    hot = e_clk > np.percentile(e_clk, 80)
    hot = binary_dilation(hot, np.ones(35, bool))  # +-150 ms guard around transients
    return ref_active & ~hot, hop


def main():
    y, _ = load_wav(os.path.join(WORK_DIR, "handcam_native.wav"))
    r, _ = load_wav(os.path.join(WORK_DIR, "ref_native.wav"))
    results = []
    for clip in manifest["clips"]:
        cid = clip["id"]
        t0 = clip["t_start_video_s"]
        mix, _ = load_wav(os.path.join(CLIP_DIR, "audio", f"{cid}_48k_stereo.wav"))
        mixm = np.mean(mix, axis=1)
        # aligned warped ref for this clip (mono)
        rw = None
        if clip["reference_covered"]:
            res = cancel_clip(y[int((t0 - 1) * SR):int((clip["t_end_video_s"] + 1) * SR)],
                              r, t0 - 1, span_ms=10.0)
            rw = np.mean(res["ref_warp"], axis=1)[int(1 * SR):int(1 * SR) + len(mixm)]
        mmask, hop = music_frames_mask(mixm, SR, rw) if rw is not None else (None, None)

        for meth, (pat, srt) in METHODS.items():
            if meth in AVAILABLE and cid not in AVAILABLE[meth]:
                continue
            path = os.path.join(OUT_DIR, "audio", pat.format(cid=cid)) \
                if "{cid}" in pat and pat.endswith(".wav") else pat.format(cid=cid)
            if not os.path.exists(path):
                continue
            out = align_len(load_method(path, SR), len(mixm))
            row = {"clip": cid, "method": meth, "sr_native": srt,
                   "out_rms_db": db(out), "mix_rms_db": db(mixm),
                   "out_minus_mix_db": db(out) - db(mixm)}
            # band retention (descriptive proxies)
            for bname, (lo, hi) in {"low_40_150": (40, 150), "mid_150_2k": (150, 2000),
                                    "click_2k_9k": (2000, 9000)}.items():
                eb = band_env_db(out, SR, lo, hi)
                row[f"ret_{bname}_db"] = db(eb) - db(band_env_db(mixm, SR, lo, hi))
            if rw is not None:
                row["coherence_ref_100_4k"] = music_proxy(out, SR, rw, t0)
                e_out = band_env_db(out, SR, 60, 6000)
                e_mix = band_env_db(mixm, SR, 60, 6000)
                row["musicframe_level_change_db"] = db(e_out[mmask]) - db(e_mix[mmask])
            # VAD speech proxy
            p_out = vad_prob_curve(out)
            p_mix = vad_prob_curve(mixm)
            row["vad_speech_sec_gt02"] = float(np.sum(p_out > 0.2) * 0.032)
            row["vad_speech_sec_gt02_mix"] = float(np.sum(p_mix > 0.2) * 0.032)
            results.append(row)
            print(f"{cid:18s} {meth:10s} ret {row['out_minus_mix_db']:+6.1f} dB  "
                  f"click {row['ret_click_2k_9k_db']:+6.1f} dB  "
                  f"coher {row.get('coherence_ref_100_4k', float('nan')):.3f}  "
                  f"musfr {row.get('musicframe_level_change_db', float('nan')):+6.1f}  "
                  f"vad {row['vad_speech_sec_gt02']:.2f}s (mix {row['vad_speech_sec_gt02_mix']:.2f}s)")
    save_json(os.path.join(LOG_DIR, "09_diagnostics.json"), {"results": results})
    print("saved 09_diagnostics.json")


def vad_prob_curve(x):
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad().to("cuda" if torch.cuda.is_available() else "cpu")
    dev = next(model.parameters()).device
    sos = sig.butter(6, [80, 7500], btype="bandpass", fs=SR, output="sos")
    xx = sig.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))
    y16 = resampy.resample(xx.astype(np.float32), SR, 16000)
    t = torch.from_numpy(y16).to(dev)
    n = len(y16) // 512
    p = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            p[i] = float(model(t[i * 512:(i + 1) * 512], 16000).item())
    return p


if __name__ == "__main__":
    main()
