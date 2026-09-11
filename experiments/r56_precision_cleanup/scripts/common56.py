# R5.6 common: reuses the R5.5 FROZEN detector/feature/audio primitives
# unchanged (common55 -> r5_common). This experiment ONLY adds candidate
# post-filtering (E1) and burst-constrained audio completion (E2).
import importlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
R56 = os.path.abspath(os.path.join(_HERE, ".."))
R56_WORK = os.path.join(R56, "work")
R56_OUT = os.path.join(R56, "outputs")
R56_LOG = os.path.join(R56, "logs")

R55 = r"D:\Code\RhythmAlign\experiments\r55_burst_event_splitting"
R55_SCRIPTS = os.path.join(R55, "scripts")
R55_OUT = os.path.join(R55, "outputs")
R55_WORK = os.path.join(R55, "work")
R55_LOG = os.path.join(R55, "logs")
R5_SCRIPTS = r"D:\Code\RhythmAlign\experiments\r5_auto_contact_detection\scripts"
R5_OUT = os.path.join(R5_SCRIPTS, "..", "outputs")
R5_WORK = os.path.join(R5_SCRIPTS, "..", "work")
R4_OUT = r"D:\Code\RhythmAlign\experiments\r4_contact_reconstruction\outputs"
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
R1_WORK = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\work"

sys.path.insert(0, R55_SCRIPTS)
sys.path.insert(0, R5_SCRIPTS)
_r5 = importlib.import_module("r5_common")
common55 = importlib.import_module("common55")
detector55 = importlib.import_module("detector55")

FPS = common55.FPS                     # 60.04
AV_OFF_MS = 0.0                        # R5 convention (frozen)
CTX_START = 3.0
SR = _r5.SR

# E0 = R5.5 frozen detector config (outputs/detector_config_r55_frozen.json)
CFG55_FROZEN = common55.load_json(os.path.join(R55_OUT, "detector_config_r55_frozen.json"))["config"]

# frozen primitives, byte-identical to R5/R5.5
extract_features = _r5.extract_features
detect_candidates = _r5.detect_candidates
comb_envelope = _r5.comb_envelope
refine_candidates = _r5.refine_candidates
build_c1 = _r5.build_c1
audio_onsets = _r5.audio_onsets
detect55 = detector55.detect55
Window = common55.Window
load_json = common55.load_json
save_json = common55.save_json
match_one_to_one = common55.match_one_to_one
prf = common55.prf
merge_spans = _r5.merge_spans


# ------------------------------------------------------------------
# the three development windows (Dev-A golden, Dev-B old holdout,
# Dev-C = R5.5 Holdout C, oracle INCOMPLETE -> qualified supervision)
# ------------------------------------------------------------------
class DevWindow:
    """features + oracle + audio sources for one development window."""

    def __init__(self, key, features_npz, oracle_json, t0_s, dur_s,
                 ctx_wav, ref_wav, raw_wav, oracle_refined_json):
        self.key = key
        d = np.load(features_npz)
        self.series = {k: d[k] for k in d.files if k != "fps"}
        self.n = len(self.series["jz_diff"])
        self.dur = dur_s
        self.t0_native = t0_s
        w = Window(key, features_npz, oracle_json, t0_s, t0_s + dur_s, dur_s)
        self.__dict__.update({k: v for k, v in w.__dict__.items()
                              if k not in ("key", "series", "n", "dur", "t0_native")})
        self.tc = w.tc
        self.ctx = ctx_wav
        self.ref = ref_wav
        self.raw = raw_wav
        self.oracle_refined = oracle_refined_json
        self.oracle_conf_complete = key in ("devA", "devB")


def load_dev_windows():
    return [
        DevWindow("devA",
                  os.path.join(R5_WORK, "dev_features.npz"),
                  os.path.join(R4_OUT, "contacts_oracle.json"),
                  22.5, 15.0,
                  os.path.join(R1_WORK, "golden_ctx.wav"),
                  os.path.join(R1_WORK, "ref_warp_fixed.wav"),
                  os.path.join(R1_OUT, "golden_raw.wav"),
                  os.path.join(R4_OUT, "contacts_refined.json")),
        DevWindow("devB",
                  os.path.join(R5_WORK, "holdout_features.npz"),
                  os.path.join(R5_OUT, "holdout_oracle_blind.json"),
                  84.0, 18.0,
                  os.path.join(R5_WORK, "holdout_ctx.wav"),
                  os.path.join(R5_WORK, "holdout_ref_warp.wav"),
                  os.path.join(R5_OUT, "holdout_raw.wav"),
                  os.path.join(R5_OUT, "holdout_oracle_refined.json")),
        DevWindow("devC",
                  os.path.join(R55_WORK, "holdoutC_features.npz"),
                  os.path.join(R55_OUT, "holdoutC_oracle_blind.json"),
                  40.0, 18.0,
                  os.path.join(R55_WORK, "holdoutC_ctx.wav"),
                  os.path.join(R55_WORK, "holdoutC_ref_warp.wav"),
                  os.path.join(R55_OUT, "holdoutC_raw.wav"),
                  os.path.join(R55_OUT, "holdoutC_oracle_refined.json")),
    ]


_comb_cache = {}


def get_comb(w):
    """Frozen multi-band comb envelope for a dev window (window timebase)."""
    if w.key not in _comb_cache:
        comb, _ = comb_envelope(w.ctx, w.ref, CTX_START, w.dur)
        _comb_cache[w.key] = comb
    return _comb_cache[w.key]


def get_comb_files(ctx_wav, ref_wav, dur_s):
    comb, _ = comb_envelope(ctx_wav, ref_wav, CTX_START, dur_s)
    return comb


def band_envelopes_files(ctx_wav, ref_wav, dur_s):
    """File-path version of band_envelopes (audit only, frozen parameters)."""
    class _W:
        pass
    w = _W()
    w.ctx, w.ref, w.dur = ctx_wav, ref_wav, dur_s
    return band_envelopes(w)


def ref_slice_mono(ref_wav, dur_s):
    import soundfile as sf
    data, _ = sf.read(ref_wav, dtype="float64", always_2d=True)
    return data[int(CTX_START * SR):int((CTX_START + dur_s) * SR)].mean(axis=1)


def ref_bands_profile(ref_wav, dur_s):
    """Band envelopes of the aligned MUSIC track itself (taiko-labelling of
    ref onsets). Same butter/hann parameters as the frozen band code."""
    from scipy import signal as sig
    r = ref_slice_mono(ref_wav, dur_s)

    def band_env(x, lo, hi):
        sosb = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
        xb = sig.sosfiltfilt(sosb, x)
        return sig.convolve(np.abs(xb), sig.windows.hann(49), mode="same")

    def norm(e):
        return e / (np.percentile(e, 99.9) + 1e-9)
    return {"bass": norm(band_env(r, 150, 500)), "mid": norm(band_env(r, 500, 2000)),
            "hi1": norm(band_env(r, 2000, 6000)), "hi2": norm(band_env(r, 6000, 14000))}


def band_envelopes(w):
    """The four R4-frozen band envelopes of comb_envelope, kept separate for
    burst/taiko AUDIT only (same butter/hann parameters as the frozen code).
    bass analyzes the raw handcam (music bass fully included); mid/hi1/hi2
    analyze the music-subtracted residual."""
    import librosa
    from scipy import signal as sig
    y, _ = librosa.load(w.ctx, sr=SR, mono=True)
    g = y[int(CTX_START * SR):int((CTX_START + w.dur) * SR)]
    ref, _ = librosa.load(w.ref, sr=SR, mono=True)
    r = ref[int(CTX_START * SR):int((CTX_START + w.dur) * SR)]
    gg, rr = g - g.mean(), r - r.mean()
    gain = float(np.dot(gg, rr) / np.dot(rr, rr))
    resid = g - gain * r

    def band_env(x, lo, hi):
        sosb = sig.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
        xb = sig.sosfiltfilt(sosb, x)
        env = np.abs(xb)
        kern = sig.windows.hann(49)
        return sig.convolve(env, kern, mode="same")

    def norm(e):
        p99 = np.percentile(e, 99.9)
        return e / (p99 + 1e-9)

    out = {"bass": norm(band_env(g, 150, 500)),
           "mid": norm(band_env(resid, 500, 2000)),
           "hi1": norm(band_env(resid, 2000, 6000)),
           "hi2": norm(band_env(resid, 6000, 14000))}
    return {k: v.astype(np.float32) for k, v in out.items()}


def refine(w, times, ids):
    """Frozen refinement on the window comb."""
    return refine_candidates(list(times), get_comb(w), AV_OFF_MS, w.dur, ids=ids)


USED_CONF = ("strong", "present", "weak")   # confidences that open windows in E0


def auto_used_times(refined, levels=USED_CONF):
    return [r["t_audio_refined"] for r in refined
            if r.get("refined_confidence") in levels]


def merge_gate_spans(t_audio_refined, dur, pre=0.015, post=0.150):
    """Merged C1 gate spans (frozen pre/post)."""
    spans = sorted((max(t - pre, 0.0), min(t + post, dur)) for t in t_audio_refined)
    return merge_spans([(max(int(s * SR), 0), min(int(e * SR), int(dur * SR)))
                        for s, e in spans])


def spans_seconds(spans_samples):
    return [(s / SR, e / SR) for s, e in spans_samples]


def false_windows(gate_spans_s, oracle_times, tol=0.080):
    """Confirmed false windows: merged spans whose center has no oracle
    contact within tol (R5.5 definition, kept for comparability)."""
    to = np.asarray(oracle_times)
    out = []
    for s, e in gate_spans_s:
        c = (s + e) / 2
        if len(to) == 0 or np.min(np.abs(to - c)) > tol:
            out.append((s, e))
    return out


def merge_spans_s(spans_s):
    out = []
    for s, e in sorted(spans_s):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def useful_window_ratio(gate_spans_s, oracle_times, pre=0.015, post=0.150):
    """NEW product metric: fraction of open gate time that lies inside the
    pre/post neighborhood of a (high-confidence) oracle contact."""
    tot = sum(e - s for s, e in gate_spans_s)
    if tot <= 0:
        return 0.0, 0.0
    sup_m = merge_spans_s([(max(t - pre, 0.0), t + post) for t in oracle_times])
    use = 0.0
    for s, e in gate_spans_s:
        for a, b in sup_m:
            if b <= s:
                continue
            if a >= e:
                break
            use += min(e, b) - max(s, a)
    return use / tot, tot


# ------------------------------------------------------------------
# E2 burst-constrained strong-audio completion (the ONE recipe).
# ------------------------------------------------------------------
E2_CFG = {
    "min_peak": 0.35,        # strong tier (mirrors frozen refinement level)
    "min_ratio": 4.0,
    "min_dist_s": 0.060,
    "resid_sum_min": 1.2,    # taiko guard: music-removed residual support
    "hi1_min": 0.15,
    "d_e1_min_ms": 90,       # not duplicating an E1 candidate
    "d_e2_min_ms": 90,       # refractory between completed events
}


def contact_neighborhoods(oracle_times, pre=0.015, post=0.150):
    return merge_spans([(int(max(t - pre, 0.0) * SR), int((t + post) * SR))
                        for t in oracle_times])


# ------------------------------------------------------------------
# E1 candidate post-filter (the ONE precision-cleanup recipe).
# Visual features come from the R5.5 FROZEN score/series; nothing here
# changes detection, it only removes or keeps existing candidates.
# ------------------------------------------------------------------
E1_CFG = {
    # echo-ratio suppression: REJECTED by the Phase-1 audit (echo FPs are NOT
    # weaker than true adjacent contacts: median echo_ratio 1.09 vs 0.75-0.84;
    # every useful threshold kills more true 90-120 ms doubles than echoes).
    # Refinement confidence already covers echoes: 87/99 echo FPs are
    # weak/no_transient/present.
    "weak_prom_frac": 0.30,     # extra visual support: high prominence
    "weak_chan_n": 2,           # ... or multi-channel agreement (>=2 of 4
    "weak_chan_floor": 0.10,    #     channel novelties above 10% of p99.5)
    "weak_burst_edge_ms": 400,  # ... or near a burst boundary AND within
    "weak_near_sp_ms": 250,     #     250 ms of a strong/present candidate
    # audio-rescue tier: a weak candidate whose transient is still clearly
    # above noise keeps its window. This is the ONLY way to keep the
    # visually-silent true contacts the Phase-1 audit found (7 wrongly-gated
    # candidates all had comb_peak 0.52-1.23 / ratio 1.6-2.5 and flat visual
    # features; rescue saved 11 TP for ~6 echo-adjacent windows reopened).
    "rescue_comb_peak": 0.80,
    "rescue_comb_ratio": 1.8,
}


def channel_support(ns, i):
    vals = {}
    for name, arr in ns.items():
        scale = np.percentile(arr, 99.5) + 1e-9
        vals[name] = float(min(arr[i] / scale, 2.5))
    return vals


def burst_position(i, ivs):
    for a, b in ivs:
        if a <= i < b:
            return True, float(min(i - a, b - i) / FPS * 1000)
    d = min([min(abs(i - a), abs(i - b)) for a, b in ivs], default=10 ** 9)
    return False, float(d / FPS * 1000)


def e1_gate_flags(cands, refined, score, ns, ivs, cfg=None):
    """Three-state E1 gate: (candidate_kept, window_opens, reason).
    strong/present: keep + window (unchanged). weak: window only with extra
    visual support or the audio-rescue tier; candidate stays either way.
    no_transient: candidate stays (R5.5 visual convention), never opens a
    window (E0 behaviour)."""
    cfg = {**E1_CFG, **(cfg or {})}
    ts_sp = sorted(r["t_audio_refined"] for r in refined
                   if r.get("refined_confidence") in ("strong", "present"))
    import bisect
    flags = {}
    for j, cd in enumerate(cands):
        r = refined[j]
        lvl = r.get("refined_confidence")
        if lvl in ("strong", "present"):
            flags[j] = (True, True, f"keep_{lvl}")
            continue
        if lvl == "no_transient":
            flags[j] = (True, False, "no_transient_no_window")
            continue
        i = cd["i"]
        in_b, edge_ms = burst_position(i, ivs)
        sup = channel_support(ns, i)
        chan_n = int(sum(v >= cfg["weak_chan_floor"] for v in sup.values()))
        prom = _prom_frac(score, i)
        kk = bisect.bisect_left(ts_sp, cd["t_video"])
        near_sp = any(abs(cd["t_video"] - t) <= cfg["weak_near_sp_ms"] / 1000
                      for t in ts_sp[max(kk - 1, 0):kk + 1])
        cp, cr = r.get("peak") or 0.0, r.get("peak_over_noise") or 0.0
        if cp >= cfg["rescue_comb_peak"] and cr >= cfg["rescue_comb_ratio"]:
            flags[j] = (True, True, f"weak_kept_rescue_p{cp:.2f}_r{cr:.1f}")
        elif prom >= cfg["weak_prom_frac"]:
            flags[j] = (True, True, f"weak_kept_prom_{prom:.2f}")
        elif chan_n >= cfg["weak_chan_n"]:
            flags[j] = (True, True, f"weak_kept_chan{chan_n}")
        elif in_b and edge_ms <= cfg["weak_burst_edge_ms"] and near_sp:
            flags[j] = (True, True, f"weak_kept_burstedge{edge_ms:.0f}_nearSP")
        else:
            why = ("not_near_sp" if in_b and edge_ms <= cfg["weak_burst_edge_ms"]
                   else "not_at_burst_edge")
            flags[j] = (True, False, f"weak_nowindow_{why}")
    return flags


def _prom_frac(score, i, valley=9):
    from detector55 import _local_prom
    pr = _local_prom(score, np.array([i]), valley)[0]
    return float(pr / (np.percentile(score, 99.5) + 1e-9))
