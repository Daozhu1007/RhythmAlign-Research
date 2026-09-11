# R5.5 two-level detector.
# LEVEL 1: burst state (hysteresis on smoothed fused activity) - answers only
#          "continuous operation area or discrete area?", never counts events.
# LEVEL 2: outside bursts -> R5 frozen rising-edge rule (unchanged, discrete
#          segments were near-perfect);
#          inside bursts  -> atomic sub-event rule, rule "A" (fused-score local
#          peaks, platform re-trigger) or rule "B" (jz_diff atomic peaks with
#          channel agreement + burst-local adaptive threshold).
# Visual feature definitions and fusion weights are R5-frozen and shared.
import numpy as np
from scipy import signal as sig

CHANNELS = [("jz_text", "jz_warm", "jz_white"),
            ("jz_diff", "jz_diff", None),
            ("glass_bright", "glass_bright", None),
            ("hand_diff", "hand_diff", None)]

DEFAULT55 = {
    # ---- R5-frozen fusion (unchanged) ----
    "smooth_frames": 3, "rise_frames": 2,
    "w_jz_text": 1.8, "w_jz_diff": 1.5, "w_glass_bright": 1.0, "w_hand_diff": 0.3,
    "norm_pct": 99.5, "novelty_clip": 2.5,
    # ---- LEVEL 1 burst state ----
    "burst_smooth_frames": 5,
    "burst_enter": 0.50,       # enter when smoothed activity >= this
    "burst_stay": 0.28,        # hysteresis: stay while >= this
    "burst_hold_frames": 20,   # exit after this many frames below stay (~333 ms)
    # ---- sparse path: R5 rising edge (frozen values) ----
    "r5_threshold": 0.40, "r5_slope_min": 0.15, "r5_min_dist_frames": 3,
    # ---- rule A: fused-score local peaks inside bursts ----
    "a_prom_frac": 0.05,       # local prominence >= frac of score p99.5
    "a_valley_frames": 9,
    "a_min_dist_frames": 4,    # ~67 ms refractory
    # ---- rule B: jz_diff atomic peaks + agreement ----
    "b_jd_floor": 0.10,        # global floor (frac of jz_diff p99.5)
    "b_jd_burst_frac": 0.35,   # burst-local adaptive thr = frac * burst p90
    "b_prom_frac": 0.04,
    "b_supp_frac": 0.10,       # glass OR hand normalized level >= this in window
    "b_supp_win_frames": 3,
    "b_min_dist_frames": 4,
    # ---- final dedup ----
    "min_sep_frames": 4,
}


def _smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def fused_score(series, cfg):
    """R5-frozen fusion, identical to r5_common.detector_score."""
    ns = {}
    for name, (key_a, key_b) in {
        "jz_text": ("jz_warm", "jz_white"), "jz_diff": ("jz_diff", None),
        "glass_bright": ("glass_bright", None), "hand_diff": ("hand_diff", None),
    }.items():
        x = series[key_a] + (series[key_b] if key_b else 0.0)
        s = _smooth(x, cfg["smooth_frames"])
        n = np.zeros(len(s))
        n[cfg["rise_frames"]:] = np.maximum(0.0, s[cfg["rise_frames"]:] - s[:-cfg["rise_frames"]])
        ns[name] = n
    score = np.zeros(len(next(iter(series.values()))))
    for name, w in [("jz_text", cfg["w_jz_text"]), ("jz_diff", cfg["w_jz_diff"]),
                    ("glass_bright", cfg["w_glass_bright"]),
                    ("hand_diff", cfg["w_hand_diff"])]:
        n = ns[name]
        scale = np.percentile(n, cfg["norm_pct"]) + 1e-9
        score += w * np.minimum(n / scale, cfg["novelty_clip"])
    return score, ns


def burst_mask(score, cfg):
    env = _smooth(score, cfg["burst_smooth_frames"])
    m = np.zeros(len(env), bool)
    state, quiet = False, 0
    for k in range(len(env)):
        if not state:
            if env[k] >= cfg["burst_enter"]:
                state, quiet = True, 0
        else:
            if env[k] >= cfg["burst_stay"]:
                quiet = 0
            else:
                quiet += 1
                if quiet > cfg["burst_hold_frames"]:
                    state = False
        m[k] = state
    # interval list
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    ivs = [(int(s), int(e)) for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]
    return m, ivs, env


def _local_prom(x, pk, valley):
    out = np.zeros(len(pk))
    for i, p in enumerate(pk):
        a, b = max(p - valley, 0), min(p + valley + 1, len(x))
        left = np.min(x[a:p + 1]) if p > a else x[p]
        right = np.min(x[p:b]) if b > p + 1 else x[p]
        out[i] = x[p] - max(left, right)
    return out


def rule_a_peaks(score, mask, cfg):
    """A: every prominent local peak of the fused score inside a burst is an
    atomic event candidate - no rising-edge requirement (platform re-trigger)."""
    pk, _ = sig.find_peaks(score, distance=cfg["a_min_dist_frames"])
    if len(pk) == 0:
        return []
    prom = _local_prom(score, pk, cfg["a_valley_frames"])
    thr = cfg["a_prom_frac"] * (np.percentile(score, 99.5) + 1e-9)
    return [int(p) for p, pr in zip(pk, prom) if pr >= thr and mask[p]]


def rule_b_peaks(series, mask, ivs, cfg):
    """B: jz_diff atomic peaks, burst-local adaptive threshold, greedy
    amplitude-ordered acceptance with refractory, glass/hand agreement as
    support (occlusion backup). hand/jz_text never veto here."""
    jd = series["jz_diff"]
    gl = series["glass_bright"]
    hd = series["hand_diff"]
    s_jd = np.percentile(jd, 99.5) + 1e-9
    s_gl = np.percentile(gl, 99.5) + 1e-9
    s_hd = np.percentile(hd, 99.5) + 1e-9
    cands = []
    for a, b in ivs:
        seg = jd[a:b]
        if len(seg) < 3:
            continue
        thr = max(cfg["b_jd_floor"] * s_jd,
                  cfg["b_jd_burst_frac"] * (np.percentile(seg, 90) + 1e-9))
        pk, _ = sig.find_peaks(seg)
        pk = pk + a
        if len(pk) == 0:
            continue
        prom = _local_prom(jd, pk, cfg["a_valley_frames"])
        ok = [(jd[p], p) for p, pr in zip(pk, prom)
              if jd[p] >= thr and pr >= cfg["b_prom_frac"] * s_jd and mask[p]]
        # greedy amplitude-ordered with refractory + support check
        ok.sort(reverse=True)
        taken = []
        for v, p in ok:
            if any(abs(p - q) < cfg["b_min_dist_frames"] for q in taken):
                continue
            w = cfg["b_supp_win_frames"]
            g_ok = gl[max(p - w, 0):p + w + 1].max() / s_gl >= cfg["b_supp_frac"]
            h_ok = hd[max(p - w, 0):p + w + 1].max() / s_hd >= cfg["b_supp_frac"]
            if g_ok or h_ok:
                taken.append(p)
        cands.extend(taken)
    return sorted(cands)


def detect55(series, fps, cfg=None, rule="A"):
    cfg = {**DEFAULT55, **(cfg or {})}
    score, _ = fused_score(series, cfg)
    mask, ivs, env = burst_mask(score, cfg)
    # sparse path: R5 frozen rising edge, outside bursts only
    rise = np.zeros(len(score))
    rise[cfg["rise_frames"]:] = score[cfg["rise_frames"]:] - score[:-cfg["rise_frames"]]
    out = []
    last = -10 ** 9
    for k in range(1, len(score)):
        if mask[k]:
            continue
        if k - last < cfg["r5_min_dist_frames"]:
            continue
        if score[k] >= cfg["r5_threshold"] and rise[k] >= cfg["r5_slope_min"]:
            out.append(k)
            last = k
    # dense path
    if rule == "A":
        pk = rule_a_peaks(score, mask, cfg)
    else:
        pk = rule_b_peaks(series, mask, ivs, cfg)
    # merge: dedup by min_sep keeping the higher fused score
    all_pk = sorted(set(out) | set(pk))
    merged = []
    for k in all_pk:
        if merged and k - merged[-1][0] < cfg["min_sep_frames"]:
            if score[k] > score[merged[-1][0]]:
                merged[-1] = (k, "dense" if mask[k] else "sparse")
            continue
        merged.append((k, "dense" if mask[k] else "sparse"))
    cands = [{"i": int(k), "t_video": round(k / fps, 4),
              "score": round(float(score[k]), 3), "path": tag}
             for k, tag in merged]
    return score, mask, ivs, env, cands
