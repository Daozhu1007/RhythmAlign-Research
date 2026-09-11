# R5-FIX PART 4/5/6 driver - evaluate every comparison variant with the
# corrected evaluator. Historical WAVs are only READ.
#
# Golden (15 s, R4 oracle timing):
#   golden_old_C1            historical defective C1
#   golden_C1_fixed          corrected C1
#   golden_old_C2            historical reversed-HPSS C2
#   golden_C2_fixed          corrected HPSS control
#   golden_music_only        CONTROL A
#   golden_C1_fixed_shift9?  -> golden_C1_fixed_shift (CONTROL B, +7.5 s roll)
# Holdout-D (18 s):
#   holdoutD_old_E0_C1       historical E0 + defective C1
#   holdoutD_E0_C1_fixed     E0 + corrected C1
#   holdoutD_old_oracle_C1   auxiliary (historical oracle + defective C1)
#   holdoutD_oracle_C1_fixed oracle + corrected C1
#   holdoutD_music_only      CONTROL A
#   holdoutD_E0_fixed_shift  CONTROL B (+9.0 s roll of the corrected E0 gate)
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx
import evaluator as ev

PRE = fx.load_json(os.path.join(fx.FIX_LOG, "predeclared_controls.json"))
assert PRE["declared_before_scoring"]


def load_wav(p):
    return sf.read(p, dtype="float64", always_2d=True)[0]


def load_or_trim(p, n):
    x = load_wav(p)
    assert len(x) == n, (p, len(x), n)
    return x


def band_envs_ref(mono):
    from scipy import signal as sig
    def band_env(x, lo, hi):
        sosb = sig.butter(4, [lo, hi], btype="bandpass", fs=fx.SR, output="sos")
        xb = sig.sosfiltfilt(sosb, x)
        return sig.convolve(np.abs(xb), sig.windows.hann(49), mode="same")
    def norm(e):
        return e / (np.percentile(e, 99.9) + 1e-9)
    return {"bass": norm(band_env(mono, 150, 500)), "mid": norm(band_env(mono, 500, 2000)),
            "hi1": norm(band_env(mono, 2000, 6000)), "hi2": norm(band_env(mono, 6000, 14000))}


# ------------------------------------------------------------------ data
class Window:
    def __init__(self, key, raw, ref, oracle_times_audio, oracle_ids,
                 oracle_times_video, visual_high, audio_strong_flags,
                 rest_spans, ref_slice):
        self.key = key
        self.raw, self.ref = raw, ref
        self.n = len(raw)
        self.dur = self.n / fx.SR
        self.oracle_t_audio = np.asarray(oracle_times_audio)
        self.oracle_ids = oracle_ids
        self.oracle_t_video = np.asarray(oracle_times_video)
        self.visual_high = visual_high
        self.audio_strong = audio_strong_flags
        self.rest_spans = rest_spans
        allowed = [(max(t - 0.040, 0.0), min(t + 0.175, self.dur)) for t in oracle_times_audio]
        self.allowed_spans = allowed
        mono_ref = ref.mean(axis=1)
        self.ref_onsets = ev.audio_onsets(mono_ref)
        self.ref_bands = band_envs_ref(mono_ref)
        self.ref_slice_mono = mono_ref if ref_slice is None else ref_slice


def make_windows():
    # ---- golden ----
    raw_g = load_or_trim(os.path.join(fx.R1_OUT, "golden_raw.wav"), 15 * fx.SR)
    ref_g = load_wav(os.path.join(fx.R1_WORK, "ref_warp_fixed.wav"))[3 * fx.SR:18 * fx.SR]
    R = fx.load_json(os.path.join(fx.R4_OUT, "contacts_refined.json"))["events"]
    used = [e for e in R if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    win_g = Window(
        "golden", raw_g, ref_g,
        oracle_times_audio=[e["t_audio_refined"] for e in used],
        oracle_ids=[e["id"] for e in used],
        oracle_times_video=[e["t_video"] for e in used],
        visual_high=[e.get("oracle_confidence") == "high" for e in used],
        audio_strong_flags=[e["refined_confidence"] == "strong" for e in used],
        rest_spans=[], ref_slice=None)

    # ---- holdout D ----
    raw_h = load_or_trim(os.path.join(fx.R56_OUT, "holdoutD_raw.wav"), 18 * fx.SR)
    ref_h = load_wav(os.path.join(fx.R56_WORK, "holdoutD_ref_warp.wav"))[3 * fx.SR:21 * fx.SR]
    O = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_oracle_refined.json"))["events"]
    win_h = Window(
        "holdoutD", raw_h, ref_h,
        oracle_times_audio=[e["t_audio_refined"] for e in O],
        oracle_ids=[e["id"] for e in O],
        oracle_times_video=[e["t_video"] for e in O],
        visual_high=[e.get("oracle_confidence") == "high" for e in O],
        audio_strong_flags=[e.get("refined_confidence") == "strong" for e in O],
        rest_spans=[(6.0, 8.0)], ref_slice=None)
    return win_g, win_h


# ------------------------------------------------------------------ cases
def build_cases(win_g, win_h):
    cases = []
    env_g_old = np.load(os.path.join(fx.FIX_WORK, "env_golden_old.npy"))
    env_g_fix = np.load(os.path.join(fx.FIX_WORK, "env_golden_oracle_fixed.npy"))
    env_h_old = np.load(os.path.join(fx.FIX_WORK, "env_holdoutD_e0_old.npy"))
    env_h_fix = np.load(os.path.join(fx.FIX_WORK, "env_holdoutD_E0_fixed.npy"))
    env_h_orc = np.load(os.path.join(fx.FIX_WORK, "env_holdoutD_oracle_fixed.npy"))

    def add(name, win, env, stem, final, preds_tvideo=None, aux=False, stem_old=None,
            final_old=None, gate_times=None, gate_note=None):
        cases.append({"name": name, "win": win.key, "env": env, "stem": stem,
                      "final": final, "preds_tvideo": preds_tvideo, "aux": aux,
                      "stem_old": stem_old, "final_old": final_old,
                      "gate_times": gate_times, "gate_note": gate_note})

    R4 = os.path.join(fx.R4_OUT)
    R56 = os.path.join(fx.R56_OUT)
    F = fx.FIX_OUT

    e0_preds = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_contacts_auto_refined.json"))["events"]
    e0_tvideo = [e["t_video"] for e in e0_preds]
    e0_gate = [e["t_audio_refined"] for e in e0_preds
               if e.get("refined_confidence") in ("strong", "present", "weak")]
    g_gate = list(win_g.oracle_t_audio)
    identity = "gate times ARE the oracle reference times (identity by construction; not informative)"

    add("golden_old_C1", win_g, env_g_old,
        load_or_trim(os.path.join(R4, "C1_interaction.wav"), win_g.n),
        load_or_trim(os.path.join(R4, "C1_final_mix.wav"), win_g.n),
        gate_times=g_gate, gate_note=identity)
    add("golden_C1_fixed", win_g, env_g_fix,
        load_or_trim(os.path.join(F, "golden_oracle_C1_fixed_interaction.wav"), win_g.n),
        load_or_trim(os.path.join(F, "golden_oracle_C1_fixed_final_mix.wav"), win_g.n),
        stem_old=load_or_trim(os.path.join(R4, "C1_interaction.wav"), win_g.n),
        final_old=load_or_trim(os.path.join(R4, "C1_final_mix.wav"), win_g.n),
        gate_times=g_gate, gate_note=identity)
    add("golden_old_C2", win_g, env_g_old,
        load_or_trim(os.path.join(R4, "C2_interaction.wav"), win_g.n),
        load_or_trim(os.path.join(R4, "C2_final_mix.wav"), win_g.n),
        gate_times=g_gate, gate_note=identity)
    add("golden_C2_fixed", win_g, env_g_fix,
        load_or_trim(os.path.join(F, "golden_C2_fixed_interaction.wav"), win_g.n),
        load_or_trim(os.path.join(F, "golden_C2_fixed_final_mix.wav"), win_g.n),
        stem_old=load_or_trim(os.path.join(R4, "C2_interaction.wav"), win_g.n),
        final_old=load_or_trim(os.path.join(R4, "C2_final_mix.wav"), win_g.n),
        gate_times=g_gate, gate_note=identity)
    zeros_g = np.zeros(win_g.n)
    add("golden_music_only", win_g, zeros_g, zeros_g, 0.5 * win_g.ref)
    sh_g = int(PRE["control_B_shifted_gate"]["shift_samples"]["golden"])
    env_gs = np.roll(env_g_fix, sh_g)
    add("golden_C1_fixed_shift", win_g, env_gs, env_gs[:, None] * win_g.raw,
        0.5 * win_g.ref + 0.5 * env_gs[:, None] * win_g.raw,
        gate_note="gate times not comparable after the predeclared roll; use contact_support/exposure_partition")

    add("holdoutD_old_E0_C1", win_h, env_h_old,
        load_or_trim(os.path.join(R56, "holdoutD_e0_C1_interaction.wav"), win_h.n),
        load_or_trim(os.path.join(R56, "holdoutD_e0_C1_final_mix.wav"), win_h.n),
        preds_tvideo=e0_tvideo, gate_times=e0_gate,
        gate_note="123 saved R5.5-E0 window times (strong/present/weak)")
    add("holdoutD_E0_C1_fixed", win_h, env_h_fix,
        load_or_trim(os.path.join(F, "holdoutD_E0_C1_fixed_interaction.wav"), win_h.n),
        load_or_trim(os.path.join(F, "holdoutD_E0_C1_fixed_final_mix.wav"), win_h.n),
        preds_tvideo=e0_tvideo,
        stem_old=load_or_trim(os.path.join(R56, "holdoutD_e0_C1_interaction.wav"), win_h.n),
        final_old=load_or_trim(os.path.join(R56, "holdoutD_e0_C1_final_mix.wav"), win_h.n),
        gate_times=e0_gate,
        gate_note="123 saved R5.5-E0 window times (strong/present/weak)")
    add("holdoutD_old_oracle_C1", win_h, None, None, None, aux=True)  # env/stem built in main()
    add("holdoutD_oracle_C1_fixed", win_h, env_h_orc,
        load_or_trim(os.path.join(F, "holdoutD_oracle_C1_fixed_interaction.wav"), win_h.n),
        load_or_trim(os.path.join(F, "holdoutD_oracle_C1_fixed_final_mix.wav"), win_h.n),
        gate_times=list(win_h.oracle_t_audio), gate_note=identity)
    zeros_h = np.zeros(win_h.n)
    add("holdoutD_music_only", win_h, zeros_h, zeros_h, 0.5 * win_h.ref)
    sh_h = int(PRE["control_B_shifted_gate"]["shift_samples"]["holdoutD"])
    env_hs = np.roll(env_h_fix, sh_h)
    add("holdoutD_E0_fixed_shift", win_h, env_hs, env_hs[:, None] * win_h.raw,
        0.5 * win_h.ref + 0.5 * env_hs[:, None] * win_h.raw)
    return cases


def env_from_hist(win_h):
    """Old oracle envelope for the auxiliary historical oracle row."""
    t = [e["t_audio_refined"] for e in fx.load_json(
        os.path.join(fx.R56_OUT, "holdoutD_oracle_refined.json"))["events"]
        if e.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    spans = fx.spans_from_times(t, 18.0)
    return fx.envelope_from_spans_OLD(18 * fx.SR, spans, 0.010, 0.060)


# ------------------------------------------------------------------ evaluate
def evaluate_case(case, win):
    env, stem, final = case["env"], case["stem"], case["final"]
    out = {"case": case["name"], "window": win.key, "auxiliary": case["aux"]}
    if stem is None:   # auxiliary historical oracle row: build stem now
        hist = np.load(os.path.join(fx.FIX_WORK, "env_holdoutD_oracle_old.npy"))
        stem = hist[:, None] * win.raw
        final = 0.5 * win.ref + 0.5 * stem
        out["note"] = "historical oracle C1 reconstructed (bit-exact to saved WAV, PART 0)"
        env = hist
    out["gate_support"] = ev.gate_support_block(env)
    out["contact_support"] = ev.contact_support_block(env, win.oracle_t_audio, win.oracle_ids)
    out["exposure_partition"] = ev.exposure_partition_block(
        env, win.allowed_spans, win.rest_spans, stem=stem, raw=win.raw)
    out["stem_onsets_vs_oracle"] = ev.output_onsets_block(stem, win.oracle_t_audio)
    mix_on = ev.audio_onsets(final.mean(axis=1))
    blk = {"n_detected_output_onsets": int(len(mix_on))}
    for tol in (0.050,):
        pairs, _, _ = ev.match_one_to_one_optimal(mix_on, win.oracle_t_audio, tol)
        blk[f"tol{int(tol*1000)}ms"] = ev.prf_from_pairs(pairs, len(mix_on), len(win.oracle_t_audio))
    out["final_mix_onsets_vs_oracle"] = blk
    stem_onsets = ev.audio_onsets(stem.mean(axis=1)) if np.any(stem) else np.array([])
    out["reference_onset_coincidence"] = ev.coincidence_block(
        stem_onsets, win.ref_onsets, win.ref_bands)
    out["dynamics"] = ev.dynamics_block(stem, win.raw, win.oracle_t_audio, env, win.oracle_ids)
    out["confidence_4B"] = ev.confidence_block(
        env, win.oracle_t_video, win.visual_high, win.audio_strong, win.oracle_t_audio)
    out["mix_rms"] = ev.mix_rms_block(final, win.raw, win.ref, stem_fixed=stem,
                                      stem_old=case["stem_old"], final_old=case["final_old"])
    if case["preds_tvideo"] is not None:
        out["detector_matching_vs_oracle_video"] = ev.event_matching_block(
            case["preds_tvideo"], win.oracle_t_video)
        # VISUAL_HIGH detector recall with one-to-one matching @50 ms
        hi = win.oracle_t_video[win.visual_high]
        pairs, _, _ = ev.match_one_to_one_optimal(np.asarray(case["preds_tvideo"]), hi, 0.050)
        out["detector_matching_vs_oracle_video"]["VISUAL_HIGH_recall_1to1_50ms"] = {
            "matched": len(pairs), "n_visual_high": int(len(hi)),
            "recall": round(len(pairs) / max(len(hi), 1), 3)}
    # gate-opening times vs oracle audio events (timing quality of the gate itself)
    if case["gate_times"] is not None:
        out["gate_times_note"] = case["gate_note"]
        out["gate_times_vs_oracle_audio"] = ev.event_matching_block(
            case["gate_times"], win.oracle_t_audio)
    return out


def main():
    win_g, win_h = make_windows()
    cases = build_cases(win_g, win_h)
    # auxiliary row: historical oracle envelope (old implementation), stem/final rebuilt
    np.save(os.path.join(fx.FIX_WORK, "env_holdoutD_oracle_old.npy"), env_from_hist(win_h))
    for c in cases:
        if c["name"] == "holdoutD_old_oracle_C1":
            c["env"] = np.load(os.path.join(fx.FIX_WORK, "env_holdoutD_oracle_old.npy"))
            c["stem"] = c["env"][:, None] * win_h.raw
            c["final"] = 0.5 * win_h.ref + 0.5 * c["stem"]
    results = {}
    for c in cases:
        if c["aux"] and c["stem"] is None:
            hist = c["env"]
            c["stem"] = hist[:, None] * win_h.raw
            c["final"] = 0.5 * win_h.ref + 0.5 * c["stem"]
        print("evaluating", c["name"], "...")
        results[c["name"]] = evaluate_case(c, win_g if c["win"] == "golden" else win_h)
    fx.save_json(os.path.join(fx.FIX_LOG, "evaluation_corrected.json"), results)
    print("saved -> logs/evaluation_corrected.json")


if __name__ == "__main__":
    main()
