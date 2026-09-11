# R5-FIX PART 2 - regenerate corrected C1 with the SAME frozen contact /
# refinement information and the SAME window policy, changing ONLY the envelope
# implementation. Historical WAVs are read but never modified.
#   A. golden oracle          -> golden_oracle_C1_fixed_{interaction,final_mix}.wav
#   B. Holdout-D oracle       -> holdoutD_oracle_C1_fixed_{interaction,final_mix}.wav
#   C. Holdout-D R5.5 E0      -> holdoutD_E0_C1_fixed_{interaction,final_mix}.wav
# Final mix arithmetic is verified after write:  final == 0.5*ref + 0.5*interaction.
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixenv as fx

OUT = fx.FIX_OUT


def load_wav(p):
    return sf.read(p, dtype="float64", always_2d=True)[0]


def golden_pair():
    raw = load_wav(os.path.join(fx.R1_OUT, "golden_raw.wav"))
    ref = load_wav(os.path.join(fx.R1_WORK, "ref_warp_fixed.wav"))[3 * fx.SR:18 * fx.SR]
    assert len(raw) == len(ref) == 15 * fx.SR
    return raw, ref


def holdout_pair():
    raw = load_wav(os.path.join(fx.R56_OUT, "holdoutD_raw.wav"))
    ref = load_wav(os.path.join(fx.R56_WORK, "holdoutD_ref_warp.wav"))[3 * fx.SR:21 * fx.SR]
    assert len(raw) == len(ref) == 18 * fx.SR
    return raw, ref


def golden_oracle_times():
    ev = [e for e in fx.load_json(os.path.join(fx.R4_OUT, "contacts_refined.json"))["events"]
          if e["refined_confidence"] != "no_transient" and e["type"] != "none"]
    return [e["t_audio_refined"] for e in ev], len(ev)


def holdout_e0_times():
    # SAVED R5.5-E0 predictions on Holdout-D (provenance re-verified in PART 0):
    # the 150 refined candidates of holdoutD_contacts_auto_refined.json; E0
    # opens windows exactly for refined_confidence in (strong, present, weak).
    ev = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_contacts_auto_refined.json"))["events"]
    t = [e["t_audio_refined"] for e in ev
         if e.get("refined_confidence") in ("strong", "present", "weak")]
    return t, len(ev)


def holdout_oracle_times():
    ev = fx.load_json(os.path.join(fx.R56_OUT, "holdoutD_oracle_refined.json"))["events"]
    t = [e["t_audio_refined"] for e in ev
         if e.get("refined_confidence") not in (None, "no_transient", "out_of_range")]
    return t, len(ev)


def build(tag, raw, ref, times, dur, env_key):
    n = len(raw)
    spans = fx.spans_from_times(times, dur)
    env = fx.envelope_from_spans_FIXED(n, spans, fx.C1_ATTACK_S, fx.C1_RELEASE_S)
    inter = fx.build_interaction(raw, env)
    mix = fx.final_mix(ref, inter)
    inter32 = inter.astype(np.float32)
    mix32 = (0.5 * ref + 0.5 * inter).astype(np.float32)
    # pre-write check: written array == the fixed arithmetic applied to the
    # full-precision interaction (identical expression -> must be exact 0)
    pre_err = float(np.max(np.abs(mix32 - (0.5 * ref + 0.5 * inter).astype(np.float32))))
    p_i = os.path.join(OUT, f"{tag}_interaction.wav")
    p_m = os.path.join(OUT, f"{tag}_final_mix.wav")
    sf.write(p_i, inter32, fx.SR, subtype="FLOAT")
    sf.write(p_m, mix32, fx.SR, subtype="FLOAT")
    # read-back check: float32 quantisation of `inter` may shift the sum by
    # <=1 float32 ULP (the historical artifacts show the same 2.98e-8 bound)
    ri = sf.read(p_i, dtype="float32", always_2d=True)[0].astype(np.float64)
    rm = sf.read(p_m, dtype="float32", always_2d=True)[0].astype(np.float64)
    err = float(np.max(np.abs(rm - (0.5 * ref + 0.5 * ri))))
    np.save(os.path.join(fx.FIX_WORK, f"env_{env_key}.npy"), env)
    return {"tag": tag, "n_events": len(times), "n_spans_after_merge": len(spans),
            "mean_env": float(env.mean()), "nonzero_env_fraction": float(np.mean(env > 0)),
            "final_mix_arith_max_err_prewrite": pre_err,
            "final_mix_arith_max_err_readback": err,
            "interaction": os.path.basename(p_i), "final_mix": os.path.basename(p_m)}


def main():
    res = {"phase": "R5-FIX PART 2 corrected C1 regeneration",
           "envelope": "FIXED: entry 0->1 (10 ms), interior 1, exit 1->0 (60 ms), merge max",
           "window_policy": "pre 15 ms / post 150 ms (unchanged frozen C1 policy)",
           "final_mix_convention": "0.5 * pristine aligned reference + 0.5 * interaction (no limiter/compressor/normalization)",
           "builds": {}}

    raw_g, ref_g = golden_pair()
    t, n = golden_oracle_times()
    res["builds"]["golden_oracle"] = build("golden_oracle_C1_fixed", raw_g, ref_g, t, 15.0, "golden_oracle_fixed")
    res["builds"]["golden_oracle"]["source"] = "R4 contacts_refined.json (84 build events, unchanged)"

    raw_h, ref_h = holdout_pair()
    t, n = holdout_e0_times()
    res["builds"]["holdoutD_E0"] = build("holdoutD_E0_C1_fixed", raw_h, ref_h, t, 18.0, "holdoutD_E0_fixed")
    res["builds"]["holdoutD_E0"]["source"] = ("R5.5 frozen E0 predictions as saved in "
                                              "r56 outputs/holdoutD_contacts_auto_refined.json "
                                              "(strong/present/weak -> 123 windows)")

    t, n = holdout_oracle_times()
    res["builds"]["holdoutD_oracle"] = build("holdoutD_oracle_C1_fixed", raw_h, ref_h, t, 18.0, "holdoutD_oracle_fixed")
    res["builds"]["holdoutD_oracle"]["source"] = ("Holdout-D video-first oracle as saved in "
                                                  "r56 outputs/holdoutD_oracle_refined.json (72 window-opening events)")

    for k, b in res["builds"].items():
        assert b["final_mix_arith_max_err_prewrite"] == 0.0, (k, b)
        assert b["final_mix_arith_max_err_readback"] <= 3e-8, (k, b)   # <=1 float32 ULP
        print(k, "->", b["interaction"], "|", b["final_mix"],
              f"| mean env {b['mean_env']:.4f} | spans {b['n_spans_after_merge']}"
              f"| arith prewrite {b['final_mix_arith_max_err_prewrite']} readback {b['final_mix_arith_max_err_readback']:.2e}")

    fx.save_json(os.path.join(fx.FIX_LOG, "part2_rebuild_summary.json"), res)
    print("summary -> logs/part2_rebuild_summary.json")


if __name__ == "__main__":
    main()
