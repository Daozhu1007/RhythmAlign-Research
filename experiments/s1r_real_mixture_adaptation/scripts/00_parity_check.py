# S1R step 00 - ZERO-SHOT PARITY REPRODUCTION (protocol section 9).
#
# Before any S1R work, the frozen zero-shot CLAPSep (exact S1E/S1W checkpoint
# family, audio query Q1, zero negative) must reproduce a prior known result on
# at least one case. Reference: S1E stored output
#   s1e_existing_corpus_feasibility/outputs/audio/clapsep_aq_c6_dense_golden_target.wav
# S1W's own parity check measured max|delta| = 0.0 against this file.
# Failure => STOP: ZERO_SHOT_PARITY_FAILURE.
import os
import sys
import json
import numpy as np
import soundfile as sf

S1R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1R, "scripts"))
import s1r_common as rc  # noqa: E402

REF_WAV = os.path.join(rc.S1W, "..", "s1e_existing_corpus_feasibility", "outputs",
                       "audio", "clapsep_aq_c6_dense_golden_target.wav")
CLIP_WAV = os.path.join(rc.S1W, "..", "s1e_existing_corpus_feasibility", "clips",
                        "audio", "c6_dense_golden_32k_mono.wav")


def main():
    import torch  # noqa: E402
    import clapsep_lib  # noqa: E402  (r3, read-only)

    assert os.path.exists(REF_WAV), f"reference missing: {REF_WAV}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = clapsep_lib.load_clapsep(device)
    emb = clapsep_lib.embed_audio_query(model, rc.Q1_WAV)
    zeros = np.zeros((1, 512), dtype=np.float32)

    mix, sr = sf.read(CLIP_WAV, dtype="float32")
    assert sr == 32000
    out = clapsep_lib.separate(model, mix, emb, zeros, device)

    ref, rsr = sf.read(REF_WAV, dtype="float32")
    assert rsr == 32000
    n = min(len(out), len(ref))
    delta = np.abs(out[:n] - ref[:n])
    res = {
        "case": "c6_dense_golden",
        "n_samples": int(n),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "out_rms_db": rc.rms_db(out[:n]),
        "ref_rms_db": rc.rms_db(ref[:n]),
        "waveform_corr": rc.wf_corr(out[:n], ref[:n]),
        "torch": torch.__version__,
        "device": str(device),
    }
    res["parity_pass"] = res["max_abs_delta"] == 0.0
    rc.save_json(os.path.join(S1R, "logs", "00_parity_check.json"), res)
    print(json.dumps(res, indent=1))
    if not res["parity_pass"]:
        print("ZERO_SHOT_PARITY_FAILURE")
        sys.exit(2)
    print("PARITY PASS (max|delta| = 0.0)")


if __name__ == "__main__":
    main()
