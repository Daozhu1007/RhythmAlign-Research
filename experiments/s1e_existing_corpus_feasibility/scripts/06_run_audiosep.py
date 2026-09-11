# S1E step 06 - AudioSep historical baseline on selected S1E clips (official
# pretrained base ckpt, vendored code in r2, used read-only; loading recipe copied
# from r2 scripts/10_audiosep_run.py). Prompt fixed a priori: r2's p2
# ("finger taps and button presses on a rhythm game cabinet") which was R2's best.
# Runs on: c1/c2 (speech), c3 (taiko prompt), c6 (golden, for continuity with r2/r3).
import json
import os
import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch

S1E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R2 = r"D:\Code\RhythmAlign\experiments\r2_target_separation"
THIRD_PARTY = os.path.join(R2, "third_party", "AudioSep")
AUDIOSEP_CKPT = os.path.join(R2, "checkpoints", "audiosep_base_4M_steps.ckpt")
CLAP_CKPT = os.path.join(R2, "checkpoints", "music_speech_audioset_epoch_15_esc_89.98.pt")

from s1e_common import CLIP_DIR, OUT_DIR, LOG_DIR, save_json

PROMPT = "finger taps and button presses on a rhythm game cabinet"  # r2 p2, fixed
CLIPS = ["c1_speech_npc_a", "c2_speech_npc_b", "c3_taiko_prompt", "c6_dense_golden"]


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def build_model(device):
    sys.path.insert(0, THIRD_PARTY)
    import models.clap_encoder as _ce

    _orig_clap = _ce.CLAP_Encoder
    _holder = {}

    def _singleton_clap(*args, **kwargs):
        if "enc" not in _holder:
            _holder["enc"] = _orig_clap(pretrained_path=CLAP_CKPT).eval()
        return _holder["enc"]

    _ce.CLAP_Encoder = _singleton_clap

    _orig_load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load_compat

    _orig_lsd = torch.nn.Module.load_state_dict

    def _lsd_compat(self, state_dict, strict=True, **kwargs):
        if isinstance(state_dict, dict):
            state_dict = {k: v for k, v in state_dict.items() if "position_ids" not in k}
        return _orig_lsd(self, state_dict, strict=strict, **kwargs)

    torch.nn.Module.load_state_dict = _lsd_compat

    from models.audiosep import AudioSep, get_model_class
    from utils import parse_yaml

    configs = parse_yaml(os.path.join(THIRD_PARTY, "config", "audiosep_base.yaml"))
    SsModel = get_model_class(model_type=configs["model"]["model_type"])
    ss_model = SsModel(
        input_channels=configs["model"]["input_channels"],
        output_channels=configs["model"]["output_channels"],
        condition_size=configs["model"]["condition_size"],
    )
    pl_model = AudioSep(
        ss_model=ss_model, waveform_mixer=None, query_encoder=_holder["enc"],
        loss_function=None, optimizer_type=None, learning_rate=None, lr_lambda_func=None,
    )
    sd = torch.load(AUDIOSEP_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    pl_model.load_state_dict(sd, strict=False)
    return pl_model.eval().to(device)


def main():
    t00 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | torch {torch.__version__}")
    pl_model = build_model(device)

    runs = []
    WINDOW = int((1.0 + 3.0 + 1.0) * 32000)  # official chunk config NL+NC+NR
    for cid in CLIPS:
        mixture, fs = librosa.load(os.path.join(CLIP_DIR, "audio", f"{cid}_32k_mono.wav"),
                                   sr=32000, mono=True)
        # Workaround for an upstream boundary bug in official chunk_inference
        # (`while current_idx + WINDOW < L` leaves the final partial window silent;
        # a 5 s input never enters the loop at all). Pad, infer, trim — the official
        # chunk algorithm itself is untouched.
        pad = WINDOW + int(3.0 * 32000)
        mixture_pad = np.concatenate([mixture, np.zeros(pad, dtype=mixture.dtype)])
        with torch.no_grad():
            conditions = pl_model.query_encoder.get_query_embed(
                modality="text", text=[PROMPT], device=device)
            input_dict = {
                "mixture": torch.Tensor(mixture_pad)[None, None, :].to(device),
                "condition": conditions.to(device),
            }
            sep = pl_model.ss_model.chunk_inference(input_dict)
        target = np.squeeze(sep).astype(np.float32)[: len(mixture)]
        name = f"audiosep_p2_{cid}"
        sf.write(os.path.join(OUT_DIR, "audio", f"{name}_target.wav"), target, fs,
                 subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, "audio", f"{name}_residual.wav"),
                 (mixture - target).astype(np.float32), fs, subtype="FLOAT")
        runs.append({"run": name, "clip": cid, "prompt": PROMPT, "sr": 32000,
                     "mix_rms_db": db(mixture), "target_rms_db": db(target),
                     "target_energy_ratio_db": db(target) - db(mixture)})
        print(f"{name}: ratio {db(target)-db(mixture):+.1f} dB")
        torch.cuda.empty_cache()
    save_json(os.path.join(LOG_DIR, "06_audiosep_runs.json"),
              {"protocol": "official chunk_inference (r2 recipe), 32 kHz mono; upstream "
                           "short-input boundary bug handled by pad->infer->trim (disclosed)",
               "checkpoint": AUDIOSEP_CKPT, "prompt": PROMPT, "runs": runs})
    print(f"done in {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
