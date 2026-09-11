# R3 Phase 2 - AudioSep AUDIO-query probe (zero-shot cross-modal: the released
# audiosep_base was trained with use_text_ratio=1.0, i.e. text-conditioned, so
# feeding an audio embedding is OFF-DISTRIBUTION. This is a cheap probe only.)
#
# Reuses the R2 environment, vendored AudioSep and checkpoints unchanged.
# Same compat shims as R2 scripts/10_audiosep_run.py (none touch vendor files):
#   1. torch>=2.6 weights_only=False for the official ckpts
#   2. drop obsolete "position_ids" buffer for strict CLAP load
#   3. CLAP_Encoder singleton factory at import time
# Conditions: get_query_embed(modality='audio', audio=query32k) for Q1, Q2.
# Input: golden_raw.wav only. Residual = mixture32k - target.
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

R3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R2_DIR = r"D:\Code\RhythmAlign\experiments\r2_target_separation"
THIRD_PARTY = os.path.join(R2_DIR, "third_party", "AudioSep")
OUT_DIR = os.path.join(R3_DIR, "outputs")
LOG_DIR = os.path.join(R3_DIR, "logs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"

AUDIOSEP_CKPT = os.path.join(R2_DIR, "checkpoints", "audiosep_base_4M_steps.ckpt")
CLAP_CKPT = os.path.join(R2_DIR, "checkpoints", "music_speech_audioset_epoch_15_esc_89.98.pt")
QUERIES = {"q1": os.path.join(R3_DIR, "queries", "query_q1.wav"),
           "q2": os.path.join(R3_DIR, "queries", "query_q2.wav")}


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    sys.path.insert(0, THIRD_PARTY)
    import models.clap_encoder as _ce

    _orig_clap = _ce.CLAP_Encoder
    _holder = {}

    def _singleton_clap(*args, **kwargs):
        if "enc" not in _holder:
            _holder["enc"] = _orig_clap(pretrained_path=CLAP_CKPT).eval()
        return _holder["enc"]

    _ce.CLAP_Encoder = _singleton_clap

    import torch

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

    import librosa
    from models.audiosep import AudioSep, get_model_class
    from utils import parse_yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | torch {torch.__version__}")

    query_encoder = _holder["enc"]
    configs = parse_yaml(os.path.join(THIRD_PARTY, "config", "audiosep_base.yaml"))
    SsModel = get_model_class(model_type=configs["model"]["model_type"])
    ss_model = SsModel(
        input_channels=configs["model"]["input_channels"],
        output_channels=configs["model"]["output_channels"],
        condition_size=configs["model"]["condition_size"],
    )
    pl_model = AudioSep(
        ss_model=ss_model, waveform_mixer=None, query_encoder=query_encoder,
        loss_function=None, optimizer_type=None, learning_rate=None, lr_lambda_func=None,
    )
    sd = torch.load(AUDIOSEP_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = pl_model.load_state_dict(sd, strict=False)
    print(f"ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
    pl_model = pl_model.eval().to(device)

    mixture, fs = librosa.load(os.path.join(R1_OUT, "golden_raw.wav"), sr=32000, mono=True)

    runs = []
    for qid, qpath in QUERIES.items():
        query32, _ = librosa.load(qpath, sr=32000, mono=True)
        with torch.no_grad():
            emb = pl_model.query_encoder.get_query_embed(
                modality="audio", audio=torch.Tensor(query32)[None, :], device=device)
            print(f"{qid}: query {len(query32)/fs:.3f}s -> embedding shape {tuple(emb.shape)} "
                  f"norm {emb.norm().item():.3f}")
            input_dict = {
                "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                "condition": emb.to(device),
            }
            sep = pl_model.ss_model.chunk_inference(input_dict)
        target = np.squeeze(sep).astype(np.float32)
        residual = (mixture - target).astype(np.float32)

        name = f"audiosep_aq_{qid}"
        sf.write(os.path.join(OUT_DIR, f"{name}_target.wav"), target, fs, subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, f"{name}_residual.wav"), residual, fs, subtype="FLOAT")

        def db(x):
            return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))

        runs.append({
            "name": name, "query": qpath,
            "mixture_rms_db": db(mixture), "target_rms_db": db(target),
            "residual_rms_db": db(residual), "target_peak": float(np.max(np.abs(target))),
            "energy_ratio_db": db(target) - db(mixture),
        })
        print(f"  {name}: target {db(target):.1f} dB vs mixture {db(mixture):.1f} dB "
              f"({db(target) - db(mixture):+.1f} dB) | residual {db(residual):.1f} dB")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(LOG_DIR, "03_audiosep_audio_query_runs.json"), "w", encoding="utf-8") as f:
        json.dump({
            "torch": torch.__version__, "device": str(device), "checkpoint": AUDIOSEP_CKPT,
            "note": "zero-shot cross-modal probe; released base model is text-conditioned "
                    "(use_text_ratio=1.0)",
            "runs": runs,
        }, f, indent=2)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
