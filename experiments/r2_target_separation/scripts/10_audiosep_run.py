# R2 - AudioSep (official pretrained) baseline on the R1 golden sample.
# SAM-Audio is BLOCKED (gated checkpoints, no HF auth on this machine) - see
# logs/00_env_audit.md. AudioSep is the only runnable model family this round.
#
# Protocol (per R2 brief):
#   inputs : golden_raw.wav + A2_residual.wav (48 kHz stereo sources; the model
#            pipeline itself loads 32 kHz mono - documented limitation)
#   prompts: exactly 2, selected from a fixed candidate list by CLAP audio-text
#            cosine similarity (diagnostic logged, selection is deterministic)
#   runs   : model.ss_model.chunk_inference (official long-audio path: 3 s core
#            + 1 s context windows, matching the 5 s training segments)
#   outputs: float32 32 kHz mono target + residual = mixture32k - target
#
# Source files are read-only; everything is written under experiments/r2_target_separation/.
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

R2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THIRD_PARTY = os.path.join(R2_DIR, "third_party", "AudioSep")
OUT_DIR = os.path.join(R2_DIR, "outputs")
LOG_DIR = os.path.join(R2_DIR, "logs")
CKPT_DIR = os.path.join(R2_DIR, "checkpoints")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"

AUDIOSEP_CKPT = os.path.join(CKPT_DIR, "audiosep_base_4M_steps.ckpt")
CLAP_CKPT = os.path.join(CKPT_DIR, "music_speech_audioset_epoch_15_esc_89.98.pt")

# Candidate prompts: the 4 SAM-Audio prompts from the brief + AudioCaps-style
# short phrases (AudioSep's CLAP text encoder was trained on AudioCaps-style
# captions, so semantically closer matches may use simpler wording).
CANDIDATE_PROMPTS = [
    "arcade button presses",
    "hand hitting arcade game buttons",
    "finger taps and button presses on a rhythm game cabinet",
    "physical hand impacts on an arcade rhythm game machine",
    "tapping",
    "tapping sounds",
    "clicking",
    "button clicks",
    "percussive taps",
]


def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Vendored AudioSep instantiates CLAP_Encoder() (default relative ckpt path)
    # at class-definition time. Intercept it with a singleton built from our
    # absolute checkpoint path so the weights load exactly once.
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

    # torch>=2.6 defaults torch.load to weights_only=True; the official CLAP
    # ckpt (from the public Audio-AGI Space) contains numpy scalars and needs
    # weights_only=False. Both checkpoints here come from the official Space.
    _orig_load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load_compat

    # Old CLAP ckpts carry a "position_ids" buffer that modern transformers'
    # RobertaModel no longer registers; drop that known-obsolete key so the
    # vendored factory's strict load succeeds. Any other mismatch still raises.
    _orig_lsd = torch.nn.Module.load_state_dict

    def _lsd_compat(self, state_dict, strict=True, **kwargs):
        if isinstance(state_dict, dict):
            state_dict = {k: v for k, v in state_dict.items() if "position_ids" not in k}
        return _orig_lsd(self, state_dict, strict=strict, **kwargs)

    torch.nn.Module.load_state_dict = _lsd_compat

    from models.audiosep import AudioSep, get_model_class
    from utils import parse_yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | torch {torch.__version__} | cuda avail {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    # ---- build model (direct construction; no Lightning hparams loading) ----
    query_encoder = _holder["enc"]   # already built by the import-time default

    configs = parse_yaml(os.path.join(THIRD_PARTY, "config", "audiosep_base.yaml"))
    SsModel = get_model_class(model_type=configs["model"]["model_type"])
    ss_model = SsModel(
        input_channels=configs["model"]["input_channels"],
        output_channels=configs["model"]["output_channels"],
        condition_size=configs["model"]["condition_size"],
    )
    pl_model = AudioSep(
        ss_model=ss_model,
        waveform_mixer=None,
        query_encoder=query_encoder,
        loss_function=None,
        optimizer_type=None,
        learning_rate=None,
        lr_lambda_func=None,
    )
    sd = torch.load(AUDIOSEP_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = pl_model.load_state_dict(sd, strict=False)
    print(f"ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys (first 5): {missing[:5]}")
    pl_model = pl_model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- prompt selection diagnostic: CLAP audio-text cosine similarity ----
    import librosa

    mixture_ref, _ = librosa.load(os.path.join(R1_OUT, "golden_raw.wav"), sr=32000, mono=True)
    with torch.no_grad():
        audio_emb = pl_model.query_encoder.get_query_embed(
            modality="audio", audio=torch.Tensor(mixture_ref)[None, :], device=device
        ).cpu().numpy()[0]
        text_embs, scores = [], []
        for p in CANDIDATE_PROMPTS:
            te = pl_model.query_encoder.get_query_embed(
                modality="text", text=[p], device=device
            ).cpu().numpy()[0]
            scores.append({"prompt": p, "cosine_vs_golden_raw": cosine(audio_emb, te)})
    scores.sort(key=lambda r: -r["cosine_vs_golden_raw"])
    # greedy pick top-2 with distinct wording (text-emb cosine < 0.9)
    chosen = [scores[0]["prompt"]]
    for row in scores[1:]:
        with torch.no_grad():
            te = pl_model.query_encoder.get_query_embed(
                modality="text", text=[row["prompt"]], device=device
            ).cpu().numpy()[0]
        te0 = pl_model.query_encoder.get_query_embed(
            modality="text", text=[chosen[0]], device=device
        ).cpu().numpy()[0]
        if cosine(te, te0) < 0.9:
            chosen.append(row["prompt"])
        if len(chosen) == 2:
            break
    print("prompt similarity ranking:")
    for row in scores:
        mark = " <== chosen" if row["prompt"] in chosen else ""
        print(f"  {row['cosine_vs_golden_raw']:.4f}  {row['prompt']}{mark}")
    print(f"chosen prompts: p1={chosen[0]!r} p2={chosen[1]!r}")
    with open(os.path.join(LOG_DIR, "prompt_selection.json"), "w", encoding="utf-8") as f:
        json.dump({"ranking": scores, "chosen": {"p1": chosen[0], "p2": chosen[1]}}, f, indent=2)

    # ---- separation runs: 2 prompts x 2 inputs ----
    inputs = [
        ("raw", os.path.join(R1_OUT, "golden_raw.wav")),
        ("a2", os.path.join(R1_OUT, "A2_residual.wav")),
    ]
    runs = []
    for tag, path in inputs:
        mixture, fs = librosa.load(path, sr=32000, mono=True)
        print(f"\n=== input '{tag}' {os.path.basename(path)} | {len(mixture)/fs:.2f}s @ {fs} Hz mono ===")
        for pi, prompt in enumerate(chosen, start=1):
            with torch.no_grad():
                conditions = pl_model.query_encoder.get_query_embed(
                    modality="text", text=[prompt], device=device
                )
                input_dict = {
                    "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                    "condition": conditions.to(device),
                }
                sep = pl_model.ss_model.chunk_inference(input_dict)
            target = np.squeeze(sep).astype(np.float32)
            residual = (mixture - target).astype(np.float32)

            name = f"audiosep_{tag}_p{pi}"
            sf.write(os.path.join(OUT_DIR, f"{name}_target.wav"), target, fs, subtype="FLOAT")
            sf.write(os.path.join(OUT_DIR, f"{name}_residual.wav"), residual, fs, subtype="FLOAT")

            def db(x):
                return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))

            stats = {
                "name": name,
                "input": path,
                "input_tag": tag,
                "prompt": prompt,
                "prompt_id": f"p{pi}",
                "sr": fs,
                "channels": 1,
                "mixture_rms_db": db(mixture),
                "target_rms_db": db(target),
                "residual_rms_db": db(residual),
                "target_peak": float(np.max(np.abs(target))),
            }
            if device.type == "cuda":
                stats["cuda_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            runs.append(stats)
            print(f"  {name}: prompt={prompt!r}")
            print(f"    mixture {stats['mixture_rms_db']:.1f} dB | target {stats['target_rms_db']:.1f} dB "
                  f"| residual {stats['residual_rms_db']:.1f} dB | peak {stats['target_peak']:.3f}")

    with open(os.path.join(LOG_DIR, "audiosep_runs.json"), "w", encoding="utf-8") as f:
        json.dump({
            "torch": torch.__version__,
            "device": str(device),
            "checkpoint": AUDIOSEP_CKPT,
            "clap_checkpoint": CLAP_CKPT,
            "chunk_inference_config": {"NL": 1.0, "NC": 3.0, "NR": 1.0, "RATE": 32000},
            "runs": runs,
        }, f, indent=2)
    print(f"\ndone in {time.time() - t0:.1f}s -> outputs/ + logs/audiosep_runs.json")


if __name__ == "__main__":
    main()
