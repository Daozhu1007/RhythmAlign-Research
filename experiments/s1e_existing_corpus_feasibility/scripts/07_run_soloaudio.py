# S1E step 07 - SoloAudio (diffusion generative TSE, official westbrook/SoloAudio v2
# checkpoints) on all challenge clips. Official inference path (tse_audioTSE.py /
# tse_languageTSE.py logic, batch-adapted; no repo file modified).
# Conditions: audio query = r3's Q1 exemplar (same as CLAPSep primary); text probe on
# the nuisance clips with the R2 p2 prompt. Fixed seed 2024 -> deterministic.
# GENERATIVE: output is a model reconstruction, not a masked copy - disclosed.
# Output 24 kHz mono (model limitation). Gain is NOT normalized.
import os
import sys
import time
import json

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
import yaml
from diffusers import DDIMScheduler
from transformers import AutoProcessor, ClapModel

S1E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.join(S1E, "third_party", "SoloAudio")
sys.path.insert(0, REPO)

from model.udit import UDiT            # noqa: E402
from vae_modules.autoencoder_wrapper import Autoencoder  # noqa: E402

from s1e_common import CLIP_DIR, OUT_DIR, LOG_DIR, save_json  # noqa: E402

R3 = r"D:\Code\RhythmAlign\experiments\r3_audio_query"
Q1 = os.path.join(R3, "queries", "query_q1.wav")
TEXT_PROMPT = "finger taps and button presses on a rhythm game cabinet"  # r2 p2, fixed
TEXT_PROBE_CLIPS = {"c1_speech_npc_a", "c2_speech_npc_b", "c3_taiko_prompt"}
SEED = 2024
STEPS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


@torch.no_grad()
def sample_diffusion(unet, autoencoder, scheduler, mixture, timbre, ddim_steps=50, eta=0, seed=SEED):
    unet.eval()
    scheduler.set_timesteps(ddim_steps)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn(mixture.shape, generator=generator, device=DEVICE)
    pred = noise
    for t in scheduler.timesteps:
        pred = scheduler.scale_model_input(pred, t)
        model_output = unet(x=pred, timesteps=t, mixture=mixture, timbre=timbre)
        pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                              eta=eta, generator=generator).prev_sample
    pred = autoencoder(embedding=pred).squeeze(1)
    return pred


def main():
    t00 = time.time()
    with open(os.path.join(REPO, "config", "SoloAudio.yaml")) as fp:
        diff_config = yaml.safe_load(fp)

    clapmodel = ClapModel.from_pretrained("laion/larger_clap_general").to(DEVICE)
    processor = AutoProcessor.from_pretrained("laion/larger_clap_general")

    autoencoder = Autoencoder(os.path.join(REPO, "pretrained_models", "audio-vae.pt"),
                              "stable_vae", quantization_first=True)
    autoencoder.eval().to(DEVICE)

    unet = UDiT(**diff_config["diffwrap"]["UDiT"]).to(DEVICE)
    sd = torch.load(os.path.join(REPO, "pretrained_models", "soloaudio_v2.pt"),
                    map_location="cpu", weights_only=False)
    unet.load_state_dict(sd["model"])
    unet.eval()

    v_pred = diff_config["ddim"]["v_prediction"]
    scheduler = DDIMScheduler(**diff_config["ddim"]["diffusers"])
    _ = scheduler.add_noise(torch.randn((1, 128, 128), device=DEVICE),
                            torch.randn((1, 128, 128), device=DEVICE),
                            torch.randint(0, 1000, (1,), device=DEVICE).long())

    # enrollment embeddings (computed once)
    import soundfile as _sf
    aud_np, fs_a = _sf.read(Q1, dtype="float32", always_2d=True)
    aud = torch.from_numpy(aud_np.T.mean(axis=0, keepdims=True))
    if fs_a != 48000:
        aud = torchaudio.transforms.Resample(fs_a, 48000)(aud)
    n = aud.shape[1]
    target_n = 48000 * 10
    if n > target_n:
        aud = aud[:, :target_n]
    else:
        aud = torch.nn.functional.pad(aud, (0, target_n - n))
    ai = processor(audios=[aud.squeeze().numpy()], sampling_rate=48000,
                   return_tensors="pt", padding=True)
    timbre_q1 = clapmodel.get_audio_features(
        input_features=ai["input_features"][0].unsqueeze(0).to(DEVICE))

    ti = processor(text=[TEXT_PROMPT], max_length=10, padding="max_length",
                   truncation=True, return_tensors="pt")
    timbre_tx = clapmodel.get_text_features(
        input_ids=ti["input_ids"][0].unsqueeze(0).to(DEVICE),
        attention_mask=ti["attention_mask"][0].unsqueeze(0).to(DEVICE))

    manifest = json.load(open(os.path.join(CLIP_DIR, "clip_manifest.json"), encoding="utf-8"))
    runs = []
    for clip in manifest["clips"]:
        cid = clip["id"]
        mixture, _ = librosa.load(os.path.join(CLIP_DIR, "audio", f"{cid}_32k_mono.wav"),
                                  sr=24000, mono=True)
        mix_t = torch.tensor(mixture).unsqueeze(0).to(DEVICE)
        mix_lat = autoencoder(audio=mix_t.unsqueeze(1))
        conds = [("aq", timbre_q1)]
        if cid in TEXT_PROBE_CLIPS:
            conds.append(("tx", timbre_tx))
        for tag, timbre in conds:
            t0 = time.time()
            pred = sample_diffusion(unet, autoencoder, scheduler, mix_lat, timbre,
                                    ddim_steps=STEPS, seed=SEED)
            out = pred.squeeze().cpu().numpy().astype(np.float32)
            name = f"soloaudio_{tag}_{cid}"
            sf.write(os.path.join(OUT_DIR, "audio", f"{name}_target.wav"), out, 24000,
                     subtype="FLOAT")
            runs.append({"run": name, "clip": cid, "condition": tag, "seed": SEED,
                         "steps": STEPS, "sr_out": 24000,
                         "mix_rms_db_24k": db(mixture), "target_rms_db": db(out),
                         "target_gain_dev_db": db(out) - db(mixture),
                         "generative": True, "runtime_s": time.time() - t0})
            print(f"{name}: gain dev {db(out)-db(mixture):+.1f} dB ({time.time()-t0:.0f}s)")
            torch.cuda.empty_cache()
    save_json(os.path.join(LOG_DIR, "07_soloaudio_runs.json"),
              {"protocol": "official westbrook/SoloAudio v2; DDIM 50 steps; seed 2024; "
                           "audio query Q1; output 24 kHz mono; generative (reconstruction)",
               "checkpoint": "pretrained_models/soloaudio_v2.pt", "runs": runs})
    print(f"done in {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
