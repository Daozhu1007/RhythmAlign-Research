# S1W step 04 - zero-shot parity check (protocol section 17).
# 1) regenerate the S1E zero-shot CLAPSep output for c6_dense_golden with the local
#    migrated checkpoint via r3's official-path clapsep_lib.separate, and compare
#    numerically against S1E's stored output WAV (local, untracked);
# 2) verify the S1W training forward path (clapsep_train_lib.train_forward) matches
#    the official inference_from_data path chunk-by-chunk (no_grad comparison);
# 3) verify checkpoint loading coverage (which keys loaded / LoRA presence).
import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa
import torch

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(S1W, "scripts"))
import clapsep_train_lib as ctl  # noqa: E402
import clapsep_lib  # noqa: E402

S1E = r"D:\Code\RhythmAlign-Research\experiments\s1e_existing_corpus_feasibility"
CLIP = os.path.join(S1E, "clips", "audio", "c6_dense_golden_32k_mono.wav")
REF = os.path.join(S1E, "outputs", "audio", "clapsep_aq_c6_dense_golden_target.wav")
LOG = os.path.join(S1W, "logs")


def stats(a, b):
    d = a - b
    return {"max_abs_diff": float(np.max(np.abs(d))),
            "rms_diff": float(np.sqrt(np.mean(d ** 2))),
            "rms_a": float(np.sqrt(np.mean(a ** 2))),
            "corr": float(np.corrcoef(a, b)[0, 1])}


def main():
    device = torch.device("cuda")
    model = ctl.load_model(device)
    ctl.set_trainable_scope(model)

    # --- checkpoint loading coverage ---
    ckpt = torch.load(ctl.CKPT_BEST, map_location="cpu", weights_only=False)
    msd = model.state_dict()
    ck = set(ckpt.keys())
    mk = set(msd.keys())
    cov = {"ckpt_keys": len(ck), "model_keys": len(mk),
           "ckpt_in_model": len(ck & mk), "ckpt_not_in_model": sorted(ck - mk)[:8],
           "n_ckpt_not_in_model": len(ck - mk),
           "model_not_in_ckpt": sorted(mk - ck)[:8],
           "n_model_not_in_ckpt": len(mk - ck),
           "lora_keys_in_ckpt": sum(1 for k in ck if "lora" in k.lower()),
           "decoder_keys_in_ckpt": sum(1 for k in ck if k.startswith("decoder_model")),
           "mismatched_shapes": [k for k in (ck & mk) if ckpt[k].shape != msd[k].shape][:8]}
    print("checkpoint coverage:", json.dumps(cov, indent=1))

    # --- 1) full zero-shot parity vs stored S1E output ---
    mix, _ = librosa.load(CLIP, sr=32000, mono=True)
    emb = ctl.embed_q1(model)
    zeros = np.zeros((1, 512), dtype=np.float32)
    target = clapsep_lib.separate(model, mix, emb, zeros, device)
    ref, ref_sr = sf.read(REF, dtype="float32")
    assert ref_sr == 32000 and len(ref) == len(target), (ref_sr, len(ref), len(target))
    p1 = stats(target, ref)
    print("zero-shot parity vs stored S1E output:", json.dumps(p1, indent=1))

    # --- 2) train-forward equivalence with official inference path ---
    ctl.set_training_mode(model, training=False)
    n = ctl.CHUNK
    chunk = torch.tensor(mix[:n], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        official = model.inference_from_data(
            chunk.clone(),
            torch.tensor(np.asarray(emb), dtype=torch.float32, device=device),
            torch.tensor(zeros, dtype=torch.float32, device=device))
        _mask, mine, _ = ctl.train_forward(model, chunk, emb, zeros)
    official = official.squeeze().cpu().numpy()
    mine = mine.squeeze().detach().cpu().numpy()
    p2 = stats(mine, official)
    print("train-forward vs official inference (same chunk):", json.dumps(p2, indent=1))

    ok = (p1["max_abs_diff"] < 2e-4 and p1["corr"] > 0.999999
          and p2["max_abs_diff"] < 2e-4)
    res = {"zero_shot_parity": p1, "forward_equivalence": p2, "checkpoint_coverage": cov,
           "parity_credible": bool(ok),
           "env": {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
           "checkpoint": {"best_model.ckpt": ctl.CKPT_BEST,
                          "clap_ckpt": ctl.CKPT_CLAP}}
    os.makedirs(LOG, exist_ok=True)
    with open(os.path.join(LOG, "04_parity_check.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("PARITY:", "CREDIBLE" if ok else "FAILED")


if __name__ == "__main__":
    main()
