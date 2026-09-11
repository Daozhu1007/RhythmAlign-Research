# R3 Phase 4 - CLAPSep positive-only audio query on golden_raw.wav.
# Negative condition = zeros (official "empty" mechanism from Space app.py).
# Queries: Q1 (single event), Q2 (nine-event exemplar) - same files as Phase 2.
import json
import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch

R3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(R3, "outputs")
LOG_DIR = os.path.join(R3, "logs")
R1_OUT = r"D:\Code\RhythmAlign\experiments\r1_golden_sample\outputs"
QUERIES = {"q1": os.path.join(R3, "queries", "query_q1.wav"),
           "q2": os.path.join(R3, "queries", "query_q2.wav")}


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | torch {torch.__version__}")

    sys_path = os.path.join(os.path.dirname(__file__))
    import sys
    sys.path.insert(0, sys_path)
    import clapsep_lib

    model = clapsep_lib.load_clapsep(device)
    if device.type == "cuda":
        print(f"after load: {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated")

    mixture, fs = librosa.load(os.path.join(R1_OUT, "golden_raw.wav"), sr=32000, mono=True)
    zeros = np.zeros((1, 512), dtype=np.float32)

    runs = []
    for qid, qpath in QUERIES.items():
        emb = clapsep_lib.embed_audio_query(model, qpath)
        print(f"{qid}: embedding shape {emb.shape} norm {np.linalg.norm(emb):.3f}")
        target = clapsep_lib.separate(model, mixture, emb, zeros, device)
        residual = (mixture - target).astype(np.float32)

        name = f"clapsep_{qid}"
        sf.write(os.path.join(OUT_DIR, f"{name}_target.wav"), target, fs, subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, f"{name}_residual.wav"), residual, fs, subtype="FLOAT")
        runs.append({"name": name, "query": qpath, "negative": "zeros",
                     "mixture_rms_db": db(mixture), "target_rms_db": db(target),
                     "residual_rms_db": db(residual),
                     "energy_ratio_db": db(target) - db(mixture),
                     "target_peak": float(np.max(np.abs(target)))})
        print(f"  {name}: target {db(target):.1f} dB vs mixture {db(mixture):.1f} dB "
              f"({db(target) - db(mixture):+.1f} dB) | residual {db(residual):.1f} dB")
        if device.type == "cuda":
            print(f"  cuda peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    with open(os.path.join(LOG_DIR, "04_clapsep_runs.json"), "w", encoding="utf-8") as f:
        json.dump({"torch": torch.__version__, "device": str(device),
                   "checkpoint": clapsep_lib.BEST_MODEL, "clap_checkpoint": clapsep_lib.CLAP_CKPT,
                   "protocol": "official Space app.py: 32 kHz mono, 10 s chunks, peak rescale 0.9, "
                               "output rescaled back to mixture domain, trimmed to 15 s",
                   "runs": runs}, f, indent=2)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
