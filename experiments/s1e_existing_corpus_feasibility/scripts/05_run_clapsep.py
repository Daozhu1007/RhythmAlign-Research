# S1E step 05 - CLAPSep on all challenge clips (official Space protocol via r3's
# clapsep_lib, used read-only). Primary condition: audio query Q1 (r3 single-event
# exemplar, fixed a priori), negative = zeros. Supplementary TEXT probe on the three
# nuisance clips with the R2 text prompt. Outputs stay in the input domain (no
# normalization; official 0.9-peak rescale is undone inside clapsep_lib.separate).
import os
import sys
import time
import json

import numpy as np
import librosa
import soundfile as sf
import torch

S1E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R3 = r"D:\Code\RhythmAlign\experiments\r3_audio_query"
sys.path.insert(0, os.path.join(R3, "scripts"))
import clapsep_lib  # r3, read-only

from s1e_common import CLIP_DIR, OUT_DIR, LOG_DIR, save_json

Q1 = os.path.join(R3, "queries", "query_q1.wav")
TEXT_PROMPT = "finger taps and button presses on a rhythm game cabinet"  # r2 p2, fixed
TEXT_PROBE_CLIPS = ["c1_speech_npc_a", "c2_speech_npc_b", "c3_taiko_prompt"]

manifest = json.load(open(os.path.join(CLIP_DIR, "clip_manifest.json"), encoding="utf-8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def main():
    t00 = time.time()
    model = clapsep_lib.load_clapsep(device)
    emb_q1 = clapsep_lib.embed_audio_query(model, Q1)
    print(f"Q1 embedding norm {np.linalg.norm(emb_q1):.3f}")
    emb_tx = model.clap_model.get_text_embedding([TEXT_PROMPT], use_tensor=False)
    emb_tx = np.asarray(emb_tx)
    print(f"text embedding norm {np.linalg.norm(emb_tx):.3f}")
    zeros = np.zeros((1, 512), dtype=np.float32)

    runs = []
    for clip in manifest["clips"]:
        cid = clip["id"]
        mix, fs = librosa.load(os.path.join(CLIP_DIR, "audio", f"{cid}_32k_mono.wav"),
                               sr=32000, mono=True)
        conds = [("aq", emb_q1)]
        if cid in TEXT_PROBE_CLIPS:
            conds.append(("tx", emb_tx))
        for tag, emb in conds:
            t0 = time.time()
            target = clapsep_lib.separate(model, mix, emb, zeros, device)
            name = f"clapsep_{tag}_{cid}"
            sf.write(os.path.join(OUT_DIR, "audio", f"{name}_target.wav"), target, fs,
                     subtype="FLOAT")
            sf.write(os.path.join(OUT_DIR, "audio", f"{name}_residual.wav"),
                     (mix - target).astype(np.float32), fs, subtype="FLOAT")
            runs.append({"run": name, "clip": cid, "condition": tag,
                         "query": Q1 if tag == "aq" else TEXT_PROMPT,
                         "negative": "zeros", "sr": 32000,
                         "mix_rms_db": db(mix), "target_rms_db": db(target),
                         "target_energy_ratio_db": db(target) - db(mix),
                         "runtime_s": time.time() - t0})
            print(f"{name}: ratio {db(target)-db(mix):+.1f} dB ({time.time()-t0:.0f}s)")
            torch.cuda.empty_cache()
    save_json(os.path.join(LOG_DIR, "05_clapsep_runs.json"),
              {"protocol": "official Space app.py via r3 clapsep_lib (32k mono, 10s chunks, "
                           "peak-0.9 rescale undone), negative=zeros",
               "q1_query": Q1, "text_prompt": TEXT_PROMPT, "runs": runs})
    print(f"done in {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
