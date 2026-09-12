# S1W step 12 - post-freeze GENERALIZATION test (protocol section 34).
# Zero-shot vs ADAPTED CLAPSep on a small set of representative windows from
# TEST_GENERALIZATION recordings. Descriptive fixed-gain diagnostics only; no
# retraining, no checkpoint choice from these results.
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

LOG = os.path.join(S1W, "logs")
OUT = os.path.join(S1W, "outputs", "audio")
WINDOWS = [(30.0, 40.0), (90.0, 100.0)]  # two song-active windows per recording


def band_stats(x32):
    S = np.abs(librosa.stft(x32, n_fft=1024, hop_length=320)) ** 2
    freqs = librosa.fft_frequencies(sr=32000, n_fft=1024)
    click = (freqs >= 2000) & (freqs <= 9000)
    p = S[click].mean(axis=0)
    f = np.exp(np.mean(np.log(S[(freqs >= 1000) & (freqs <= 6000)] + 1e-12), axis=0)) \
        / np.mean(S[(freqs >= 1000) & (freqs <= 6000)] + 1e-12, axis=0)
    return (float(20 * np.log10(np.sqrt(np.mean(p)) + 1e-12)),
            float(np.mean(f > 0.3)),
            float(20 * np.log10(np.sqrt(np.mean(x32 ** 2)) + 1e-12)))


def main():
    device = torch.device("cuda")
    best = os.path.join(S1W, "checkpoints", "best.ckpt")
    split = json.load(open(os.path.join(S1W, "work", "private", "DATA_SPLIT.private.json"),
                           encoding="utf-8"))
    rows = []
    zs_model = ad_model = emb = None
    for e in split["test_generalization"]:
        x, _ = sf.read(e["work32k"], dtype="float32")
        dur = len(x) / 32000
        for (w0, w1) in WINDOWS:
            if w1 > dur - 1:
                continue
            seg = x[int(w0 * 32000):int(w1 * 32000)]
            if zs_model is None:
                zs_model = ctl.load_model(device)
                ctl.set_trainable_scope(zs_model)
                emb = ctl.embed_q1(zs_model)
                ad_model = ctl.load_model(device)
                ctl.set_trainable_scope(ad_model)
                ck = torch.load(best, map_location=device, weights_only=False)
                ad_model.load_state_dict(ck["model_state"], strict=True)
            zeros = np.zeros((1, 512), np.float32)
            z = clapsep_lib.separate(zs_model, seg, emb, zeros, device)
            a = clapsep_lib.separate(ad_model, seg, emb, zeros, device)
            torch.cuda.empty_cache()
            row = {"recording_id": e["recording_id"], "window": [w0, w1]}
            for tag, y in (("raw", seg), ("zeroshot", z), ("adapted", a)):
                click_db, flat, rms = band_stats(y)
                row[tag] = {"click_band_db": round(click_db, 2),
                            "flat_frac": round(flat, 3), "rms_db": round(rms, 2)}
            for tag, y in (("zeroshot", z), ("adapted", a)):
                row[f"{tag}_ret_vs_raw_db"] = round(band_stats(y)[0] - band_stats(seg)[0], 2)
            rows.append(row)
            print(json.dumps(row), flush=True)
            sf.write(os.path.join(OUT, f"gen_{e['recording_id']}_{int(w0)}_adapted.wav"),
                     a, 32000, subtype="FLOAT")
    with open(os.path.join(LOG, "12_generalization.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "descriptive post-freeze generalization diagnostics; no ground truth",
                   "rows": rows}, f, indent=1)
    print("generalization test done:", len(rows), "windows")


if __name__ == "__main__":
    main()
