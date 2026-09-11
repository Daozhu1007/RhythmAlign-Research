"""S1a 92 — exact-convention receipt (RUN IN THE R2 TORCH VENV).

Compares the numpy reproduction (stft_clapsep.py) against the ACTUAL torchlibrosa
STFT/ISTFT modules that the local CLAPSep signal path uses, including the exact
magphase + mask + wav_reconstruct sequence from CLAPSep.py. Writes
logs/stft_equivalence_check.json.

Usage (from any cwd):
  D:/Code/RhythmAlign/experiments/r2_target_separation/.venv/Scripts/python.exe \
      92_stft_equivalence_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C


def main() -> None:
    import torch
    import torchlibrosa
    from torchlibrosa import ISTFT, STFT
    from torchlibrosa.stft import magphase

    st = STFT(n_fft=1024, hop_length=320, win_length=1024, window="hann",
              center=True, pad_mode="reflect", freeze_parameters=True)
    ist = ISTFT(n_fft=1024, hop_length=320, win_length=1024, window="hann",
                center=True, pad_mode="reflect", freeze_parameters=True)

    sys.path.insert(0, HERE)
    from stft_clapsep import ClapSepSTFT

    npst = ClapSepSTFT()
    rng = np.random.default_rng(7)
    results = {"torch": torch.__version__,
               "torchlibrosa": getattr(torchlibrosa, "__version__", "0.1.0"),
               "tolerances": {"stft_rel": 1e-4, "recon_rel": 1e-4},
               "cases": []}
    all_ok = True
    for L in (8000, 32000 * 5, 160000):
        x = (rng.standard_normal(L) * 0.1).astype(np.float32)
        with torch.no_grad():
            real, imag = st(torch.from_numpy(x)[None, :])
            mag, cos, sin = magphase(real, imag)
            mask = torch.sigmoid(torch.randn_like(mag))  # arbitrary bounded mask
            mag_y = torch.nn.functional.relu_(mag * mask)
            pred_t = ist(mag_y * cos, mag_y * sin, length=L).numpy().ravel()
            Xt = (real + 1j * imag).numpy()[0, 0]  # (F, K)
        Xn = npst.stft(x.astype(np.float64))
        Mn = 1.0 / (1.0 + np.exp(-rng.standard_normal(Xn.shape)))  # same role as sigmoid
        # apply the SAME torch mask to the numpy path for exact comparison
        Mn = mask.numpy()[0, 0]
        pred_n = npst.wav_reconstruct(Mn, Xn, L)
        stft_rel = float(np.max(np.abs(Xn - Xt)) / (np.max(np.abs(Xt)) + 1e-30))
        recon_rel = float(np.max(np.abs(pred_n - pred_t)) / (np.max(np.abs(pred_t)) + 1e-30))
        ok = stft_rel < 1e-4 and recon_rel < 1e-4
        all_ok &= ok
        results["cases"].append({"length": L, "stft_rel_err": stft_rel,
                                 "wav_reconstruct_rel_err": recon_rel, "ok": bool(ok)})
        print(f"L={L}: stft_rel={stft_rel:.3e} recon_rel={recon_rel:.3e} ok={ok}")

    results["all_ok"] = bool(all_ok)
    C.save_json(results, os.path.join(C.LOG_DIR, "stft_equivalence_check.json"))
    print("receipt -> logs/stft_equivalence_check.json")
    assert all_ok, "STOP: numpy reproduction does not match torchlibrosa"


if __name__ == "__main__":
    main()
