# RTA1 step 07 - complex-ratio / phase-capable diagnostic oracle.
#
# G = S_p / (S_y + eps) is COMPLEX: it carries phase freedom, is NOT a bounded
# real mask, and is NOT deployable. Its only job is diagnosis: if it clearly
# succeeds where bounded real masks fail, the evidence implicates
# representation/phase limitation (DECISION_RULES Outcome B).
#
# --selftest validates the implementation on synthesized material (software
# validation only).
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import rta1_lib as rl  # noqa: E402
from importlib import import_module  # noqa: E402

bo = import_module("06_run_bounded_oracles")  # same dir; shares metric helpers

N_FFT = 1024
HOP = 256


def complex_ratio_mask(p, y, n_fft=N_FFT, hop=HOP, eps=1e-8):
    Sp = rl.stft(p, n_fft, hop)
    Sy = rl.stft(y, n_fft, hop)
    return Sp / (Sy + eps)


def evaluate(p, n, y, n_fft=N_FFT, hop=HOP):
    G = complex_ratio_mask(p, y, n_fft, hop)
    y_hat = rl.istft(G * rl.stft(y, n_fft, hop), n_fft, hop, length=len(p))
    y_hat = y_hat.numpy() if hasattr(y_hat, "numpy") else np.asarray(y_hat)
    n_hat = rl.istft(G * rl.stft(n, n_fft, hop), n_fft, hop, length=len(n))
    n_hat = n_hat.numpy() if hasattr(n_hat, "numpy") else np.asarray(n_hat)
    return {
        "oracle": "complex_ratio_diagnostic",
        "recon_snr_db": bo.snr_db(p, y_hat),
        "target_nmse_db": bo.nmse_db(p, y_hat),
        "nuisance_attenuation_db": bo.nuisance_attenuation_db(n, n_hat),
        "note": "unbounded complex mask; diagnostic only, never deployable",
    }


def selftest():
    rng = np.random.default_rng(20260915)
    sr = rl.SR
    t = np.arange(int(3.0 * sr)) / sr
    p = (0.3 * np.sin(2 * np.pi * 3000 * t)
         * (np.sin(2 * np.pi * 0.7 * t) > 0)).astype(np.float32)
    n = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    y = p + n
    row = evaluate(p, n, y)
    print(row)
    assert row["recon_snr_db"] > 20, "complex ratio should near-perfectly isolate"
    print("SELFTEST OK (tooling validation only)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, metavar="DIR")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.run:
        print("nothing to do: pass --selftest and/or --run DIR")
        return 1
    print("real runs require captured material + mixture recipes (04); "
          "see ORACLE_PROTOCOL.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
