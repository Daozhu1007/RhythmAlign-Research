# RTA1 step 02 - native-channel diagnostic.
#
# Question: do the two native channels carry meaningful independent spatial
# information, or are they near-duplicates? MEASURED, never assumed.
#
# For each multi-channel take in the (fixture-or-real) private ingest manifest:
#   - inter-channel waveform correlation
#   - inter-channel level difference (dB)
#   - mid/side energy ratio
#   - frequency-dependent coherence summarized in bands (Welch/CSD)
# The result is a per-take verdict of INDEPENDENT / PARTIAL / DUPLICATE based
# on recorded thresholds, plus raw numbers. No claim about product usefulness
# follows automatically from a verdict.
import argparse
import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(__file__))
import rta1_lib as rl  # noqa: E402

BANDS_HZ = [(0, 200), (200, 1000), (1000, 4000), (4000, 10000), (10000, 20000)]
# verdict thresholds (recorded in the output, tunable ONLY on session-01
# material or declared before the real run - never on held-out data)
DUP_CORR = 0.995      # corr >= this and level diff <= 0.5 dB -> DUPLICATE
PARTIAL_CORR = 0.97


def coherence_bands(a, b, sr, nperseg=4096):
    f, C = signal.coherence(a, b, fs=sr, nperseg=nperseg)
    out = []
    for lo, hi in BANDS_HZ:
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            continue
        out.append({"band_hz": [lo, hi],
                    "coherence_mean": round(float(C[sel].mean()), 4)})
    return out


def diagnose(x, sr):
    """x: (n, ch) float. Diagnoses ch0 vs every other channel."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_ch = x.shape[1]
    res = {"native_channels": n_ch, "sample_rate": sr}
    if n_ch < 2:
        res["verdict"] = "MONO_ONLY"
        res["note"] = "single native channel; no spatial information question"
        return res
    pairs = []
    a = x[:, 0]
    for c in range(1, n_ch):
        b = x[:, c]
        n = min(len(a), len(b))
        av, bv = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
        na, nb = np.linalg.norm(av), np.linalg.norm(bv)
        corr = float(np.dot(av, bv) / (na * nb + 1e-30))
        ld_db = round(rl.rms_db(b[:n]) - rl.rms_db(a[:n]), 3)
        mid, side = (a[:n] + b[:n]) / 2.0, (a[:n] - b[:n]) / 2.0
        ms_ratio_db = round(rl.rms_db(side) - rl.rms_db(mid), 3)
        coh = coherence_bands(a[:n], b[:n], sr)
        if corr >= DUP_CORR and abs(ld_db) <= 0.5:
            verdict = "DUPLICATE"
        elif corr >= PARTIAL_CORR:
            verdict = "PARTIAL"
        else:
            verdict = "INDEPENDENT"
        pairs.append({
            "channels": [0, c], "waveform_corr": round(corr, 5),
            "level_diff_db": ld_db, "side_minus_mid_db": ms_ratio_db,
            "coherence_by_band": coh, "verdict": verdict,
        })
    res["channel_pairs"] = pairs
    res["verdict"] = "MULTI_" + "_".join(sorted({p["verdict"] for p in pairs}))
    res["note"] = ("verdict thresholds are recorded; stereo usefulness for the "
                   "product is NOT inferred automatically")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.fixture_dir
    if root:
        manifest_path = os.path.join(root, "_manifests",
                                     "ingest_manifest.private.json")
    else:
        manifest_path = os.path.join(rl.PRIVATE, "ingest_manifest.private.json")
    if not os.path.exists(manifest_path):
        print(f"no ingest manifest at {manifest_path} - run 01_ingest_capture first")
        return 1
    manifest = rl.load_json(manifest_path)
    inbox_root = root or rl.INBOX

    results = []
    for e in manifest["files"]:
        path = os.path.join(inbox_root, e["session"], e["original_filename"])
        if args.dry_run:
            print(f"DRY-RUN would decode+diagnose [{e['session']}] "
                  f"{e['original_filename']}")
            continue
        print(f"diagnosing [{e['session']}] {e['original_filename']} ...",
              flush=True)
        x, sr = rl.decode_to_float32(path)
        res = diagnose(x, sr)
        results.append({"session": e["session"],
                        "original_filename": e["original_filename"],
                        "sha256": e["sha256"], **res})
        print(f"  verdict: {res['verdict']}")

    if args.dry_run:
        return 0
    out = {"fixture": bool(root), "takes": results}
    out_dir = os.path.join(root, "_manifests") if root else rl.PRIVATE
    rl.save_json(os.path.join(out_dir, "channel_diagnostics.private.json"), out)
    public = {"fixture": bool(root), "takes": [
        {k: t[k] for k in ("session", "native_channels", "verdict")}
        for t in results]}
    findings = rl.scan_private_text(public)
    if findings:
        print(f"PRIVACY WARNING: {findings}")
        return 2
    rl.save_json(os.path.join(out_dir, "channel_diagnostics.public.json"), public)
    print("channel diagnostics written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
