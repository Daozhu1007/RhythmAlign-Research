"""S1a 04 — sync map fitting from anchors.csv (see SYNC_PROTOCOL.md).

Usage: python 04_sync_fit.py --session <id> [--root <dir>]
Expects <root>/<session>/anchors.csv and working PCM from 02 (32 kHz).
Target channel = phone; every other audio channel gets t_phone = a + b*t_aux.
"""
import argparse
import csv
import os
import sys

import common as C
import sync_fit as SF

AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif"}


def find_channel_wav(workroot: str, channel: str) -> str | None:
    """Match by file basename OR parent directory name (raw/ layout uses both)."""
    for dirpath, _d, files in os.walk(workroot):
        dir_match = os.path.basename(dirpath).lower() == channel.lower()
        for fn in files:
            b = os.path.splitext(fn)[0].lower()
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS and \
                    (b == channel.lower() or (dir_match and b.startswith(channel.lower()))):
                return os.path.join(dirpath, fn)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tolerance-s", type=float, default=0.002)
    args = ap.parse_args()
    root = os.path.abspath(args.root) if args.root else C.RAW_DIR
    tag = os.path.basename(root)
    sess = root if tag == args.session else os.path.join(root, args.session)
    anchors_csv = os.path.join(sess, "anchors.csv")
    inner = "" if tag == args.session else args.session
    workroot = os.path.join(C.S1A_DIR, "work", f"pcm_{tag}", "32k", inner)
    if not os.path.isdir(workroot):
        raise SystemExit(f"working PCM not found at {workroot}; run 02_extract_pcm.py "
                         f"--root {root} first")

    rows = list(csv.DictReader(open(anchors_csv, encoding="utf-8")))
    events = {}
    for r in rows:
        events.setdefault(r["channel"], {})[r["event"]] = \
            (float(r["approx_time_s"]) if r["approx_time_s"] else None)

    target = "phone"
    target_wav = find_channel_wav(workroot, target)
    assert target_wav, f"target channel '{target}' not found under {workroot}"
    xt, srt = C.read_wav(target_wav)
    xt = C.to_mono(xt)

    result = {"session": args.session,
              "target_channel": target,
              "map_convention": "t_phone = a + b * t_aux",
              "tolerance_s": args.tolerance_s,
              "channels": {}}
    ok_all = True
    for channel in sorted(events):
        if channel == target:
            continue
        wav = find_channel_wav(workroot, channel)
        if wav is None:
            result["channels"][channel] = {"status": "missing (optional channel)"}
            continue
        xa, sra = C.read_wav(wav)
        assert sra == srt, "02_extract_pcm must produce uniform 32 kHz working PCM"
        xa = C.to_mono(xa)
        pairs, details = [], {}
        for ev, t_hint in sorted(events[channel].items()):
            if t_hint is None:
                continue
            at = SF.find_onset(xa, srt, t_hint)
            pt = SF.find_onset(xt, srt, t_hint)
            details[ev] = {"aux": at, "target": pt}
            if at and pt:
                pairs.append((at["t"], pt["t"]))
        if len(pairs) < 2:
            result["channels"][channel] = {"status": "insufficient anchors",
                                           "anchor_details": details}
            ok_all = False
            continue
        m = SF.fit_clock_map(pairs)
        v = SF.sync_verdict(m, args.tolerance_s)
        result["channels"][channel] = {"status": "fitted", "map": m,
                                       "verdict": v, "anchor_details": details}
        ok_all &= v["sync_ok"]

    result["session_sync_ok"] = bool(ok_all)
    out = os.path.join(C.MANIFEST_DIR, f"sync_map_{args.session}.json")
    C.save_json(result, out)
    C.log(f"sync map for {args.session} -> {out}; ok={ok_all}")
    for ch, rec in result["channels"].items():
        if rec.get("status") == "fitted":
            m = rec["map"]
            C.log(f"  {ch}: a={m['a_s']*1e3:.3f} ms, drift={m['drift_ppm']:.1f} ppm, "
                  f"max_resid={m['max_abs_residual_s']*1e3:.3f} ms, "
                  f"sync_ok={rec['verdict']['sync_ok']}")
        else:
            C.log(f"  {ch}: {rec['status']}")


if __name__ == "__main__":
    main()
