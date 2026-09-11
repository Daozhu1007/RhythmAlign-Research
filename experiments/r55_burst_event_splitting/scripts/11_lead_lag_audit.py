# R5.5 Phase 1b - resolve the -116 ms locked-average lead:
# (a) per-contact transient presence/offset per channel, TP vs FN populations,
# (b) fully-silent contacts (no channel transient), (c) oracle audio side:
# how many dense contacts have no audio transient either (likely oracle-side
# over-split, not detector failure). Dev-B audio comb is reused frozen.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import signal as sig

from common55 import (R55_OUT, R55_LOG, R5_WORK, load_windows, save_json,
                      load_json, detect_candidates, R5_FROZEN_CFG, FPS,
                      match_one_to_one)

CHANNELS = [("jz_text", "jz_warm", "jz_white"),
            ("jz_diff", "jz_diff", None),
            ("glass_bright", "glass_bright", None),
            ("hand_diff", "hand_diff", None)]
WIN = 12          # +/- frames searched for a transient (200 ms)
PROM_FRAC = 0.03  # local prominence >= 3% of channel p99.5
VALLEY = 9


def local_prom(x, pk):
    out = np.zeros(len(pk))
    for i, p in enumerate(pk):
        a, b = max(p - VALLEY, 0), min(p + VALLEY + 1, len(x))
        left = np.min(x[a:p + 1]) if p > a else x[p]
        right = np.min(x[p:b]) if b > p + 1 else x[p]
        out[i] = x[p] - max(left, right)
    return out


def main():
    windows = load_windows()
    report = {}
    for w in windows:
        score, cands = detect_candidates(w.series, FPS, R5_FROZEN_CFG)
        tp_t = np.array([c["t_video"] for c in cands])
        tc = w.tc()
        mc, mp, pairs = match_one_to_one(tp_t, tc, 0.050)
        tp_of = {k: j for j, k, _ in pairs}

        ch = {n: w.series[a] + (w.series[b] if b else 0.0) for n, a, b in CHANNELS}
        scale = {n: float(np.percentile(x, 99.5)) + 1e-9 for n, x in ch.items()}

        # per-contact nearest-peak offset per channel
        rows = []
        for i, e in enumerate(w.contacts):
            if not e["dense"]:
                continue
            k = int(round(e["t_video"] * FPS))
            row = {"id": e["id"], "t": e["t_video"],
                   "is_tp": i in mc, "conf": e.get("confidence"), "type": e["type"],
                   "chan": {}}
            n_transient = 0
            for name, x in ch.items():
                pk, _ = sig.find_peaks(x)
                if len(pk) == 0:
                    continue
                prom = local_prom(x, pk)
                good = pk[prom / scale[name] >= PROM_FRAC]
                if len(good) == 0:
                    continue
                d = good - k
                j = int(np.argmin(np.abs(d)))
                off_ms = d[j] / FPS * 1000
                within = abs(d[j]) <= WIN
                row["chan"][name] = {"off_ms": round(off_ms, 1),
                                     "within": bool(within),
                                     "h": round(float(x[good[j]] / scale[name]), 2)}
                n_transient += int(within)
            row["n_chan_transient"] = n_transient
            row["silent"] = n_transient == 0
            rows.append(row)

        n = len(rows)
        silent = [r for r in rows if r["silent"]]
        tp_rows = [r for r in rows if r["is_tp"]]
        fn_rows = [r for r in rows if not r["is_tp"]]
        for name in ch:
            offs = [abs(r["chan"][name]["off_ms"]) for r in rows
                    if name in r["chan"] and r["chan"][name]["within"]]
            report.setdefault(w.key, {})[name] = {
                "contacts_with_transient_within200ms": len(offs),
                "frac_of_dense": round(len(offs) / n, 3),
                "offset_ms_median": round(float(np.median(offs)), 1) if offs else None}
        # locked avg split TP vs FN for jz_diff
        HALF = 18
        lock = {}
        for tag, subset in (("TP", tp_rows), ("FN", fn_rows)):
            acc = []
            for r in subset:
                k = int(round(r["t"] * FPS))
                if k - HALF >= 0 and k + HALF < w.n:
                    v = ch["jz_diff"][k - HALF:k + HALF + 1] / scale["jz_diff"]
                    acc.append(v)
            lock[tag] = (np.mean(acc, axis=0).tolist() if acc else None,
                         len(acc))
        # audio-side silence (devB only: frozen comb from R5 work)
        audio_silent = None
        if w.key == "devB":
            comb = np.load(os.path.join(R5_WORK, "holdout_comb.npy"))
            sr = 48000
            no_tr = 0
            for e in w.contacts:
                if not e["dense"]:
                    continue
                t0 = int(e["t_video"] * sr)
                seg = comb[max(t0 - int(0.04 * sr), 0):t0 + int(0.04 * sr)]
                if len(seg) == 0:
                    continue
                a, b = max(t0 - int(0.25 * sr), 0), min(t0 + int(0.25 * sr), len(comb))
                excl = np.ones(b - a, bool)
                lo = max(t0 - int(0.04 * sr), 0) - a
                excl[max(lo, 0):lo + len(seg)] = False
                med = np.median(comb[a:b][excl]) if excl.any() else 0
                if seg.max() / (med + 1e-6) < 1.6:
                    no_tr += 1
            audio_silent = no_tr

        report[w.key].update({
            "n_dense": n,
            "silent_contacts": [r["id"] for r in silent],
            "n_silent": len(silent),
            "silent_frac": round(len(silent) / n, 3),
            "n_silent_conf_low": sum(1 for r in silent if r["conf"] == "low"),
            "tp_vs_fn_jzdiff_locked": lock,
            "audio_silent_dense_contacts": audio_silent,
        })
        print(f"\n=== {w.key} ({n} dense contacts)")
        for name in ch:
            r = report[w.key][name]
            print(f"  {name}: transient within 200ms for {r['frac_of_dense']:.0%} "
                  f"of dense contacts | median |offset| {r['offset_ms_median']} ms")
        print(f"  fully silent (no channel transient): {len(silent)}/{n} "
              f"({report[w.key]['silent_frac']:.0%}), of which conf=low: "
              f"{report[w.key]['n_silent_conf_low']}")
        print(f"  silent ids: {report[w.key]['silent_contacts']}")
        if audio_silent is not None:
            print(f"  audio-side no-transient dense contacts: {audio_silent}/{n}")
        for tag, (prof, cnt) in lock.items():
            if prof:
                p = np.array(prof)
                print(f"  jz_diff locked [{tag}] n={cnt}: at -116ms {p[np.argmin(np.abs(np.linspace(-300,300,len(p)))-(-116))]:.2f}, "
                      f"at 0 {p[len(p)//2]:.2f}, peak {p.max():.2f} @ {np.linspace(-300,300,len(p))[np.argmax(p)]:.0f} ms")
    save_json(os.path.join(R55_LOG, "phase1b_lead_lag.json"), report)


if __name__ == "__main__":
    main()
