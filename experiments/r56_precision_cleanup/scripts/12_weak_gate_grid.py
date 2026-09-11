# R5.6 Phase 1c - grid over visual-support clauses for weak gating (devA+B only).
# Goal: a SIMPLE rule that closes many weak-FP windows while losing few weak-TP
# windows. Dev-C used only as a sanity anchor (high-confidence oracle).
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common56 as c


def load_aud(key):
    return c.load_json(os.path.join(c.R56_LOG, f"phase1_audit_{key}.json"))


def near_sp_factory(cands, tol):
    ts = sorted(x["t"] for x in cands
                if x["refined_confidence"] in ("strong", "present"))

    def f(x):
        import bisect
        i = bisect.bisect_left(ts, x["t"])
        return any(abs(x["t"] - t) <= tol for t in ts[max(i - 1, 0):i + 1])
    return f


def main():
    auds = {k: load_aud(k) for k in ("devA", "devB", "devC")}
    allc = {k: auds[k]["cands"] for k in auds}
    weak = {k: [dict(x, win=k) for x in allc[k] if x["refined_confidence"] == "weak"]
            for k in allc}
    nsp = {k: near_sp_factory(allc[k], 1.0) for k in allc}  # tol set inside

    hi_ts = np.array([e["t"] for e in auds["devC"]["oracle"] if e["conf"] == "high"])
    med_ts = np.array([e["t"] for e in auds["devC"]["oracle"] if e["conf"] == "med"])

    rows = []
    for tol, prom, chan, score_thr, edge in itertools.product(
            (0.120, 0.180, 0.250, 0.350), (0.20, 0.25, 0.30, 0.40),
            (2, 3), (0.60, 0.70, 10.0), (400.0, 700.0, 10 ** 9)):
        nsp = {k: near_sp_factory(allc[k], tol) for k in allc}

        def fn(x, prom=prom, chan=chan, score_thr=score_thr, edge=edge, nsp=nsp):
            sp = nsp[x["win"]]
            return (x["prom_frac_p995"] >= prom
                    or x["chan"]["n_ge_0.10"] >= chan
                    or x["score"] >= score_thr
                    or (x["in_burst"] and x["burst_edge_ms"] <= edge
                        and sp(x)))
        # devA+B effect
        kt = sum(fn(x) for x in weak["devA"] + weak["devB"] if x["is_tp"])
        kf = sum(fn(x) for x in weak["devA"] + weak["devB"] if not x["is_tp"])
        nt, nf = len(weak["devA"]) + len(weak["devB"]), 0
        ktp_all, kfp_all = kt, kf
        # devC: weak kept near high/med oracle (positives only; FP unknown)
        kc = sum(fn(dict(x, win="devC")) for x in weak["devC"])
        kc_hi = sum(fn(dict(x, win="devC")) and bool(
            len(hi_ts) and np.min(np.abs(hi_ts - x["t"])) <= 0.050) for x in weak["devC"])
        rows.append({"tol": tol, "prom": prom, "chan": chan, "score": score_thr,
                     "edge": edge, "keepTP": f"{kt}/23", "keepFP": f"{kf}/56",
                     "fp_closed": 56 - kf, "tp_lost": 23 - kt,
                     "devC_weak_kept": kc, "devC_kept_near_hi": kc_hi})

    # Pareto view: sort by (tp_lost asc, fp_closed desc)
    rows.sort(key=lambda r: (r["tp_lost"], -r["fp_closed"]))
    print("tol   prom chan score edge | keepTP  keepFP | fpClosed tpLost | devC weak kept (near hi)")
    for r in rows:
        if r["tp_lost"] <= 4 or (r["fp_closed"] >= 25 and r["tp_lost"] <= 8):
            print(f"{r['tol']:.3f} {r['prom']:.2f} {r['chan']:3d} {r['score']:.2f} "
                  f"{r['edge']:7.0f} | {r['keepTP']}  {r['keepFP']} | "
                  f"{r['fp_closed']:8d} {r['tp_lost']:6d} | {r['devC_weak_kept']:4d} ({r['devC_kept_near_hi']})")


if __name__ == "__main__":
    main()
