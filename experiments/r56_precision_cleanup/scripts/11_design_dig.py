# R5.6 Phase 1b - design dig: which features separate weak-TP from weak-FP,
# and can ANY feature rescue echo suppression? Feeds the single E1 recipe.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common56 as c


def load_aud(key):
    return c.load_json(os.path.join(c.R56_LOG, f"phase1_audit_{key}.json"))


def med(xs, k):
    v = [x[k] for x in xs if x.get(k) is not None]
    return round(float(np.median(v)), 3) if v else None


def qtab(groups, fields):
    out = {}
    for gname, xs in groups.items():
        out[gname] = {"n": len(xs)}
        for f in fields:
            out[gname][f] = med(xs, f)
    return out


FIELDS = ["score", "prom_frac_p995", "echo_ratio", "comb_peak", "comb_ratio",
          "burst_edge_ms", "prev_peak_dist_ms"]


def chan_n(x):
    return x["chan"]["n_ge_0.10"]


def main():
    auds = {k: load_aud(k) for k in ("devA", "devB", "devC")}
    allc = {k: auds[k]["cands"] for k in auds}

    # ---- 1. weak candidates: TP vs FP (devA/B trusted only) ----
    weak_tp, weak_fp = [], []
    for k in ("devA", "devB"):
        for x in allc[k]:
            if x["refined_confidence"] == "weak":
                (weak_tp if x["is_tp"] else weak_fp).append(dict(x, win=k))
    print("=== weak candidates (devA+B): TP", len(weak_tp), "FP", len(weak_fp))
    print(qtab({"weakTP": weak_tp, "weakFP": weak_fp},
               FIELDS + ["n_ge_0.10_dummy"]))
    for f in ("score", "prom_frac_p995", "comb_peak", "comb_ratio", "burst_edge_ms"):
        print(f"  {f}: weakTP med {med(weak_tp, f)} vs weakFP med {med(weak_fp, f)}")
    print("  chan n>=0.10: weakTP", np.median([chan_n(x) for x in weak_tp]),
          "weakFP", np.median([chan_n(x) for x in weak_fp]))
    inb_tp = sum(x["in_burst"] for x in weak_tp) / len(weak_tp)
    inb_fp = sum(x["in_burst"] for x in weak_fp) / len(weak_fp)
    print(f"  in_burst: weakTP {inb_tp:.2f} vs weakFP {inb_fp:.2f}")

    # ---- 2. echo FP vs adjacent TP on AUDIO features ----
    echo_fp, adj_tp = [], []
    for k in ("devA", "devB"):
        for x in allc[k]:
            if x["is_tp"] or (x["near_oracle_ms"] is not None and x["near_oracle_ms"] <= 150):
                (adj_tp if x["is_tp"] else echo_fp).append(x)
    print("\n=== echo FP (<=150ms of oracle) vs all TP (devA+B): echo", len(echo_fp), "TP", len(adj_tp))
    for f in ("comb_peak", "comb_ratio", "score", "prom_frac_p995"):
        print(f"  {f}: echoFP med {med(echo_fp, f)} vs TP med {med(adj_tp, f)}")
    ct_fp = {}
    for x in echo_fp:
        ct_fp[x["refined_confidence"]] = ct_fp.get(x["refined_confidence"], 0) + 1
    ct_tp = {}
    for x in adj_tp:
        ct_tp[x["refined_confidence"]] = ct_tp.get(x["refined_confidence"], 0) + 1
    print("  echoFP refined conf:", ct_fp, "| TP refined conf:", ct_tp)

    # ---- 3. E1 weak-gate variants: support = prominence OR multichan OR
    #         (burst AND near strong/present candidate) ----
    # near strong/present: within 250 ms of a candidate whose refinement is
    # strong/present (in the same window)
    near_sp = {}
    for k in allc:
        ts_sp = sorted(x["t"] for x in allc[k]
                       if x["refined_confidence"] in ("strong", "present"))
        near_sp[k] = ts_sp

    def near_strong_present(x, k, tol=0.250):
        ts = near_sp[k]
        import bisect
        i = bisect.bisect_left(ts, x["t"])
        cands = ts[max(i - 1, 0):i + 1]
        return any(abs(x["t"] - t) <= tol for t in cands)

    variants = {
        "V1 prom>=0.35": lambda x: x["prom_frac_p995"] >= 0.35,
        "V2 chan>=3": lambda x: chan_n(x) >= 3,
        "V3 burst+nearSP": lambda x: x["in_burst"] and near_strong_present(x, x["win"]),
        "V4 prom>=0.35 or chan>=3": lambda x: x["prom_frac_p995"] >= 0.35 or chan_n(x) >= 3,
        "V5 (prom>=0.35 or chan>=3) or (burst & nearSP)":
            lambda x: (x["prom_frac_p995"] >= 0.35 or chan_n(x) >= 3
                       or (x["in_burst"] and near_strong_present(x, x["win"]))),
        "V6 prom>=0.50 or chan>=3 or (burst & nearSP)":
            lambda x: (x["prom_frac_p995"] >= 0.50 or chan_n(x) >= 3
                       or (x["in_burst"] and near_strong_present(x, x["win"]))),
        "V7 (prom>=0.35 and in_burst) or chan>=3 or (burst & nearSP)":
            lambda x: ((x["prom_frac_p995"] >= 0.35 and x["in_burst"]) or chan_n(x) >= 3
                       or (x["in_burst"] and near_strong_present(x, x["win"]))),
    }
    print("\n=== weak-gate variants (devA+B weak only): keep-TP / keep-FP ===")
    for name, fn in variants.items():
        kt = sum(fn(x) for x in weak_tp)
        kf = sum(fn(x) for x in weak_fp)
        print(f"  {name}: keeps TP {kt}/{len(weak_tp)}  FP {kf}/{len(weak_fp)}  "
              f"(FP removed {len(weak_fp) - kf}, TP lost {len(weak_tp) - kt})")

    # ---- 4. devC effect of best-style variants (weak only, anchor = high-conf oracle) ----
    print("\n=== devC weak candidates kept / near-highconf-oracle among kept ===")
    hi_ts = [e["t"] for e in auds["devC"]["oracle"] if e["conf"] == "high"]
    hi_ts = np.array(hi_ts)
    for name, fn in list(variants.items())[4:7]:
        kept = [x for x in allc["devC"] if x["refined_confidence"] != "weak" or fn(x)]
        kw = [x for x in kept if x["refined_confidence"] == "weak"]
        n_near_hi = sum(bool(len(hi_ts) and np.min(np.abs(hi_ts - x["t"])) <= 0.050) for x in kw)
        print(f"  {name}: devC cands {len(kept)} (was {len(allc['devC'])}), "
              f"weak kept {len(kw)}, weak near high-conf oracle {n_near_hi}")

    # ---- 5. what do weak-TP have that no_transient-TP have? (devA/B) ----
    nt_tp = [x for x in allc["devA"] + allc["devB"]
             if x["refined_confidence"] == "no_transient" and x["is_tp"]]
    print("\n=== no_transient TPs (devA+B):", len(nt_tp), "===")
    for f in ("score", "prom_frac_p995"):
        print(f"  {f} med {med(nt_tp, f)}")
    print("  chan n med", np.median([chan_n(x) for x in nt_tp]) if nt_tp else None)

    # ---- 6. per-window weak gate effect preview (windows opened) ----
    print("\n=== E0 weak windows that would close under V5/V6/V7 (devA+B, weak-FP only) ===")
    for name in ("V5 (prom>=0.35 or chan>=3) or (burst & nearSP)",
                 "V6 prom>=0.50 or chan>=3 or (burst & nearSP)",
                 "V7 (prom>=0.35 and in_burst) or chan>=3 or (burst & nearSP)"):
        fn = variants[name]
        closed = sum(1 for k in ("devA", "devB") for x in allc[k]
                     if x["refined_confidence"] == "weak" and not x["is_tp"] and not fn(x))
        lost = sum(1 for k in ("devA", "devB") for x in allc[k]
                   if x["refined_confidence"] == "weak" and x["is_tp"] and not fn(x))
        print(f"  {name}: weak-FP windows closed {closed}, weak-TP windows lost {lost}")


if __name__ == "__main__":
    main()
