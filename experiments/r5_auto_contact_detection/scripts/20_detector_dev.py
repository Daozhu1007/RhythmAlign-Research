# R5 Phase 3 - automatic contact candidate detector: development tuning + eval.
# Detector code lives in r5_common (identical code path will run on the holdout).
# Tuning is deliberately limited: a handful of thresholds x one weight revision,
# judged on the dev oracle. No large grid search.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from r5_common import (R4_OUT, WORK, OUT, LOG, load_json, save_json,
                       detect_candidates, detector_score)

FPS = 60.04
MATCH_TOLS = (0.033, 0.050)


def load_dev():
    d = np.load(os.path.join(WORK, "dev_features.npz"))
    series = {k: d[k] for k in d.files if k != "fps"}
    return series


def oracle_events():
    o = load_json(os.path.join(R4_OUT, "contacts_oracle.json"))
    contacts = [e for e in o["events"]
                if e["type"] in ("press", "touch", "slide")
                and e.get("status") != "rejected_noncontact"]
    noncontacts = [e for e in o["events"]
                   if e["type"] in ("release", "rest", "none")
                   or e.get("status") == "rejected_noncontact"]
    return o, contacts, noncontacts


def eval_candidates(cands, contacts, regions, noncontacts=(), dur_s=15.0):
    tc = np.array([e["t_video"] for e in contacts])
    tp = np.array([c["t_video"] for c in cands])
    res = {"n_pred": len(cands), "n_oracle": len(tc)}
    for tol in MATCH_TOLS:
        matched_c, matched_p, errs = set(), set(), []
        for j, t in enumerate(tp):
            d = np.abs(tc - t)
            k = int(np.argmin(d))
            if d[k] <= tol:
                # one-to-one: each prediction claims its nearest oracle event
                if k in matched_c:
                    # allow the second-closest unmatched oracle within tol
                    order = np.argsort(d)
                    for k2 in order[:4]:
                        if d[k2] <= tol and k2 not in matched_c:
                            k = k2
                            break
                if k in matched_c:
                    continue
                matched_c.add(k)
                matched_p.add(j)
                errs.append(tp[j] - tc[k])
        n_hit = len(matched_c)
        prec = n_hit / max(len(tp), 1)
        rec = n_hit / len(tc)
        res[f"tol{int(tol*1000)}"] = {
            "match": n_hit, "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "err_ms_median": round(float(np.median(np.abs(errs)) * 1000), 1) if errs else None,
            "err_ms_p95": round(float(np.percentile(np.abs(errs), 95) * 1000), 1) if errs else None,
        }
    strong = [e for e in contacts if e["confidence"] == "high"]
    ts = np.array([e["t_video"] for e in strong])
    hit_s = sum(1 for t in ts if len(tp) and np.min(np.abs(tp - t)) <= 0.050)
    res["strong"] = {"n": len(ts), "recall_50ms": round(hit_s / len(ts), 4)}
    # unmatched predictions => FP list (with noncontact flag if near one)
    tnc = np.array([e["t_video"] for e in noncontacts]) if noncontacts else np.array([])
    fp = []
    for j, t in enumerate(tp):
        if not len(tp) or (np.min(np.abs(tc - t)) > 0.050):
            near_nc = bool(len(tnc) and np.min(np.abs(tnc - t)) <= 0.100)
            fp.append({"t": round(float(t), 3), "score": cands[j]["score"],
                       "near_noncontact": near_nc})
    res["false_positives"] = fp
    fn = [contacts[i]["id"] for i in range(len(tc))
          if len(tp) == 0 or np.min(np.abs(tp - tc[i])) > 0.050]
    res["false_negatives"] = fn
    # region coverage: holes > 0.3 s inside a contact region without any
    # prediction within 0.3 s; plus false-open duration outside all
    # contact spans (regions + events +/-0.15 s)
    contact_spans = []
    for e in contacts:
        contact_spans.append((e["t_video"] - 0.15, e["t_video"] + 0.15))
    for rg in regions:
        if not rg["id"].startswith("G"):
            contact_spans.append((rg["t0"], rg["t1"]))
    miss_tot, reg_tot = 0.0, 0.0
    per_region = {}
    for rg in regions:
        if rg["id"].startswith("G"):
            continue
        holes, t = 0.0, rg["t0"]
        while t < rg["t1"]:
            near = len(tp) and np.min(np.abs(tp - t)) <= 0.3
            if near:
                # skip ahead past this prediction's shelter zone
                nxt = tp[np.argmin(np.abs(tp - t))] + 0.3
                t = max(t + 1 / FPS, min(nxt, rg["t1"]))
            else:
                # accumulate hole until a prediction comes within 0.3 s
                t2 = t
                while t2 < rg["t1"] and not (len(tp) and np.min(np.abs(tp - t2)) <= 0.3):
                    t2 += 1 / FPS
                holes += min(t2, rg["t1"]) - t
                t = t2
        per_region[rg["id"]] = {"dur": round(rg["t1"] - rg["t0"], 2),
                                "missed_s": round(holes, 2),
                                "coverage": round(1 - holes / (rg["t1"] - rg["t0"]), 3)}
        miss_tot += holes
        reg_tot += rg["t1"] - rg["t0"]
    res["regions"] = per_region
    res["region_coverage_overall"] = round(1 - miss_tot / reg_tot, 4)
    # false-open duration: union of pred activity spans (t +/-0.15) minus contact spans
    act = []
    for t in tp:
        act.append((max(t - 0.15, 0), min(t + 0.15, dur_s)))
    act.sort()
    merged = []
    for s, e in act:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    contact_merged = []
    for s, e in sorted(contact_spans):
        if contact_merged and s <= contact_merged[-1][1]:
            contact_merged[-1][1] = max(contact_merged[-1][1], e)
        else:
            contact_merged.append([s, e])
    fo = 0.0
    for s, e in merged:
        cur = s
        for cs, ce in contact_merged:
            if ce <= cur:
                continue
            if cs >= e:
                break
            if cs > cur:
                fo += cs - cur          # uncovered gap before this contact span
            cur = max(cur, min(ce, e))
            if cur >= e:
                break
        if cur < e:
            fo += e - cur               # tail beyond the last covering span
    res["false_open_s"] = round(fo, 3)
    return res


def main():
    series = load_dev()
    oracle, contacts, noncontacts = oracle_events()
    print(f"dev: {len(contacts)} contacts, {len(noncontacts)} non-contacts")

    # ---- one reasoned weight revision: text evidence up, hand motion down ----
    rows = []
    for thr in [0.40, 0.45, 0.50]:
        for w in [(1.5, 1.2, 1.0, 0.35), (1.5, 1.5, 1.0, 0.3), (1.8, 1.5, 1.0, 0.3)]:
            cfg = {"threshold": thr, "slope_min": 0.15,
                   "w_jz_text": w[0], "w_jz_diff": w[1],
                   "w_glass_bright": w[2], "w_hand_diff": w[3]}
            score, cands = detect_candidates(series, FPS, cfg)
            r = eval_candidates(cands, contacts, oracle["regions"], noncontacts)
            t5 = r["tol50"]
            print(f"thr {thr:.2f} w{w}: pred {r['n_pred']:3d}  "
                  f"P {t5['precision']:.3f} R {t5['recall']:.3f} F1 {t5['f1']:.3f} | "
                  f"strong {r['strong']['recall_50ms']:.3f} | err med {t5['err_ms_median']} "
                  f"p95 {t5['err_ms_p95']} | FP {len(r['false_positives'])} "
                  f"| regcov {r['region_coverage_overall']}")
            rows.append({"cfg": cfg, "metrics": r})
    save_json(os.path.join(LOG, "dev_sweep_weights.json"), rows)


if __name__ == "__main__":
    main()
