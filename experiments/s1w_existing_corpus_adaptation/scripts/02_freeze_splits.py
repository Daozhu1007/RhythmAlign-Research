# S1W step 02 - FREEZE the recording-identity split (protocol order enforced:
# split freezing happens BEFORE any target-proxy / nuisance mining).
#
# Rule (declared a priori, seed 20260912):
#   SEALED_TEST_PRIMARY: 共感怪物AP (entire S1E source recording; excluded from everything)
#   TEST_GENERALIZATION (14): stratified diversity picks + seeded fill
#   DEV (7): stratified diversity picks from the remainder + seeded fill
#   TRAIN: everything else
# One identity = one provenance family (raw + all derived copies + byte-identical dups).
import os
import json
import random

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIVATE = os.path.join(S1W, "work", "private")
PUB = S1W
SEED = 20260912

SEALED_FAMILY = "共感怪物AP"
N_TEST_GEN = 14
N_DEV = 7


def load():
    recs = json.load(open(os.path.join(PRIVATE, "RECORDING_DIAGNOSTICS.private.json"), encoding="utf-8"))["recordings"]
    disc = json.load(open(os.path.join(PRIVATE, "CORPUS_DISCOVERY.private.json"), encoding="utf-8"))
    return recs, disc


def main():
    recs, disc = load()
    # stable IDs: sort usable identities by family name (deterministic across runs)
    usable = [r for r in recs if r["usable"]]
    assert len(usable) == 58
    usable.sort(key=lambda r: r["family"])
    ids = {r["family"]: f"recording_{i:04d}" for i, r in enumerate(usable, 1)}

    sealed = [r for r in usable if r["family"] == SEALED_FAMILY]
    assert len(sealed) == 1
    pool = [r for r in usable if r["family"] != SEALED_FAMILY]
    assert len(pool) == 57

    def top(pool, key, k, desc=True):
        s = sorted(pool, key=key, reverse=desc)
        take, seen_fam = [], set()
        for r in s:
            take.append(r)
            if len(take) >= k:
                break
        return take

    def folder_group(r):
        return r["rel_path"].split(os.sep)[0]

    picks = []
    def take_from(pool, cand):
        for c in cand:
            if c in pool:
                pool.remove(c)
                picks.append(c)
                return True
        return False

    # ---- TEST_GENERALIZATION stratified picks (in declared priority order) ----
    tpool = list(pool)
    tpick = []
    def take_t(cands):
        for c in cands:
            if c in tpool:
                tpool.remove(c)
                tpick.append(c)
                return True
        return False

    take_t(top(tpool, lambda r: r["diagnostics"]["speech_frac"], 3))          # speech-contaminated
    take_t(top(tpool, lambda r: r["diagnostics"]["flat_frac"], 3))            # slide/friction-heavy
    take_t(top(tpool, lambda r: r["diagnostics"]["onset_rate_per_s"], 2))     # dense interaction
    take_t(top(tpool, lambda r: r["diagnostics"]["floor_dbfs"], 1, desc=False))  # quietest floor
    take_t(top(tpool, lambda r: r["diagnostics"]["clip_frac"], 1))            # loudest/hottest capture
    take_t(top(tpool, lambda r: r["diagnostics"]["onset_rate_per_s"], 1, desc=False))  # sparsest interaction
    take_t(top(tpool, lambda r: -r["diagnostics"]["rms_dbfs"], 1, desc=False))  # weakest overall level
    # folder-group coverage fill (seeded order)
    rng = random.Random(SEED)
    groups = sorted({folder_group(r) for r in tpool})
    rng.shuffle(groups)
    for g in groups:
        if len(tpick) >= N_TEST_GEN:
            break
        members = [r for r in tpool if folder_group(r) == g]
        if members:
            take_t([rng.choice(members)])
    while len(tpick) < N_TEST_GEN and tpool:                                   # seeded remainder
        take_t([rng.choice(tpool)])
    assert len(tpick) == N_TEST_GEN

    # ---- DEV stratified picks from the remainder ----
    dpick = []
    def take_d(cands):
        for c in cands:
            if c in tpool:
                tpool.remove(c)
                dpick.append(c)
                return True
        return False

    take_d(top(tpool, lambda r: r["diagnostics"]["speech_frac"], 1))
    take_d(top(tpool, lambda r: r["diagnostics"]["flat_frac"], 1))
    take_d(top(tpool, lambda r: r["diagnostics"]["onset_rate_per_s"], 1))
    take_d(top(tpool, lambda r: r["diagnostics"]["floor_dbfs"], 1, desc=False))
    take_d(top(tpool, lambda r: r["diagnostics"]["clip_frac"], 1))
    take_d(top(tpool, lambda r: r["diagnostics"]["onset_rate_per_s"], 1, desc=False))
    if len(dpick) < N_DEV:
        take_d([rng.choice(tpool)])
    assert len(dpick) == N_DEV

    train = [r for r in pool if r not in tpick and r not in dpick]
    assert len(train) == 57 - N_TEST_GEN - N_DEV

    def entry(r, split):
        return {"recording_id": ids[r["family"]], "family": r["family"], "split": split,
                "rel_path": r["rel_path"], "abs_path": r["abs_path"],
                "duration_s": r["diagnostics"]["duration_s"], "work32k": r.get("work32k"),
                "diagnostics": r["diagnostics"]}

    split_doc = {
        "seed": SEED,
        "rule": "sealed S1E source; stratified diversity picks (speech/friction/dense/quiet/hot/sparse"
                " strata, declared order) + folder-coverage fill + seed fill; DEV re-stratified from remainder;"
                " identity = provenance family; one byte-identical duplicate copy excluded",
        "sealed_primary": [entry(r, "SEALED_TEST_PRIMARY") for r in sealed],
        "test_generalization": [entry(r, "TEST_GENERALIZATION") for r in tpick],
        "dev": [entry(r, "DEV") for r in dpick],
        "train": [entry(r, "TRAIN") for r in train],
        "excluded_duplicates": [
            {"rel_path": r"DATASET\海底谭DATASET\海底谭0.mp4", "family": "海底谭",
             "reason": "byte-identical duplicate (sha256 6f49020b…) of 已发/海底谭/海底谭.mp4"}],
    }
    with open(os.path.join(PRIVATE, "DATA_SPLIT.private.json"), "w", encoding="utf-8") as f:
        json.dump(split_doc, f, ensure_ascii=False, indent=1)

    # public: drop paths/diagnostics to safe metadata only
    def pub(e):
        return {"recording_id": e["recording_id"], "split": e["split"],
                "duration_s": e["duration_s"], "sample_rate": 48000, "channels": 2,
                "provenance_class": "ORIGINAL_RAW_HANDCAM"}
    pub_doc = {"seed": SEED, "unit": "recording identity (provenance family)",
               "counts": {"SEALED_TEST_PRIMARY": 1, "TEST_GENERALIZATION": N_TEST_GEN,
                          "DEV": N_DEV, "TRAIN": len(train)},
               "recordings": [pub(e) for e in
                              split_doc["sealed_primary"] + split_doc["test_generalization"]
                              + split_doc["dev"] + split_doc["train"]]}
    with open(os.path.join(PUB, "DATA_SPLIT.public.json"), "w", encoding="utf-8") as f:
        json.dump(pub_doc, f, ensure_ascii=False, indent=1)

    print("split frozen. TRAIN", len(train), "| DEV", N_DEV, "| TEST_GEN", N_TEST_GEN, "| SEALED 1")
    for name, lst in (("SEALED", split_doc["sealed_primary"]), ("TEST_GEN", split_doc["test_generalization"]),
                      ("DEV", split_doc["dev"])):
        print(f"\n{name}:")
        for e in lst:
            d = e["diagnostics"]
            print(f"  {e['recording_id']} {e['family']:<12} rms {d['rms_dbfs']:>6} speech {d['speech_frac']:.3f} "
                  f"onset/s {d['onset_rate_per_s']:.2f} flat {d['flat_frac']:.2f} clip {d['clip_frac']:.4f}")


if __name__ == "__main__":
    main()
