# S2A step 1: reference availability audit (protocol sections 8-9).
#
# Pairs ORIGINAL_RAW_HANDCAM recordings (S1W frozen split, read-only) with
# credible pristine song references found conservatively INSIDE the handcam
# corpus tree (owner-collected reference assets, S1W provenance class C).
# Explicitly forbidden sources are not used: no _synced product output, no
# product-repo artifacts, no song extracted from the handcam itself, no
# downloads.
#
# Output: work/private/REFERENCE_MAP.private.json (paths) + console summary.
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
import s2a_common as sc  # noqa: E402

SEP = chr(92)
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".aac")


def main():
    corpus = json.load(open(os.path.join(sc.S1W_PRIVATE, "CORPUS_MANIFEST.private.json"),
                            encoding="utf-8"))
    split = sc.load_split()
    by_split = {r["recording_id"]: r for r in split["recordings"]}

    # candidate pristine references: corpus class C audio (owner-collected song files)
    cands = [r for r in corpus["records"]
             if r["provenance_class"] == "C" and r["ext"].lower() in AUDIO_EXTS]

    excluded_dups = {e["rel_path"] for e in
                     json.load(open(os.path.join(sc.S1W_PRIVATE, "DATA_SPLIT.private.json"),
                                    encoding="utf-8"))["excluded_duplicates"]}

    def dir_segments(p):
        return os.path.dirname(p).split(SEP)

    def shared_k(a, b):
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        return k

    # stable anonymized ref ids: sort by path
    cands_sorted = sorted(cands, key=lambda r: r["rel_path"].lower())
    ref_id_of, song_of = {}, {}
    # group Override's two files into one song identity by (version, song folder)
    song_key_id, next_song, next_ref = {}, 1, 1
    for r in cands_sorted:
        rid = f"ref_{next_ref:02d}"
        next_ref += 1
        ref_id_of[r["rel_path"]] = rid
        skey = tuple(dir_segments(r["rel_path"]))  # version + song folder = song identity
        if skey not in song_key_id:
            song_key_id[skey] = f"song_{next_song:02d}"
            next_song += 1
        song_of[r["rel_path"]] = song_key_id[skey]

    pairs = []
    for e in sorted(split["recordings"], key=lambda r: r["recording_id"]):
        raw_rel = e["rel_path"]
        if raw_rel in excluded_dups:
            continue
        raw_dir = dir_segments(raw_rel)
        raw_stem = os.path.splitext(os.path.basename(raw_rel))[0]
        fam = e["family"].lower()
        best = None
        for c in cands_sorted:
            k = shared_k(raw_dir, dir_segments(c["rel_path"]))
            ref_stem = os.path.splitext(os.path.basename(c["rel_path"]))[0].lower()
            # folder-local reference (same song folder) -> strong candidate
            if k >= 2:
                score = (k, 2)
            # name-based: ref stem or ref parent folder matches the raw family name
            elif fam and len(fam) >= 2 and (fam in ref_stem or ref_stem in fam
                                            or fam in os.path.dirname(c["rel_path"]).lower().split(SEP)[-1]):
                score = (k, 1)
            else:
                continue
            if best is None or score > best[0]:
                best = (score, c)
        if best is None:
            continue
        c = best[1]
        pairs.append({
            "recording_id": e["recording_id"],
            "family": e["family"],
            "split": e["split"],
            "raw_rel_path": raw_rel,
            "raw_abs_path": e["abs_path"],
            "raw_work32k": e.get("work32k"),
            "duration_s": e["duration_s"],
            "ref_id": ref_id_of[c["rel_path"]],
            "song_id": song_of[c["rel_path"]],
            "ref_rel_path": c["rel_path"],
            "ref_abs_path": c["abs_path"],
            "ref_bytes": c["bytes"],
            "match_rule": "folder-local" if best[0][1] == 2 else "name-based",
            "provenance": "in-tree owner-collected pristine song reference (corpus class C asset, never a product output)",
        })

    songs_by_split = {}
    for p in pairs:
        songs_by_split.setdefault(p["split"], set()).add(p["song_id"])
    summary = {
        "n_candidate_pairs": len(pairs),
        "recordings_per_split": {k: sum(1 for p in pairs if p["split"] == k)
                                 for k in ("TRAIN", "DEV", "TEST_GENERALIZATION", "SEALED_TEST_PRIMARY")},
        "songs_per_split": {k: len(v) for k, v in songs_by_split.items()},
    }
    sc.save_json(os.path.join(sc.S2A_PRIVATE, "REFERENCE_MAP.private.json"),
                 {"pairs": pairs, "summary": summary, "candidates_total": len(cands_sorted)})
    print(json.dumps(summary, indent=1))
    for p in pairs:
        print(f'{p["split"]:20s} {p["recording_id"]} {p["family"]:16s} '
              f'{p["ref_id"]}/{p["song_id"]} {p["match_rule"]:12s} {p["ref_rel_path"]}')


if __name__ == "__main__":
    main()
