# S1W step 00 - discover + provenance-classify the existing real handcam corpus.
# Read-only over the source tree. Private output only (absolute paths, device
# metadata); the sanitized public manifest is derived later in 02.
import os
import re
import json
import subprocess
import datetime

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_ROOT = r"D:\Daozh\Videos\舞萌手元"
PRIVATE = os.path.join(S1W, "work", "private")
LOG_DIR = os.path.join(S1W, "logs")

# S1E's primary test source (protocol section 12): sealed, entire recording.
SEALED_SOURCES = {
    r"13.2\共感觉\AP\共感怪物AP.mp4": "SEALED_TEST_PRIMARY (S1E primary source, owner-provided)",
}

# Derived markers per S1W protocol section 6 (context-confirmed product export suffix).
DERIVED_RE = re.compile(r"(_synced|-synced|\s synced|_aligned|-aligned|processed|rhythmalign)",
                        re.IGNORECASE)
AUDIO_MEDIA = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mp3", ".flac", ".wav", ".aac", ".m4a", ".ogg", ".opus"}
IMAGE_MEDIA = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_media(ff, path):
    """ffmpeg -i metadata parse. Device-ish metadata keys stay PRIVATE."""
    try:
        p = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=120)
    except Exception as e:
        return {"probe_ok": False, "probe_error": str(e)}
    err = p.stderr
    out = {"probe_ok": True}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
    if m:
        out["duration_s"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 3)
    m = re.search(r"bitrate:\s*(\d+)\s*kb/s", err)
    if m:
        out["container_bitrate_kbps"] = int(m.group(1))
    vcodec = re.findall(r"Stream #0:\d+.*?: Video: (\w+)", err)
    acodec = re.findall(r"Stream #0:\d+.*?: Audio: (\w+)", err)
    out["video_codec"] = vcodec[0] if vcodec else None
    out["n_video_streams"] = len(vcodec)
    out["audio_codec"] = acodec[0] if acodec else None
    m = re.search(r"Stream #0:\d+.*?: Audio:\s*[^,]+,\s*(\d+)\s*Hz", err)
    out["sample_rate"] = int(m.group(1)) if m else None
    m = re.search(r"Stream #0:\d+.*?: Audio: .*?,\s*(mono|stereo|5\.1|\d+)\s*,", err)
    ch_txt = m.group(1) if m else None
    out["channels_txt"] = ch_txt
    out["channels"] = {"mono": 1, "stereo": 2}.get(ch_txt, None)
    m = re.search(r"Stream #0:\d+.*?: Video:.*?, (\d+)x(\d+)", err)
    if m:
        out["resolution"] = [int(m.group(1)), int(m.group(2))]
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    out["fps"] = float(m.group(1)) if m else None
    # metadata block (private only)
    meta = {}
    in_meta = False
    for line in err.splitlines():
        if line.strip() == "Metadata:":
            in_meta = True
            continue
        if in_meta:
            mm = re.match(r"\s{4}([A-Za-z0-9_.\- ]+?)\s*:\s*(.*)$", line)
            if mm:
                meta[mm.group(1).strip()] = mm.group(2).strip()
            elif line.strip() and not line.startswith("    "):
                in_meta = False
    out["metadata"] = meta
    # provenance support flags from container metadata (phone capture vs ffmpeg export)
    md = out.get("metadata", {})
    out["provenance_support"] = {
        "phone_capture_metadata": bool(md.get("com.android.manufacturer") or md.get("com.android.model")),
        "ffmpeg_export_encoder": str(md.get("encoder", "")).startswith("Lavf"),
        "creation_time": md.get("creation_time"),
    }
    out["sha256"] = None
    out["metadata"] = meta
    return out


def family_key(stem):
    return DERIVED_RE.sub("", stem).strip()


def classify(rel, fname):
    ext = os.path.splitext(fname)[1].lower()
    stem = os.path.splitext(fname)[0]
    rec = {"ext": ext, "stem": stem}
    if ext in IMAGE_MEDIA:
        rec.update(provenance_class="C", provenance="IMAGE_OR_NON_AUDIO_ASSET",
                   reason=f"image/cover asset ({ext})")
    elif ext == ".mp3":
        rec.update(provenance_class="C", provenance="IMAGE_OR_NON_AUDIO_ASSET",
                   reason="pristine song reference audio (non-handcam; excluded from evidence)")
    elif ext in AUDIO_MEDIA:
        if DERIVED_RE.search(stem):
            rec.update(provenance_class="B", provenance="RHYTHMALIGN_DERIVED",
                       reason="filename carries product aligned-export marker",
                       family=family_key(stem))
        else:
            rec.update(provenance_class="A", provenance="ORIGINAL_RAW_HANDCAM",
                       reason="media recording without derived marker (candidate raw; verify plausibility)",
                       family=family_key(stem))
    else:
        rec.update(provenance_class="D", provenance="UNKNOWN", reason=f"unhandled extension {ext}")
    return rec


def main():
    ff = ffmpeg_exe()
    records = []
    for dirpath, dirnames, filenames in os.walk(CORPUS_ROOT):
        for fname in sorted(filenames):
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, CORPUS_ROOT)
            st = os.stat(full)
            rec = {"abs_path": full, "rel_path": rel, "file_name": fname,
                   "parent_dir": os.path.basename(dirpath),
                   "bytes": st.st_size,
                   "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
            rec.update(classify(rel, fname))
            if rec["rel_path"] in SEALED_SOURCES:
                rec["seal"] = SEALED_SOURCES[rec["rel_path"]]
            if rec["ext"] in AUDIO_MEDIA:
                rec.update(probe_media(ff, full))
            records.append(rec)
        dirnames.sort()

    # sibling derived relationships for raw candidates
    by_dir = {}
    for r in records:
        by_dir.setdefault(r["parent_dir"] + "::" + os.path.dirname(r["rel_path"]), []).append(r)
    for r in records:
        if r.get("provenance_class") in ("A", "B"):
            sibs = [s["file_name"] for s in records
                    if s["parent_dir"] == r["parent_dir"] and s is not r
                    and s.get("family") == r.get("family")]
            r["derived_siblings"] = [s for s in sibs if DERIVED_RE.search(os.path.splitext(s)[0])]
            r["raw_siblings"] = [s for s in sibs if not DERIVED_RE.search(os.path.splitext(s)[0])]

    os.makedirs(PRIVATE, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # full sha256 only where byte-identity is suspected (same-size files) — duplicate guard
    import hashlib
    from collections import defaultdict
    by_size = defaultdict(list)
    for r in records:
        if r.get("provenance_class") in ("A", "B"):
            by_size[r["bytes"]].append(r)
    for size, group in by_size.items():
        if len(group) > 1:
            for r in group:
                h = hashlib.sha256()
                with open(r["abs_path"], "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        h.update(chunk)
                r["sha256"] = h.hexdigest()

    out = {"corpus_root": CORPUS_ROOT, "discovered_utc": datetime.datetime.now().isoformat(),
           "n_files": len(records), "records": records}
    with open(os.path.join(PRIVATE, "CORPUS_DISCOVERY.private.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    from collections import Counter
    cnt = Counter(r["provenance"] for r in records)
    print("files:", len(records), dict(cnt))
    raws = [r for r in records if r.get("provenance_class") == "A"]
    print("raw candidate families:", sorted({r["family"] for r in raws}))
    bad_probe = [r["rel_path"] for r in records if r.get("probe_ok") is False]
    print("probe failures:", bad_probe)
    no_audio_stream = [r["rel_path"] for r in records
                       if r.get("provenance_class") in ("A", "B") and r.get("audio_codec") is None]
    print("audio-less media:", no_audio_stream)


if __name__ == "__main__":
    main()
