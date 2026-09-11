"""S1a 01 — media inventory: formats, rates, channels, durations, hashes.

Usage:
    python 01_media_inventory.py [--root <dir>] [--out <json>]
Default root: ../raw. Fixture runs pass --root ../fixtures.
"""
import argparse
import os
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff", ".mp3", ".m4a", ".aac", ".ogg",
              ".opus", ".wma", ".amr", ".3gp", ".caf"}


def inventory_file(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    entry = {
        "path": os.path.relpath(path, C.S1A_DIR).replace("\\", "/"),
        "size_bytes": os.path.getsize(path),
        "sha256": C.sha256_file(path),
    }
    if ext in AUDIO_EXTS:
        try:
            info = sf.info(path)
            entry.update({
                "kind": "audio",
                "format": info.format,
                "subtype": info.subtype,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "duration_s": round(info.frames / info.samplerate, 6),
            })
        except Exception as e:  # unreadable/corrupt -> record, don't crash
            entry.update({"kind": "audio_unreadable", "error": str(e)})
    else:
        entry.update({"kind": "other", "note": "video/other media: inventory only; "
                                               "decode requires ffmpeg (not installed)"})
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.RAW_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            files.append(inventory_file(os.path.join(dirpath, fn)))
    out = {
        "root": os.path.relpath(root, C.S1A_DIR).replace("\\", "/"),
        "n_files": len(files),
        "n_audio": sum(1 for f in files if f.get("kind") == "audio"),
        "files": files,
    }
    out_path = args.out or os.path.join(C.MANIFEST_DIR, "media_inventory.json")
    C.save_json(out, out_path)
    C.log(f"inventory: {out['n_files']} files ({out['n_audio']} audio) -> {out_path}")


if __name__ == "__main__":
    main()
