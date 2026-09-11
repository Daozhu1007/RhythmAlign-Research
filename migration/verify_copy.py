"""Verify the RhythmAlign research migration copy and write MIGRATION_MANIFEST.json.

Re-walks source and destination with the same exclusions used for the copy
(dirs named .venv, __pycache__, .pytest_cache), compares file sets, sizes and
SHA-256 digests, then writes the manifest. Exits non-zero on any mismatch.
"""
import hashlib
import json
import os
import sys
import time

SRC_ROOT = r"D:\Code\RhythmAlign\experiments"
DST_ROOT = r"D:\Code\RhythmAlign-Research\experiments"
IGNORED_LIST = r"D:\Code\RhythmAlign-Research\migration\.git_ignored_list.txt"
MANIFEST = r"D:\Code\RhythmAlign-Research\migration\MIGRATION_MANIFEST.json"
EXCLUDE_DIRS = {".venv", "__pycache__", ".pytest_cache"}

AUDIO_VIDEO_EXTS = {".wav", ".mp4", ".flac", ".npy", ".npz", ".jpg", ".jpeg", ".png"}
MODEL_EXTS = {".pt", ".ckpt", ".pth", ".onnx", ".safetensors", ".gz", ".bin"}
TEXT_EXTS = {".py", ".md", ".json", ".yaml", ".yml", ".csv", ".txt", ".ipynb", ".log", ".gitignore"}


def category_for(rel, ext):
    if "/third_party/" in "/" + rel or ext in MODEL_EXTS:
        return "D_model_checkpoint_third_party"
    if ext in AUDIO_VIDEO_EXTS:
        return "C_research_audio_video_outputs"
    if ext in TEXT_EXTS:
        return "B_research_source_scripts_reports"
    return "F_unknown"


def walk(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            out[rel] = os.path.getsize(full)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    src = walk(SRC_ROOT)
    dst = walk(DST_ROOT)
    with open(IGNORED_LIST, encoding="utf-8") as f:
        ignored = {line.strip()[len("experiments/"):] for line in f if line.strip()}

    missing_in_dst = sorted(set(src) - set(dst))
    extra_in_dst = sorted(set(dst) - set(src))
    size_mismatch = sorted(r for r in set(src) & set(dst) if src[r] != dst[r])

    errors = []
    if missing_in_dst:
        errors.append(f"{len(missing_in_dst)} files missing in destination")
    if extra_in_dst:
        errors.append(f"{len(extra_in_dst)} unexpected files in destination")
    if size_mismatch:
        errors.append(f"{len(size_mismatch)} size mismatches")

    files = []
    hash_mismatch = []
    for i, rel in enumerate(sorted(src)):
        s_path = os.path.join(SRC_ROOT, rel.replace("/", os.sep))
        d_path = os.path.join(DST_ROOT, rel.replace("/", os.sep))
        s_hash = sha256(s_path)
        if rel not in dst or src[rel] != dst[rel]:
            continue  # already recorded as an error; skip hashing dest
        d_hash = sha256(d_path)
        if s_hash != d_hash:
            hash_mismatch.append(rel)
        ext = os.path.splitext(rel)[1].lower()
        status = "untracked_ignored" if rel in ignored else "untracked"
        files.append({
            "source": s_path,
            "destination": d_path,
            "size": src[rel],
            "sha256": d_hash,
            "category": category_for(rel, ext),
            "git_status": status,
            "action": "copied (robocopy /E /COPY:DAT; .venv, __pycache__, .pytest_cache excluded)",
        })
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(src)}", flush=True)

    if hash_mismatch:
        errors.append(f"{len(hash_mismatch)} SHA-256 mismatches")

    manifest = {
        "migration_date": time.strftime("%Y-%m-%d"),
        "product_repo": "D:\\Code\\RhythmAlign",
        "research_workspace": "D:\\Code\\RhythmAlign-Research",
        "source_root": SRC_ROOT,
        "destination_root": DST_ROOT,
        "excluded_dir_names": sorted(EXCLUDE_DIRS),
        "copy_method": "robocopy /E /COPY:DAT /DCOPY:DAT (timestamps preserved)",
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files),
        "category_counts": {},
        "git_status_counts": {},
        "verification": {
            "file_sets_match": not missing_in_dst and not extra_in_dst,
            "sizes_match": not size_mismatch,
            "sha256_match": not hash_mismatch,
            "errors": errors,
        },
        "files": files,
    }
    for f in files:
        manifest["category_counts"][f["category"]] = manifest["category_counts"].get(f["category"], 0) + 1
        manifest["git_status_counts"][f["git_status"]] = manifest["git_status_counts"].get(f["git_status"], 0) + 1

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"files={len(files)} bytes={manifest['total_bytes']} elapsed={time.time() - t0:.0f}s")
    print("categories:", json.dumps(manifest["category_counts"]))
    print("git_status:", json.dumps(manifest["git_status_counts"]))
    if errors:
        print("VERIFICATION: FAIL")
        for e in errors:
            print(" -", e)
        if missing_in_dst[:10]:
            print(" missing_in_dst sample:", missing_in_dst[:10])
        if hash_mismatch[:10]:
            print(" hash_mismatch sample:", hash_mismatch[:10])
        sys.exit(1)
    print("VERIFICATION: PASS")


if __name__ == "__main__":
    main()
