# RTA1 step 01 - capture ingestion.
#
# Discovers media under work/capture_inbox/session_*/, probes each file
# (duration / sample rate / codec / native channels), hashes the ORIGINAL
# bytes, decodes an in-memory analysis copy for clipping statistics, and
# writes:
#   work/private/ingest_manifest.private.json   (real file names, hashes, notes)
#   work/private/CAPTURE_MANIFEST.public.json   (anonymized; tested by
#                                                tests/test_manifest_privacy.py)
#
# HARD RULES (protocol section: ingest):
#   - originals are NEVER transcoded, normalized, denoised, resampled in place,
#     or metadata-touched; the source file is opened read-only
#   - no downmix at ingestion: native channel count is recorded and preserved
#   - --dry-run lists what would happen and writes nothing
#
# Fixture mode (--fixture-dir) exists ONLY for software validation with
# synthetic files; such manifests are tagged "fixture": true and are not
# scientific evidence.
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import rta1_lib as rl  # noqa: E402


def discover(inbox):
    found = []
    if not os.path.isdir(inbox):
        return found
    for session in sorted(os.listdir(inbox)):
        sdir = os.path.join(inbox, session)
        if not os.path.isdir(sdir):
            continue
        for name in sorted(os.listdir(sdir)):
            p = os.path.join(sdir, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in rl.MEDIA_EXTS:
                found.append((session, name, p))
    return found


def ingest_one(session, name, path, fixture=False):
    entry = {
        "session": session,
        "original_filename": name,
        "sha256": rl.sha256_file(path),
    }
    probe = rl.probe_media(path)
    entry.update({
        "duration_s": round(probe.get("duration_s") or 0.0, 3),
        "sample_rate": probe.get("sample_rate"),
        "channels": probe.get("channels"),
        "codec": probe.get("codec"),
        "probe_source": probe.get("probe_source"),
        "size_bytes": probe.get("size_bytes"),
    })
    # clipping statistics from an in-memory decode of the ORIGINAL file
    x, sr_dec = rl.decode_to_float32(path)
    entry["clipping"] = rl.clipping_stats(x)
    entry["decoded_sample_rate"] = sr_dec
    entry["fixture"] = bool(fixture)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inbox", default=rl.INBOX)
    ap.add_argument("--fixture-dir", default=None,
                    help="ingest a synthetic validation directory instead of "
                         "the capture inbox (marks manifests fixture=true)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fixture = args.fixture_dir is not None
    root = args.fixture_dir or args.inbox
    files = discover(root)
    print(f"discovered {len(files)} media file(s) under {os.path.basename(root)}/")
    for session, name, path in files:
        print(f"  [{session}] {name}")

    if args.dry_run:
        print("DRY-RUN: no probing, no hashing, no manifests written")
        return 0
    if not files:
        print("nothing to ingest")
        return 0

    entries = []
    for session, name, path in files:
        print(f"ingesting [{session}] {name} ...", flush=True)
        entries.append(ingest_one(session, name, path, fixture=fixture))

    private_manifest = {
        "fixture": bool(fixture),
        "n_files": len(entries),
        "files": entries,
    }
    public_manifest = {
        "fixture": bool(fixture),
        "n_files": len(entries),
        "files": [rl.anonymize_take_entry(e) for e in entries],
    }
    out_dir = args.fixture_dir and os.path.join(root, "_manifests") or rl.PRIVATE
    rl.save_json(os.path.join(out_dir, "ingest_manifest.private.json"),
                 private_manifest)
    rl.save_json(os.path.join(out_dir, "CAPTURE_MANIFEST.public.json"),
                 public_manifest)
    findings = rl.scan_private_text(public_manifest)
    if findings:
        print(f"PRIVACY WARNING in public manifest: {findings}")
        return 2
    print(f"manifests written under {out_dir} "
          f"(private + anonymized public; privacy scan clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
