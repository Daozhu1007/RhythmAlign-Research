# Manifest privacy tests: the public anonymized manifest must not leak
# absolute paths, device/private metadata, or raw file identities beyond what
# the anonymization whitelist allows.
import os
import sys

RESEARCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(RESEARCH, "experiments",
                                "rta1_real_target_headroom_pilot", "scripts"))
import rta1_lib as rl  # noqa: E402


def _fake_private_manifest():
    return {
        "fixture": False,
        "n_files": 1,
        "files": [{
            "session": "session_01",
            "original_filename": "REC_20260915_142233.MOV",
            "sha256": "ab" * 32,
            "duration_s": 181.2,
            "sample_rate": 48000,
            "channels": 2,
            "codec": "aac",
            "probe_source": "ffprobe",
            "size_bytes": 123456789,
            "clipping": {"clip_frac_per_channel": [0.0, 0.0],
                         "clip_frac_max": 0.0, "peak_abs": 0.71},
            "decoded_sample_rate": 48000,
            "fixture": False,
            # private fields that must NEVER reach the public manifest:
            "absolute_path": "D:\\Daozh\\Videos\\raw\\REC_20260915_142233.MOV",
            "device_serial": "F2LX9Q8Z",
            "gps_latitude": 12.3456,
        }],
    }


def test_anonymized_entry_drops_private_fields():
    entry = _fake_private_manifest()["files"][0]
    pub = rl.anonymize_take_entry(entry)
    forbidden = ("original_filename", "absolute_path", "device_serial",
                 "gps_latitude")
    for key in forbidden:
        assert key not in pub, key
    assert pub["sha256"] == entry["sha256"]
    assert pub["duration_s"] == entry["duration_s"]


def test_private_scan_finds_what_it_should():
    priv = _fake_private_manifest()
    findings = rl.scan_private_text(priv)
    assert findings, "scanner must flag the absolute path"
    assert any("absolute_path" in f["field"] for f in findings)
    assert any("serial" in f["field"] or "latitude" in f["field"]
               for f in findings)


def test_public_manifest_scans_clean():
    priv = _fake_private_manifest()
    public = {"fixture": False, "n_files": 1,
              "files": [rl.anonymize_take_entry(e) for e in priv["files"]]}
    assert rl.scan_private_text(public) == [], rl.scan_private_text(public)


def test_scan_catches_drive_letter_in_any_field():
    doc = {"note": "see D:\\Secret\\dir\\file.wav"}
    assert rl.scan_private_text(doc), "drive-letter path must be flagged"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("test_manifest_privacy: ALL PASS")
