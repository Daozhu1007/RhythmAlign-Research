# RTA1 step 04 - controlled mixture builder (EVALUATION FIXTURES ONLY).
#
# Constructed mixtures y = p + g*n with exact bookkeeping. They are NOT
# ordinary real arcade mixtures and are never described as such.
#
# Modes:
#   --plan       write the pre-registered 24-mixture plan (8/session; strata
#                weak / strong_button / friction_tail / dense; nuisance levels
#                -10 dB and 0 dB) WITHOUT building audio. Legal at any time.
#   --build      build mixtures. REQUIRES: ingest manifest + truth-QC PASS for
#                the selected takes. Refuses otherwise. Never legal before the
#                real capture exists.
#   --fixture-dir  software validation with synthetic material; outputs are
#                tagged fixture=true and are NOT scientific evidence.
#
# Bookkeeping per mixture (mandatory): target take sha256 + region, nuisance
# take sha256 + region + shift, target/nuisance RMS, gain, sample rate,
# mono-derivation formula, output sha256.
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import rta1_lib as rl  # noqa: E402

STRATA = ("weak", "strong_button", "friction_tail", "dense")
NUISANCE_LEVELS_DB = (-10.0, 0.0)
MIX_S = 6.0
WORK_SR = 32000


def load_state(fixture_dir):
    manifest_dir = (os.path.join(fixture_dir, "_manifests")
                    if fixture_dir else rl.PRIVATE)
    manifest = rl.load_json(os.path.join(manifest_dir,
                                         "ingest_manifest.private.json"))
    qc_path = os.path.join(manifest_dir, "truth_qc.private.json")
    qc = rl.load_json(qc_path) if os.path.exists(qc_path) else None
    regions_path = os.path.join(manifest_dir, "scored_regions.private.json")
    regions = rl.load_json(regions_path) if os.path.exists(regions_path) else {}
    return manifest_dir, manifest, qc, regions


def takes_by_type(manifest, qc, root):
    """QC-passed take A (target) and B (nuisance) per session. Type comes from
    the session's CAPTURE_SESSION.json when present, else from sorted file
    order (first=A, second=B, third=C)."""
    verdicts = {}
    if qc:
        for r in qc["takes"]:
            verdicts[(r["session"], r["original_filename"])] = r["verdict"]
    out = {}
    for e in sorted(manifest["files"],
                    key=lambda x: (x["session"], x["original_filename"])):
        key = (e["session"], e["original_filename"])
        session_json = os.path.join(root, e["session"], "CAPTURE_SESSION.json")
        etype = None
        if os.path.exists(session_json):
            sj = rl.load_json(session_json)
            for t in sj.get("takes", []):
                if t.get("original_filename") == e["original_filename"]:
                    etype = t.get("type", "")
        if not etype:
            ordinal = len(out.get(e["session"], {}))
            etype = ("A_quiet_interaction", "B_playback_only",
                     "C_normal_gameplay")[min(ordinal, 2)]
        out.setdefault(e["session"], {})[etype] = (e, verdicts.get(key))
    return out


def plan(recipes_out):
    plan_rows = []
    for session in ("session_01", "session_02", "session_03"):
        panel = "tuning" if session == "session_01" else "heldout"
        for stratum in STRATA:
            for level_db in NUISANCE_LEVELS_DB:
                plan_rows.append({
                    "session": session, "panel": panel, "stratum": stratum,
                    "nuisance_level_db": level_db, "duration_s": MIX_S,
                    "target_region": "TBD_after_QC",
                    "nuisance_region": "TBD_after_QC",
                })
    rl.save_json(recipes_out, {"fixture": False, "built": False,
                               "plan": plan_rows})
    print(f"plan written: {len(plan_rows)} mixtures "
          f"(8/session x 3; 4 strata x 2 nuisance levels; "
          f"session_01=tuning, sessions 02+03=held-out)")


def pick_region(sr, n_samples, regions, key, stratum, rng_seed):
    """Deterministic ~6 s region. Prefers declared stratum regions; falls back
    to an even-coverage window and marks the recipe 'stratum_unverified'."""
    want = int(MIX_S * sr)
    entry = regions.get(key, {})
    spans = (entry.get("strata", {}) or {}).get(stratum) or []
    for t0, t1 in spans:
        if (t1 - t0) * sr >= want:
            return float(t0), float(t0) + MIX_S, "declared"
    if n_samples > want:
        t0 = int(rng_seed * 37 % max(1, (n_samples - want))) / sr
        return t0, t0 + MIX_S, "stratum_unverified"
    return 0.0, n_samples / sr, "stratum_unverified"


def build(manifest_dir, manifest, qc, regions, fixture_dir, out_root):
    assert qc is not None, "--build requires truth-QC results (03_truth_qc)"
    root = fixture_dir or rl.INBOX
    by_session = takes_by_type(manifest, qc, root)
    recipes, wav_rows = [], []
    skipped = {}
    for session in sorted(by_session):
        types = by_session[session]
        tgt = next((v for k, v in types.items()
                    if k.startswith("A_") and v[1] == "PASS"), None)
        nui = next((v for k, v in types.items()
                    if k.startswith("B_") and v[1] in ("PASS", "REVIEW")), None)
        if tgt is None or nui is None:
            skipped[session] = ("no QC-PASS take A" if tgt is None
                                else "no QC-passed take B")
            print(f"{session}: SKIPPED ({skipped[session]})")
            continue
        panel = "tuning" if session == "session_01" else "heldout"

        p_full, sr_p = rl.decode_to_float32(
            os.path.join(fixture_dir or rl.INBOX, session,
                         tgt[0]["original_filename"]), sr=WORK_SR)
        n_full, sr_n = rl.decode_to_float32(
            os.path.join(fixture_dir or rl.INBOX, session,
                         nui[0]["original_filename"]), sr=WORK_SR)
        assert sr_p == sr_n == WORK_SR, (sr_p, sr_n)
        p_full, n_full = rl.mono(p_full), rl.mono(n_full)

        for stratum in STRATA:
            for level_db in NUISANCE_LEVELS_DB:
                idx = len(recipes)
                t0, t1, tag = pick_region(WORK_SR, len(p_full), regions,
                                          f'{session}/{tgt[0]["original_filename"]}',
                                          stratum, idx)
                p = p_full[int(t0 * WORK_SR):int(t1 * WORK_SR)]
                shift = int((idx * 1.3) % max(1.0, len(n_full) / WORK_SR - MIX_S)
                            * WORK_SR)
                n_seg = n_full[shift:shift + len(p)]
                if len(n_seg) < len(p):
                    n_seg = np.pad(n_seg, (0, len(p) - len(n_seg)))
                g = (10.0 ** (level_db / 20.0)) * (rl.rms(p) / (rl.rms(n_seg) + 1e-12))
                y = p + g * n_seg
                name = f'{session}_{stratum}_{"m" if level_db < 0 else "0"}{abs(level_db):.0f}.wav'
                out_dir = os.path.join(out_root, panel)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, name)
                sf.write(out_path, y, WORK_SR, subtype="FLOAT")
                recipes.append({
                    "fixture": bool(fixture_dir), "panel": panel,
                    "mixture": name, "stratum": stratum,
                    "nuisance_level_db": level_db,
                    "target": {"sha256": tgt[0]["sha256"],
                               "region_s": [round(t0, 3), round(t1, 3)],
                               "region_tag": tag,
                               "rms": round(rl.rms(p), 6)},
                    "nuisance": {"sha256": nui[0]["sha256"],
                                 "region_s": [round(shift / WORK_SR, 3),
                                              round((shift + len(p)) / WORK_SR, 3)],
                                 "rms_pre_gain": round(rl.rms(n_seg), 6)},
                    "gain": round(float(g), 6),
                    "sample_rate": WORK_SR,
                    "alignment": "integer-sample shift; mono = mean(native ch)",
                    "output_sha256": rl.sha256_file(out_path),
                })
                wav_rows.append(name)
        print(f"{session}: 8 mixtures built ({panel})")

    rl.save_json(os.path.join(manifest_dir, "mixture_recipes.private.json"),
                 {"fixture": bool(fixture_dir), "built": True,
                  "skipped_sessions": skipped, "recipes": recipes})
    print(f"total: {len(recipes)} evaluation-fixture mixtures; recipes recorded"
          f"{' (FIXTURE - not scientific evidence)' if fixture_dir else ''}"
          + (f"; skipped: {skipped}" if skipped else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fixture-dir", default=None)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    if args.plan:
        manifest_dir = (os.path.join(args.fixture_dir, "_manifests")
                        if args.fixture_dir else rl.PRIVATE)
        plan(os.path.join(manifest_dir, "mixture_plan.json"))
        return 0
    if args.build:
        manifest_dir, manifest, qc, regions = load_state(args.fixture_dir)
        out_root = args.out_root or (
            os.path.join(args.fixture_dir, "_outputs")
            if args.fixture_dir else rl.ORACLE)
        build(manifest_dir, manifest, qc, regions, args.fixture_dir, out_root)
        return 0
    print("nothing to do: pass --plan and/or --build")
    return 1


if __name__ == "__main__":
    sys.exit(main())
