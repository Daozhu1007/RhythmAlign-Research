# S1W step 05b - mine DEV-recording target proxies (for DEV evaluation mixtures ONLY;
# never used in training). Same miner, DEV split, separate bank file.
import os
import sys
import json
import importlib.util

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIVATE = os.path.join(S1W, "work", "private")
PROXY_DIR = os.path.join(S1W, "work", "audio", "proxy_dev")

spec = importlib.util.spec_from_file_location("miner", os.path.join(S1W, "scripts", "05_mine_target_proxy.py"))
miner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(miner)


def main():
    from silero_vad import load_silero_vad
    import soundfile as sf
    vad = load_silero_vad().cuda().eval()
    split = json.load(open(os.path.join(PRIVATE, "DATA_SPLIT.private.json"), encoding="utf-8"))
    os.makedirs(PROXY_DIR, exist_ok=True)

    bank = []
    for e in split["dev"]:
        rid, wav = e["recording_id"], e["work32k"]
        res = miner.mine_recording(wav, vad)
        rec_dir = os.path.join(PROXY_DIR, rid)
        os.makedirs(rec_dir, exist_ok=True)
        x, _ = sf.read(wav, dtype="float32")
        for kind, evs in res["impacts"].items():
            for i, ev in enumerate(evs):
                t0 = ev["t"] - miner.PRE_S
                a, b = int(max(0, t0) * miner.SR), int(t0 * miner.SR + 1.5 * miner.SR)
                name = f"{kind}_{i:02d}_t{int(ev['t']*10):05d}.wav"
                sf.write(os.path.join(rec_dir, name), x[a:b], miner.SR, subtype="FLOAT")
                ev["file"] = os.path.join("work", "audio", "proxy_dev", rid, name)
                ev["seconds"] = round((b - a) / miner.SR, 3)
        for i, ev in enumerate(res["friction"]):
            a, b = int(ev["t"] * miner.SR), int((ev["t"] + ev["dur"]) * miner.SR)
            name = f"friction_{i:02d}_t{int(ev['t']*10):05d}.wav"
            sf.write(os.path.join(rec_dir, name), x[a:b], miner.SR, subtype="FLOAT")
            ev["file"] = os.path.join("work", "audio", "proxy_dev", rid, name)
            ev["seconds"] = round((b - a) / miner.SR, 3)
        bank.append({"recording_id": rid, **res})
        print(rid, e["family"], sum(len(v) for v in res["impacts"].values()), len(res["friction"]))

    with open(os.path.join(PRIVATE, "TARGET_PROXY_BANK_DEV.private.json"), "w", encoding="utf-8") as f:
        json.dump({"label": "TARGET_PROXY (DEV evaluation mixtures only; never used in training)",
                   "recordings": bank, "gates": "identical to 05_mine_target_proxy.py"},
                  f, ensure_ascii=False, indent=1)
    n = sum(sum(len(v) for v in b["impacts"].values()) + len(b["friction"]) for b in bank)
    s = sum(ev["seconds"] for b in bank for grp in list(b["impacts"].values()) + [b["friction"]] for ev in grp)
    print("DEV proxy events:", n, "seconds:", round(s, 1))


if __name__ == "__main__":
    main()
