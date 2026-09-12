# S1W shared mixture-example loading (exact stored components).
import os
import json
import numpy as np
import soundfile as sf
import torch

S1W = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIX_DIR = os.path.join(S1W, "work", "audio", "mixtures")
SR = 32000


def load_recipes():
    doc = json.load(open(os.path.join(S1W, "work", "private", "MIXTURES.private.json"), encoding="utf-8"))
    out = {"train": [], "dev": []}
    for r in doc["examples"]:
        out[r["split"]].append(r)
    return doc, out


class MixtureExample:
    def __init__(self, rec):
        self.rec = rec
        d = os.path.join(MIX_DIR, rec["split"])
        self.y = sf.read(os.path.join(d, rec["y_file"]), dtype="float32")[0]
        self.s = sf.read(os.path.join(d, rec["s_file"]), dtype="float32")[0] if "s_file" in rec else None
        self.n = sf.read(os.path.join(d, rec["n_file"]), dtype="float32")[0] if "n_file" in rec else None

    def spans(self):
        return [(e["t_start"], e["t_start"] + e["seconds"]) for e in self.rec.get("target", {}).get("events", [])]

    def tensors(self, device):
        y = torch.tensor(self.y, dtype=torch.float32, device=device).unsqueeze(0)
        s = torch.tensor(self.s, dtype=torch.float32, device=device).unsqueeze(0) if self.s is not None else None
        n = torch.tensor(self.n, dtype=torch.float32, device=device).unsqueeze(0) if self.n is not None else None
        return y, s, n


def collate_examples(split, device, limit=None):
    _, recipes = load_recipes()
    recs = recipes[split][:limit] if limit else recipes[split]
    return [MixtureExample(r) for r in recs]
