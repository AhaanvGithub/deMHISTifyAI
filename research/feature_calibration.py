#!/usr/bin/env python3
"""Feature-score calibration for the literature judge (backend/config/calibration.json) - TRAINING images only.

For each WHO-2019 feature, CONCH compares the image embedding with a text description. Per feature, on the 2,175
MHIST training images (frozen CONCH embeddings, data/emb/mhist/conch.npz):
  mu, sd         mean and spread of the similarity (z-scoring in the app)
  auc_train      AUC of the similarity for SSL vs HP
  points_to      SSA if higher similarity goes with SSL, else HP
  reliability    |2 x (auc_train - 0.5)|
  usable         points the way the literature says AND reliability >= 0.1
Two text versions are compared: the single description (research/mhist_prompts.json) and the average of 5
phrasings (research/mhist_feature_variants.json, prompt ensembling). Decision rule, fixed in advance: use the
ensemble if it gives at least as many usable features and a higher mean reliability over the usable ones.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from models import load_conch  # noqa: E402
from select_prompts import load_labels  # noqa: E402

LITERATURE = {  # which class each feature supports in the WHO-2019 criteria / literature
    "serration_to_base": "SSA", "serration_upper_only": "HP", "dilated_crypt_base": "SSA", "boot_shaped_crypts": "SSA",
    "horizontal_growth": "SSA", "inverted_t_crypts": "SSA", "narrow_crypt_base": "HP", "mucin_at_base": "SSA",
    "goblet_cells": "HP", "microvesicular_mucin": "HP", "crypt_branching": "SSA", "architectural_distortion": "SSA",
    "normal_proliferative_zone": "HP", "dysplasia": "SSA"}


def calib(sims, y):
    auc = float(roc_auc_score(y, sims))
    out = {"mu": float(sims.mean()), "sd": float(sims.std() + 1e-6), "auc_train": round(auc, 3),
           "points_to": "SSA" if auc > 0.5 else "HP", "reliability": round(abs(2 * (auc - 0.5)), 3)}
    return out


def main():
    z = np.load(ROOT / "data/emb/mhist/conch.npz", allow_pickle=False)
    names = list(z["names"])
    lab = load_labels(ROOT / "data/mhist/annotations.csv").set_index("image")
    tr = [i for i, n in enumerate(names) if n in lab.index and lab.loc[n, "split"] == "train"]
    img = z["img_emb"][list(z["views"]).index("orig")][tr]
    y = (lab.loc[[names[i] for i in tr], "label"] == "SSA").to_numpy().astype(int)
    prompts = json.loads((ROOT / "research/mhist_prompts.json").read_text())
    variants = json.loads((ROOT / "research/mhist_feature_variants.json").read_text())
    conch = load_conch(device="cuda" if torch.cuda.is_available() else "cpu")

    def text(ts):
        e = np.asarray(conch.encode_texts(ts), dtype=np.float32)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        v = e.mean(0)
        return v / np.linalg.norm(v)

    results = {}
    for mode in ("single", "ensemble"):
        res = {}
        for k in [k for k in prompts if k.startswith("crit:")]:
            c = calib(img @ text([prompts[k]] if mode == "single" else variants[k]), y)
            lit = LITERATURE[k.split(":", 1)[1]]
            c.update({"literature_points_to": lit, "usable": c["points_to"] == lit and c["reliability"] >= 0.1})
            res[k] = c
        usable = [v["reliability"] for v in res.values() if v["usable"]]
        results[mode] = {"features": res, "n_usable": len(usable), "mean_reliability_usable": round(float(np.mean(usable)), 3) if usable else 0}
        print(f"{mode:8}: {len(usable)} usable, mean reliability {results[mode]['mean_reliability_usable']}")
        for k, v in res.items():
            print(f"   {k:34} auc {v['auc_train']:.3f} -> {v['points_to']:3} (lit {v['literature_points_to']:3}) rel {v['reliability']:.3f}{'  usable' if v['usable'] else ''}")
    s, e = results["single"], results["ensemble"]
    use = "ensemble" if (e["n_usable"] >= s["n_usable"] and e["mean_reliability_usable"] > s["mean_reliability_usable"]) else "single"
    out = {k: {**v, "text": "ensemble" if use == "ensemble" else "single"} for k, v in results[use]["features"].items()}
    (ROOT / "backend/config/calibration.json").write_text(json.dumps(out, indent=2))
    (ROOT / "docs/results/feature_calibration.json").write_text(json.dumps(
        {"chosen": use, "rule": "ensemble if >= usable features and higher mean reliability (training images)",
         "single": {k: v for k, v in s.items() if k != "features"}, "ensemble": {k: v for k, v in e.items() if k != "features"},
         "features": results}, indent=2))
    print(f"\nchosen: {use}; wrote backend/config/calibration.json (statistics from {len(tr)} training images only)")


if __name__ == "__main__":
    main()
