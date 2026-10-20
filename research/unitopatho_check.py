#!/usr/bin/env python3
"""Real other-hospital check: UniToPatho hyperplastic polyps (Univ. of Turin; Hamamatsu NanoZoomer S210,
20x, 0.4415 um/px; 800 um patches of 1812 px). All 545 patches are HP, so the share called HP is the measure.

Each patch is shrunk to MHIST's ~2.0 um/px (~400 px), cut into 224 px fields (448 um, one MHIST tile), and the
fields' scores are averaged. The source stain is measured from each patch alone (what the app
sees); the target is MHIST's averaged reference. Model: ConvNeXt-Base final model (relative comparison
of colour methods; the full panel is checked once it is final). Writes docs/results/unitopatho_check.json.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
import stain  # noqa: E402
from kfold_cv import ConvNextBaseFT  # noqa: E402

SRC = ROOT / "data/unitopatho/unitopath-public/800/HP"
MPP_SRC, MPP_MHIST = 0.4415, 2.0


def fields(a):
    """224 px fields covering the (shrunk) patch: 4 corners of the ~400 px image"""
    h, w = a.shape[:2]
    return [a[y:y + 224, x:x + 224] for y in (0, h - 224) for x in (0, w - 224)]


def main():
    ref = json.loads(stain.REF_PATH.read_text())
    mac_ref = (np.array(ref["stain_matrix"]), np.array(ref["max_conc"]))
    files = sorted(SRC.glob("*.png"))
    by_slide = defaultdict(list)
    for f in files:
        by_slide[re.split(r"\.ndpi", f.name)[0]].append(f)
    print(f"{len(files)} HP patches from {len(by_slide)} slides")
    size = round(1812 * MPP_SRC / MPP_MHIST)
    model = ConvNextBaseFT(-1)
    model.load_state_dict(torch.load(ROOT / "data/models/cv/convnext_base/final.pt")["state"], strict=False)
    model = model.cuda().eval()
    methods = ["none", "colour averaging", "Macenko", "Vahadane", "derived"]
    logit = {m: [] for m in methods}
    for s, fs in by_slide.items():
        imgs = [np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.LANCZOS)) for f in fs]
        # SOURCE stain measured from each image alone (what the app sees); TARGET = MHIST's averaged reference
        versions = {m: [] for m in ("none", "colour averaging", "Macenko", "Vahadane", "derived")}
        for a in imgs:
            versions["none"].append(a)
            mac = stain.estimate(a)
            vh = stain.estimate_vahadane(a)
            if mac is None or vh is None:  # almost no tissue: leave the image unchanged
                for m in ("colour averaging", "Macenko", "Vahadane", "derived"):
                    versions[m].append(a)
                continue
            vh_med, vh_p99 = stain.conc_stats(a, vh)
            versions["colour averaging"].append(stain.reinhard(a, stain.reinhard_stats(a), ref))
            versions["Macenko"].append(stain.normalize(a, mac, mac_ref))
            versions["Vahadane"].append(stain.vahadane(a, vh, vh_p99, ref))
            versions["derived"].append(stain.derived(a, vh, (vh_med, vh_p99), ref))
        with torch.no_grad():
            for m, ims in versions.items():
                for a in ims:
                    x = torch.stack([model.pre(Image.fromarray(np.ascontiguousarray(f))) for f in fields(a)]).cuda()
                    logit[m].append(float(model(x).float().mean().cpu()))
        print(f"  slide {s}: {len(fs)} patches done", flush=True)
    res = {m: {"called_HP": round(float(np.mean(np.array(v) < 0)), 4),
               "mean_P_SSL": round(float(np.mean(1 / (1 + np.exp(-np.array(v))))), 4)} for m, v in logit.items()}
    (ROOT / "docs/results/unitopatho_check.json").write_text(json.dumps(
        {"what": "UniToPatho 800um HP patches (all HP), shrunk to 2.0 um/px, ConvNeXt-Base final model, stain measured per patch",
         "n_patches": len(files), "n_slides": len(by_slide), "results": res}, indent=2))
    for m in methods:
        print(f"{m:17} called HP {res[m]['called_HP']:.1%}   mean P(SSL) {res[m]['mean_P_SSL']:.3f}")


if __name__ == "__main__":
    main()
