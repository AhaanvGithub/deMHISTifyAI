#!/usr/bin/env python3
"""Which stain normalisation (if any) should the app use? Training images only - the test split is not used.

Each ConvNeXt-Base fold model (research/kfold_cv.py, tag convnext_base) scores only its own held-out fold, so
all 2,175 training images are judged by a model that never trained on them. Conditions: the original images
and four simulated "other labs" (one fixed stain shift per lab, applied in haematoxylin/eosin space like a real
lab's staining, plus a scanner shift outside the two-dye model). Methods: none, colour averaging (Reinhard), Macenko, Vahadane, and the derived method, all
toward MHIST's own reference (backend/config/stain_mhist.json). The source stain is measured from each image
alone (what the app sees); the target is MHIST's averaged reference. A method should not hurt the original images and should recover accuracy on
the simulated labs. Writes docs/results/stain_compare.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
import stain  # noqa: E402
from kfold_cv import ConvNextBaseFT  # noqa: E402
from select_prompts import load_labels  # noqa: E402

# Simulated labs. Each applies ONE fixed change to every image of that "lab":
#  - dye amount: scale the haematoxylin/eosin amounts (stronger/weaker staining, longer/shorter dye baths)
#  - dye shade: rotate a dye's colour vector by ~10-15 degrees (typical H&E shade variation between labs)
#  - scanner: white balance + colour curve applied in RGB - outside the two-dye model, so it does not favour
#    the dye-based normalisers (a shift built inside their own model would be easy for them to undo)
LABS = {
    "original": None,
    "pinker": dict(amount=(0.9, 1.4), rotate={"E": 12}),
    "bluer, darker": dict(amount=(1.4, 0.85), rotate={"H": -12}),
    "faded": dict(amount=(0.6, 0.6)),
    "scanner shift": dict(scanner=dict(gain=(1.06, 0.97, 0.90), gamma=0.85)),
    "everything": dict(amount=(1.25, 1.2), rotate={"H": -10, "E": 10}, scanner=dict(gain=(1.04, 0.98, 0.93), gamma=0.9)),
}


def _rotate(v, deg):
    """rotate a stain colour vector (in optical-density RGB space) about the grey axis by `deg` degrees"""
    k = np.ones(3) / np.sqrt(3)
    t = np.deg2rad(deg)
    v = v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1 - np.cos(t))
    v = np.clip(v, 1e-3, None)
    return v / np.linalg.norm(v)


def restain(a, lab):
    from skimage.color import combine_stains, rgb_from_hed, separate_stains, hed_from_rgb
    if lab is None:
        return a
    out = a
    if "amount" in lab or "rotate" in lab:
        hed = separate_stains(a, hed_from_rgb)
        h_amt, e_amt = lab.get("amount", (1, 1))
        hed[..., 0] *= h_amt
        hed[..., 1] *= e_amt
        m = rgb_from_hed.copy()
        for name, deg in lab.get("rotate", {}).items():
            row = 0 if name == "H" else 1
            m[row] = _rotate(m[row], deg)
        out = (np.clip(combine_stains(hed, m), 0, 1) * 255).astype(np.uint8)
    if "scanner" in lab:
        sc = lab["scanner"]
        f = (out / 255.0) ** sc["gamma"] * np.array(sc["gain"])
        out = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    return out


def main():
    ref = json.loads(stain.REF_PATH.read_text())
    assert "vahadane_matrix" in ref, "run research/stain_reference.py first"
    mac_ref = (np.array(ref["stain_matrix"]), np.array(ref["max_conc"]))
    lab = load_labels(ROOT / "data/mhist/annotations.csv")
    names = np.array(sorted(lab["image"]))
    lab = lab.set_index("image").loc[names]
    y = (lab["label"] == "SSA").to_numpy().astype(int)
    tr_all = np.flatnonzero((lab["split"] == "train").to_numpy())
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr_all, y[tr_all]))  # = kfold_cv.py
    raw = {i: np.asarray(Image.open(ROOT / "data/mhist/images" / names[i]).convert("RGB")) for i in tr_all}

    methods = ["none", "colour averaging", "Macenko", "Vahadane", "derived"]
    scores = {(c, m): np.full(len(names), np.nan) for c in LABS for m in methods}
    t0 = time.time()
    for k, (_, fi_va) in enumerate(folds):
        va = tr_all[fi_va]
        model = ConvNextBaseFT(-1)
        model.load_state_dict(torch.load(ROOT / f"data/models/cv/convnext_base/fold{k}.pt")["state"], strict=False)
        model = model.cuda().eval()
        for cname, labp in LABS.items():
            imgs = [restain(raw[i], labp) for i in va]
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
                    out = []
                    for b in range(0, len(ims), 64):
                        x = torch.stack([model.pre(Image.fromarray(a)) for a in ims[b:b + 64]]).cuda()
                        out.append(model(x).float().cpu().numpy())
                    scores[(cname, m)][va] = np.concatenate(out)
            print(f"fold {k + 1}/5 {cname:20} done ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    res = {}
    yt = y[tr_all]
    for (c, m), s in scores.items():
        s = s[tr_all]
        res.setdefault(c, {})[m] = {"auc": round(float(roc_auc_score(yt, s)), 4),
                                    "bal_acc": round(float(balanced_accuracy_score(yt, s > 0)), 4)}
    (ROOT / "docs/results/stain_compare.json").write_text(json.dumps(
        {"what": "ConvNeXt-Base fold models on their own held-out training folds; simulated labs; test split unused",
         "results": res}, indent=2))
    print(f"\n{'condition':20} " + " ".join(f"{m:>17}" for m in methods) + "   (AUC / balanced acc)")
    for c in LABS:
        print(f"{c:20} " + " ".join(f"{res[c][m]['auc']:.3f} / {res[c][m]['bal_acc']:.3f}".rjust(17) for m in methods))


if __name__ == "__main__":
    main()
