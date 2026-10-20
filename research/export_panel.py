#!/usr/bin/env python3
"""Export what the app needs to run the chosen panel (docs/results/panel.json) - TRAINING images only.

panel.json records the member weights and calibrations but not the standardisation statistics or the combiner's
intercept, so this recomputes the chosen group exactly as research/panel.py does (same data, same deterministic
fits) and checks the weights match. It also writes:
  - each member's own displayed-percentage calibration (so each judge can show its own %),
  - the panel's cross-fitted training scores on the displayed scale (the cut-off chart in the app),
  - the "unlike training data" reference: fine-tuned ConvNeXt-Base features of the 2,175 training images and the
    distance limit that 1% of them exceed (research note: flags 100% of junk images, 14% of UniToPatho HP).

Outputs: backend/config/panel_deploy.json, app/cutoff_data.json, data/models/cv/ood_ref_convnext_base.npz
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from panel import balanced_lr, natural_lr  # noqa: E402
from select_prompts import load_labels  # noqa: E402

P = json.loads((ROOT / "docs/results/panel.json").read_text())
members = P["chosen"].split(" + ")

lab = load_labels(ROOT / "data/mhist/annotations.csv")
names = np.array(sorted(lab["image"]))
lab = lab.set_index("image").loc[names]
y = (lab["label"] == "SSA").to_numpy().astype(int)
tr = np.flatnonzero((lab["split"] == "train").to_numpy())
ytr, ystr = y[tr], lab["ssa_votes"].to_numpy(float)[tr] / 7

# same order and statistics as panel.py: all members are standardised together, then the group's columns used
O_all = np.stack([np.load(ROOT / "data/models/cv" / t / "oof_logits.npy")[tr] for t in P["members"]], 1)
mu_all, sd_all = O_all.mean(0), O_all.std(0) + 1e-9
g = [P["members"].index(t) for t in members]
Oz = ((O_all - mu_all) / sd_all)[:, g]

stack = balanced_lr(Oz, ystr)
for t, w in zip(members, stack.coef_[0]):
    assert abs(w - P["weights"][t]) < 1e-6, f"weight mismatch for {t}: {w} vs {P['weights'][t]}"

# cross-fitted panel scores (panel.py step 1 for the chosen group) -> displayed scale, for the cut-off chart
pred = np.zeros(len(tr))
for i, j in StratifiedKFold(5, shuffle=True, random_state=7).split(tr, ytr):
    pred[j] = balanced_lr(Oz[i], ystr[i]).decision_function(Oz[j])
cs = P["calibration"]["shown"]
shown = 1 / (1 + np.exp(-(cs["coef"] * pred + cs["intercept"])))

per_member = {}
for k, t in enumerate(members):
    show_k = natural_lr(O_all[:, [P["members"].index(t)]], ytr)
    per_member[t] = {"mu": float(mu_all[g[k]]), "sd": float(sd_all[g[k]]), "weight": float(stack.coef_[0][k]),
                     "shown": {"coef": float(show_k.coef_[0][0]), "intercept": float(show_k.intercept_[0])}}

deploy = {"what": "chosen panel for the app; every number from MHIST training images only (test split unused)",
          "chosen": P["chosen"], "members": per_member, "intercept": float(stack.intercept_[0]),
          "calibration": P["calibration"], "cut_off": P["cut_off"], "shown_cut_off": P["shown_cut_off"],
          "train_oof": P["train_oof"], "test": P["TEST_chosen_panel"]}

# ---- "unlike training data" reference (fine-tuned ConvNeXt-Base features of the training images) ----
from kfold_cv import ConvNextBaseFT  # noqa: E402
m = ConvNextBaseFT(-1)
m.load_state_dict(torch.load(ROOT / "data/models/cv/convnext_base/final.pt", map_location="cpu")["state"], strict=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(dev).eval()
feats = []
with torch.no_grad():
    for b in range(0, len(tr), 64):
        x = torch.stack([m.pre(Image.open(ROOT / "data/mhist/images" / n).convert("RGB")) for n in names[tr[b:b + 64]]]).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = m.m(x)
        feats.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu().numpy())
F = np.concatenate(feats).astype(np.float32)
S = F @ F.T
np.fill_diagonal(S, -np.inf)
K = 5
d_train = 1 - np.sort(S, axis=1)[:, -K:].mean(1)
limit = float(np.quantile(d_train, 0.99))
np.savez_compressed(ROOT / "data/models/cv/ood_ref_convnext_base.npz", feats=F.astype(np.float16), limit=limit, k=K)
deploy["ood"] = {"features": "convnext_base (fine-tuned)", "k": K, "limit": round(limit, 5),
                 "rule": "1 - mean cosine similarity to the 5 nearest MHIST training images; flag above the 99th "
                         "percentile of the training images' own distances"}

(ROOT / "backend/config/panel_deploy.json").write_text(json.dumps(deploy, indent=2))
chart = {"model": "Panel (ConvNeXt-Base + MUSK)",
         "source": "cross-fitted panel scores on the 2,175 MHIST training images, displayed scale; test split not used",
         "default_cut": P["shown_cut_off"],
         "hp": [round(float(v), 4) for v in np.sort(shown[ytr == 0])],
         "ssl": [round(float(v), 4) for v in np.sort(shown[ytr == 1])]}
(ROOT / "app/cutoff_data.json").write_text(json.dumps(chart))  # for app/cutoff_preview.html
(ROOT / "app/cutoff_data.js").write_text(  # the app loads this copy (Electron pages can't fetch local JSON)
    "// generated by research/export_panel.py - panel training scores for the cut-off chart\n"
    f"window.CUTOFF_DATA = {json.dumps(chart)};\n")
print("weights match panel.json; intercept", round(float(stack.intercept_[0]), 4), "; OOD limit", round(limit, 4))
print("shown cut-off", P["shown_cut_off"], "; train balanced acc at it:",
      round(((shown[ytr == 1] > P["shown_cut_off"]).mean() + (shown[ytr == 0] <= P["shown_cut_off"]).mean()) / 2, 4))
