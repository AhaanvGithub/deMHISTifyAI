"""Early look at the panel using ONLY out-of-fold predictions on the training images (the test split is
not loaded). Same group search as research/panel.py: every group of members, one learned weight per member,
scored by AUC (and balanced accuracy at the best training cut-off) on a second 5-fold split of train."""
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(r"C:\Users\Ahaan\Desktop\deMHISTifyAI")
sys.path.insert(0, str(ROOT / "research"))
from panel import balanced_lr  # noqa: E402
from select_prompts import load_labels  # noqa: E402

members = sys.argv[1:]
lab = load_labels(ROOT / "data/mhist/annotations.csv")
names = np.array(sorted(lab["image"]))
lab = lab.set_index("image").loc[names]
tr = np.flatnonzero((lab["split"] == "train").to_numpy())
y = (lab["label"] == "SSA").to_numpy().astype(int)[tr]
ys = lab["ssa_votes"].to_numpy(float)[tr] / 7
O = np.stack([np.load(ROOT / "data/models/cv" / m / "oof_logits.npy")[tr] for m in members], 1)
Oz = (O - O.mean(0)) / O.std(0)
outer = list(StratifiedKFold(5, shuffle=True, random_state=7).split(tr, y))
rows = []
for r in range(1, len(members) + 1):
    for g in itertools.combinations(range(len(members)), r):
        pred = np.zeros(len(tr))
        for i, j in outer:
            pred[j] = balanced_lr(Oz[i][:, g], ys[i]).decision_function(Oz[j][:, g])
        auc = roc_auc_score(y, pred)
        ba = max(balanced_accuracy_score(y, pred > c) for c in np.quantile(pred, np.linspace(0.3, 0.8, 101)))
        rows.append((auc, ba, " + ".join(members[k] for k in g)))
print(f"{'AUC':>7} {'bal acc':>8}  group  (training images only; test split not used)")
for auc, ba, name in sorted(rows, reverse=True):
    print(f"{auc:7.4f} {ba:8.4f}  {name}")
