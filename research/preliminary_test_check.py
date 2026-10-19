#!/usr/bin/env python3
"""Preliminary test-split check (2026-10-19, before the final line-up was settled), recorded for honesty.

For every group of cross-validated members: weights, calibration and the balanced-accuracy cut-off are fit
on the training images' out-of-fold predictions exactly as in research/panel.py, then applied to the test
logits averaged over the 5 fold models (the final models retrained on all of train come later). The member
line-up and the group used by the app are still chosen by research/panel.py's training-data rule; this
check does not feed into any choice. Writes docs/results/preliminary_test_check.json.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from panel import balanced_lr, metrics  # noqa: E402
from select_prompts import load_labels  # noqa: E402

members = sys.argv[1:]
lab = load_labels(ROOT / "data/mhist/annotations.csv")
names = np.array(sorted(lab["image"]))
lab = lab.set_index("image").loc[names]
y = (lab["label"] == "SSA").to_numpy().astype(int)
votes = lab["ssa_votes"].to_numpy(float)
tr = np.flatnonzero((lab["split"] == "train").to_numpy())
ts = np.flatnonzero((lab["split"] == "test").to_numpy())
O = np.stack([np.load(ROOT / "data/models/cv" / m / "oof_logits.npy")[tr] for m in members], 1)
T = np.stack([np.mean([np.load(ROOT / "data/models/cv" / m / f"fold{k}_test_logits.npy") for k in range(5)], 0)
              for m in members], 1)
mu, sd = O.mean(0), O.std(0)
Oz, Tz = (O - mu) / sd, (T - mu) / sd
ytr, ystr = y[tr], votes[tr] / 7
outer = list(StratifiedKFold(5, shuffle=True, random_state=7).split(tr, ytr))
cuts = np.linspace(0.2, 0.8, 121)

out = {}
for r in range(1, len(members) + 1):
    for g in itertools.combinations(range(len(members)), r):
        pred = np.zeros(len(tr))
        for i, j in outer:
            pred[j] = balanced_lr(Oz[i][:, g], ystr[i]).decision_function(Oz[j][:, g])
        stack, cal = balanced_lr(Oz[:, g], ystr), balanced_lr(pred[:, None], ystr)
        p_tr = cal.predict_proba(pred[:, None])[:, 1]
        cut = float(cuts[np.argmax([balanced_accuracy_score(ytr, p_tr > c) for c in cuts])])
        p_ts = cal.predict_proba(stack.decision_function(Tz[:, g])[:, None])[:, 1]
        out[" + ".join(members[k] for k in g)] = {"members": len(g), "cut_off": round(cut, 3), **metrics(p_ts, y[ts], votes[ts], cut)}
app = np.mean([np.load(f) for f in sorted((ROOT / "data/models").glob("conch_ft_seed*_testpred.npy"))], axis=0)
out["app today (4 partly fine-tuned CONCH)"] = {"members": 4, **metrics(app, y[ts], votes[ts])}
(ROOT / "docs/results/preliminary_test_check.json").write_text(json.dumps(out, indent=2))
print(f"{'bal acc':>7} {'AUC':>6} {'acc':>6} {'SSL':>5} {'HP':>5} {'ECE':>6} {'rho':>6}  group (TEST split, fold-model average)")
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["bal_acc"]):
    print(f"{v['bal_acc']:7.4f} {v['auc']:6.4f} {v['acc_vs_majority']:6.4f} {v['ssl_caught']:5.2f} {v['hp_caught']:5.2f} "
          f"{v['ece']:6.4f} {v['spearman_vs_votes']:6.3f}  {k}")
