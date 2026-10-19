#!/usr/bin/env python3
"""Frozen-feature members for the panel: a logistic head (soft labels, 50/50 class weight) on each
cached embedding, cross-validated on the SAME 5 folds as research/kfold_cv.py.

For each backbone it writes data/models/cv/frozen_<name>/oof_logits.npy (out-of-fold logits on
train) and test_logits.npy (mean of the 5 fold heads), so research/panel.py can treat frozen and
fine-tuned members alike. Regularisation is chosen per fold on that fold's own training part
(inner 4-fold CV), so the out-of-fold predictions never see their own labels.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from select_prompts import load_labels  # noqa: E402

EMB = {"conch": "conch", "plip": "hfclip_vinid_plip", "quilt": "openclip_wisdomik_QuiltNet-B-32", "musk": "musk"}
CS = [0.001, 0.003, 0.01, 0.03, 0.1]


def fit(x, ys, C):
    prev = ys.mean()
    w = np.r_[ys * .5 / prev, (1 - ys) * .5 / (1 - prev)]
    return LogisticRegression(C=C, max_iter=5000).fit(np.vstack([x, x]), np.r_[np.ones(len(x)), np.zeros(len(x))],
                                                     sample_weight=w)


def main():
    lab = load_labels(ROOT / "data/mhist/annotations.csv")
    names = np.array(sorted(lab["image"]))
    lab = lab.set_index("image").loc[names]
    y = (lab["label"] == "SSA").to_numpy().astype(int)
    ys = lab["ssa_votes"].to_numpy(float) / 7
    tr_all = np.flatnonzero((lab["split"] == "train").to_numpy())
    ts = np.flatnonzero((lab["split"] == "test").to_numpy())
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr_all, y[tr_all]))  # = kfold_cv.py

    for name, f in EMB.items():
        path = ROOT / f"data/emb/mhist/{f}.npz"
        if not path.exists():
            print("skip", name, "(no embeddings)"); continue
        z = np.load(path)
        assert (z["names"] == names).all()
        x = z["img_emb"][list(z["views"]).index("orig")].astype(np.float64)
        oof, test, chosen = np.full(len(names), np.nan), [], []
        for fi_tr, fi_va in folds:
            tr, va = tr_all[fi_tr], tr_all[fi_va]
            sc = StandardScaler().fit(x[tr]); xs = sc.transform(x)
            inner = list(StratifiedKFold(4, shuffle=True, random_state=1).split(tr, y[tr]))
            score = {C: np.mean([roc_auc_score(y[tr[b]], fit(xs[tr[a]], ys[tr[a]], C).decision_function(xs[tr[b]]))
                                 for a, b in inner]) for C in CS}
            C = max(score, key=score.get); chosen.append(C)
            m = fit(xs[tr], ys[tr], C)
            oof[va] = m.decision_function(xs[va])
            test.append(m.decision_function(xs[ts]))
        out = ROOT / f"data/models/cv/frozen_{name}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "oof_logits.npy", oof)
        np.save(out / "test_logits.npy", np.mean(test, axis=0))
        auc = roc_auc_score(y[tr_all], oof[tr_all])
        (ROOT / "docs/results/cv").mkdir(parents=True, exist_ok=True)
        (ROOT / f"docs/results/cv/frozen_{name}.json").write_text(json.dumps({"oof_auc": round(float(auc), 4), "C_per_fold": chosen}))
        print(f"frozen {name:6} OOF AUC {auc:.4f}  (C per fold {chosen})", flush=True)


if __name__ == "__main__":
    main()
