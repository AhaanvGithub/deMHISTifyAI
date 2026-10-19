#!/usr/bin/env python3
"""Two questions about training, answered on cached frozen-CONCH embeddings (CPU, minutes):

1. Class weighting: does weighting SSA vs HP differently in training (30/70 ... 70/30) beat
   training 50/50 and then simply moving the decision cut-off? Each weighting is a logistic head
   on soft labels (SSA votes / 7); its cut-off is either 0.5 or the one that maximises balanced
   accuracy on the validation slice. Everything is chosen on validation, scored once on test.
2. Learning curve: does test performance still climb when training on 25/50/75/100% of the
   training split? If it does, more data (e.g. HunCRC) should help; if it is flat, it won't.

These are linear heads on frozen features, not the fine-tuned CONCH the app uses; they show the
trend cheaply, before spending GPU time on the real model.
Writes docs/results/weight_sweep_learning_curve.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from select_prompts import load_labels  # noqa: E402
from train_judges import ece  # noqa: E402

OUT = ROOT / "docs" / "results" / "weight_sweep_learning_curve.json"


def fit(x, y_soft, ssa_share, C):
    """Logistic head on soft labels: each image appears as SSA with weight y and as HP with weight
    1-y; class weights put `ssa_share` of the total weight on SSA."""
    prev = y_soft.mean()
    w_ssa, w_hp = ssa_share / prev, (1 - ssa_share) / (1 - prev)
    X = np.vstack([x, x])
    Y = np.r_[np.ones(len(x)), np.zeros(len(x))]
    W = np.r_[y_soft * w_ssa, (1 - y_soft) * w_hp]
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(X, Y, sample_weight=W)
    return m


def score(p, y, votes, cut=0.5):
    conf = np.where(p > cut, p, 1 - p)
    split = (votes == 3) | (votes == 4)
    return {"bal_acc": round(float(balanced_accuracy_score(y, p > cut)), 4),
            "acc_vs_majority": round(float(((p > cut) == y).mean()), 4),
            "ssl_caught": round(float((p[y == 1] > cut).mean()), 4),
            "hp_caught": round(float((p[y == 0] <= cut).mean()), 4),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "ece": round(ece(p, y), 4),
            "spearman_vs_votes": round(float(spearmanr(p, votes)[0]), 4),
            "acc_on_4v3": round(float(((p > cut) == y)[split].mean()), 4)}


def best_cut(p, y):
    cuts = np.linspace(0.05, 0.95, 181)
    return float(cuts[np.argmax([balanced_accuracy_score(y, p > c) for c in cuts])])


def main():
    z = np.load(ROOT / "data/emb/mhist/conch.npz", allow_pickle=False)
    names = z["names"]
    x = z["img_emb"][list(z["views"]).index("orig")]
    lab = load_labels(ROOT / "data/mhist/annotations.csv").set_index("image").loc[names]
    votes = lab["ssa_votes"].to_numpy(float)
    y_soft, y = votes / 7, (lab["label"] == "SSA").to_numpy().astype(int)
    train_all = (lab["split"] == "train").to_numpy()
    ts = (lab["split"] == "test").to_numpy()
    va = train_all & (np.random.default_rng(0).random(len(lab)) < 0.2)  # same slice as the other scripts
    tr = train_all & ~va

    # regularisation chosen once, on validation, for the 50/50 head
    Cs = [0.03, 0.1, 0.3, 1, 3, 10]
    val_auc = {C: roc_auc_score(y[va], fit(x[tr], y_soft[tr], .5, C).predict_proba(x[va])[:, 1]) for C in Cs}
    C = max(val_auc, key=val_auc.get)
    print(f"C={C} (val AUC {val_auc[C]:.4f})", flush=True)

    sweep = {}
    for share in np.round(np.arange(0.30, 0.701, 0.05), 2):
        m = fit(x[tr], y_soft[tr], share, C)
        pv, pt = m.predict_proba(x[va])[:, 1], m.predict_proba(x[ts])[:, 1]
        cut = best_cut(pv, y[va])
        sweep[f"{int(share * 100)}/{int(round((1 - share) * 100))}"] = {
            "val_bal_acc_at_0.5": round(float(balanced_accuracy_score(y[va], pv > .5)), 4),
            "val_best_cut": round(cut, 3),
            "test_at_0.5": score(pt, y[ts], votes[ts]),
            "test_at_val_cut": score(pt, y[ts], votes[ts], cut)}
        r = sweep[list(sweep)[-1]]
        print(f"SSA:HP {list(sweep)[-1]:6}  test@0.5 bal {r['test_at_0.5']['bal_acc']:.3f} auc {r['test_at_0.5']['auc']:.3f} "
              f"ssl {r['test_at_0.5']['ssl_caught']:.2f} hp {r['test_at_0.5']['hp_caught']:.2f} | val cut {cut:.2f} -> test bal "
              f"{r['test_at_val_cut']['bal_acc']:.3f}", flush=True)
    pick = max(sweep, key=lambda k: sweep[k]["val_bal_acc_at_0.5"])

    curve = {}
    tr_idx = np.flatnonzero(tr)
    for frac in (0.25, 0.5, 0.75, 1.0):
        runs = []
        for rep in range(5 if frac < 1 else 1):
            sub = np.random.default_rng(100 + rep).choice(tr_idx, int(round(frac * len(tr_idx))), replace=False)
            pt = fit(x[sub], y_soft[sub], .5, C).predict_proba(x[ts])[:, 1]
            runs.append(score(pt, y[ts], votes[ts]))
        curve[str(frac)] = {"n_train": int(round(frac * len(tr_idx))),
                            **{k: round(float(np.mean([r[k] for r in runs])), 4) for k in ("auc", "bal_acc", "spearman_vs_votes", "ece")},
                            "auc_sd": round(float(np.std([r["auc"] for r in runs])), 4)}
        print(f"train {frac:.0%} (n={curve[str(frac)]['n_train']}): test AUC {curve[str(frac)]['auc']:.4f} "
              f"± {curve[str(frac)]['auc_sd']:.4f}, bal acc {curve[str(frac)]['bal_acc']:.3f}", flush=True)

    OUT.write_text(json.dumps({
        "what": "frozen CONCH (orig view) + logistic head on soft labels; chosen on validation, scored on MHIST test",
        "C": C, "class_weight_sweep": sweep, "picked_on_val_at_0.5": pick, "learning_curve": curve}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
