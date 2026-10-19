#!/usr/bin/env python3
"""Choose the final panel from the cross-validated members using ONLY the training images, then score
it once on the MHIST test split.

Members: data/models/cv/<tag>/ for each --members tag, with out-of-fold logits on train
(oof_logits.npy, research/kfold_cv.py; all members share the same 5 folds) and test logits: the final
model retrained on all of train (final_test_logits.npy, the version the app will run) if it exists,
else the mean of the 5 fold models.

1. Every group of members (all 2^n - 1) gets one learned weight per member: a logistic regression on
   the members' standardised out-of-fold logits (soft labels, SSA and HP equal total weight). Each
   group is scored by AUC with a second 5-fold split over the training images, so a group's weights
   never see the images it is scored on.
2. Luck check: among groups within --tolerance AUC of the best, keep the one with the fewest members
   (the simpler, faster panel); the per-fold wins are reported.
3. Calibrate the chosen group's score (Platt scaling on its out-of-fold scores, equal class weight)
   and set the SSL/HP cut-off that maximises balanced accuracy on the training images.
4. Only then score the test split: the chosen panel, plus each member alone and the app's current
   models for context. Writes docs/results/panel.json.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from select_prompts import load_labels  # noqa: E402
from train_judges import ece  # noqa: E402

OUT = ROOT / "docs" / "results" / "panel.json"


def balanced_lr(X, ys):
    """logistic regression on soft labels, SSA and HP carrying equal total weight"""
    hard = ys > .5
    w_ssa = (~hard).sum() / max(hard.sum(), 1)
    w = np.r_[ys * w_ssa, 1 - ys]
    return LogisticRegression(C=1.0, max_iter=5000).fit(np.vstack([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))],
                                                        sample_weight=w)


def metrics(p, y, votes, cut=0.5):
    split = (votes == 3) | (votes == 4)
    return {"bal_acc": round(float(balanced_accuracy_score(y, p > cut)), 4), "auc": round(float(roc_auc_score(y, p)), 4),
            "acc_vs_majority": round(float(((p > cut) == y).mean()), 4),
            "ssl_caught": round(float((p[y == 1] > cut).mean()), 4), "hp_caught": round(float((p[y == 0] <= cut).mean()), 4),
            "ece": round(ece(p, y), 4), "spearman_vs_votes": round(float(spearmanr(p, votes)[0]), 4),
            "acc_on_4v3": round(float(((p > cut) == y)[split].mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="tags under data/models/cv/")
    ap.add_argument("--tolerance", type=float, default=0.003, help="AUC margin within which the smaller group wins")
    a = ap.parse_args()
    hold = ROOT / "data/models/cv/HOLD_PANEL"
    if hold.exists():  # the member line-up is still being decided; the test split must be scored only once
        sys.exit(f"panel on hold: {hold.read_text().strip()}")

    lab = load_labels(ROOT / "data/mhist/annotations.csv")
    names = np.array(sorted(lab["image"]))
    lab = lab.set_index("image").loc[names]
    y = (lab["label"] == "SSA").to_numpy().astype(int)
    votes = lab["ssa_votes"].to_numpy(float)
    tr = np.flatnonzero((lab["split"] == "train").to_numpy())
    ts = np.flatnonzero((lab["split"] == "test").to_numpy())
    ytr, ystr = y[tr], votes[tr] / 7

    O, T, source = [], [], {}
    for tag in a.members:
        d = ROOT / "data/models/cv" / tag
        oof = np.load(d / "oof_logits.npy")[tr]
        assert not np.isnan(oof).any(), f"{tag}: cross-validation not finished"
        if (d / "final_test_logits.npy").exists():
            t, source[tag] = np.load(d / "final_test_logits.npy"), "final model (all of train)"
        else:
            t, source[tag] = np.mean([np.load(d / f"fold{k}_test_logits.npy") for k in range(5)], axis=0), "mean of 5 fold models"
        O.append(oof); T.append(t)
    O, T = np.stack(O, 1), np.stack(T, 1)
    mu, sd = O.mean(0), O.std(0) + 1e-9
    Oz, Tz = (O - mu) / sd, (T - mu) / sd  # standardised with training statistics only

    # 1. every group, scored with a second split over the training images
    outer = list(StratifiedKFold(5, shuffle=True, random_state=7).split(tr, ytr))
    groups = [g for r in range(1, len(a.members) + 1) for g in itertools.combinations(range(len(a.members)), r)]
    res_groups, cv_pred = {}, {}
    for g in groups:
        pred, fold_auc = np.zeros(len(tr)), []
        for i, j in outer:
            m = balanced_lr(Oz[i][:, g], ystr[i])
            pred[j] = m.decision_function(Oz[j][:, g])
            fold_auc.append(roc_auc_score(ytr[j], pred[j]))
        name = " + ".join(a.members[k] for k in g)
        res_groups[name] = {"members": len(g), "cv_auc": round(float(roc_auc_score(ytr, pred)), 4),
                            "fold_aucs": [round(float(x), 4) for x in fold_auc]}
        cv_pred[name] = (g, pred)
    ranked = sorted(res_groups, key=lambda n: -res_groups[n]["cv_auc"])
    best = ranked[0]

    # 2. luck check: the smallest group within tolerance of the best
    close = [n for n in ranked if res_groups[n]["cv_auc"] >= res_groups[best]["cv_auc"] - a.tolerance]
    pick = min(close, key=lambda n: (res_groups[n]["members"], -res_groups[n]["cv_auc"]))
    wins = sum(fa >= fb for fa, fb in zip(res_groups[pick]["fold_aucs"], res_groups[ranked[1] if pick == best else best]["fold_aucs"]))
    print("groups by AUC on training images (second 5-fold split):")
    for n in ranked:
        print(f"  {res_groups[n]['cv_auc']:.4f}  {n}")
    print(f"chosen: {pick}  (best was {best}; chosen wins {wins}/5 folds vs the runner-up)")

    # 3. weights on all training images, calibration and cut-off from out-of-fold scores
    g, pred = cv_pred[pick]
    stack = balanced_lr(Oz[:, g], ystr)
    cal = balanced_lr(pred[:, None], ystr)
    p_tr = cal.predict_proba(pred[:, None])[:, 1]
    cuts = np.linspace(0.2, 0.8, 121)
    cut = float(cuts[np.argmax([balanced_accuracy_score(ytr, p_tr > c) for c in cuts])])

    # 4. the one test evaluation of the chosen panel, plus context
    p_ts = cal.predict_proba(stack.decision_function(Tz[:, g])[:, None])[:, 1]
    out = {"members": a.members, "test_logits_from": source, "groups": res_groups, "best_group": best, "chosen": pick,
           "weights": dict(zip([a.members[k] for k in g], map(float, stack.coef_[0]))), "cut_off": round(cut, 3),
           "train_oof": metrics(p_tr, ytr, votes[tr], cut), "TEST_chosen_panel": metrics(p_ts, y[ts], votes[ts], cut)}
    ctx = {}
    for k, tag in enumerate(a.members):  # each member alone, calibrated on its own out-of-fold logits
        c = balanced_lr(O[:, [k]], ystr)
        ctx[tag] = metrics(c.predict_proba(T[:, [k]])[:, 1], y[ts], votes[ts])
    app = np.mean([np.load(f) for f in sorted((ROOT / "data/models").glob("conch_ft_seed*_testpred.npy"))], axis=0)
    ctx["app now (4 partly fine-tuned CONCH)"] = metrics(app, y[ts], votes[ts])
    out["test_context"] = ctx
    OUT.write_text(json.dumps(out, indent=2))
    print("TEST chosen panel:", json.dumps(out["TEST_chosen_panel"]))
    print("TEST app now:     ", json.dumps(ctx["app now (4 partly fine-tuned CONCH)"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
