#!/usr/bin/env python3
"""Build the best panel from the cross-validated members, choosing ONLY on training data, then score
the chosen panel once on the MHIST test split.

Members: every data/models/cv/<tag>/ with out-of-fold logits on train (oof_logits.npy) and test
logits (fold*_test_logits.npy averaged over folds, or test_logits.npy for frozen heads). All use the
same 5 folds (research/kfold_cv.py, research/frozen_cv_heads.py).

Candidate panels (each scored by AUC on out-of-fold predictions; the ones with fitted weights are
scored with an extra cross-validation over the training images so the weights never see the images
they are scored on):
  best single member | equal-weight mean of standardised logits (all members, or the top-k by OOF
  AUC) | greedy forward selection with replacement (Caruana) | L2 logistic stacking.
The winner is refit on all out-of-fold predictions, calibrated (Platt on OOF, 50/50 class prior as in
training), and only then applied to test. Other members/baselines are reported on test as context.
Writes docs/results/panel.json.
"""

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


def load_members(n):
    mem = {}
    for d in sorted((ROOT / "data/models/cv").iterdir()):
        oof_p = d / "oof_logits.npy"
        tests = sorted(d.glob("fold*_test_logits.npy")) or ([d / "test_logits.npy"] if (d / "test_logits.npy").exists() else [])
        if not oof_p.exists() or not tests:
            continue
        oof = np.load(oof_p)
        if len(tests) > 1 and len(tests) < 5:
            print(f"skip {d.name}: only {len(tests)}/5 folds finished"); continue
        mem[d.name] = (oof, np.mean([np.load(t) for t in tests], axis=0))
    return mem


def balanced_lr(X, ys, C=1.0):
    """logistic regression on soft labels with a 50/50 class prior (same as training)"""
    prev = ys.mean()
    w = np.r_[ys * .5 / prev, (1 - ys) * .5 / (1 - prev)]
    return LogisticRegression(C=C, max_iter=5000).fit(np.vstack([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))], sample_weight=w)


def metrics(p, y, votes):
    split = (votes == 3) | (votes == 4)
    return {"auc": round(float(roc_auc_score(y, p)), 4), "bal_acc": round(float(balanced_accuracy_score(y, p > .5)), 4),
            "acc_vs_majority": round(float(((p > .5) == y).mean()), 4),
            "ssl_caught": round(float((p[y == 1] > .5).mean()), 4), "hp_caught": round(float((p[y == 0] <= .5).mean()), 4),
            "ece": round(ece(p, y), 4), "spearman_vs_votes": round(float(spearmanr(p, votes)[0]), 4),
            "acc_on_4v3": round(float(((p > .5) == y)[split].mean()), 4)}


def greedy(Z, y, rounds=25):
    """Caruana forward selection with replacement on AUC; returns member weights"""
    w = np.zeros(Z.shape[1]); cur = None
    for _ in range(rounds):
        best = None
        for j in range(Z.shape[1]):
            w2 = w.copy(); w2[j] += 1
            a = roc_auc_score(y, Z @ (w2 / w2.sum()))
            if best is None or a > best[0]:
                best = (a, j)
        if cur is not None and best[0] <= cur + 1e-5:
            break
        cur = best[0]; w[best[1]] += 1
    return w / w.sum()


def main():
    lab = load_labels(ROOT / "data/mhist/annotations.csv")
    names = np.array(sorted(lab["image"]))
    lab = lab.set_index("image").loc[names]
    y = (lab["label"] == "SSA").to_numpy().astype(int)
    votes = lab["ssa_votes"].to_numpy(float)
    ys = votes / 7
    tr = np.flatnonzero((lab["split"] == "train").to_numpy())
    ts = np.flatnonzero((lab["split"] == "test").to_numpy())

    mem = load_members(len(names))
    keys = list(mem)
    O = np.stack([mem[k][0][tr] for k in keys], 1)   # out-of-fold logits on train
    T = np.stack([mem[k][1] for k in keys], 1)       # test logits
    mu, sd = O.mean(0), O.std(0) + 1e-9
    Oz, Tz = (O - mu) / sd, (T - mu) / sd            # standardise with TRAIN statistics only
    ytr, ystr = y[tr], ys[tr]
    member_auc = {k: round(float(roc_auc_score(ytr, O[:, j])), 4) for j, k in enumerate(keys)}
    order = sorted(range(len(keys)), key=lambda j: -member_auc[keys[j]])
    print("members (OOF AUC on train):", json.dumps(dict(sorted(member_auc.items(), key=lambda kv: -kv[1])), indent=1))

    # ---- candidate panels, each with a weight vector over members (or a stacker) ----
    outer = list(StratifiedKFold(5, shuffle=True, random_state=7).split(tr, ytr))
    cands = {}
    cands[f"single: {keys[order[0]]}"] = ("weights", np.eye(len(keys))[order[0]], None)
    for k in range(2, len(keys) + 1):
        w = np.zeros(len(keys)); w[order[:k]] = 1 / k
        cands[f"mean of top-{k}"] = ("weights", w, None)

    def cv_auc(fit_fn):
        """AUC of a fitted panel, scored on held-out parts of train (weights refit per part)"""
        pred = np.zeros(len(tr))
        for a, b in outer:
            pred[b] = fit_fn(a)(Oz[b])
        return roc_auc_score(ytr, pred)

    cv_score = {name: roc_auc_score(ytr, Oz @ w) for name, (_, w, _) in cands.items()}
    # top-k ordering itself was chosen on all of train; re-check the best mean-of-top-k honestly
    best_k_name = max([c for c in cands if c.startswith("mean of top")], key=cv_score.get)
    k_best = int(best_k_name.split("-")[1])

    def fit_topk(a):
        o = sorted(range(len(keys)), key=lambda j: -roc_auc_score(ytr[a], Oz[a, j]))[:k_best]
        w = np.zeros(len(keys)); w[o] = 1 / k_best
        return lambda Z: Z @ w
    cv_score[best_k_name] = cv_auc(fit_topk)

    def fit_greedy(a):
        w = greedy(Oz[a], ytr[a]); return lambda Z: Z @ w
    cv_score["greedy selection"] = cv_auc(fit_greedy)
    cands["greedy selection"] = ("weights", greedy(Oz, ytr), None)

    for C in (0.01, 0.1, 1.0):
        def fit_stack(a, C=C):
            m = balanced_lr(Oz[a], ystr[a], C); return lambda Z: m.decision_function(Z)
        name = f"logistic stacking C={C}"
        cv_score[name] = cv_auc(fit_stack)
        cands[name] = ("stack", None, balanced_lr(Oz, ystr, C))

    ranked = sorted(cv_score, key=lambda c: -cv_score[c])
    for c in ranked:
        print(f"  {c:32} train-CV AUC {cv_score[c]:.4f}")
    pick = ranked[0]
    kind, w, stack = cands[pick]
    score_tr = Oz @ w if kind == "weights" else stack.decision_function(Oz)
    score_ts = Tz @ w if kind == "weights" else stack.decision_function(Tz)
    cal = balanced_lr(score_tr[:, None], ystr)        # Platt scaling on out-of-fold scores, 50/50 prior
    p_tr, p_ts = cal.predict_proba(score_tr[:, None])[:, 1], cal.predict_proba(score_ts[:, None])[:, 1]

    # ---- the one test evaluation of the chosen panel, plus context ----
    res = {"members_oof_auc": member_auc, "panels_train_cv_auc": {c: round(float(cv_score[c]), 4) for c in ranked},
           "chosen": pick, "weights": dict(zip(keys, map(float, w))) if kind == "weights" else
           dict(zip(keys, map(float, stack.coef_[0]))), "chosen_train_oof": metrics(p_tr, ytr, votes[tr]),
           "TEST_chosen_panel": metrics(p_ts, y[ts], votes[ts])}
    ctx = {}
    for j, k in enumerate(keys):  # each member alone on test, Platt-calibrated on its own OOF
        c = balanced_lr(O[:, [j]], ystr)
        ctx[k] = metrics(c.predict_proba(T[:, [j]])[:, 1], y[ts], votes[ts])
    app_now = np.mean([np.load(f) for f in sorted((ROOT / "data/models").glob("conch_ft_seed*_testpred.npy"))], axis=0)
    ctx["APP NOW (4 fine-tuned CONCH seeds)"] = metrics(app_now, y[ts], votes[ts])
    res["test_context"] = ctx
    OUT.write_text(json.dumps(res, indent=2))
    print("\nchosen panel:", pick)
    print("TEST:", json.dumps(res["TEST_chosen_panel"]))
    print("APP NOW:", json.dumps(ctx["APP NOW (4 fine-tuned CONCH seeds)"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
