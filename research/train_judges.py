#!/usr/bin/env python3
"""Train the 8 gutfeeling judges and evaluate them on the held-out test split.

Judge = (prompt set p, image condition c). Each judge is a small logistic head on
frozen CONCH image features that STARTS from its prompt's zero-shot score:

    logit_SSA = (w . x_c + b + alpha * z_p(x_c)) / T

z_p is the zero-shot logit for prompt set p (SSA prompt vs HP prompt), alpha and w
are learned on the training split (each judge on its own bootstrap sample, so the
panel isn't 8 copies of one model), and T is a temperature fitted on a validation
slice so "82%" means roughly 82%. Targets are the pathologists' vote share
(SSA votes / 7) when official MHIST labels are available, else hard labels.

Run research/select_prompts.py first (it writes backend/config/judges.json).
Writes backend/config/heads.npz and docs/results/judges_eval.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from select_prompts import CLASSES, load_labels

ROOT = Path(__file__).resolve().parent.parent
COND_VIEW = {"raw": "orig", "normalized": "macenko"}
STRESS_VIEWS = ["gray", "jitter_mild", "jitter_strong", "jpeg20"]


def zs_logit(x, te, prompt_ids, scale):
    return scale * (x @ te[prompt_ids["SSA"]] - x @ te[prompt_ids["HP"]])


def fit_head(x, z, y, seed, l2=1e-5, steps=300):
    """Logistic head with the zero-shot logit as a learnable anchor; soft targets ok.
    Class-balanced: SSA and HP count equally (MHIST is ~31% SSA), i.e. a 50/50 prior."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(x), (len(x),), generator=g)  # bootstrap
    X, Z, Y = (torch.tensor(a, dtype=torch.float32)[idx] for a in (x, z, y))
    prev = float(Y.mean())
    sw = Y / (2 * prev) + (1 - Y) / (2 * (1 - prev))  # per-sample weight for soft targets
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    alpha = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b, alpha], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = X @ w + b + alpha * Z
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, Y, reduction="none") * sw).mean() + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach()), float(alpha.detach())


def fit_temperature(logit, y):
    t = torch.ones(1, requires_grad=True)
    L, Y = torch.tensor(logit, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    prev = float(Y.mean())
    sw = Y / (2 * prev) + (1 - Y) / (2 * (1 - prev))  # same 50/50 prior as training
    opt = torch.optim.LBFGS([t], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(L / t.clamp(min=0.05), Y, reduction="none") * sw).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.clamp(min=0.05))


def ece(p, y, bins=10):
    conf, pred = np.maximum(p, 1 - p), (p > 0.5).astype(int)
    edges, e = np.linspace(0.5, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(ROOT / "data/emb/mhist/conch.npz"))
    ap.add_argument("--labels", required=True)
    ap.add_argument("--logit-scale", type=float, default=56.3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()

    z = np.load(a.emb, allow_pickle=False)
    names, views = list(z["names"]), list(z["views"])
    te = dict(zip(z["text_keys"], z["text_emb"]))
    lab = load_labels(a.labels).set_index("image")
    keep = [i for i, n in enumerate(names) if n in lab.index]
    lab = lab.loc[[names[i] for i in keep]]
    V = {v: z["img_emb"][views.index(v)][keep] for v in views}

    y_hard = (lab["label"] == "SSA").to_numpy().astype(float)
    votes = lab["ssa_votes"].to_numpy(dtype=float)
    soft = not np.isnan(votes).all()
    y_train_target = votes / 7.0 if soft else y_hard

    train_all = (lab["split"] == "train").to_numpy()
    test = (lab["split"] == "test").to_numpy()
    rng = np.random.default_rng(0)
    val = train_all & (rng.random(len(lab)) < a.val_frac)
    tr = train_all & ~val
    print(f"{len(lab)} images | train {tr.sum()} val {val.sum()} test {test.sum()} | targets: {'pathologist vote share' if soft else 'hard labels'}")

    sets = json.loads((ROOT / "backend/config/judges.json").read_text())["prompt_sets"]
    judges = [(p, c) for p in sets for c in ("raw", "normalized")]

    W, B, A, T, rows = [], [], [], [], []
    P_zs, P_tr, P_stress = [], [], {v: [] for v in STRESS_VIEWS}
    for n, (p, c) in enumerate(judges):
        x = V[COND_VIEW[c]]
        zl = zs_logit(x, te, p, a.logit_scale)
        # choose regularisation on the validation slice (never on test)
        best = None
        for l2 in (1e-6, 1e-5, 1e-4, 1e-3):
            cand = fit_head(x[tr], zl[tr], y_train_target[tr], seed=n, l2=l2)
            v_logit = x[val] @ cand[0] + cand[1] + cand[2] * zl[val]
            score = roc_auc_score(y_hard[val], v_logit)
            if best is None or score > best[0]:
                best = (score, l2, cand)
        _, l2_used, (w, b, alpha) = best
        raw_logit = x @ w + b + alpha * zl
        t = fit_temperature(raw_logit[val], y_hard[val])
        W.append(w); B.append(b); A.append(alpha); T.append(t)

        p_zs, p_tr = sigmoid(zl[test]), sigmoid(raw_logit[test] / t)
        P_zs.append(p_zs); P_tr.append(p_tr)
        for sv in STRESS_VIEWS:  # same judge, image altered in non-biological ways
            xs = V[sv][test]
            P_stress[sv].append(sigmoid((xs @ w + b + alpha * zs_logit(xs, te, p, a.logit_scale)) / t))
        rows.append({"judge": f"J{n + 1}", "prompt_set": p, "condition": c, "alpha": round(alpha, 3), "T": round(t, 3), "l2": l2_used,
                     "zero_shot": {"bal_acc": balanced_accuracy_score(y_hard[test], p_zs > .5), "auc": roc_auc_score(y_hard[test], p_zs)},
                     "trained": {"bal_acc": balanced_accuracy_score(y_hard[test], p_tr > .5), "auc": roc_auc_score(y_hard[test], p_tr)}})

    yt = y_hard[test].astype(int)
    P_zs, P_tr = np.array(P_zs), np.array(P_tr)
    report = {"n_test": int(test.sum()), "soft_targets": soft, "judges": rows}
    for name, P in (("zero_shot_panel", P_zs), ("trained_panel", P_tr)):
        m = P.mean(0)
        report[name] = {"bal_acc": balanced_accuracy_score(yt, m > .5), "auc": roc_auc_score(yt, m),
                        "ece": ece(m, yt), "mean_confidence": float(np.maximum(m, 1 - m).mean())}

    panel = P_tr.mean(0)
    ai_votes = (P_tr > .5).sum(0)
    agree = np.minimum(ai_votes, 8 - ai_votes) <= 1
    correct = (panel > .5) == yt
    report["agreement_vs_accuracy"] = {"acc_when_judges_agree": float(correct[agree].mean()) if agree.any() else None,
                                       "n_agree": int(agree.sum()),
                                       "acc_when_judges_split": float(correct[~agree].mean()) if (~agree).any() else None,
                                       "n_split": int((~agree).sum())}
    report["stress"] = {}
    for sv, Ps in P_stress.items():
        ps = np.array(Ps).mean(0)
        report["stress"][sv] = {"bal_acc": balanced_accuracy_score(yt, ps > .5),
                                "flip_rate": float(((ps > .5) != (panel > .5)).mean())}
    if soft:
        from scipy.stats import spearmanr
        pv = votes[test]
        rho, pval = spearmanr(ai_votes, pv)
        hard_cases = (np.minimum(pv, 7 - pv) >= 3)  # 4-3 splits among pathologists
        report["vs_pathologists"] = {
            "spearman_ai_votes_vs_pathologist_votes": float(rho), "p": float(pval),
            "ai_split_rate_on_4v3_cases": float((~agree)[hard_cases].mean()),
            "ai_split_rate_on_unanimous_cases": float((~agree)[(pv == 0) | (pv == 7)].mean()),
        }

    out = ROOT / "backend/config/heads.npz"
    np.savez(out, W=np.array(W, np.float32), b=np.array(B, np.float32), alpha=np.array(A, np.float32),
             T=np.array(T, np.float32), logit_scale=np.float32(a.logit_scale))
    res = ROOT / "docs/results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "judges_eval.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "judges"}, indent=2, default=float))
    print(f"\nsaved {out} and {res / 'judges_eval.json'}")


if __name__ == "__main__":
    main()
