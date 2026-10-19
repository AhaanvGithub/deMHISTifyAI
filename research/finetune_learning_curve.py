#!/usr/bin/env python3
"""Learning curve for the fine-tuned CONCH judges: retrain on 25% / 50% of MHIST train (2 seeds
each), score on MHIST test, and compare with the full-data seeds in docs/results/finetune_seeds.json.
If test AUC is still climbing at 100%, more SSL/HP data is worth finding; if flat, it isn't.

Uses research/finetune_conch.run() unchanged except that nothing is saved: run() normally writes
data/models/conch_ft_seed{N}.pt, and the app loads every file matching that name as a judge.
Writes docs/results/finetune_learning_curve.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import finetune_conch as ft  # noqa: E402
from select_prompts import load_labels  # noqa: E402
from train_judges import ece  # noqa: E402

ft.torch.save = lambda *a, **k: None  # never touch data/models (the deployed judges)
ft.STRESS = []                        # only the plain test view is needed here
OUT = ROOT / "docs" / "results" / "finetune_learning_curve.json"


def main():
    names = np.array(sorted(p.name for p in (ROOT / "data/mhist/images").iterdir()))
    lab = load_labels(ROOT / "data/mhist/annotations.csv").set_index("image").loc[names]
    y_hard = (lab["label"] == "SSA").to_numpy().astype(int)
    votes = lab["ssa_votes"].to_numpy(dtype=float)
    train_all = (lab["split"] == "train").to_numpy()
    ts = (lab["split"] == "test").to_numpy()
    va = train_all & (np.random.default_rng(0).random(len(lab)) < 0.2)  # same slice as training
    tr_idx = np.flatnonzero(train_all & ~va)

    full = json.loads((ROOT / "docs/results/finetune_seeds.json").read_text())
    curve = {"1.0": {"n_train": len(tr_idx), "seeds": {s: v["test"] for s, v in full["seeds"].items()},
                     "auc_mean": full["mean_auc"], "source": "finetune_seeds.json"}}
    for frac in (0.25, 0.5):
        runs = {}
        for seed in (0, 1):
            sub = np.random.default_rng(1000 + seed).choice(tr_idx, int(round(frac * len(tr_idx))), replace=False)
            tr = np.zeros(len(names), bool); tr[sub] = True
            print(f"=== {frac:.0%} of train (n={tr.sum()}), seed {seed}", flush=True)
            preds, val_auc = ft.run(seed, names, votes / 7.0, y_hard, tr, va, ts)
            p, yt = preds["orig"], y_hard[ts]
            runs[seed] = {"val_auc": round(float(val_auc), 4), "auc": round(float(roc_auc_score(yt, p)), 4),
                          "bal_acc": round(float(balanced_accuracy_score(yt, p > .5)), 4), "ece": round(ece(p, yt), 4)}
            print(f"    test AUC {runs[seed]['auc']:.4f}  bal acc {runs[seed]['bal_acc']:.3f}", flush=True)
        curve[str(frac)] = {"n_train": int(round(frac * len(tr_idx))), "seeds": runs,
                            "auc_mean": round(float(np.mean([r["auc"] for r in runs.values()])), 4)}
        OUT.write_text(json.dumps({"what": "fine-tuned CONCH, test AUC vs training-set size", "curve": curve}, indent=2))
    for k in ("0.25", "0.5", "1.0"):
        print(f"train {float(k):.0%} (n={curve[k]['n_train']}): mean test AUC {curve[k]['auc_mean']:.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
