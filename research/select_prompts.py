#!/usr/bin/env python3
"""Pick the 4 best CONCH prompt sets on the TRAINING split, calibrate the
criteria scores, and report the 8-judge panel on the held-out TEST split.

Inputs:
  --emb     research embeddings for CONCH (conch.npz from embed_views.py)
  --labels  either MHIST's annotations.csv
              (Image Name, Majority Vote Label, Number of Annotators who Selected SSA (Out of 7), Partition)
            or a simple CSV with columns image,label (e.g. the 160 PathMMU labels; split is then random 50/50)
Writes backend/config/judges.json and backend/config/calibration.json, prints a report.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["SSA", "HP"]


def load_labels(path):
    df = pd.read_csv(path)
    if "Image Name" in df.columns:  # official MHIST annotations
        out = pd.DataFrame({
            "image": df["Image Name"],
            "label": df["Majority Vote Label"].str.upper().map(lambda s: "HP" if s.startswith("HP") else "SSA"),
            "ssa_votes": df[[c for c in df.columns if "Number of Annotators" in c][0]],
            "split": df["Partition"].str.lower(),
        })
    else:
        rng = np.random.default_rng(0)
        out = pd.DataFrame({
            "image": df["image"],
            "label": df["label"].str.lower().map(lambda s: "HP" if "hyper" in s else "SSA"),
            "ssa_votes": np.nan,
            "split": np.where(rng.random(len(df)) < 0.5, "train", "test"),
        })
    return out


def probs(img, te, keys, scale):
    t = np.stack([te[keys[c]] for c in CLASSES])
    z = scale * img @ t.T
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--logit-scale", type=float, default=56.3)  # CONCH learned logit scale (measured: 56.3)
    ap.add_argument("--top-k", type=int, default=4)
    a = ap.parse_args()

    z = np.load(a.emb, allow_pickle=False)
    names = list(z["names"])
    views = list(z["views"])
    te = dict(zip(z["text_keys"], z["text_emb"]))
    txt = dict(zip(z["text_keys"], z["texts"]))
    lab = load_labels(a.labels).set_index("image")
    idx = [i for i, n in enumerate(names) if n in lab.index]
    lab = lab.loc[[names[i] for i in idx]]
    emb = {v: z["img_emb"][views.index(v)][idx] for v in ("orig", "macenko")}
    y = (lab["label"] == "SSA").to_numpy().astype(int)
    tr, ts = (lab["split"] == "train").to_numpy(), (lab["split"] == "test").to_numpy()
    print(f"{len(idx)} labelled images: {tr.sum()} train / {ts.sum()} test, SSA rate {y.mean():.2f}")

    # candidate prompt sets: class:SSA:NN paired with class:HP:NN
    ids = sorted({k.split(":")[2] for k in te if k.startswith("class:SSA:")} &
                 {k.split(":")[2] for k in te if k.startswith("class:HP:")})
    scores = []
    for i in ids:
        keys = {c: f"class:{c}:{i}" for c in CLASSES}
        # rank on TRAIN only, averaged over both image conditions
        ba = np.mean([balanced_accuracy_score(y[tr], 1 - probs(emb[v][tr], te, keys, a.logit_scale).argmax(1))
                      for v in emb])
        scores.append((ba, i))
    scores.sort(reverse=True)
    print("\nPrompt sets ranked on TRAIN (balanced accuracy, mean of raw+normalised):")
    for ba, i in scores:
        print(f"  {i}: {ba:.3f}   SSA='{txt[f'class:SSA:{i}']}' | HP='{txt[f'class:HP:{i}']}'")
    best = [i for _, i in scores[: a.top_k]]

    cfg = ROOT / "backend" / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "judges.json").write_text(json.dumps(
        {"prompt_sets": [{c: f"class:{c}:{i}" for c in CLASSES} for i in best],
         "selected_on": f"{tr.sum()} training images", "train_bal_acc": {i: round(s, 4) for s, i in scores}}, indent=2))

    # criteria calibration: mean/sd of similarity over ALL images (raw), no labels used
    all_raw = z["img_emb"][views.index("orig")]
    calib = {}
    for k, v in te.items():
        if k.startswith("crit:"):
            s = all_raw @ v
            calib[k] = {"mu": float(s.mean()), "sd": float(s.std() + 1e-6)}
    (cfg / "calibration.json").write_text(json.dumps(calib, indent=2))

    # ---- held-out TEST report: 8 judges = best 4 prompt sets x {raw, normalised}
    P = []
    for i in best:
        for v in ("orig", "macenko"):
            P.append(probs(emb[v][ts], te, {c: f"class:{c}:{i}" for c in CLASSES}, a.logit_scale)[:, 0])
    P = np.array(P)  # [8, n_test] P(SSA)
    yt = y[ts]
    print(f"\nTEST ({ts.sum()} images)")
    for n, (i, v) in enumerate((i, v) for i in best for v in ("raw", "norm")):
        print(f"  J{n + 1} set {i} {v:4s}: bal-acc {balanced_accuracy_score(yt, P[n] > .5):.3f}  AUC {roc_auc_score(yt, P[n]):.3f}")
    panel = P.mean(0)
    votes = (P > .5).sum(0)
    print(f"  PANEL mean prob : bal-acc {balanced_accuracy_score(yt, panel > .5):.3f}  AUC {roc_auc_score(yt, panel):.3f}")
    print(f"  PANEL majority  : bal-acc {balanced_accuracy_score(yt, votes >= 4):.3f}")
    split = (np.minimum(votes, 8 - votes) >= 2)
    if split.any() and (~split).any():
        print(f"  accuracy when judges agree (>=7/8): {((panel > .5) == yt)[~split].mean():.3f}  (n={(~split).sum()})")
        print(f"  accuracy when judges split        : {((panel > .5) == yt)[split].mean():.3f}  (n={split.sum()})")
    sv = lab["ssa_votes"].to_numpy()[ts]
    if not np.isnan(sv).all():
        from scipy.stats import spearmanr
        rho, p = spearmanr(votes, sv)
        print(f"  AI judges' SSA votes vs pathologists' SSA votes (/7): Spearman rho={rho:.3f} (p={p:.1e})")
    print(f"\nwrote {cfg / 'judges.json'} and {cfg / 'calibration.json'}")


if __name__ == "__main__":
    main()
