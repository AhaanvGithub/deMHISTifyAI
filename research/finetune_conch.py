#!/usr/bin/env python3
"""Partial fine-tune of CONCH (last 2 ViT blocks + pooling/projection + linear head),
repeated over several seeds, to check whether the gain over the frozen linear judge is real.

For each seed: train on MHIST train (minus the fixed validation slice), early-stop and
temperature-scale on validation, then score the 977 test images once (plus stress views).
Also checks alignment with the 7 pathologists: rank correlation between P(SSA) and their
SSA vote count, and confidence on 4-3 split cases vs unanimous ones.

Saves each seed's weights to data/models/conch_ft_seed{N}.pt (local only: CONCH's licence
forbids sharing derivatives) and a summary to docs/results/finetune_seeds.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from compare_judges import STRESS, FineTuned, Tiles, balanced_bce, predict_ft, sig  # noqa: E402
from select_prompts import load_labels  # noqa: E402
from train_judges import ece, fit_temperature  # noqa: E402

DEV = "cuda"


def alignment(p, votes, y):
    conf = np.maximum(p, 1 - p)
    split = (votes == 3) | (votes == 4)
    unanimous = (votes == 0) | (votes == 7)
    return {"spearman_p_vs_votes": round(float(spearmanr(p, votes)[0]), 4),
            "mean_conf_on_4v3": round(float(conf[split].mean()), 4),
            "mean_conf_on_unanimous": round(float(conf[unanimous].mean()), 4),
            "acc_on_4v3": round(float(((p > .5) == y)[split].mean()), 4),
            "acc_on_unanimous": round(float(((p > .5) == y)[unanimous].mean()), 4)}


def run(seed, names, y_soft, y_hard, tr, va, ts, epochs=6):
    from conch.open_clip_custom import create_model_from_pretrained

    torch.manual_seed(seed); np.random.seed(seed)
    conch, pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH")
    model = FineTuned(conch).to(DEV)
    head_p = list(model.cls.parameters())
    body_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls")]
    opt = torch.optim.AdamW([{"params": body_p, "lr": 1e-5}, {"params": head_p, "lr": 1e-3}], weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(Tiles(names[tr], y_soft[tr], pre, augment=True), batch_size=16, shuffle=True, generator=g)
    best, state = -1, None
    for ep in range(epochs):
        model.train()
        for x, yb in dl:
            loss = balanced_bce(model(x.to(DEV)), yb.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        auc = roc_auc_score(y_hard[va], predict_ft(model, names[va], y_hard[va], pre))
        print(f"  seed {seed} epoch {ep + 1}: val AUC {auc:.4f}", flush=True)
        if auc > best:
            best, state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    T = fit_temperature(predict_ft(model, names[va], y_hard[va], pre), y_hard[va])
    trainable = {k: v for k, v in state.items() if k.startswith("cls") or any(
        s in k for s in ("trunk.blocks.10.", "trunk.blocks.11.", "trunk.norm.", "attn_pool_contrast", "ln_contrast", "visual.head"))}
    out_dir = ROOT / "data/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": trainable, "T": T, "seed": seed, "val_auc": best}, out_dir / f"conch_ft_seed{seed}.pt")

    preds = {"orig": sig(predict_ft(model, names[ts], y_hard[ts], pre) / T)}
    for sv in STRESS:
        preds[sv] = sig(predict_ft(model, names[ts], y_hard[ts], pre, view=sv) / T)
    del model, conch
    torch.cuda.empty_cache()
    return preds, best


def main():
    seeds = [int(s) for s in (sys.argv[1:] or ["0", "1", "2"])]
    names = np.array(sorted(p.name for p in (ROOT / "data/mhist/images").iterdir()))
    lab = load_labels(ROOT / "data/mhist/annotations.csv").set_index("image").loc[names]
    y_hard = (lab["label"] == "SSA").to_numpy().astype(int)
    votes = lab["ssa_votes"].to_numpy(dtype=float)
    y_soft = votes / 7.0
    train_all = (lab["split"] == "train").to_numpy()
    ts = (lab["split"] == "test").to_numpy()
    va = train_all & (np.random.default_rng(0).random(len(lab)) < 0.2)  # same slice as the other scripts
    tr = train_all & ~va
    yt, vt = y_hard[ts], votes[ts]

    summary, all_p = {"seeds": {}}, []
    for s in seeds:
        preds, val_auc = run(s, names, y_soft, y_hard, tr, va, ts)
        p = preds["orig"]
        all_p.append(p)
        np.save(ROOT / f"data/models/conch_ft_seed{s}_testpred.npy", p)
        summary["seeds"][s] = {
            "val_auc": round(float(val_auc), 4),
            "test": {"bal_acc": round(float(balanced_accuracy_score(yt, p > .5)), 4),
                     "auc": round(float(roc_auc_score(yt, p)), 4), "ece": round(ece(p, yt), 4)},
            "stress_auc": {sv: round(float(roc_auc_score(yt, preds[sv])), 4) for sv in STRESS},
            "pathologist_alignment": alignment(p, vt, yt),
        }
        print(json.dumps(summary["seeds"][s]), flush=True)
    aucs = [v["test"]["auc"] for v in summary["seeds"].values()]
    bas = [v["test"]["bal_acc"] for v in summary["seeds"].values()]
    summary["mean_auc"], summary["std_auc"] = round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)
    summary["mean_bal_acc"], summary["std_bal_acc"] = round(float(np.mean(bas)), 4), round(float(np.std(bas)), 4)
    ens = np.mean(all_p, 0)
    summary["ensemble_of_seeds"] = {"auc": round(float(roc_auc_score(yt, ens)), 4),
                                    "bal_acc": round(float(balanced_accuracy_score(yt, ens > .5)), 4),
                                    "pathologist_alignment": alignment(ens, vt, yt)}
    (ROOT / "docs/results/finetune_seeds.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
