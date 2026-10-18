#!/usr/bin/env python3
"""Is a heavier judge worth it? Same data, same splits, same metrics:

  A. linear   : current judge type (logistic head on frozen CONCH features, prompt-anchored)
  B. mlp      : small neural network (512 -> 256 -> 1, dropout) on the same frozen features
  C. finetune : partial fine-tune of CONCH itself (last 2 of 12 ViT blocks + pooling/projection) + linear head

All use: MHIST train split minus a fixed validation slice for training, validation for early
stopping / regularisation / temperature, and the 977-image test split exactly once.
Targets = pathologists' SSA vote share; class-balanced (50/50 prior). Stress tests: grayscale,
strong colour jitter, JPEG q=20. Writes docs/results/compare_judges.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from embed_views import VIEWS  # noqa: E402
from select_prompts import load_labels  # noqa: E402
from train_judges import ece, fit_head, fit_temperature  # noqa: E402

STRESS = ["gray", "jitter_strong", "jpeg20"]
DEV = "cuda"
torch.manual_seed(0)


def metrics(y, p):
    return {"bal_acc": round(float(balanced_accuracy_score(y, p > 0.5)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4), "ece": round(ece(p, y), 4)}


def sig(a):
    return 1 / (1 + np.exp(-a))


def balanced_bce(logit, y):
    prev = y.mean().clamp(0.05, 0.95)
    w = y / (2 * prev) + (1 - y) / (2 * (1 - prev))
    return (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()


# ---------------- A + B: heads on cached frozen embeddings ----------------
class MLP(nn.Module):
    def __init__(self, d=513, h=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xva, yva_hard, seed, epochs=200):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    Xt, Yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV, dtype=torch.float32)
    Xv = torch.tensor(Xva, device=DEV)
    best, state, patience = -1, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            loss = balanced_bce(m(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            auc = roc_auc_score(yva_hard, m(Xv).cpu().numpy())
        if auc > best:
            best, state, patience = auc, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 20:
                break
    m.load_state_dict(state); m.eval()
    return m


# ---------------- C: partial fine-tune of CONCH ----------------
class FineTuned(nn.Module):
    def __init__(self, conch, n_train_blocks=2):
        super().__init__()
        self.conch = conch
        for p in conch.parameters():
            p.requires_grad = False
        v = conch.visual
        for mod in [*v.trunk.blocks[-n_train_blocks:], v.trunk.norm, v.attn_pool_contrast, v.ln_contrast, v.head]:
            for p in mod.parameters():
                p.requires_grad = True
        self.cls = nn.Linear(512, 1)

    def forward(self, x):
        z = self.conch.encode_image(x, proj_contrast=True, normalize=True)
        return self.cls(z).squeeze(-1) * 10.0  # scale: normalised features have small norms


class Tiles(torch.utils.data.Dataset):
    def __init__(self, names, y, pre, view=None, augment=False):
        self.names, self.y, self.pre, self.view, self.augment = names, y, pre, view, augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(ROOT / "data/mhist/images" / self.names[i]).convert("RGB")
        if self.view:
            img = VIEWS[self.view](img, np.random.default_rng(i * 100 + 7))
        if self.augment:  # histology is orientation-free: flips + 90-degree rotations
            if np.random.rand() < .5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.rotate(90 * np.random.randint(4))
        return self.pre(img), torch.tensor(self.y[i], dtype=torch.float32)


def predict_ft(model, names, y, pre, view=None):
    dl = torch.utils.data.DataLoader(Tiles(names, y, pre, view), batch_size=32, num_workers=0)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            out.append(model(x.to(DEV)).cpu().numpy())
    return np.concatenate(out)


def run_finetune(names, y_soft, y_hard, tr, va, ts, epochs=6):
    from conch.open_clip_custom import create_model_from_pretrained

    conch, pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH")
    model = FineTuned(conch).to(DEV)
    head_p = list(model.cls.parameters())
    body_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls")]
    opt = torch.optim.AdamW([{"params": body_p, "lr": 1e-5}, {"params": head_p, "lr": 1e-3}], weight_decay=1e-2)
    dl = torch.utils.data.DataLoader(Tiles(names[tr], y_soft[tr], pre, augment=True), batch_size=16, shuffle=True, num_workers=0)
    best, state = -1, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for x, yb in dl:
            loss = balanced_bce(model(x.to(DEV)), yb.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        v = predict_ft(model, names[va], y_hard[va], pre)
        auc = roc_auc_score(y_hard[va], v)
        print(f"  finetune epoch {ep + 1}: val AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best = auc
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    T = fit_temperature(predict_ft(model, names[va], y_hard[va], pre), y_hard[va])
    out = {"test": sig(predict_ft(model, names[ts], y_hard[ts], pre) / T)}
    for sv in STRESS:
        out[sv] = sig(predict_ft(model, names[ts], y_hard[ts], pre, view=sv) / T)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return out, n_train, best


def main():
    z = np.load(ROOT / "data/emb/mhist/conch.npz")
    names, views = np.array(z["names"]), list(z["views"])
    te = dict(zip(z["text_keys"], z["text_emb"]))
    lab = load_labels(ROOT / "data/mhist/annotations.csv").set_index("image").loc[names]
    y_hard = (lab["label"] == "SSA").to_numpy().astype(int)
    y_soft = lab["ssa_votes"].to_numpy(dtype=float) / 7.0
    train_all = (lab["split"] == "train").to_numpy()
    ts = (lab["split"] == "test").to_numpy()
    rng = np.random.default_rng(0)  # same validation slice as train_judges.py
    va = train_all & (rng.random(len(lab)) < 0.2)
    tr = train_all & ~va

    sets = json.loads((ROOT / "backend/config/judges.json").read_text())["prompt_sets"]
    p = sets[0]  # the best prompt set; raw condition (like judge J1)
    X = {v: z["img_emb"][views.index(v)] for v in ["orig", *STRESS]}
    zl = {v: 56.3 * (X[v] @ te[p["SSA"]] - X[v] @ te[p["HP"]]) for v in X}
    results, preds = {}, {}

    # A. linear (current judge)
    best = None
    for l2 in (1e-6, 1e-5, 1e-4, 1e-3):
        w, b, a = fit_head(X["orig"][tr], zl["orig"][tr], y_soft[tr], seed=0, l2=l2)
        s = roc_auc_score(y_hard[va], X["orig"][va] @ w + b + a * zl["orig"][va])
        if best is None or s > best[0]:
            best = (s, (w, b, a))
    w, b, a = best[1]
    lin = {v: X[v] @ w + b + a * zl[v] for v in X}
    T = fit_temperature(lin["orig"][va], y_hard[va])
    preds["linear"] = {v: sig(lin[v][ts] / T) for v in X}
    results["linear"] = {"trainable_params": 512 + 2}

    # B. MLP on [features, zero-shot logit]
    F_ = {v: np.concatenate([X[v], zl[v][:, None] / 10.0], 1).astype(np.float32) for v in X}
    m = train_mlp(F_["orig"][tr], y_soft[tr], F_["orig"][va], y_hard[va], seed=0)
    with torch.no_grad():
        lg = {v: m(torch.tensor(F_[v], device=DEV)).cpu().numpy() for v in X}
    T = fit_temperature(lg["orig"][va], y_hard[va])
    preds["mlp"] = {v: sig(lg[v][ts] / T) for v in X}
    results["mlp"] = {"trainable_params": sum(q.numel() for q in m.parameters())}

    # C. partial fine-tune
    print("fine-tuning CONCH (last 2 blocks)...", flush=True)
    t0 = time.time()
    ft, n_ft, best_val = run_finetune(names, y_soft, y_hard, tr, va, ts)
    preds["finetune"] = {"orig": ft["test"], **{sv: ft[sv] for sv in STRESS}}
    results["finetune"] = {"trainable_params": n_ft, "train_minutes": round((time.time() - t0) / 60, 1)}

    yt = y_hard[ts]
    for k in preds:
        results[k]["test"] = metrics(yt, preds[k]["orig"])
        results[k]["stress"] = {sv: {**metrics(yt, preds[k][sv]),
                                     "flip_rate": round(float(((preds[k][sv] > .5) != (preds[k]["orig"] > .5)).mean()), 4)}
                                for sv in STRESS}
    out = ROOT / "docs/results/compare_judges.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
