#!/usr/bin/env python3
"""5-fold cross-validated fine-tuning on the MHIST training split (the test split is never used
for any choice).

For one configuration (backbone, unfrozen blocks, learning rate) this trains 5 models, each on 4/5
of train with the 5th as validation (early stopping on val AUC, up to --epochs). It saves, per fold:
the trainable weights, the out-of-fold logits (for choosing settings and fitting the temperature)
and the test logits at the chosen epoch (only read later for the configuration picked on OOF AUC).

    python research/kfold_cv.py --backbone conch --blocks 2 --lr 1e-5 --tag conch_b2_lr1e-5
    python research/kfold_cv.py --backbone plip  --blocks 2 --lr 1e-5 --tag plip_b2_lr1e-5
    MUSK_SRC=<clone of lilab-stanford/MUSK> python research/kfold_cv.py --backbone musk --blocks 2 --lr 1e-5 --tag musk_b2_lr1e-5

Resumable: folds whose weights + test logits already exist are skipped (the overnight queue stops
jobs at a cut-off time and restarts them later).

Outputs go to data/models/cv/<tag>/ (not data/models/, so the app never loads them as judges)
plus a summary in docs/results/cv/<tag>.json. Soft labels (votes / 7) and the balanced loss are
the same as research/finetune_conch.py.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from compare_judges import balanced_bce  # noqa: E402
from select_prompts import load_labels  # noqa: E402

DEV = "cuda"


class ConchFT(nn.Module):
    """CONCH with its last `blocks` ViT blocks + pooling/projection trainable, linear head."""
    def __init__(self, blocks):
        super().__init__()
        from conch.open_clip_custom import create_model_from_pretrained
        self.m, self.pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH")
        for p in self.m.parameters():
            p.requires_grad = False
        v = self.m.visual
        for mod in [*v.trunk.blocks[-blocks:], v.trunk.norm, v.attn_pool_contrast, v.ln_contrast, v.head]:
            for p in mod.parameters():
                p.requires_grad = True
        self.cls = nn.Linear(512, 1)

    def forward(self, x):
        z = self.m.encode_image(x, proj_contrast=True, normalize=True)
        return self.cls(z).squeeze(-1) * 10.0


class PlipFT(nn.Module):
    """PLIP (CLIP ViT-B/32 tuned on pathology) with its last `blocks` vision layers + projection trainable."""
    def __init__(self, blocks):
        super().__init__()
        from transformers import CLIPImageProcessor, CLIPModel
        clip = CLIPModel.from_pretrained("vinid/plip")
        self.vision, self.proj = clip.vision_model, clip.visual_projection
        for p in [*self.vision.parameters(), *self.proj.parameters()]:
            p.requires_grad = False
        for mod in [*self.vision.encoder.layers[-blocks:], self.vision.post_layernorm, self.proj]:
            for p in mod.parameters():
                p.requires_grad = True
        proc = CLIPImageProcessor.from_pretrained("vinid/plip")
        self.pre = lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]
        self.cls = nn.Linear(self.proj.out_features, 1)

    def forward(self, x):
        z = nn.functional.normalize(self.proj(self.vision(pixel_values=x).pooler_output), dim=-1)
        return self.cls(z).squeeze(-1) * 10.0


class MuskFT(nn.Module):
    """MUSK (BEiT-3, ViT-L, 384 px) with its last `blocks` encoder layers trainable, linear head on the
    [CLS] token. Run under bf16 autocast to fit an 8 GB GPU."""
    def __init__(self, blocks):
        super().__init__()
        sys.path.insert(0, os.environ["MUSK_SRC"])
        from musk import modeling, utils  # noqa: F401
        import torchvision.transforms as T
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from timm.models import create_model
        self.m = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", self.m, "model|module", "")
        for p in self.m.parameters():
            p.requires_grad = False
        enc = self.m.beit3.encoder
        for mod in [*enc.layers[-blocks:], *([enc.layer_norm] if getattr(enc, "layer_norm", None) is not None else [])]:
            for p in mod.parameters():
                p.requires_grad = True
        self.pre = T.Compose([T.Resize(384, interpolation=3, antialias=True), T.CenterCrop((384, 384)), T.ToTensor(),
                              T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)])
        self.cls = nn.Linear(1024, 1)

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.m(image=x, with_head=False, out_norm=False, ms_aug=False, return_global=True)[0]
        return self.cls(nn.functional.normalize(z.float(), dim=-1)).squeeze(-1) * 10.0


BACKBONES = {"conch": ConchFT, "plip": PlipFT, "musk": MuskFT}


class Tiles(torch.utils.data.Dataset):
    def __init__(self, names, y, pre, augment):
        self.names, self.y, self.pre, self.augment = names, y, pre, augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(ROOT / "data/mhist/images" / self.names[i]).convert("RGB")
        if self.augment:  # orientation-free: flips + 90-degree rotations (as in finetune_conch.py)
            if np.random.rand() < .5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.rotate(90 * np.random.randint(4))
        return self.pre(img), torch.tensor(self.y[i], dtype=torch.float32)


@torch.no_grad()
def logits(model, names, pre):
    model.eval()
    dl = torch.utils.data.DataLoader(Tiles(names, np.zeros(len(names)), pre, False), batch_size=32)
    return np.concatenate([model(x.to(DEV)).float().cpu().numpy() for x, _ in dl])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    lab = load_labels(ROOT / "data/mhist/annotations.csv")
    names = np.array(sorted(lab["image"]))
    lab = lab.set_index("image").loc[names]
    y = (lab["label"] == "SSA").to_numpy().astype(int)
    y_soft = lab["ssa_votes"].to_numpy(float) / 7.0
    tr_all = np.flatnonzero((lab["split"] == "train").to_numpy())
    ts = np.flatnonzero((lab["split"] == "test").to_numpy())

    out_dir = ROOT / "data/models/cv" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = ROOT / "docs/results/cv" / f"{a.tag}.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(res_path.read_text()) if res_path.exists() else {"config": vars(a), "folds": {}}
    summary["folds"] = {int(k): v for k, v in summary["folds"].items()}
    oof = np.load(out_dir / "oof_logits.npy") if (out_dir / "oof_logits.npy").exists() else np.full(len(names), np.nan)

    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=0)
    for k, (fi_tr, fi_va) in enumerate(skf.split(tr_all, y[tr_all])):
        if k in summary["folds"] and (out_dir / f"fold{k}.pt").exists() and (out_dir / f"fold{k}_test_logits.npy").exists():
            print(f"[{a.tag}] fold {k + 1} already done, skipping", flush=True)
            continue
        t0 = time.time()
        torch.manual_seed(k); np.random.seed(k)
        model = BACKBONES[a.backbone](a.blocks).to(DEV)
        tr, va = tr_all[fi_tr], tr_all[fi_va]
        head = list(model.cls.parameters())
        body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls")]
        opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": 1e-3}], weight_decay=1e-2)
        dl = torch.utils.data.DataLoader(Tiles(names[tr], y_soft[tr], model.pre, True), batch_size=a.batch, shuffle=True,
                                         generator=torch.Generator().manual_seed(k))
        best, state, curve = -1, None, []
        for ep in range(a.epochs):
            model.train()
            for x, yb in dl:
                loss = balanced_bce(model(x.to(DEV)), yb.to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()
            lv = logits(model, names[va], model.pre)
            auc = roc_auc_score(y[va], lv)
            curve.append(round(float(auc), 4))
            print(f"[{a.tag}] fold {k + 1}/{a.folds} epoch {ep + 1}: val AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if auc > best:
                best, best_ep, best_lv = auc, ep + 1, lv
                state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(state, strict=False)
        oof[va] = best_lv
        np.save(out_dir / f"fold{k}_test_logits.npy", logits(model, names[ts], model.pre))
        torch.save({"state": state, "config": vars(a), "fold": k, "best_epoch": best_ep, "val_auc": best},
                   out_dir / f"fold{k}.pt")
        summary["folds"][k] = {"best_epoch": best_ep, "val_auc": round(float(best), 4), "curve": curve,
                               "minutes": round((time.time() - t0) / 60, 1)}
        np.save(out_dir / "oof_logits.npy", oof)
        res_path.write_text(json.dumps(summary, indent=2))
        del model, opt
        torch.cuda.empty_cache()

    m = ~np.isnan(oof[tr_all])
    summary["oof_auc"] = round(float(roc_auc_score(y[tr_all][m], oof[tr_all][m])), 4)
    summary["mean_fold_auc"] = round(float(np.mean([f["val_auc"] for f in summary["folds"].values()])), 4)
    res_path.write_text(json.dumps(summary, indent=2))
    print(f"[{a.tag}] OOF AUC {summary['oof_auc']}  mean fold AUC {summary['mean_fold_auc']}", flush=True)


if __name__ == "__main__":
    main()
