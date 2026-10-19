#!/usr/bin/env python3
"""5-fold cross-validated fine-tuning on the MHIST training split (the test split is never used
for any choice), plus the final retrain on all of train.

For one configuration (backbone, trainable blocks, learning rate) this trains 5 models, each on 4/5
of train with the 5th as validation. Each model first trains only its linear head on the frozen
backbone (--head-epochs), then fine-tunes (early stopping on val AUC, up to --epochs). Saved per
fold: trainable weights, out-of-fold logits (for choosing the panel and calibration) and test logits.
With --final-epochs N it instead trains once on ALL training images for N fine-tune epochs (the
median best epoch from the folds) and saves final.pt + final_test_logits.npy.

Loss: soft labels (SSA votes / 7) with a FIXED class weight so SSA and HP carry equal total weight:
SSA examples x (n_HP / n_SSA) of that training set (2.45 on full train). compare_judges.balanced_bce
(used for the app's current models) re-estimated it per batch of 16 from soft labels: median 1.87,
swinging 1.2-3.0. Training images get flips/90-degree rotations and mild colour jitter; prediction
uses images as-is, so the same image always gets the same score. Forward passes run in bf16 autocast
(weights and optimiser stay fp32) so full fine-tuning fits on an 8 GB GPU.

    python research/kfold_cv.py --backbone conch    --blocks -1 --lr 1e-5 --tag conch_full
    python research/kfold_cv.py --backbone plip     --blocks -1 --lr 1e-5 --tag plip_full
    python research/kfold_cv.py --backbone convnext --blocks -1 --lr 1e-4 --tag convnext_full
    MUSK_SRC=<clone of lilab-stanford/MUSK> python research/kfold_cv.py --backbone musk --blocks 2 --lr 1e-5 --batch 8 --tag musk_b2
    python research/kfold_cv.py --backbone conch --blocks -1 --lr 1e-5 --tag conch_full --final-epochs 5
(--blocks -1 = every layer trainable.)

Resumable: folds whose weights + test logits already exist are skipped. Outputs go to
data/models/cv/<tag>/ (not data/models/, so the app never loads them as judges) and a summary to
docs/results/cv/<tag>.json.
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
from select_prompts import load_labels  # noqa: E402

DEV = "cuda"


class ConchFT(nn.Module):
    """CONCH vision tower; `blocks` < 0 trains every layer, otherwise the last `blocks` ViT blocks +
    pooling/projection. Linear head on the normalised embedding."""
    def __init__(self, blocks):
        super().__init__()
        from conch.open_clip_custom import create_model_from_pretrained
        self.m, self.pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH")
        for p in self.m.parameters():
            p.requires_grad = False
        v = self.m.visual
        mods = [v] if blocks < 0 else [*v.trunk.blocks[-blocks:], v.trunk.norm, v.attn_pool_contrast, v.ln_contrast, v.head]
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad = True
        self.cls = nn.Linear(512, 1)

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16 activations: full fine-tune fits in 8 GB
            z = self.m.encode_image(x, proj_contrast=True, normalize=True)
        return self.cls(z.float()).squeeze(-1) * 10.0


class PlipFT(nn.Module):
    """PLIP (CLIP ViT-B/32 tuned on pathology); `blocks` < 0 trains the whole vision tower."""
    def __init__(self, blocks):
        super().__init__()
        from transformers import CLIPImageProcessor, CLIPModel
        clip = CLIPModel.from_pretrained("vinid/plip")
        self.vision, self.proj = clip.vision_model, clip.visual_projection
        for p in [*self.vision.parameters(), *self.proj.parameters()]:
            p.requires_grad = False
        mods = [self.vision, self.proj] if blocks < 0 else [*self.vision.encoder.layers[-blocks:], self.vision.post_layernorm, self.proj]
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad = True
        proc = CLIPImageProcessor.from_pretrained("vinid/plip")
        self.pre = lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]
        self.cls = nn.Linear(self.proj.out_features, 1)

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.proj(self.vision(pixel_values=x).pooler_output)
        return self.cls(nn.functional.normalize(z.float(), dim=-1)).squeeze(-1) * 10.0


class MuskFT(nn.Module):
    """MUSK (BEiT-3, ViT-L, 384 px) with its last `blocks` encoder layers trainable (too large to train
    fully on 8 GB), linear head on the [CLS] token, bf16 autocast."""
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
        for mod in [*enc.layers[-max(blocks, 1):], *([enc.layer_norm] if getattr(enc, "layer_norm", None) is not None else [])]:
            for p in mod.parameters():
                p.requires_grad = True
        self.pre = T.Compose([T.Resize(384, interpolation=3, antialias=True), T.CenterCrop((384, 384)), T.ToTensor(),
                              T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)])
        self.cls = nn.Linear(1024, 1)

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.m(image=x, with_head=False, out_norm=False, ms_aug=False, return_global=True)[0]
        return self.cls(nn.functional.normalize(z.float(), dim=-1)).squeeze(-1) * 10.0


class ConvNextFT(nn.Module):
    """ConvNeXt pretrained on ImageNet-22k (timm, not gated); `blocks` < 0 trains every layer,
    otherwise the last `blocks` stages."""
    ARCH = "convnext_tiny.fb_in22k_ft_in1k"

    def __init__(self, blocks):
        super().__init__()
        import timm
        self.m = timm.create_model(self.ARCH, pretrained=True, num_classes=0)
        for p in self.m.parameters():
            p.requires_grad = blocks < 0
        if blocks >= 0:
            for mod in [*self.m.stages[-blocks:], self.m.head]:
                for p in mod.parameters():
                    p.requires_grad = True
        cfg = timm.data.resolve_data_config({}, model=self.m)
        self.pre = timm.data.create_transform(**cfg)  # eval transform: resize/centre-crop 224, ImageNet norm
        self.cls = nn.Linear(self.m.num_features, 1)

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.m(x)
        return self.cls(nn.functional.normalize(z.float(), dim=-1)).squeeze(-1) * 10.0


class ConvNextBaseFT(ConvNextFT):
    """ConvNeXt-Base (89M parameters) vs Tiny (28M): same family and pretraining, larger."""
    ARCH = "convnext_base.fb_in22k_ft_in1k"


BACKBONES = {"conch": ConchFT, "plip": PlipFT, "musk": MuskFT, "convnext": ConvNextFT, "convnext_base": ConvNextBaseFT}


def weighted_bce(logit, y_soft, w_ssa):
    """BCE on soft labels; the SSA side weighted by the fixed w_ssa (= n_HP / n_SSA of the training set)."""
    w = y_soft * w_ssa + (1 - y_soft)
    return (nn.functional.binary_cross_entropy_with_logits(logit, y_soft, reduction="none") * w).mean()


_JITTER = []


def jitter(img):
    """mild colour jitter (training only): brightness/contrast/saturation +-10%, hue +-0.02"""
    if not _JITTER:
        import torchvision.transforms as T
        _JITTER.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02))
    return _JITTER[0](img)


class Tiles(torch.utils.data.Dataset):
    def __init__(self, names, y, pre, augment):
        self.names, self.y, self.pre, self.augment = names, y, pre, augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(ROOT / "data/mhist/images" / self.names[i]).convert("RGB")
        if self.augment:  # orientation-free: flips + 90-degree rotations, plus mild colour jitter
            if np.random.rand() < .5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.rotate(90 * np.random.randint(4))
            img = jitter(img)
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
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch x accum)")
    ap.add_argument("--blocks", type=int, default=-1, help="-1 = every layer trainable")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=8, help="fine-tune epochs (after the head-only epochs)")
    ap.add_argument("--head-epochs", type=int, default=1, help="epochs training only the linear head first")
    ap.add_argument("--final-epochs", type=int, default=0, help="train once on all of train for N fine-tune epochs")
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

    def train(seed, tr, va, epochs):
        """head-only epochs, then fine-tune; early stopping on val AUC when `va` is given"""
        torch.manual_seed(seed); np.random.seed(seed)
        model = BACKBONES[a.backbone](a.blocks).to(DEV)
        w_ssa = float((y[tr] == 0).sum() / (y[tr] == 1).sum())  # fixed: SSA and HP carry equal total weight
        body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cls")]
        dl = torch.utils.data.DataLoader(Tiles(names[tr], y_soft[tr], model.pre, True), batch_size=a.batch, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
        where = "final" if va is None else f"fold {seed + 1}/{a.folds}"
        t0, curve, best, state, best_ep, best_lv = time.time(), [], -1.0, None, 0, None
        for p in body:
            p.requires_grad = False
        opt = torch.optim.AdamW(model.cls.parameters(), lr=1e-3, weight_decay=1e-2)
        for stage, n_ep in (("head", a.head_epochs), ("fine-tune", epochs)):
            if stage == "fine-tune":
                for p in body:
                    p.requires_grad = True
                opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": list(model.cls.parameters()), "lr": 1e-3}],
                                        weight_decay=1e-2)
            for ep in range(n_ep):
                model.train()
                opt.zero_grad()
                for i, (x, yb) in enumerate(dl):
                    loss = weighted_bce(model(x.to(DEV)), yb.to(DEV), w_ssa) / a.accum
                    loss.backward()
                    if (i + 1) % a.accum == 0 or i + 1 == len(dl):
                        opt.step(); opt.zero_grad()
                if stage == "head" or va is None:
                    print(f"[{a.tag}] {where} {stage} epoch {ep + 1} done ({time.time() - t0:.0f}s)", flush=True)
                    continue
                lv = logits(model, names[va], model.pre)
                torch.cuda.empty_cache()  # free validation buffers before the next training epoch
                auc = roc_auc_score(y[va], lv)
                curve.append(round(float(auc), 4))
                print(f"[{a.tag}] {where} fine-tune epoch {ep + 1}: val AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if auc > best:
                    best, best_ep, best_lv = auc, ep + 1, lv
                    state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        if va is None:
            state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        else:
            model.load_state_dict(state, strict=False)
        return model, state, best, best_ep, best_lv, curve, w_ssa, time.time() - t0

    if a.final_epochs:
        model, state, _, _, _, _, w_ssa, sec = train(99, tr_all, None, a.final_epochs)
        np.save(out_dir / "final_test_logits.npy", logits(model, names[ts], model.pre))
        torch.save({"state": state, "config": vars(a), "epochs": a.final_epochs, "w_ssa": w_ssa}, out_dir / "final.pt")
        summary["final"] = {"epochs": a.final_epochs, "w_ssa": round(w_ssa, 3), "minutes": round(sec / 60, 1)}
        res_path.write_text(json.dumps(summary, indent=2))
        print(f"[{a.tag}] final model trained on all {len(tr_all)} training images", flush=True)
        return

    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=0)
    for k, (fi_tr, fi_va) in enumerate(skf.split(tr_all, y[tr_all])):
        if k in summary["folds"] and (out_dir / f"fold{k}.pt").exists() and (out_dir / f"fold{k}_test_logits.npy").exists():
            print(f"[{a.tag}] fold {k + 1} already done, skipping", flush=True)
            continue
        tr, va = tr_all[fi_tr], tr_all[fi_va]
        model, state, best, best_ep, best_lv, curve, w_ssa, sec = train(k, tr, va, a.epochs)
        oof[va] = best_lv
        np.save(out_dir / f"fold{k}_test_logits.npy", logits(model, names[ts], model.pre))
        torch.save({"state": state, "config": vars(a), "fold": k, "best_epoch": best_ep, "val_auc": best},
                   out_dir / f"fold{k}.pt")
        summary["folds"][k] = {"best_epoch": best_ep, "val_auc": round(float(best), 4), "curve": curve,
                               "w_ssa": round(w_ssa, 3), "minutes": round(sec / 60, 1)}
        np.save(out_dir / "oof_logits.npy", oof)
        res_path.write_text(json.dumps(summary, indent=2))
        del model
        torch.cuda.empty_cache()

    m = ~np.isnan(oof[tr_all])
    summary["oof_auc"] = round(float(roc_auc_score(y[tr_all][m], oof[tr_all][m])), 4)
    summary["mean_fold_auc"] = round(float(np.mean([f["val_auc"] for f in summary["folds"].values()])), 4)
    summary["median_best_epoch"] = int(np.median([f["best_epoch"] for f in summary["folds"].values()]))
    res_path.write_text(json.dumps(summary, indent=2))
    print(f"[{a.tag}] OOF AUC {summary['oof_auc']}  mean fold AUC {summary['mean_fold_auc']}  "
          f"median best epoch {summary['median_best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
