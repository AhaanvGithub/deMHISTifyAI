#!/usr/bin/env python3
"""Embed all MHIST images with MUSK (frozen) -> data/emb/mhist/musk.npz (same layout as the other
cached embeddings, view "orig" only).

MUSK's code is not pip-installed (it pins torch 2.0 / timm 0.9); point MUSK_SRC at a clone of
https://github.com/lilab-stanford/MUSK. Needs Hugging Face access to xiangjx/musk.
Settings follow the MUSK README for linear probing: 384 px, Inception mean/std, ms_aug=True,
with_head=False, return_global=True.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ["MUSK_SRC"])
from musk import modeling, utils  # noqa: E402,F401  (registers musk_large_patch16_384 with timm)
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD  # noqa: E402
from timm.models import create_model  # noqa: E402


def main():
    model = create_model("musk_large_patch16_384")
    utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
    model.to("cuda", dtype=torch.float16).eval()
    tf = T.Compose([T.Resize(384, interpolation=3, antialias=True), T.CenterCrop((384, 384)), T.ToTensor(),
                    T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)])
    names = np.array(sorted(p.name for p in (ROOT / "data/mhist/images").iterdir()))
    embs = []
    with torch.inference_mode():
        for i in range(0, len(names), 16):
            x = torch.stack([tf(Image.open(ROOT / "data/mhist/images" / n).convert("RGB")) for n in names[i:i + 16]])
            e = model(image=x.to("cuda", dtype=torch.float16), with_head=False, out_norm=False, ms_aug=True,
                      return_global=True)[0]
            embs.append(e.float().cpu().numpy())
            if i % 320 == 0:
                print(f"  {i}/{len(names)}", flush=True)
    emb = np.concatenate(embs)
    out = ROOT / "data/emb/mhist/musk.npz"
    np.savez(out, names=names, views=np.array(["orig"]), img_emb=emb[None])
    print("wrote", out, emb.shape)


if __name__ == "__main__":
    main()
