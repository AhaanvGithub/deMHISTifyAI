#!/usr/bin/env python3
"""Embed every tile in a folder with several frozen pathology models, once on
the original image and once on each non-biological "view" (grayscale, stain
colour jitter, JPEG re-compression). Also embeds a list of text prompts
(class descriptions + diagnostic criteria) with each model's text encoder.

Run on a GPU node (see embed_views.sbatch), never the login node.
Labels are NOT needed here; everything after this runs on CPU.

Output per model: <out_dir>/<model_tag>.npz with
    names       [N]            image file names (sorted)
    views       [V]            view names, "orig" first
    img_emb     [V, N, D]      L2-normalised image embeddings
    texts       [T]            prompt strings
    text_keys   [T]            prompt ids, e.g. "class:SSA" or "crit:serrated_crypts"
    text_emb    [T, D]         L2-normalised text embeddings (if the model has a text encoder)
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

# models.py (from ncsa/ashby-hackathon-2026, MIT) sits next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import resolve_adapter  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DEFAULT_MODELS = [
    "conch",
    "openclip:wisdomik/QuiltNet-B-32",
    "hfclip:vinid/plip",
    "openclip:ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt",
]


# ---------- non-biological views ----------
def view_gray(img, rng):
    return ImageOps.grayscale(img).convert("RGB")


def _jitter(img, rng, strength):
    """Random per-channel gain + offset: shifts stain colour, keeps structure."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    gain = 1.0 + rng.uniform(-strength, strength, size=3)
    bias = rng.uniform(-strength / 2, strength / 2, size=3)
    a = np.clip(a * gain + bias, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def view_jitter_mild(img, rng):
    return _jitter(img, rng, 0.10)


def view_jitter_strong(img, rng):
    return _jitter(img, rng, 0.25)


def view_jpeg(img, rng):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# Macenko stain normalisation to a fixed reference (standard reference values
# from Macenko et al. 2009 as used in common implementations). Maps every tile's
# H&E stain colours onto the same reference stain, keeping tissue structure.
_HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
_MAXC_REF = np.array([1.9705, 1.0308])


def view_macenko(img, rng, Io=240, alpha=1, beta=0.15):
    a = np.asarray(img, dtype=np.float64)
    h, w, _ = a.shape
    od = -np.log((a.reshape(-1, 3) + 1) / Io)
    od_hat = od[~np.any(od < beta, axis=1)]
    if len(od_hat) < 50:  # mostly background: nothing to normalise
        return img
    try:
        _, vecs = np.linalg.eigh(np.cov(od_hat.T))
        t = od_hat @ vecs[:, 1:3]
        phi = np.arctan2(t[:, 1], t[:, 0])
        min_phi, max_phi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
        v1 = vecs[:, 1:3] @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = vecs[:, 1:3] @ np.array([np.cos(max_phi), np.sin(max_phi)])
        he = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
        conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
        max_c = np.percentile(conc, 99, axis=1)
        conc = conc * (_MAXC_REF / max_c)[:, None]
        out = Io * np.exp(-_HE_REF @ conc)
        out = np.clip(out, 0, 255).T.reshape(h, w, 3).astype(np.uint8)
        return Image.fromarray(out)
    except Exception:
        return img


VIEWS = {
    "orig": lambda img, rng: img,
    "macenko": view_macenko,
    "gray": view_gray,
    "jitter_mild": view_jitter_mild,
    "jitter_strong": view_jitter_strong,
    "jpeg20": view_jpeg,
}


def model_tag(spec):
    return spec.replace(":", "_").replace("/", "_").replace("@", "_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompts", required=True, help="JSON file: {key: prompt text}")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible; this will be slow. Are you on a compute node?", flush=True)

    names = sorted(f for f in os.listdir(args.input_dir) if Path(f).suffix.lower() in IMG_EXTS)
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names)} images, {len(VIEWS)} views, {len(args.models)} models, device={device}", flush=True)

    prompts = json.load(open(args.prompts))
    text_keys = list(prompts.keys())
    texts = [prompts[k] for k in text_keys]

    os.makedirs(args.out_dir, exist_ok=True)

    for spec in args.models:
        out_path = Path(args.out_dir) / f"{model_tag(spec)}.npz"
        if out_path.exists():
            print(f"[skip] {out_path} already exists", flush=True)
            continue
        t0 = time.time()
        print(f"\n=== {spec} ===", flush=True)
        try:
            adapter = resolve_adapter(spec, device=device)
        except Exception as e:  # keep going with the other models
            print(f"[FAILED to load {spec}] {e}", flush=True)
            continue

        all_views = []
        for vidx, (vname, vfn) in enumerate(VIEWS.items()):
            embs = []
            for i in range(0, len(names), args.batch_size):
                batch_names = names[i : i + args.batch_size]
                tensors = []
                for j, n in enumerate(batch_names):
                    img = Image.open(os.path.join(args.input_dir, n)).convert("RGB")
                    # same seed per (image, view) for every model -> identical perturbations
                    rng = np.random.default_rng((i + j) * 100 + vidx)
                    tensors.append(adapter.preprocess(vfn(img, rng)))
                embs.append(adapter.encode_images(torch.stack(tensors)))
            all_views.append(np.concatenate(embs))
            print(f"  view {vname}: done ({time.time() - t0:.0f}s)", flush=True)

        save = dict(
            names=np.array(names),
            views=np.array(list(VIEWS)),
            img_emb=np.stack(all_views).astype(np.float32),
            texts=np.array(texts),
            text_keys=np.array(text_keys),
        )
        if adapter.encode_texts is not None:
            te = adapter.encode_texts(texts)
            save["text_emb"] = (te / np.linalg.norm(te, axis=1, keepdims=True)).astype(np.float32)
        np.savez_compressed(out_path, **save)
        print(f"  saved {out_path}  img_emb={save['img_emb'].shape}  ({time.time() - t0:.0f}s)", flush=True)

        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
