"""Crypt finders under test. Each takes one field (PIL image) and returns crypts as dicts with "outline" (0..1 of
the field, [[x, y], ...]) and "tags" (from backend/glands.label). Only how outlines are found differs; measurement
and labelling stay those of backend/glands.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import glands as GL  # noqa: E402


def classical(field):
    return GL.label(GL.measure(field))


FINDERS = {"classical": classical}


_SAM = None


def _sam():
    global _SAM
    if _SAM is None:
        import torch
        from transformers import SamModel, SamProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _SAM = (SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval(),
                SamProcessor.from_pretrained("facebook/sam-vit-base"), dev)
    return _SAM


def sam(field):
    """SAM (ViT-base) prompted with one point at each classical crypt centre; per point the largest of SAM's
    three masks that is crypt-sized (not the whole tissue); overlapping masks keep the first."""
    import numpy as np
    import torch
    from PIL import Image
    model, proc, dev = _sam()
    f = field.convert("RGB").resize((224, 224), Image.LANCZOS)
    seeds = [g["center"] for g in GL.measure(f)]
    if not seeds:
        return []
    pts = [[[[x * 224, y * 224]] for x, y in seeds]]
    inp = proc(f, input_points=pts, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = model(**inp, multimask_output=True)
    masks = proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
    scores = o.iou_scores.cpu()[0]
    taken = np.zeros((224, 224), bool)
    keep = []
    for k in range(masks.shape[0]):
        best = None
        # SAM's three masks are part / larger part / whole; a crypt is the largest one that is still crypt-sized
        for j in sorted(range(masks.shape[1]), key=lambda j: -int(masks[k, j].sum())):
            m = masks[k, j].numpy()
            if 150 <= m.sum() <= 0.12 * 224 * 224:       # ~28 um to ~170 um across at 2 um/px
                best = m
                break
        if best is None or (best & taken).sum() > 0.3 * best.sum():
            continue
        best = best & ~taken
        taken |= best
        keep.append(best)
    return GL.label(GL.measure_masks(f, keep))


def unet_out_of_patient(field, fold):
    """U-Net trained without this image's half of the GlaS patients (for scoring on GlaS only)."""
    import torch
    import gland_net as GN
    key = f"unet{fold}"
    if key not in _NETS:
        net = GN.build(False)
        net.load_state_dict(torch.load(GN.WEIGHTS.with_name(f"gland_net_fold{fold}.pt"), map_location="cpu"))
        _NETS[key] = net.cuda().eval()
    return GL.label(GL.measure_masks(field, GN.outlines(field, _NETS[key], "cuda")), GL.thresholds("unet") or None)


def unet_measure(field):
    import gland_net as GN
    return GL.measure_masks(field, GN.outlines(field))


def unet(field):
    return GL.label(unet_measure(field), GL.thresholds("unet") or None)


def unet_g(field):
    """U-Net outlines, labels from the outline itself (basis "gland") instead of the pale interior."""
    return GL.label(unet_measure(field), GL.thresholds("unet_g") or None)


_NETS = {}
FINDERS.update(sam=sam, unet=unet, unet_g=unet_g)
MEASURERS = {"unet": unet_measure, "unet_g": unet_measure}
