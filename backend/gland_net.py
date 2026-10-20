"""Learned gland outliner: a U-Net (ImageNet ConvNeXt-Tiny encoder) trained on the GlaS colon gland outlines
(Warwick, expert-annotated) shrunk to MHIST's ~2 um/px (research/train_gland_net.py).

Any field (raw tile or screen snip) is first brought to MHIST's own size, 224 px for 448 um, so a snip and a tile
carry the same detail, then enlarged 2x for the network (1 um/px). The network marks each pixel as stroma, gland,
or gland border; glands are the gland pixels split along the borders, then grown back to the full gland.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

FIELD_PX = 224                 # one 448 um field at MHIST's ~2 um/px
UP = 2                         # the network works at 1 um/px
MIN_GLAND_UM = 20              # smaller blobs are not crypts
# shown outlines: fewer and right beats many and shaky. On GlaS (benign, out-of-patient) these lift precision
# 0.91 -> 0.94 for recall 0.88 -> 0.85; on MHIST they mainly hide two touching crypts merged into a U or W shape
# (solidity ~0.73 on MHIST_btr, where single crypts are >= 0.84)
MIN_AREA_UM2 = 600             # ~28 um across
MIN_SOLIDITY = 0.8
WEIGHTS = Path(__file__).resolve().parent.parent / "data" / "models" / "gland_net.pt"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build(pretrained: bool = False):
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Up(nn.Module):
        def __init__(self, cin, cskip, cout):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                      nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.conv(torch.cat([x, skip], 1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=pretrained, features_only=True)
            c = self.enc.feature_info.channels()          # strides 4, 8, 16, 32
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.u3 = Up(c[3], c[2], 256)
            self.u2 = Up(256, c[1], 128)
            self.u1 = Up(128, c[0], 64)
            self.u0 = Up(64, 32, 32)
            self.head = nn.Conv2d(32, 3, 1)                # stroma / gland / gland border

        def forward(self, x):
            f = self.enc(x)
            y = self.u3(f[3], f[2])
            y = self.u2(y, f[1])
            y = self.u1(y, f[0])
            y = self.u0(y, self.stem(x))
            return self.head(y)

    return Net()


def to_field(img: Image.Image) -> Image.Image:
    """Bring any field to MHIST's size (224 px per 448 um), then to the network's 1 um/px."""
    f = img.convert("RGB")
    if f.size != (FIELD_PX, FIELD_PX):
        f = f.resize((FIELD_PX, FIELD_PX), Image.LANCZOS)
    return f.resize((FIELD_PX * UP, FIELD_PX * UP), Image.BICUBIC)


def instances(prob: np.ndarray, um_per_px: float) -> np.ndarray:
    """Per-pixel probabilities (3, H, W) -> labelled glands. Seeds = gland minus border, grown back over the gland."""
    from scipy import ndimage as ndi
    from skimage import measure, morphology, segmentation

    gland = (prob[1] + prob[2]) > 0.5
    seeds = (prob[1] > 0.5) & (prob[2] < 0.35)
    seeds = morphology.opening(seeds, morphology.disk(2))
    min_area = np.pi * (MIN_GLAND_UM / um_per_px / 2) ** 2
    seeds = morphology.remove_small_objects(seeds, max_size=int(min_area * 0.25))
    markers = measure.label(seeds)
    lab = segmentation.watershed(-prob[1], markers, mask=gland)
    out = np.zeros_like(lab)
    k = 0
    for p in measure.regionprops(lab):
        if p.area < min_area:
            continue
        k += 1
        m = ndi.binary_fill_holes(lab == p.label)
        out[m & (out == 0)] = k
    return out


_NET = None


def _net():
    global _NET
    if _NET is None:
        import torch
        net = build(False)
        net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _NET = (net.to(dev).eval(), dev)
    return _NET


def predict(img: Image.Image, net=None, device: Optional[str] = None, tta: bool = True) -> np.ndarray:
    """Class probabilities (3, 448, 448) for one field, averaged over the 8 flips/rotations when tta."""
    import torch
    if net is None:
        net, device = _net()
    a = (np.asarray(to_field(img), np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(device)
    acc = 0
    views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    with torch.no_grad():
        for k, f in views:
            v = torch.rot90(x, k, (2, 3))
            v = torch.flip(v, (3,)) if f else v
            p = torch.softmax(net(v).float(), 1)
            p = torch.flip(p, (3,)) if f else p
            acc = acc + torch.rot90(p, -k, (2, 3))
    return (acc / len(views))[0].cpu().numpy()


def outlines(img: Image.Image, net=None, device=None) -> List[np.ndarray]:
    """Gland masks for one field at MHIST size (224 x 224 bool each), without implausible shapes."""
    from skimage.measure import regionprops
    from skimage.transform import resize
    prob = predict(img, net, device)
    lab = instances(prob, 448 / (FIELD_PX * UP))
    um2 = (448 / FIELD_PX) ** 2
    masks = []
    for i in range(1, lab.max() + 1):
        m = resize((lab == i).astype(np.float32), (FIELD_PX, FIELD_PX), order=1, anti_aliasing=True) >= 0.5
        if m.sum() * um2 >= MIN_AREA_UM2 and regionprops(m.astype(np.uint8))[0].solidity >= MIN_SOLIDITY:
            masks.append(m)
    return masks
