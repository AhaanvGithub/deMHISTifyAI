#!/usr/bin/env python3
"""Score crypt finders against expert gland outlines: GlaS (Warwick; 165 colon H&E images, every gland outlined by
a pathologist), each image shrunk from 0.62 to MHIST's 2.0 um/px and placed in a 448 um field (224 px) the way an
MHIST tile is (centre crop if wider; glass beyond the image).

Per finder: gland detection precision / recall / F1 at IoU >= 0.5 (one-to-one), and mean IoU of matched outlines.
"interior" = glands cut by the image edge are don't-care (the classical finder skips them by design); "all" counts
them. Reported for healthy colon, adenoma (benign) and cancer images separately: healthy and adenoma are the
relevant ones for serrated polyps. The learned finder is scored out-of-patient (each image by the fold model that
never saw its patient). GlaS has no dilated / serrated tags, so labels are not scored here.

    python research/score_crypt_finders.py classical unet [sam]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypt_gt as GT  # noqa: E402
import train_gland_net as TG  # noqa: E402

F = 224


def glas_field(name):
    """GlaS image -> MHIST-sized field, ground-truth glands (outline 0..1) and their edge flags."""
    from skimage import measure
    im, an = TG.load(name)
    h, w = an.shape
    lo = (round(w * TG.MPP_GLAS / TG.MPP_MHIST), round(h * TG.MPP_GLAS / TG.MPP_MHIST))
    small = np.asarray(Image.fromarray(im).resize(lo, Image.LANCZOS))
    lab = np.asarray(Image.fromarray(an.astype(np.int32)).resize(lo, Image.NEAREST))
    W, H = lo
    c0, r0 = max(0, (W - F) // 2), max(0, (H - F) // 2)
    small, lab = small[r0:r0 + F, c0:c0 + F], lab[r0:r0 + F, c0:c0 + F]
    h2, w2 = lab.shape
    oy, ox = (F - h2) // 2, (F - w2) // 2
    field = np.full((F, F, 3), 240, np.uint8)
    field[oy:oy + h2, ox:ox + w2] = small
    full = np.zeros((F, F), np.int32)
    full[oy:oy + h2, ox:ox + w2] = lab
    gt, edge = [], []
    for v in np.unique(full):
        if v == 0:
            continue
        m = full == v
        if m.sum() < 20:              # slivers of a gland cut by the crop
            continue
        cs = measure.find_contours(np.pad(m, 1).astype(float), 0.5)
        c = max(cs, key=len) - 1
        gt.append({"outline": [[float(x) / F, float(y) / F] for y, x in c], "tags": []})
        rr, cc = np.nonzero(m)
        edge.append(bool(rr.min() <= oy + 1 or cc.min() <= ox + 1 or rr.max() >= oy + h2 - 2 or cc.max() >= ox + w2 - 2))
    return Image.fromarray(field), gt, edge


def finders(names):
    import crypt_finders as CF
    out = {}
    for n in names:
        if n == "unet":
            out[n] = CF.unet_out_of_patient
        else:
            out[n] = CF.FINDERS[n]
    return out


def main(names):
    g = TG.glas_table()
    fs = finders(names)
    rows = {n: [] for n in fs}
    for k, r in enumerate(g.itertuples(), 1):
        field, gt, edge = glas_field(r.name)
        for n, f in fs.items():
            pred = f(field, fold=r.fold) if n == "unet" else f(field)
            for scope in ("interior", "all"):
                s = GT.score_field(gt, pred, scope == "interior", edge)
                rows[n].append({**s, "scope": scope, "group": r.subtype if r.grade == "benign" else "cancer", "image": r.name})
        print(f"{k}/{len(g)} {r.name}", flush=True)
    res = {"what": "GlaS expert gland outlines at MHIST scale (2.0 um/px, 448 um field); detection at IoU >= 0.5",
           "groups": {"healthy": "benign, healthy colon", "adenomatous": "benign adenoma", "cancer": "adenocarcinoma"},
           "finders": {}}
    for n, rs in rows.items():
        d = pd.DataFrame(rs)
        res["finders"][n] = {}
        for scope in ("interior", "all"):
            for grp in ("healthy", "adenomatous", "cancer", "benign (healthy + adenoma)"):
                sel = d[(d.scope == scope) & ((d.group == grp) if grp in ("healthy", "adenomatous", "cancer") else (d.group != "cancer"))]
                s = GT.summarise(sel.to_dict("records"))
                res["finders"][n][f"{scope} / {grp}"] = {k: s[k] for k in ("fields", "gt_crypts", "found", "precision", "recall", "f1", "mean_iou_matched")}
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or ["classical"]
    res = main(names)
    out = Path(__file__).resolve().parent.parent / "docs/results" / f"crypt_finders_glas_{'_'.join(names)}.json"
    out.write_text(json.dumps(res, indent=2))
    for n, v in res["finders"].items():
        print(n)
        for k, s in v.items():
            print(f"  {k:38s} P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} IoU {s['mean_iou_matched']}  ({s['gt_crypts']} glands)")
