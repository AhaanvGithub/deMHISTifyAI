"""Hand-traced crypt ground truth: which MHIST fields, where the traces live, and how a crypt finder is scored.

Fields: 32 MHIST TRAINING images, 4 per number of SSL votes (0..7), none of the 600 images that set the gland
thresholds (research/gland_thresholds.py). Within each vote count 2 fields are "dev" (allowed for tuning a finder)
and 2 are "test" (scored only, never tuned on).

Traces: data/crypt_traces/traces.json, {image: {"crypts": [{"outline": [[x, y], ...] (0..1 of the field),
"tags": [...]}], "done": bool, "note": str}}. Tags: "dilated", "serrated", "round regular"; no tag = a crypt with
nothing clear.

Score (per finder, per split): crypt detection precision / recall / F1 at IoU >= 0.5 (one-to-one matching), mean IoU
of the matched outlines, and label agreement on the matched crypts. Crypts cut by the field edge are "don't care" in
the interior score (the finder skips them by design): missing them is not a miss and finding them is not a false hit.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAGS = ("dilated", "serrated", "round regular")
EDGE = 2 / 224          # an outline within 2 tile pixels of the border is cut by the field edge


def data_root() -> Path:
    """data/ of this checkout, or of the main checkout when running from a git worktree (data is not in git)."""
    if (ROOT / "data" / "mhist").exists():
        return ROOT / "data"
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (ROOT / common).resolve().parent / "data"


DATA = data_root()
TRACES = DATA / "crypt_traces" / "traces.json"


def fields() -> pd.DataFrame:
    ann = pd.read_csv(DATA / "mhist/annotations.csv")
    train = ann[ann.Partition == "train"]
    used = set(pd.concat([d.sample(min(len(d), 300), random_state=0)
                          for _, d in train.groupby("Majority Vote Label")])["Image Name"])
    pool = train[~train["Image Name"].isin(used)]
    vcol = "Number of Annotators who Selected SSA (Out of 7)"
    rows = []
    for v, d in pool.groupby(vcol):
        pick = d.sample(4, random_state=1)
        for k, name in enumerate(pick["Image Name"]):
            rows.append({"image": name, "votes": int(v), "split": "dev" if k < 2 else "test"})
    return pd.DataFrame(rows)


def load_traces() -> dict:
    return json.loads(TRACES.read_text()) if TRACES.exists() else {}


def main_label(tags) -> str:
    for t in TAGS:
        if t in tags:
            return t
    return "none"


def _mask(outline, size):
    from skimage.draw import polygon
    m = np.zeros((size, size), bool)
    p = np.asarray(outline, float) * size
    if len(p) >= 3:
        rr, cc = polygon(p[:, 1], p[:, 0], (size, size))
        m[rr, cc] = True
    return m


def at_edge(outline) -> bool:
    p = np.asarray(outline, float)
    return bool((p.min() <= EDGE) or (p.max() >= 1 - EDGE))


def match(gt: list, pred: list, size: int = 448, thr: float = 0.5):
    """One-to-one matching by IoU (Hungarian). Returns (pairs [(gi, pi, iou)], iou matrix)."""
    from scipy.optimize import linear_sum_assignment
    gm = [_mask(c["outline"], size) for c in gt]
    pm = [_mask(c["outline"], size) for c in pred]
    iou = np.zeros((len(gm), len(pm)))
    for i, a in enumerate(gm):
        for j, b in enumerate(pm):
            u = (a | b).sum()
            iou[i, j] = (a & b).sum() / u if u else 0.0
    pairs = []
    if iou.size:
        r, c = linear_sum_assignment(-iou)
        pairs = [(i, j, float(iou[i, j])) for i, j in zip(r, c) if iou[i, j] >= thr]
    return pairs, iou


def score_field(gt: list, pred: list, interior_only: bool, edge=None):
    """Counts for one field. pred items need "outline" and "tags". edge: per traced crypt, cut by the field edge
    (default: its outline touches the field border)."""
    pairs, iou = match(gt, pred)
    matched_g = {i for i, _, _ in pairs}
    matched_p = {j for _, j, _ in pairs}
    edge_g = list(edge) if edge is not None else [at_edge(c["outline"]) for c in gt]
    tp = fn = fp = 0
    ious, labs = [], []
    for i, j, v in pairs:
        if interior_only and edge_g[i]:
            continue
        tp += 1
        ious.append(v)
        labs.append((main_label(gt[i].get("tags", [])), main_label(pred[j].get("tags", [])),
                     set(gt[i].get("tags", [])), set(pred[j].get("tags", []))))
    for i in range(len(gt)):
        if i not in matched_g and not (interior_only and edge_g[i]):
            fn += 1
    for j in range(len(pred)):
        if j in matched_p:
            continue
        # a hit on an edge-cut crypt (or one that overlaps it most) is don't-care in the interior score
        if interior_only and len(gt) and iou[:, j].max() >= 0.5 and edge_g[int(iou[:, j].argmax())]:
            continue
        fp += 1
    return {"tp": tp, "fp": fp, "fn": fn, "ious": ious, "labels": labs}


def summarise(per_field: list) -> dict:
    tp = sum(f["tp"] for f in per_field)
    fp = sum(f["fp"] for f in per_field)
    fn = sum(f["fn"] for f in per_field)
    ious = [v for f in per_field for v in f["ious"]]
    labs = [x for f in per_field for x in f["labels"]]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"fields": len(per_field), "gt_crypts": tp + fn, "found": tp + fp, "tp": tp, "fp": fp, "fn": fn,
           "precision": round(prec, 3), "recall": round(rec, 3),
           "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
           "mean_iou_matched": round(float(np.mean(ious)), 3) if ious else None}
    if labs:
        from sklearn.metrics import cohen_kappa_score
        g = [a for a, _, _, _ in labs]
        p = [b for _, b, _, _ in labs]
        out["label_agreement"] = round(float(np.mean([a == b for a, b in zip(g, p)])), 3)
        out["label_kappa"] = round(float(cohen_kappa_score(g, p)), 3) if len(set(g) | set(p)) > 1 else None
        out["label_confusion"] = pd.crosstab(pd.Series(g, name="traced"), pd.Series(p, name="finder")).to_dict()
        tags = {}
        for t in TAGS:
            tg = np.array([t in a for _, _, a, _ in labs])
            tq = np.array([t in b for _, _, _, b in labs])
            tags[t] = {"traced": int(tg.sum()), "finder": int(tq.sum()), "both": int((tg & tq).sum())}
        out["tags_on_matched"] = tags
    return out
