#!/usr/bin/env python3
"""Tiny vs Base encoder for the gland outliner, on the same unseen GlaS patients: both trained on one patient half
(train_gland_net.py fold0 [--enc base]) and scored on the other half's images, with the app's outline rules.

    python research/compare_gland_nets.py   ->  docs/results/gland_net_tiny_vs_base.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import crypt_gt as GT  # noqa: E402
import gland_net as GN  # noqa: E402
import score_crypt_finders as S  # noqa: E402
import train_gland_net as TG  # noqa: E402
import glands as GL  # noqa: E402

g = TG.glas_table()
g = g[g.fold == 0]
res = {"what": __doc__.split("\n")[0], "images": len(g)}
for name, enc, path in (("tiny", TG.ENCODERS["tiny"], "gland_net_fold0.pt"), ("base", TG.ENCODERS["base"], "gland_net_base_fold0.pt")):
    net = GN.build(False, enc)
    net.load_state_dict(torch.load(GN.WEIGHTS.with_name(path), map_location="cpu"))
    net = net.cuda().eval()
    rows = []
    for r in g.itertuples():
        field, gt, edge = S.glas_field(r.name)
        pred = GL.measure_masks(field, GN.outlines(field, net, "cuda"))
        grp = r.subtype if r.grade == "benign" else "cancer"
        for scope in ("interior", "all"):
            rows.append({**GT.score_field(gt, pred, scope == "interior", edge), "scope": scope, "group": grp})
    res[name] = {}
    for scope in ("interior", "all"):
        for grp in ("benign", "healthy", "adenomatous", "cancer"):
            sel = [x for x in rows if x["scope"] == scope and (x["group"] != "cancer" if grp == "benign" else x["group"] == grp)]
            s = GT.summarise(sel)
            res[name][f"{scope} / {grp}"] = {k: s[k] for k in ("gt_crypts", "precision", "recall", "f1", "mean_iou_matched")}
    del net
    torch.cuda.empty_cache()
out = Path(__file__).resolve().parent.parent / "docs/results/gland_net_tiny_vs_base.json"
out.write_text(json.dumps(res, indent=2))
for k in res["tiny"]:
    a, b = res["tiny"][k], res["base"][k]
    print(f"{k:24s} tiny F1 {a['f1']:.3f} IoU {a['mean_iou_matched']}  |  base F1 {b['f1']:.3f} IoU {b['mean_iou_matched']}  ({a['gt_crypts']} glands)")
