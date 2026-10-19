#!/usr/bin/env python3
"""Does the in-app pipeline lose accuracy compared with training-time evaluation?

Takes a fixed random sample of held-out MHIST test images, renders each the way a
researcher's on-screen snip could look, and runs the real backend Engine.analyze() on it
(auto zoom via backend/scale.py, tiling, tissue mask, raw + Macenko judges, full fp32):

    native        the 224 px MHIST PNG as stored (engine path, no resampling)
    x2_bicubic    upscaled x2 (bicubic)
    x3_bicubic    upscaled x3 (bicubic)
    screenshot    upscaled x1.5 (150% display scaling) then PNG encode/decode
    jpeg_q70      upscaled x2 then JPEG quality 70 encode/decode

The panel's P(SSA) (mean over the 8 judges and tissue tiles, as shown in the app) is
scored against the pathologists' majority label and SSA vote count. The training-time
reference is the 4-seed ensemble from research/finetune_conch.py on the SAME images
(data/models/conch_ft_seed*_testpred.npy), plus docs/results/finetune_seeds.json.

Usage: python research/eval_app_pipeline.py [n_images] [views,comma,separated] [extra_image,...]   (stop the app's backend first)
Output: docs/results/app_pipeline_eval.json
"""

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
from engine import Engine  # noqa: E402
from finetune_conch import alignment  # noqa: E402
from select_prompts import load_labels  # noqa: E402
from train_judges import ece  # noqa: E402

OUT = Path(os.environ.get("APP_EVAL_OUT", ROOT / "docs" / "results" / "app_pipeline_eval.json"))


def up(img, f):
    return img.resize((round(img.width * f), round(img.height * f)), Image.BICUBIC)


def roundtrip(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


VIEWS = {
    "native": lambda im: im,
    "x2_bicubic": lambda im: up(im, 2),
    "x3_bicubic": lambda im: up(im, 3),
    "screenshot": lambda im: roundtrip(up(im, 1.5), "PNG"),
    "jpeg_q70": lambda im: roundtrip(up(im, 2), "JPEG", quality=70),
}


def metrics(p, y, votes):
    return {"n": int(len(p)),
            "bal_acc": round(float(balanced_accuracy_score(y, p > .5)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "ece": round(ece(p, y), 4),
            "pathologist_alignment": alignment(p, votes, y)}


def panel_p_ssa(res):
    """Exact panel P(SSA): engine's mean_tile over tissue tiles (the % shown is this, rounded)."""
    t = [x for x in res["tiles"] if x["is_tissue"]]
    return float(np.mean([x["probs"]["SSA"] for x in t]))


def raw_judges_p_ssa(res):
    """Mean P(SSA) of the 4 raw-condition judges (1% rounding from the panel JSON)."""
    ps = [(j["top_conf"] if j["top"] == "SSA" else j["alt_conf"]) / 100 for j in res["judges"] if j["condition"] == "raw"]
    return float(np.mean(ps))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    lab = load_labels(ROOT / "data/mhist/annotations.csv")
    test_names = np.array(sorted(lab.loc[lab["split"] == "test", "image"]))
    lab = lab.set_index("image")
    pick = np.random.default_rng(0).choice(len(test_names), n, replace=False)
    extra = sys.argv[3].split(",") if len(sys.argv) > 3 else []  # e.g. the feature-check tiles
    pick = np.sort(np.unique(np.concatenate([pick, np.searchsorted(test_names, extra)]).astype(int)))
    n = len(pick)
    views = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VIEWS)
    names = test_names[pick]
    y = (lab.loc[names, "label"] == "SSA").to_numpy().astype(int)
    votes = lab.loc[names, "ssa_votes"].to_numpy(dtype=float)

    # training-time predictions on the same images (test preds are stored in sorted test-name order)
    seeds = sorted((ROOT / "data/models").glob("conch_ft_seed*_testpred.npy"))
    train_p = np.mean([np.load(s)[pick] for s in seeds], axis=0)

    eng = Engine()
    assert eng.ft, "fine-tuned judges not found in data/models"
    per_view, rows = {}, {k: [] for k in views}
    for view in views:
        fn = VIEWS[view]
        p_panel, p_raw, t0 = [], [], time.time()
        for i, name in enumerate(names):
            img = fn(Image.open(ROOT / "data/mhist/images" / name).convert("RGB"))
            res = eng.analyze(img)
            if "error" in res:  # no tissue found: app shows no answer; score as undecided
                pp = pr = 0.5
            else:
                pp, pr = panel_p_ssa(res), raw_judges_p_ssa(res)
            p_panel.append(pp)
            p_raw.append(pr)
            if os.environ.get("APP_EVAL_VERBOSE"):  # live per-image line: truth, raw-judge and panel calls, running accuracy
                ok = lambda q: "ok " if (q > .5) == y[i] else "XX "
                ar = np.mean([(q > .5) == t for q, t in zip(p_raw, y)])
                ap = np.mean([(q > .5) == t for q, t in zip(p_panel, y)])
                print(f"{view} {i + 1:4d}/{n} {name}  truth {'SSA' if y[i] else 'HP '} ({int(votes[i])}/7)  "
                      f"raw {pr:.2f} {ok(pr)} panel {pp:.2f} {ok(pp)}  running acc raw {ar:.3f} panel {ap:.3f}", flush=True)
            mag = res.get("magnification", {})
            rows[view].append({"image": name, "p_panel": round(pp, 4), "p_raw_judges": round(pr, 4),
                               "size": list(img.size), "tile_px": mag.get("tile_px"),
                               "zoom_est": mag.get("zoom_vs_training"),
                               "n_tiles": len(res["tiles"]), "error": res.get("error"),
                               "criteria": {k: v["score"] for k, v in res.get("criteria", {}).items()}})
        p_panel, p_raw = np.array(p_panel), np.array(p_raw)
        zooms = [r["zoom_est"] for r in rows[view] if r["zoom_est"] is not None]
        per_view[view] = {
            "panel_8_judges": metrics(p_panel, y, votes),
            "raw_judges_only": metrics(p_raw, y, votes),
            "mean_abs_shift_vs_training_p": round(float(np.abs(p_panel - train_p).mean()), 4),
            "label_flips_vs_training": int(((p_panel > .5) != (train_p > .5)).sum()),
            "no_tissue_errors": int(sum(r["error"] is not None for r in rows[view])),
            "zoom_estimate_median": round(float(np.median(zooms)), 2) if zooms else None,
            "zoom_estimate_iqr": [round(float(q), 2) for q in np.percentile(zooms, [25, 75])] if zooms else None,
            "seconds_per_image": round((time.time() - t0) / n, 3),
        }
        print(view, json.dumps(per_view[view]["panel_8_judges"]), flush=True)

    summary = json.loads((ROOT / "docs/results/finetune_seeds.json").read_text())
    out = {
        "what": "Engine.analyze() on rendered held-out MHIST test images vs training-time scoring",
        "n_images": n, "sample": "numpy default_rng(0) choice without replacement from the 977 MHIST test images, plus the feature-check tiles: " + ",".join(extra),
        "precision": "fp32", "device": eng.device,
        "views": {
            "native": "224px PNG as stored", "x2_bicubic": "bicubic x2", "x3_bicubic": "bicubic x3",
            "screenshot": "bicubic x1.5 then PNG round trip", "jpeg_q70": "bicubic x2 then JPEG q70 round trip"},
        "training_time_same_images": metrics(train_p, y, votes),
        "training_time_full_test_set": {"mean_auc_4_seeds": summary["mean_auc"], "std_auc": summary["std_auc"],
                                        "mean_bal_acc": summary["mean_bal_acc"],
                                        "ensemble_of_seeds": summary["ensemble_of_seeds"]},
        "app_pipeline": per_view,
        "per_image": rows,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
