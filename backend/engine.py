"""CONCH side of deMHISTifyAI: tiling, the 8 judges, heatmap and criteria scores.

Judges = 4 class-prompt sets x 2 image conditions (raw, Macenko colour-normalised).
Which 4 prompt sets are used comes from config/judges.json (picked on the MHIST
training split by research/select_prompts.py); until that exists we fall back to
the first 4 prompt ids for each class.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from embed_views import view_macenko  # noqa: E402  (same normalisation as the research embeddings)

PROMPTS_PATH = ROOT / "research" / "mhist_prompts.json"
JUDGES_PATH = ROOT / "backend" / "config" / "judges.json"
CALIB_PATH = ROOT / "backend" / "config" / "calibration.json"

CLASSES = ["SSA", "HP"]
CLASS_NAMES = {"SSA": "Sessile serrated lesion", "HP": "Hyperplastic polyp"}
CONDITIONS = ["raw", "normalized"]


def conch_tokenize(tokenizer, texts: List[str]) -> torch.Tensor:
    """Same as conch.open_clip_custom.tokenize, but works with transformers 5
    (which removed batch_encode_plus): 127 tokens, padded, plus one pad column."""
    ids = tokenizer(texts, max_length=127, add_special_tokens=True, return_token_type_ids=False,
                    truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
    return torch.nn.functional.pad(ids, (0, 1), value=tokenizer.pad_token_id)


@dataclass
class Tile:
    row: int
    col: int
    box: tuple  # (x0, y0, x1, y1) in snip pixels
    is_tissue: bool = True
    tissue_score: float = 1.0
    probs: Dict[str, float] = field(default_factory=dict)  # mean over judges
    criteria: Dict[str, float] = field(default_factory=dict)


class Engine:
    def __init__(self, device: Optional[str] = None):
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH"
        )
        self.model = self.model.to(self.device).eval()
        self._tokenizer = get_tokenizer()
        self._tokenize = tokenize
        self.logit_scale = float(self.model.logit_scale.exp().item())

        self.prompts: Dict[str, str] = json.loads(PROMPTS_PATH.read_text())
        self.text_emb = self._encode_texts(list(self.prompts))
        self.judges = self._load_judges()
        heads_path = ROOT / "backend" / "config" / "heads.npz"
        self.heads = dict(np.load(heads_path)) if heads_path.exists() else None  # trained by research/train_judges.py
        self.calib = json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else {}
        self.criteria_keys = [k for k in self.prompts if k.startswith("crit:")]

    # ---------- encoders ----------
    @torch.no_grad()
    def _encode_texts(self, keys: List[str]) -> Dict[str, np.ndarray]:
        texts = [self.prompts[k] for k in keys]
        tok = conch_tokenize(self._tokenizer, texts).to(self.device)
        emb = self.model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return dict(zip(keys, emb.float().cpu().numpy()))

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i : i + batch_size]]).to(self.device)
            out.append(self.model.encode_image(x, proj_contrast=True, normalize=True).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def _load_judges(self) -> List[dict]:
        if JUDGES_PATH.exists():
            sets = json.loads(JUDGES_PATH.read_text())["prompt_sets"]
        else:  # placeholder until prompt selection has run
            sets = [{c: f"class:{c}:{i:02d}" for c in CLASSES} for i in range(4)]
        return [
            {"id": f"J{n + 1}", "prompts": s, "condition": cond}
            for n, (s, cond) in enumerate((s, c) for s in sets for c in CONDITIONS)
        ]

    # ---------- tiling ----------
    @staticmethod
    def make_grid(img: Image.Image, tile_px: int = 224, max_tiles: int = 64) -> List[Tile]:
        w, h = img.size
        # shrink the tile grid if the snip is huge, grow tiles if it is tiny
        side = max(min(tile_px, w, h), 64)
        while (w // side) * (h // side) > max_tiles:
            side = int(side * 1.25)
        cols, rows = max(w // side, 1), max(h // side, 1)
        tw, th = w / cols, h / rows
        return [
            Tile(r, c, (int(c * tw), int(r * th), int((c + 1) * tw), int((r + 1) * th)))
            for r in range(rows)
            for c in range(cols)
        ]

    # ---------- scoring ----------
    def _probs(self, img_emb: np.ndarray, prompt_ids: Dict[str, str]) -> np.ndarray:
        """[N, len(CLASSES)] softmax over classes for one prompt set."""
        t = np.stack([self.text_emb[prompt_ids[c]] for c in CLASSES])
        logits = self.logit_scale * img_emb @ t.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def _judge_probs(self, n: int, img_emb: np.ndarray, prompt_ids: Dict[str, str]) -> np.ndarray:
        """Trained judge if heads.npz exists, else pure zero-shot. Columns follow CLASSES."""
        if self.heads is None:
            return self._probs(img_emb, prompt_ids)
        h = self.heads
        z = float(h["logit_scale"]) * (img_emb @ self.text_emb[prompt_ids["SSA"]] - img_emb @ self.text_emb[prompt_ids["HP"]])
        logit = (img_emb @ h["W"][n] + h["b"][n] + h["alpha"][n] * z) / h["T"][n]
        p_ssa = 1 / (1 + np.exp(-logit))
        return np.stack([p_ssa, 1 - p_ssa], axis=1)

    def _criteria(self, img_emb: np.ndarray) -> np.ndarray:
        """[N, n_criteria] presence scores in 0..1, z-scored against MHIST if calibrated."""
        t = np.stack([self.text_emb[k] for k in self.criteria_keys])
        sims = img_emb @ t.T
        mu = np.array([self.calib.get(k, {}).get("mu", np.nan) for k in self.criteria_keys])
        sd = np.array([self.calib.get(k, {}).get("sd", np.nan) for k in self.criteria_keys])
        if np.isnan(mu).any():  # uncalibrated: compare within this snip only
            mu, sd = sims.mean(0), sims.std(0) + 1e-6
        return 1 / (1 + np.exp(-(sims - mu) / sd))

    @staticmethod
    def _tissue(crops: List[Image.Image]) -> np.ndarray:
        """Fraction of stained (non-white, coloured) pixels per tile. The region
        finder already isolated the slide, so inside the box this only has to
        skip empty glass / white space. (CONCH text gating was unreliable here.)"""
        out = []
        for c in crops:
            a = np.asarray(c.resize((64, 64)), dtype=np.float32) / 255.0
            stained = (a.mean(2) < 0.85) & ((a.max(2) - a.min(2)) > 0.08)
            out.append(float(stained.mean()))
        return np.array(out)

    def analyze(self, snip: Image.Image, tile_px: int = 224) -> dict:
        snip = snip.convert("RGB")
        tiles = self.make_grid(snip, tile_px)
        crops = [snip.crop(t.box) for t in tiles]
        emb = {
            "raw": self.encode_images(crops),
            "normalized": self.encode_images([view_macenko(c, None) for c in crops]),
        }

        tissue = self._tissue(crops)
        for t, s in zip(tiles, tissue):
            t.tissue_score, t.is_tissue = float(s), bool(s >= 0.25)
        mask = np.array([t.is_tissue for t in tiles])
        if not mask.any():
            return {"error": "no_tissue", "tiles": [t.__dict__ for t in tiles]}

        judge_results, tile_probs = [], []
        for n, j in enumerate(self.judges):
            p = self._judge_probs(n, emb[j["condition"]], j["prompts"])  # [N, C]
            tile_probs.append(p)
            snip_p = p[mask].mean(0)
            order = np.argsort(-snip_p)
            judge_results.append(
                {
                    "id": j["id"],
                    "condition": j["condition"],
                    "prompts": {c: self.prompts[j["prompts"][c]] for c in CLASSES},
                    "top": CLASSES[order[0]],
                    "top_conf": round(100 * float(snip_p[order[0]])),
                    "alt": CLASSES[order[1]],
                    "alt_conf": round(100 * float(snip_p[order[1]])),
                }
            )
        mean_tile = np.mean(tile_probs, axis=0)  # [N, C]
        crit = self._criteria(emb["raw"])
        for i, t in enumerate(tiles):
            t.probs = {c: float(mean_tile[i, k]) for k, c in enumerate(CLASSES)}
            t.criteria = {k.split(":", 1)[1]: float(crit[i, n]) for n, k in enumerate(self.criteria_keys)}

        votes = {c: sum(r["top"] == c for r in judge_results) for c in CLASSES}
        panel_p = mean_tile[mask].mean(0)
        order = np.argsort(-panel_p)
        snip_crit = crit[mask].mean(0)
        return {
            "grid": {"rows": max(t.row for t in tiles) + 1, "cols": max(t.col for t in tiles) + 1},
            "tiles": [t.__dict__ for t in tiles],
            "judges": judge_results,
            "panel": {
                "votes": votes,
                "top": CLASSES[order[0]],
                "top_conf": round(100 * float(panel_p[order[0]])),
                "alt": CLASSES[order[1]],
                "alt_conf": round(100 * float(panel_p[order[1]])),
                "agreement": max(votes.values()) / len(judge_results),
            },
            "criteria": {
                k.split(":", 1)[1]: {"text": self.prompts[k], "score": round(float(snip_crit[n]), 3)}
                for n, k in enumerate(self.criteria_keys)
            },
            "calibrated": bool(self.calib),
            "judge_mode": "trained" if self.heads is not None else "zero-shot",
        }
