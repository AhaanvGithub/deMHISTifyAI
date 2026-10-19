"""CONCH side of deMHISTifyAI: tiling, the 8 judges, heatmap and criteria scores.

Judges = 4 class-prompt sets x 2 image conditions (raw, Macenko colour-normalised).
Which 4 prompt sets are used comes from config/judges.json (picked on the MHIST
training split by research/select_prompts.py); until that exists we fall back to
the first 4 prompt ids for each class.
"""

import json
import os
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

from concurrent.futures import ThreadPoolExecutor

_POOL = ThreadPoolExecutor(max_workers=max(4, (os.cpu_count() or 8) - 2))

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
        # Fine-tuned CONCH versions (research/finetune_conch.py). If present they ARE the 8 judges:
        # 4 versions x {raw, colour-normalised}. Original CONCH stays for features/outlines.
        self.ft = self._load_finetuned()
        if self.ft:
            self.judges = [{"id": f"J{n + 1}", "seed": f["seed"], "version": i + 1, "condition": c}
                           for n, (i, f, c) in enumerate((i, f, c) for i, f in enumerate(self.ft) for c in CONDITIONS)]

    def _load_finetuned(self) -> List[dict]:
        """Each fine-tuned version only changed CONCH's last 2 blocks + pooling (+ a linear head),
        so we keep one shared copy of blocks 0-9 and a small branch per version."""
        import copy

        v = self.model.visual
        out = []
        for path in sorted((ROOT / "data" / "models").glob("conch_ft_seed*.pt")):
            ck = torch.load(path, map_location="cpu")
            branch = torch.nn.ModuleDict({
                "blocks": copy.deepcopy(v.trunk.blocks[10:]), "norm": copy.deepcopy(v.trunk.norm),
                "pool": copy.deepcopy(v.attn_pool_contrast), "ln": copy.deepcopy(v.ln_contrast)})
            rename = {"conch.visual.trunk.blocks.": "blocks.",  # sliced Sequential keeps indices 10, 11
                      "conch.visual.trunk.norm.": "norm.", "conch.visual.attn_pool_contrast.": "pool.",
                      "conch.visual.ln_contrast.": "ln."}
            sd = {}
            for k, val in ck["state"].items():
                for a, b in rename.items():
                    if k.startswith(a):
                        sd[b + k[len(a):]] = val
            missing, unexpected = branch.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"{path.name}: weights don't match CONCH ({len(missing)} missing, {len(unexpected)} unexpected)")
            cls = torch.nn.Linear(512, 1)
            cls.load_state_dict({"weight": ck["state"]["cls.weight"], "bias": ck["state"]["cls.bias"]})
            out.append({"seed": ck["seed"], "branch": branch.to(self.device).eval(), "cls": cls.to(self.device).eval(),
                        "T": float(ck["T"])})
        return out

    # ---------- encoders ----------
    @torch.no_grad()
    def _encode_texts(self, keys: List[str]) -> Dict[str, np.ndarray]:
        texts = [self.prompts[k] for k in keys]
        tok = conch_tokenize(self._tokenizer, texts).to(self.device)
        emb = self.model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return dict(zip(keys, emb.float().cpu().numpy()))

    @torch.no_grad()
    def encode(self, images: List[Image.Image], batch_size: int = 32):
        """One pass through CONCH's shared blocks 0-9, then branch:
        original blocks 10-11 -> text-aligned embedding (features, outlines, prompt judges);
        each fine-tuned version's blocks 10-11 -> P(SSA) for that judge.
        Returns (embeddings [N,512], ft_probs [N, n_versions] or None). Full fp32 on purpose."""
        v, t = self.model.visual, self.model.visual.trunk
        F = torch.nn.functional
        embs, fts = [], []
        tensors = list(_POOL.map(self.preprocess, images))  # CPU preprocessing is the bottleneck: parallelise it
        for i in range(0, len(images), batch_size):
            x = torch.stack(tensors[i : i + batch_size]).to(self.device)
            h = t.norm_pre(t.patch_drop(t._pos_embed(t.patch_embed(x))))
            for blk in t.blocks[:10]:
                h = blk(h)
            o = h
            for blk in t.blocks[10:]:
                o = blk(o)
            o = v.ln_contrast(v.attn_pool_contrast(t.norm(o))[:, 0]) @ v.proj_contrast
            embs.append(F.normalize(o, dim=-1).float().cpu().numpy())
            if self.ft:
                cols = []
                for f in self.ft:
                    b, q = f["branch"], h
                    for blk in b["blocks"]:
                        q = blk(q)
                    z = F.normalize(b["ln"](b["pool"](b["norm"](q))[:, 0]) @ v.proj_contrast, dim=-1)
                    cols.append(torch.sigmoid(f["cls"](z).squeeze(-1) * 10.0 / f["T"]))
                fts.append(torch.stack(cols, 1).float().cpu().numpy())
        emb = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
        return emb, (np.concatenate(fts) if fts else None)

    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        return self.encode(images, batch_size)[0]

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

    def _dense_heat(self, snip: Image.Image, tiles: List[Tile], sub: int = 4) -> dict:
        """Fine heatmap: slide a tile-sized window in steps of 1/sub of a tile, score
        every window with the raw-condition judges, and give each fine cell the mean
        of all windows covering it (up to sub*sub of them). Cells line up exactly
        with the major tile grid: each major tile = sub x sub cells."""
        w, h = snip.size
        rows, cols = max(t.row for t in tiles) + 1, max(t.col for t in tiles) + 1
        tw, th = w / cols, h / rows
        R, C = rows * sub, cols * sub
        cw, ch = w / C, h / R
        starts = [(r, c) for r in range(R - sub + 1) for c in range(C - sub + 1)]
        crops = [snip.crop((int(c * cw), int(r * ch), int(c * cw + tw), int(r * ch + th))) for r, c in starts]
        emb, ftp = self.encode(crops)
        if self.ft:  # fine-tuned judges: mean P(SSA) over versions on the raw image
            p_win = ftp.mean(axis=1)
        else:
            raw_judges = [n for n, j in enumerate(self.judges) if j["condition"] == "raw"]
            p_win = np.mean([self._judge_probs(n, emb, self.judges[n]["prompts"])[:, 0] for n in raw_judges], axis=0)

        crit_win = self._criteria(emb)  # [n_windows, n_criteria] feature presence per window
        acc, cnt = np.zeros((R, C)), np.zeros((R, C))
        cacc = np.zeros((len(self.criteria_keys), R, C))
        for (r, c), p, cv in zip(starts, p_win, crit_win):
            acc[r : r + sub, c : c + sub] += p
            cnt[r : r + sub, c : c + sub] += 1
            cacc[:, r : r + sub, c : c + sub] += cv[:, None, None]
        p_ssa = acc / np.maximum(cnt, 1)
        crit_maps = cacc / np.maximum(cnt, 1)[None]

        # stained-pixel fraction per fine cell (same rule as the tile tissue check)
        a = np.asarray(snip.resize((C * 8, R * 8)), dtype=np.float32) / 255.0
        stained = ((a.mean(2) < 0.85) & ((a.max(2) - a.min(2)) > 0.08)).reshape(R, 8, C, 8).mean(axis=(1, 3))
        return {"rows": R, "cols": C, "sub": sub, "windows": len(starts),
                "p_ssa": np.round(p_ssa, 4).ravel().tolist(), "tissue": np.round(stained, 3).ravel().tolist(),
                # per-cell presence (0..1) of each diagnostic feature, for outlining features on the image
                "features": {k.split(":", 1)[1]: np.round(crit_maps[i], 3).ravel().tolist()
                             for i, k in enumerate(self.criteria_keys)}}

    def refine_heat(self, snip: Image.Image, tile_px: int, sub: int = 8) -> dict:
        """Denser heatmap on the same major grid (called after the first answer is on screen)."""
        snip = snip.convert("RGB")
        return self._dense_heat(snip, self.make_grid(snip, tile_px), sub=sub)

    def analyze(self, snip: Image.Image, tile_px: Optional[int] = None) -> dict:
        snip = snip.convert("RGB")
        if tile_px is None:  # match the judges' training field of view (see scale.py)
            from scale import tile_px_for
            tile_px, mag = tile_px_for(snip)
        else:
            mag = {"auto": False, "tile_px": tile_px}
        tiles = self.make_grid(snip, tile_px)
        crops = [snip.crop(t.box) for t in tiles]
        emb = {
            "raw": self.encode(crops),
            "normalized": self.encode(list(_POOL.map(lambda c: view_macenko(c, None), crops))),
        }

        tissue = self._tissue(crops)
        for t, s in zip(tiles, tissue):
            t.tissue_score, t.is_tissue = float(s), bool(s >= 0.25)
        mask = np.array([t.is_tissue for t in tiles])
        if not mask.any():
            return {"error": "no_tissue", "tiles": [t.__dict__ for t in tiles]}

        judge_results, tile_probs = [], []
        for n, j in enumerate(self.judges):
            if self.ft:
                ps = emb[j["condition"]][1][:, j["version"] - 1]
                p = np.stack([ps, 1 - ps], axis=1)  # [N, C]
            else:
                p = self._judge_probs(n, emb[j["condition"]][0], j["prompts"])  # [N, C]
            tile_probs.append(p)
            snip_p = p[mask].mean(0)
            order = np.argsort(-snip_p)
            judge_results.append(
                {
                    "id": j["id"],
                    "condition": j["condition"],
                    "prompts": ({c: self.prompts[j["prompts"][c]] for c in CLASSES} if "prompts" in j
                                else {"SSA": f"fine-tuned CONCH v{j['version']}", "HP": f"seed {j['seed']}"}),
                    "top": CLASSES[order[0]],
                    "top_conf": round(100 * float(snip_p[order[0]])),
                    "alt": CLASSES[order[1]],
                    "alt_conf": round(100 * float(snip_p[order[1]])),
                }
            )
        mean_tile = np.mean(tile_probs, axis=0)  # [N, C]
        crit = self._criteria(emb["raw"][0])
        for i, t in enumerate(tiles):
            t.probs = {c: float(mean_tile[i, k]) for k, c in enumerate(CLASSES)}
            t.criteria = {k.split(":", 1)[1]: float(crit[i, n]) for n, k in enumerate(self.criteria_keys)}

        heat = self._dense_heat(snip, tiles)

        votes = {c: sum(r["top"] == c for r in judge_results) for c in CLASSES}
        panel_p = mean_tile[mask].mean(0)
        order = np.argsort(-panel_p)
        snip_crit = crit[mask].mean(0)
        return {
            "grid": {"rows": max(t.row for t in tiles) + 1, "cols": max(t.col for t in tiles) + 1},
            "tiles": [t.__dict__ for t in tiles],
            "heat": heat,
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
            "magnification": {**mag, "tile_px": tile_px},
            "calibrated": bool(self.calib),
            "judge_mode": "fine-tuned" if self.ft else ("trained" if self.heads is not None else "zero-shot"),
        }
