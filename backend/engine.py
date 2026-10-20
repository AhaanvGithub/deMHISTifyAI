"""Image side of deMHISTifyAI: tiling, the image judges, heatmap and criteria scores.

Judges = the chosen panel (config/panel_deploy.json, written by research/export_panel.py): fine-tuned
ConvNeXt-Base + MUSK (LoRA), combined with learned weights and calibrated on MHIST training images. CONCH
(original weights) still provides the feature scores and outlines. Without panel_deploy.json the older
fine-tuned CONCH judges are used.

Older notes:

Judges = 4 class-prompt sets x 2 image conditions (raw, Macenko colour-normalised).
Which 4 prompt sets are used comes from config/judges.json (picked on the MHIST
training split by research/select_prompts.py); until that exists we fall back to
the first 4 prompt ids for each class.
"""

import json
import os
import sys

# _POOL runs Macenko (numpy linalg) on many tiles at once while OpenBLAS spawns its own threads per
# call; on Windows that crashes with an access violation once tiles reach ~444 px. The pool is the
# parallelism, so keep OpenBLAS single-threaded. Must be set before numpy loads (threadpoolctl at
# runtime did not stop the crash).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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
PANEL_PATH = ROOT / "backend" / "config" / "panel_deploy.json"
OOD_FAR = 0.65  # unlike-training distance beyond which the field is not colon tissue at all (see Engine._ood)
OOD_REF_PATH = ROOT / "data" / "models" / "cv" / "ood_ref_convnext_base.npz"
os.environ.setdefault("MUSK_SRC", str(Path(os.environ.get("TEMP", "")) / "claude" / "musk_src" / "MUSK"))
MEMBER_INFO = {  # how each panel member is shown in the app
    "convnext_base": {"id": "ConvNeXt", "about": "ConvNeXt-Base (ImageNet CNN), fine-tuned on MHIST"},
    "musk_lora": {"id": "MUSK", "about": "MUSK pathology model, fine-tuned on MHIST (LoRA)"},
}

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
        # The chosen panel (if exported) replaces the older fine-tuned CONCH judges.
        self.panel = self._load_panel()
        # Fine-tuned CONCH versions (research/finetune_conch.py). If present they ARE the 8 judges:
        # 4 versions x {raw, colour-normalised}. Original CONCH stays for features/outlines.
        self.ft = [] if self.panel else self._load_finetuned()
        if self.panel:
            self.judges = [{"id": MEMBER_INFO.get(t, {"id": t})["id"], "tag": t, "condition": "raw"} for t in self.panel["members"]]
        elif self.ft:
            self.judges = [{"id": f"J{n + 1}", "seed": f["seed"], "version": i + 1, "condition": c}
                           for n, (i, f, c) in enumerate((i, f, c) for i, f in enumerate(self.ft) for c in CONDITIONS)]

    def _load_panel(self) -> Optional[dict]:
        """Load the panel members' final weights (trained on all 2,175 MHIST training images)."""
        if not PANEL_PATH.exists():
            return None
        from kfold_cv import ConvNextBaseFT, MuskLoRAFT
        cfg = json.loads(PANEL_PATH.read_text())
        kinds = {"convnext_base": ConvNextBaseFT, "musk_lora": MuskLoRAFT}
        nets = {}
        for tag in cfg["members"]:
            net = kinds[tag](-1)
            ck = torch.load(ROOT / "data" / "models" / "cv" / tag / "final.pt", map_location="cpu")
            net.load_state_dict(ck["state"], strict=False)  # LoRA checkpoints hold only the trained weights
            nets[tag] = net.to(self.device).eval()
        cfg["nets"] = nets
        if OOD_REF_PATH.exists():
            ref = np.load(OOD_REF_PATH)
            cfg["ood_ref"] = torch.from_numpy(ref["feats"].astype(np.float32)).to(self.device)
        return cfg

    @torch.no_grad()
    def _panel_logits(self, images: List[Image.Image], batch_size: int = 16):
        """Each member's logit per image, plus the ConvNeXt features (for the unlike-training-data check)."""
        F = torch.nn.functional
        dev_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        logits, feats = {}, []
        for tag, net in self.panel["nets"].items():
            tensors = list(_POOL.map(net.pre, images))
            out = []
            for i in range(0, len(tensors), batch_size):
                x = torch.stack(tensors[i : i + batch_size]).to(self.device)
                with torch.autocast(dev_type, dtype=torch.bfloat16):
                    if tag.startswith("musk"):
                        z = net.m(image=x, with_head=False, out_norm=False, ms_aug=False, return_global=True)[0]
                    else:
                        z = net.m(x)
                z = F.normalize(z.float(), dim=-1)
                out.append((net.cls(z).squeeze(-1) * 10.0).float().cpu().numpy())
                if tag == "convnext_base":
                    feats.append(z)
            logits[tag] = np.concatenate(out)
        return logits, (torch.cat(feats) if feats else None)

    def _panel_score(self, logits: Dict[str, np.ndarray]) -> np.ndarray:
        """Combined score (standardised member logits x learned weights), same as research/panel.py."""
        m = self.panel["members"]
        return self.panel["intercept"] + sum(m[t]["weight"] * (logits[t] - m[t]["mu"]) / m[t]["sd"] for t in m)

    def _shown(self, z):
        """Displayed P(SSL): calibrated to the 7-pathologist majority call on MHIST training images."""
        c = self.panel["calibration"]["shown"]
        return 1 / (1 + np.exp(-(c["coef"] * np.asarray(z) + c["intercept"])))

    def _ood(self, feats) -> Optional[dict]:
        """Unlike-training-data check: distance to the 5 nearest MHIST training images (ConvNeXt features)."""
        if feats is None or "ood_ref" not in self.panel:
            return None
        o = self.panel["ood"]
        sims = feats.float() @ self.panel["ood_ref"].T
        d = float((1 - sims.topk(o["k"], dim=1).values.mean(1)).mean())
        # "far": beyond anything real tissue reached. Measured on outside data: UniToPatho colon tissue (another lab)
        # reached 0.57 at most; photos and noise 0.71+; a brown immunostain 0.55 (tissue, wrong stain: a caution only)
        return {"distance": round(d, 4), "limit": o["limit"], "flag": d > o["limit"], "far": d > OOD_FAR}

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
        if self.panel:  # cover-up evidence map (ConvNeXt) + finer CONCH feature map, at the trained field size
            return self._evidence_heat(snip, tiles, G=sub if sub in self.LEVELS else 16)
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

    def analyze(self, snip: Image.Image, tile_px: Optional[int] = None, cut: Optional[float] = None) -> dict:
        """`cut`: the app's decision cut-off on the displayed P(SSL) scale (default: the balanced one)."""
        snip = snip.convert("RGB")
        if tile_px is None:
            # The researcher snips one field, framed like an MHIST image: score the whole snip as one
            # image (CONCH resizes it to 224). A long snip is split into squares along its long side,
            # because CONCH's preprocessing centre-crops and would silently drop the ends.
            # (Nucleus-size auto zoom, scale.py, is no longer used: it read anything up to ~1.5x as 1x
            # and multi-tile snips as up to 3x too zoomed.)
            tile_px = min(snip.size)
            mag = {"auto": False, "mode": "whole snip = one field", "tile_px": tile_px}
        else:
            mag = {"auto": False, "tile_px": tile_px}
        tiles = self.make_grid(snip, tile_px)
        crops = [snip.crop(t.box) for t in tiles]
        emb = {"raw": self.encode(crops)}
        if not self.panel:  # colour-normalised checks only for the older CONCH judges (the panel uses raw images)
            emb["normalized"] = self.encode(list(_POOL.map(lambda c: view_macenko(c, None), crops)))

        tissue = self._tissue(crops)
        for t, s in zip(tiles, tissue):
            t.tissue_score, t.is_tissue = float(s), bool(s >= 0.25)
        mask = np.array([t.is_tissue for t in tiles])
        if not mask.any():
            return {"error": "no_tissue", "tiles": [t.__dict__ for t in tiles]}

        if self.panel:
            return self._analyze_panel(snip, tiles, crops, mask, emb, mag, tile_px, cut)

        judge_results, tile_probs = [], []
        # The fine-tuned versions only ever saw raw images. Fed Macenko-normalised tiles they drift to HP
        # (977 MHIST test images: panel SSA sensitivity 54% with them vs 79% raw-only, which also exactly
        # matches training-time scores; docs/results/app_pipeline_eval_full_native.json). So only the raw
        # judges vote; the normalised ones are still reported as a colour-robustness check.
        votes_on = lambda j: j["condition"] == "raw" or not self.ft
        for n, j in enumerate(self.judges):
            if self.ft:
                ps = emb[j["condition"]][1][:, j["version"] - 1]
                p = np.stack([ps, 1 - ps], axis=1)  # [N, C]
            else:
                p = self._judge_probs(n, emb[j["condition"]][0], j["prompts"])  # [N, C]
            if votes_on(j):
                tile_probs.append(p)
            snip_p = p[mask].mean(0)
            order = np.argsort(-snip_p)
            judge_results.append(
                {
                    "id": j["id"],
                    "condition": j["condition"],
                    "votes": votes_on(j),
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

        voters = [r for r in judge_results if r["votes"]]
        votes = {c: sum(r["top"] == c for r in voters) for c in CLASSES}
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
                "agreement": max(votes.values()) / len(voters),
                "n_voters": len(voters),
            },
            "criteria": {
                k.split(":", 1)[1]: {"text": self.prompts[k], "score": round(float(snip_crit[n]), 3),
                                     # measured on MHIST train: can CONCH actually read this feature? (research/ calibration)
                                     "reliability": self.calib.get(k, {}).get("reliability"),
                                     "usable": self.calib.get(k, {}).get("usable", False),
                                     "points_to": self.calib.get(k, {}).get("points_to")}
                for n, k in enumerate(self.criteria_keys)
            },
            "magnification": {**mag, "tile_px": tile_px},
            "calibrated": bool(self.calib),
            "judge_mode": "fine-tuned" if self.ft else ("trained" if self.heads is not None else "zero-shot"),
        }

    def _analyze_panel(self, snip, tiles, crops, mask, emb, mag, tile_px, cut) -> dict:
        cut = float(self.panel["shown_cut_off"] if cut is None else cut)
        logits, feats = self._panel_logits(crops)
        z = self._panel_score(logits)
        p_tile = self._shown(z)
        p_snip = float(self._shown(z[mask].mean()))
        call = lambda p: "SSA" if p > cut else "HP"
        judge_results = []
        for j in self.judges:
            c = self.panel["members"][j["tag"]]["shown"]
            p = float(1 / (1 + np.exp(-(c["coef"] * logits[j["tag"]][mask].mean() + c["intercept"]))))
            top = call(p)
            judge_results.append({
                "id": j["id"], "condition": "raw", "votes": True,
                "prompts": {"SSA": MEMBER_INFO.get(j["tag"], {}).get("about", j["tag"]),
                            "HP": f"weight in panel {self.panel['members'][j['tag']]['weight']:.2f}"},
                "top": top, "top_conf": round(100 * (p if top == "SSA" else 1 - p)),
                "alt": "HP" if top == "SSA" else "SSA", "alt_conf": round(100 * (1 - p if top == "SSA" else p)),
                "p_ssl": round(p, 4)})
        crit = self._criteria(emb["raw"][0])
        for i, t in enumerate(tiles):
            t.probs = {"SSA": float(p_tile[i]), "HP": float(1 - p_tile[i])}
            t.criteria = {k.split(":", 1)[1]: float(crit[i, n]) for n, k in enumerate(self.criteria_keys)}
        heat = self._dense_heat(snip, tiles)
        logits_ood = self._ood(feats[torch.from_numpy(mask).to(feats.device)] if feats is not None else None)
        # nothing to measure on a photo or noise: crypt outlines there would be shapes in random texture
        glands = [] if (logits_ood and logits_ood["far"]) else self._glands(snip, tiles, heat)
        votes = {c: sum(r["top"] == c for r in judge_results) for c in CLASSES}
        top = call(p_snip)
        snip_crit = crit[mask].mean(0)
        return {
            "grid": {"rows": max(t.row for t in tiles) + 1, "cols": max(t.col for t in tiles) + 1},
            "tiles": [t.__dict__ for t in tiles],
            "heat": heat,
            "judges": judge_results,
            "panel": {
                "votes": votes, "top": top,
                "top_conf": round(100 * (p_snip if top == "SSA" else 1 - p_snip)),
                "alt": "HP" if top == "SSA" else "SSA",
                "alt_conf": round(100 * (1 - p_snip if top == "SSA" else p_snip)),
                "agreement": max(votes.values()) / len(judge_results), "n_voters": len(judge_results),
                "p_ssl": round(p_snip, 4), "cut": round(cut, 4), "default_cut": self.panel["shown_cut_off"],
            },
            "ood": logits_ood,
            "glands": glands,
            "criteria": {
                k.split(":", 1)[1]: {"text": self.prompts[k], "score": round(float(snip_crit[n]), 3),
                                     "reliability": self.calib.get(k, {}).get("reliability"),
                                     "usable": self.calib.get(k, {}).get("usable", False),
                                     "points_to": self.calib.get(k, {}).get("points_to")}
                for n, k in enumerate(self.criteria_keys)
            },
            "magnification": {**mag, "tile_px": tile_px},
            "calibrated": bool(self.calib),
            "judge_mode": "panel",
        }

    # ---------- "where": cover-up (occlusion) evidence map, refined over time ----------
    # cells per field side -> (covered window in cells, CONCH feature window in cells, feature window step).
    # ConvNeXt sees the central 87.5% of a field (its centre crop, as in training) = the inner 14/16 of the grid;
    # the cover-up window stays ~30-40 px of the model's 224 px input (~60-80 um on an MHIST-scale field).
    LEVELS = {16: (3, 8, 2), 32: (5, 16, 2), 64: (8, 16, 4), 128: (16, 32, 8)}
    OCC_STEP = {16: 1, 32: 1, 64: 2, 128: 4}  # finer levels skip positions (~4x/16x faster); windows still overlap every cell

    @torch.no_grad()
    def _occlusion(self, field: Image.Image, G: int) -> np.ndarray:
        """[G, G] change in ConvNeXt's logit when each region is blurred out: > 0 = the region pushed the call
        toward SSL, < 0 = toward HP. The field is scored exactly as in training (one field, centre crop)."""
        from PIL import ImageFilter
        net = self.panel["nets"]["convnext_base"]
        dev_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        occ = self.LEVELS[G][0]
        n = G * 14 // 16; c = 224 // n; b0 = (G - n) // 2                    # inner cells, px per cell, border
        x = net.pre(field).to(self.device)                                   # [3, 224, 224]
        blur = net.pre(field.filter(ImageFilter.GaussianBlur(radius=max(field.size) / 40))).to(self.device)
        st = self.OCC_STEP[G]
        last = n - occ
        grid = sorted(set(list(range(0, last + 1, st)) + [last]))            # include the last row/column
        pos = [(r, q) for r in grid for q in grid]
        score = lambda t: (net.cls(torch.nn.functional.normalize(net.m(t).float(), dim=-1)).squeeze(-1) * 10.0).float()
        with torch.autocast(dev_type, dtype=torch.bfloat16):
            base = float(score(x.unsqueeze(0))[0])
        delta = []
        for i in range(0, len(pos), 64):                                     # build covered copies chunk by chunk
            chunk = pos[i : i + 64]
            batch = x.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
            for j, (r, q) in enumerate(chunk):
                sl = (slice(None), slice(r * c, (r + occ) * c), slice(q * c, (q + occ) * c))
                batch[j][sl] = blur[sl]
            with torch.autocast(dev_type, dtype=torch.bfloat16):
                delta.append(base - score(batch).cpu())
        delta = torch.cat(delta).numpy()
        acc, cnt = np.zeros((n, n)), np.zeros((n, n))
        for d, (r, q) in zip(delta, pos):
            acc[r : r + occ, q : q + occ] += d
            cnt[r : r + occ, q : q + occ] += 1
        full = np.zeros((G, G))
        full[b0 : b0 + n, b0 : b0 + n] = acc / np.maximum(cnt, 1)            # outer ring: not seen by the model
        return full

    def _feature_maps(self, field: Image.Image, G: int) -> np.ndarray:
        """[n_criteria, G, G] CONCH feature presence from sliding windows smaller than the field."""
        _, win, step = self.LEVELS[G]
        w, h = field.size
        cw, ch = w / G, h / G
        starts = [(r, q) for r in range(0, G - win + 1, step) for q in range(0, G - win + 1, step)]
        crops = [field.crop((int(q * cw), int(r * ch), int((q + win) * cw), int((r + win) * ch))) for r, q in starts]
        crit = self._criteria(self.encode(crops)[0])                         # [n_windows, n_criteria]
        acc, cnt = np.zeros((len(self.criteria_keys), G, G)), np.zeros((G, G))
        for (r, q), cv in zip(starts, crit):
            acc[:, r : r + win, q : q + win] += cv[:, None, None]
            cnt[r : r + win, q : q + win] += 1
        return acc / np.maximum(cnt, 1)[None]

    def _evidence_heat(self, snip: Image.Image, tiles: List[Tile], G: int = 16) -> dict:
        """Heatmap for the panel: per field, where covering the tissue changes the call (cover-up / occlusion map
        from ConvNeXt, the panel's main member), plus CONCH feature maps, on a G x G grid per field (16 first,
        then 32 and 64 while the literature judge runs). p_ssa holds 0.5 + evidence (0.5 = no effect,
        > 0.5 = pushed toward SSL), scaled to the strongest regions."""
        rows, cols = max(t.row for t in tiles) + 1, max(t.col for t in tiles) + 1
        R, C = rows * G, cols * G
        ev = np.zeros((R, C))
        feats = np.zeros((len(self.criteria_keys), R, C))
        for t in tiles:
            field = snip.crop(t.box)
            r0, c0 = t.row * G, t.col * G
            ev[r0 : r0 + G, c0 : c0 + G] = self._occlusion(field, G)
            feats[:, r0 : r0 + G, c0 : c0 + G] = self._feature_maps(field, G)
        k = max(1, 256 // G)                                                 # px per cell for the tissue check
        a = np.asarray(snip.resize((C * k, R * k)), dtype=np.float32) / 255.0
        stained = ((a.mean(2) < 0.85) & ((a.max(2) - a.min(2)) > 0.08)).reshape(R, k, C, k).mean(axis=(1, 3))
        scale = float(np.percentile(np.abs(ev[stained >= 0.25]), 98)) if (stained >= 0.25).any() else 0.0
        p = 0.5 + 0.5 * np.tanh(ev / scale) if scale > 1e-6 else np.full_like(ev, 0.5)
        return {"rows": R, "cols": C, "sub": G, "kind": "evidence", "level": G,
                "p_ssa": np.round(p, 3).ravel().tolist(), "tissue": np.round(stained, 2).ravel().tolist(),
                "features": {key.split(":", 1)[1]: np.round(feats[i], 2).ravel().tolist()
                             for i, key in enumerate(self.criteria_keys)}}

    @staticmethod
    def _glands(snip: Image.Image, tiles: List[Tile], heat: dict) -> List[dict]:
        """Crypts found and measured per field (backend/glands.py), in 0..1 coordinates of the whole snip, each
        with the cover-up evidence under it (> 0 = that region pushed the call toward SSL)."""
        import glands as GL
        w, h = snip.size
        ev = (np.array(heat["p_ssa"]).reshape(heat["rows"], heat["cols"]) - 0.5) * 2
        out = []
        for t in tiles:
            x0, y0, x1, y1 = t.box
            for g in GL.label(GL.measure(snip.crop(t.box))):
                to_snip = lambda x, y: [round((x0 + x * (x1 - x0)) / w, 4), round((y0 + y * (y1 - y0)) / h, 4)]
                g["outline"] = [to_snip(x, y) for x, y in g["outline"]]
                g["center"] = to_snip(*g["center"])
                r = min(heat["rows"] - 1, int(g["center"][1] * heat["rows"]))
                c = min(heat["cols"] - 1, int(g["center"][0] * heat["cols"]))
                g["evidence"] = round(float(ev[r, c]), 3)
                out.append(g)
        return out
