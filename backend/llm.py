"""Lumen (NCSA) side of deMHISTifyAI: paper retrieval, the research LLM judge,
the final verdict, and parsing what the researcher typed.

The LLMs never see the image. They see only what CONCH measured (judge votes,
criteria presence scores) plus excerpts from the curated paper library, and may
cite only those excerpts.
"""

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = ROOT / "papers" / "library.json"

JUDGE_MODEL = os.environ.get("DEMHIST_JUDGE_MODEL", "gemma-4-31b-it")
VERDICT_MODEL = os.environ.get("DEMHIST_VERDICT_MODEL", "nemotron-3-super-120b-a12b")
N_SAMPLES = int(os.environ.get("DEMHIST_JUDGE_SAMPLES", "5"))

CLASSES = {"SSA": "Sessile serrated lesion (SSA/SSL)", "HP": "Hyperplastic polyp (HP)"}


def client() -> OpenAI:
    key = os.environ.get("LUMEN_API_KEY")
    if not key:
        raise RuntimeError("LUMEN_API_KEY is not set")
    return OpenAI(api_key=key, base_url="https://lumen.ncsa.illinois.edu/v1")


def _json_from(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON in model output")
    return json.loads(m.group(0))


# ---------- paper library ----------
class Library:
    """papers/library.json (built by research/build_library.py): evidence passages from the
    OpenEvidence syntheses, each with the primary papers it cites:
    [{"id","question","topic","heading","text","sources":[{"n","citation","title","year","doi"}]}]"""

    def __init__(self):
        self.papers = json.loads(LIBRARY_PATH.read_text(encoding="utf-8")) if LIBRARY_PATH.exists() else []
        self._vec = None
        if self.papers:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self._mat = self._vec.fit_transform([f"{p['topic']} {p['heading']} {p['text']}" for p in self.papers])

    def search(self, query: str, k: int = 8) -> List[dict]:
        if not self.papers:
            return []
        sims = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
        hits = []
        for rank, i in enumerate(sims.argsort()[::-1][:k]):
            p = self.papers[i]
            hits.append({"ref": f"P{rank + 1}", "id": p["id"], "title": f"{p['topic']}: {p['heading']}",
                         "excerpt": p["text"][:1800], "sources": p.get("sources", [])})
        return hits


# ---------- researcher input ----------
def parse_researcher(text: str) -> dict:
    t = text.lower()
    pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", t) or re.search(r"\b(\d{1,3})\s*(?:percent|pct)\b", t)
    certainty = min(float(pct.group(1)), 100.0) if pct else None
    if re.search(r"\bhyperplastic\b|\bhp\b|\bmvhp\b|\bgchp\b", t):
        call = "HP"
    elif re.search(r"\bsessile\b|\bssa\b|\bssl\b|\bssp\b|serrated (adenoma|lesion)", t):
        call = "SSA"
    else:
        call = None
    return {"text": text, "call": call, "certainty": certainty, "in_scope": call is not None}


# ---------- research LLM judge ----------
JUDGE_SYSTEM = """You are a gastrointestinal pathology research assistant comparing two diagnoses for a colon polyp image:
SSA = sessile serrated lesion (sessile serrated adenoma/polyp); HP = hyperplastic polyp.
You CANNOT see the image. An image model measured how strongly each histologic feature appears (0 = much weaker than typical, 0.5 = typical, 1 = much stronger than typical, relative to a reference set of colon polyp images). These measurements are noisy.
You also get numbered literature excerpts [P1], [P2], ... (evidence syntheses that cite primary papers with their own [n] markers; cite an excerpt as [P#]).
Reason about THIS case: decide which features matter most here and why, the way a pathologist weighs evidence; the field is contested, so do not apply a fixed checklist.
Cite only the provided excerpts, by their [P#] labels. Never invent citations. If the excerpts do not support a claim, say so.
Respond with JSON only:
{"call": "SSA" or "HP", "confidence": 0-100, "alt": the other class, "alt_confidence": 0-100,
 "key_features": [{"feature_id": "the [bracketed id] from the measurements", "feature": "short readable name", "points_to": "SSA" or "HP", "why": str, "cites": ["P#"]}]  (2-4 items),
 "reasoning": "at most 120 words, with [P#] citations",
 "caveats": "one sentence"}"""


def _judge_once(cli: OpenAI, payload: str, temperature: float) -> Optional[dict]:
    try:
        r = cli.chat.completions.create(
            model=JUDGE_MODEL, temperature=temperature, max_tokens=1200,
            messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": payload}],
        )
        out = _json_from(r.choices[0].message.content or "")
        return out if out.get("call") in CLASSES else None
    except Exception:
        return None


def research_judge(criteria: dict, papers: List[dict]) -> dict:
    feats = "\n".join(f"- [{k}] {v['text']}: {v['score']:.2f}" for k, v in criteria.items())
    refs = "\n\n".join(f"[{p['ref']}] {p['title']}\n{p['excerpt']}" for p in papers) or "(no excerpts available)"
    payload = f"Feature measurements:\n{feats}\n\nLiterature excerpts:\n{refs}"

    cli = client()
    with ThreadPoolExecutor(N_SAMPLES) as ex:
        samples = [s for s in ex.map(lambda i: _judge_once(cli, payload, 0.7), range(N_SAMPLES)) if s]
    if not samples:
        return {"error": "llm_unavailable"}

    votes = Counter(s["call"] for s in samples)
    top, n_top = votes.most_common(1)[0]
    agreeing = sorted((s for s in samples if s["call"] == top), key=lambda s: s.get("confidence", 0))
    rep = agreeing[len(agreeing) // 2]  # median-confidence sample as the shown explanation
    valid = {p["ref"] for p in papers}
    for f in rep.get("key_features", []):
        f["cites"] = [c for c in f.get("cites", []) if c in valid]
    alt = "HP" if top == "SSA" else "SSA"
    return {
        "call": top,
        "consistency": round(100 * n_top / len(samples)),  # measured: how often it gave this answer
        "stated_confidence": int(rep.get("confidence", 0)),  # what it claims
        "alt": alt,
        "alt_consistency": round(100 * votes.get(alt, 0) / len(samples)),
        "alt_stated_confidence": int(rep.get("alt_confidence", 0)),
        "n_samples": len(samples),
        "key_features": rep.get("key_features", []),
        "reasoning": rep.get("reasoning", ""),
        "caveats": rep.get("caveats", ""),
        "citations": [p for p in papers if p["ref"] in set(re.findall(r"P\d+", json.dumps(rep)))],
    }


# ---------- final verdict ----------
VERDICT_SYSTEM = """You write the final summary for a pathology research tool. You cannot see the image.
Compare the researcher's call and stated certainty with (a) a panel of 8 image-model judges and (b) a literature-grounded reasoning model.
Be direct and brief. Point out disagreement, over- or under-confidence, and what the researcher should look at again. This is a research tool, not a diagnosis.
Respond with JSON only:
{"alignment": "agrees" | "partly agrees" | "disagrees", "headline": "max 10 words",
 "conclusion": "2-4 sentences", "look_again_at": "features or regions to re-check, or empty",
 "certainty_note": "one sentence comparing the researcher's certainty with the evidence"}"""


def final_verdict(researcher: dict, panel: dict, judge: dict, flags: List[str]) -> dict:
    payload = json.dumps({"researcher": researcher, "image_judges": panel, "literature_judge": {
        k: judge.get(k) for k in ("call", "consistency", "stated_confidence", "alt", "key_features", "reasoning")},
        "trust_flags": flags}, indent=1)
    try:
        r = client().chat.completions.create(
            model=VERDICT_MODEL, temperature=0.2, max_tokens=4000,
            messages=[{"role": "system", "content": VERDICT_SYSTEM}, {"role": "user", "content": payload}],
        )
        return _json_from(r.choices[0].message.content or "")
    except Exception as e:
        return {"error": f"verdict_failed: {type(e).__name__}"}
