"""Lumen (NCSA) side of deMHISTifyAI: paper retrieval, the research LLM judge,
the final verdict, and parsing what the researcher typed.

The LLMs never see the image. They see only what the image models measured (the panel's
calibrated P(SSL), its cut-off and the members' votes; CONCH feature scores) plus excerpts
from the curated paper library, and may cite only those excerpts.
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
    """The JSON object in a model answer; tolerant of reasoning tags, code fences and trailing commas."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON in model output")
    raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))


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

    def search(self, query: str, k: int = 8, pool: int = 30, diversity: float = 0.6) -> List[dict]:
        """Top-k passages for the query, chosen for relevance AND variety (maximal marginal relevance), so the
        judge doesn't get eight near-copies of one synthesis."""
        if not self.papers:
            return []
        sims = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
        cand = list(sims.argsort()[::-1][:pool])
        chosen = []
        while cand and len(chosen) < k:
            def score(i):
                red = max(((self._mat[i] @ self._mat[j].T).toarray()[0, 0] for j in chosen), default=0.0)
                return (1 - diversity) * sims[i] - diversity * red
            best = max(cand, key=score)
            chosen.append(best)
            cand.remove(best)
        hits = []
        for rank, i in enumerate(chosen):
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
JUDGE_SYSTEM = """You are a gastrointestinal pathology research assistant. A researcher is deciding between two diagnoses for one
field of a colon polyp: SSA = sessile serrated lesion (sessile serrated adenoma/polyp) and HP = hyperplastic polyp.
You CANNOT see the image. Your job is to check whether the image panel's call is consistent with what was MEASURED on
this field and with the numbered literature excerpts, and to say what a pathologist should verify by eye.

Evidence you get, strongest first:
1. IMAGE PANEL: two fine-tuned image models combined (test AUC 0.956, balanced accuracy 0.888 on the MHIST benchmark).
   Its percentage is calibrated to a 7-pathologist majority; it calls SSA above its stated cut-off, not above 50%.
   This is the strongest evidence about this specific field. Do not overrule it without a specific, cited reason.
2. MEASURED CRYPTS: crypts outlined and measured on this field (ids like [crypts:dilated]). Each alone is weak evidence
   (they separate SSA from HP at AUC ~0.6 on held-out images). Serration is common to BOTH HP and SSA.
3. FEATURE SCORES: CONCH image-text scores (ids like [crit:...]), 0-1 relative to typical polyps. Weak hints only.
4. LITERATURE: numbered excerpts [P1], [P2], ... from evidence syntheses. Cite them by label.

Rules:
- Only refer to measured findings by their [bracketed id]. Anything listed under "not measured" was NOT observed: never
  say it is present or absent; put it under check_by_eye instead.
- Never say the evidence "confirms", is "diagnostic", "definitive" or "pathognomonic", and never call weak evidence strong.
- WHO 2019: one unequivocally architecturally distorted crypt (basal dilation, boot/L/inverted-T shapes, horizontal
  growth along the muscularis mucosae, serration into the base) is enough for SSA. Say plainly when this field cannot
  show crypt bases.
- If the field is flagged as unlike the training data, say so in your FIRST sentence: the image evidence may not apply.
- When you suggest what to check by eye, say WHERE when the measurements give a position (e.g. "the dilated crypt at
  the upper right", "the region that pushed toward SSA, lower left"). A measured crypt that sits where the panel's
  evidence map pushes toward its call is the best first place to look.
- Cite only the given excerpts, by [P#]. If they don't support a claim, don't make it.
Respond with JSON only:
{"call": "SSA" or "HP", "confidence": 0-100, "alt": the other class, "alt_confidence": 0-100,
 "agrees_with_panel": true or false,
 "key_features": [{"feature_id": "a [bracketed id] from MEASURED CRYPTS or FEATURE SCORES", "feature": "short name",
                   "points_to": "SSA" or "HP" or "neither", "strength": "weak" or "moderate", "why": "one sentence", "cites": ["P#"]}] (0-3 items),
 "check_by_eye": ["a not-measured feature the pathologist should verify, and why, with [P#]"] (1-3 items),
 "reasoning": "at most 110 words, with [P#] citations",
 "caveats": "one sentence"}"""

# "confirms" is an overclaim; "verify to confirm" and "confirm whether" are instructions, not claims
OVERCLAIM = re.compile(r"\b(?<!to )(?<!would )(?<!can )(?<!help )(confirm(s|ed|ing)?\b(?!\s+(?:if|whether))|diagnostic of|definitive(ly)?|pathognomonic|"
                       r"strongly support(s|ed)?|clearly (shows|demonstrates)|proves?|certainly)\b", re.I)
SEEN = re.compile(r"\b(is|are|were|was) (present|seen|visible|observed|evident|noted)\b|\b(shows?|demonstrates?|reveals?)\b", re.I)


def _where(x: float, y: float) -> str:
    """Plain position in the field from 0..1 coordinates: 'upper left', 'centre', 'lower edge', ..."""
    v = "upper" if y < 0.34 else "lower" if y > 0.66 else ""
    h = "left" if x < 0.34 else "right" if x > 0.66 else ""
    return (f"{v} {h}".strip() or "centre") if (v and h) or not (v or h) else (v + " middle" if v else "middle " + h)


def _evidence_regions(heat: dict) -> List[str]:
    """Where the evidence map says the call came from: the strongest SSA- and HP-pushing regions."""
    if not heat or heat.get("kind") != "evidence":
        return []
    import numpy as np
    R, C = heat["rows"], heat["cols"]
    p = np.array(heat["p_ssa"]).reshape(R, C) - 0.5
    t = np.array(heat["tissue"]).reshape(R, C) >= 0.25
    out = []
    for sign, word in ((1, "SSA"), (-1, "HP")):
        v = np.where(t, sign * p, -1)
        if v.max() < 0.15:                                   # nothing clearly pushed this way
            continue
        r, c = np.unravel_index(int(v.argmax()), v.shape)
        out.append(f"- The region that pushed the call most toward {word} is at the {_where((c + 0.5) / C, (r + 0.5) / R)} of the field.")
    return out


def _overclaims(text: str) -> int:
    """Overclaiming phrases, not counting negated ones ("does not confirm", "can't be diagnostic")."""
    n = 0
    for m in OVERCLAIM.finditer(text or ""):
        before = text[max(0, m.start() - 30) : m.start()].lower()
        if not re.search(r"\b(not|n't|no|cannot|without|never)\b", before):
            n += 1
    return n


def _measured(conch: dict) -> tuple:
    """Text block of what was measured on this field, and the ids the judge may cite as evidence."""
    pnl = conch.get("panel", {})
    ids = {}
    lines = []
    if pnl.get("p_ssl") is not None:
        lines.append(f"- Panel: P(SSA) {round(100 * pnl['p_ssl'])}%, calls SSA above {round(100 * pnl['cut'])}%, so its call "
                     f"is {pnl['top']}. Members: " + ", ".join(f"{j['id']} {round(100 * j['p_ssl'])}%" for j in conch.get("judges", [])
                                                           if j.get("p_ssl") is not None) + ".")
    else:
        p = pnl.get("top_conf") if pnl.get("top") == "SSA" else pnl.get("alt_conf")
        lines.append(f"- Panel: P(SSA) {p}%, call {pnl.get('top')}, votes {pnl.get('votes')}.")
    ood = conch.get("ood")
    if ood and ood.get("flag"):
        lines.append("- WARNING: this field looks unlike the training images (other tissue, stain, zoom or not tissue).")
    glands = [g for g in conch.get("glands", []) if g.get("label")]
    allg = conch.get("glands", [])
    crypt_lines = []
    if allg:
        n_clear = sum(1 for g in allg if g.get("ring", 0) >= 0.85)
        for tag, key, what in (("dilated", "crypts:dilated", "dilated (the whole crypt wider than the field's other crypts; "
                                                             "whether the widening is at the BASE is not measured)"),
                               ("serrated", "crypts:serrated", "serrated lumen (saw-toothed border)"),
                               ("round regular", "crypts:round", "round, regular crypt (HP- or normal-like)")):
            n = sum(tag in g.get("tags", []) for g in glands)
            if n:
                widths = [g["width_um"] for g in glands if tag in g.get("tags", [])]
                where = sorted({_where(*g["center"]) for g in glands if tag in g.get("tags", [])})
                size = f"width {widths[0]:.0f} um" if len(widths) == 1 else f"widths {min(widths):.0f}-{max(widths):.0f} um"
                evs = [g.get("evidence", 0) for g in glands if tag in g.get("tags", [])]
                push = sum(e > 0.1 for e in evs), sum(e < -0.1 for e in evs)
                lean = (f"; the image panel's own evidence map pushes toward SSA over {push[0]} and toward HP over {push[1]} of them"
                        if conch.get("heat", {}).get("kind") == "evidence" else "")
                crypt_lines.append(f"- [{key}] {n} of {n_clear} clearly ringed crypts {what}; {size}; at the {', '.join(where)}{lean}")
                ids[key] = what
        if not crypt_lines:
            crypt_lines.append(f"- {n_clear} clearly ringed crypts found; none clearly dilated, serrated or round-regular.")
    else:
        crypt_lines.append("- No crypts could be outlined on this field (cut lengthwise, open to the surface, or blurred).")
    crypt_lines += _evidence_regions(conch.get("heat"))
    feats = []
    for k, v in conch.get("criteria", {}).items():
        if v.get("usable"):
            feats.append(f"- [crit:{k}] {v['text']}: score {v['score']:.2f} (reliability {v['reliability']:.2f})")
            ids[f"crit:{k}"] = v["text"]
    not_measured = [v["text"] for v in conch.get("criteria", {}).values() if not v.get("usable")]
    block = ("IMAGE PANEL\n" + "\n".join(lines) + "\n\nMEASURED CRYPTS (outlined and measured on this field)\n" + "\n".join(crypt_lines)
             + "\n\nFEATURE SCORES\n" + ("\n".join(feats) or "(none usable)")
             + "\n\nNOT MEASURED on this field (never say present or absent): " + "; ".join(not_measured))
    return block, ids, not_measured


def judge_query(conch: dict) -> str:
    """Literature query from what this field actually shows, plus the WHO criteria the judge must weigh."""
    q = ["sessile serrated lesion hyperplastic polyp WHO 2019 criteria architecturally distorted crypt base dilation"]
    tags = {t for g in conch.get("glands", []) for t in g.get("tags", [])}
    if "dilated" in tags:
        q.append("dilated crypt base boot L-shaped inverted T horizontal growth muscularis mucosae")
    if "serrated" in tags:
        q.append("serration extending to crypt base versus upper third serration hyperplastic")
    if "round regular" in tags:
        q.append("regular straight crypts narrow base hyperplastic polyp microvesicular goblet")
    if not conch.get("glands"):
        q.append("tangential sectioning orientation crypt bases not visible small biopsy pitfalls")
    q.append("interobserver agreement reproducibility distinguishing features")
    return " ".join(q)


READABLE = {"crypts:dilated": "dilated crypts (measured)", "crypts:serrated": "serrated crypt lumens (measured)",
            "crypts:round": "round regular crypts (measured)"}


def _readable(text, criteria=None):
    """Evidence ids the judge cites ([crypts:round], [crit:goblet_cells]) shown as plain words."""
    if not isinstance(text, str):
        return text
    def name(m):
        k = m.group(1)
        kind = "measured" if k.startswith("crypts:") else "feature score"
        full = READABLE.get(k) or (f"{criteria[k[5:]]['text']} (feature score)" if k.startswith("crit:") and criteria
                                   and k[5:] in criteria else k.split(":", 1)[-1].replace("_", " "))
        # already named just before the id ("a round, regular crypt [crypts:round]"): only say where it came from
        stem = re.sub(r"[^a-z]", "", k.split(":", 1)[1].split("_")[0])[:5]
        before = text[max(0, m.start() - 45) : m.start()].lower()
        return f"({kind})" if stem and stem in before else full
    def group(m):
        ids = re.findall(r"(?:crypts|crit):[a-z_]+", m.group(0))
        parts = [name(_Id(i, m.start())) for i in ids]
        return " " + (" and ".join(dict.fromkeys(parts)))
    return re.sub(r"\s*\[\s*(?:crypts|crit):[a-z_]+(?:\s*[,;&]\s*(?:crypts|crit):[a-z_]+)*\s*\]", group, text).replace("  ", " ")


class _Id:
    """minimal stand-in for a regex match (group(1), start()) so one id inside a bracket group can be named"""
    def __init__(self, k, start):
        self.k, self.s = k, start

    def group(self, i):
        return self.k

    def start(self):
        return self.s


def _clean_citations(text, valid):
    """Drop [P#] references that are not among the given excerpts (e.g. invented [P12])."""
    if not isinstance(text, str):
        return text
    def fix(m):
        kept = [r for r in re.findall(r"P\d+", m.group(0)) if r in valid]
        return "[" + ", ".join(kept) + "]" if kept else ""
    return re.sub(r"\[\s*P\d+(?:\s*[,;&]\s*P\d+)*\s*\]", fix, text).replace("  ", " ").strip()


def _violations(s: dict, valid: set, ids: dict, not_measured: List[str]) -> int:
    """Rule breaks in one judge answer: invented citations, overclaiming, evidence that wasn't measured, and
    claims that a not-measured feature was seen."""
    text = " ".join([s.get("reasoning", ""), s.get("caveats", "")] + [f.get("why", "") for f in s.get("key_features", [])]
                    + [c for c in s.get("check_by_eye", []) if isinstance(c, str)])
    v = sum(1 for r in re.findall(r"P\d+", json.dumps(s)) if r not in valid)
    v += _overclaims(text)
    v += sum(1 for f in s.get("key_features", []) if f.get("feature_id", "").strip("[]") not in ids)
    low = text.lower()
    for nm in not_measured:
        key = nm.lower().split(" or ")[0].split(",")[0][:24]
        for m in re.finditer(re.escape(key), low):
            if SEEN.search(low[max(0, m.start() - 60) : m.end() + 60]):
                v += 1
                break
    return v


def _judge_once(cli: OpenAI, payload: str, temperature: float) -> Optional[dict]:
    try:
        r = cli.chat.completions.create(
            model=JUDGE_MODEL, temperature=temperature, max_tokens=1200,
            messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": payload}],
        )
        out = _json_from(r.choices[0].message.content or "")
        call = str(out.get("call", "")).strip().upper()
        call = {"SSL": "SSA", "SSA/SSL": "SSA", "SESSILE SERRATED LESION": "SSA", "HYPERPLASTIC POLYP": "HP"}.get(call, call)
        if call not in CLASSES:
            return None
        out["call"] = call
        return out
    except Exception:
        return None


def research_judge(conch: dict, papers: List[dict], panel: Optional[dict] = None) -> dict:
    if "criteria" not in conch:          # older call style: research_judge(criteria, papers, panel)
        conch = {"criteria": conch, "panel": panel or {}}
    block, ids, not_measured = _measured(conch)
    refs = "\n\n".join(f"[{p['ref']}] {p['title']}\n{p['excerpt']}" for p in papers) or "(no excerpts available)"
    payload = f"{block}\n\nLITERATURE EXCERPTS\n{refs}"

    try:
        cli = client()
    except Exception:  # no or invalid key: the app shows "literature judge unavailable" and carries on
        return {"error": "llm_unavailable"}
    with ThreadPoolExecutor(N_SAMPLES) as ex:
        samples = [s for s in ex.map(lambda i: _judge_once(cli, payload, 0.7), range(N_SAMPLES)) if s]
        if not samples:  # every sample failed to parse: one calmer retry before giving up
            samples = [s for s in ex.map(lambda i: _judge_once(cli, payload, 0.3), range(3)) if s]
    if not samples:
        return {"error": "llm_unavailable"}

    valid = {p["ref"] for p in papers}
    votes = Counter(s["call"] for s in samples)
    top, n_top = votes.most_common(1)[0]
    agreeing = [s for s in samples if s["call"] == top]
    for s in agreeing:
        s["_v"] = _violations(s, valid, ids, not_measured)
    # shown explanation: the cleanest answer that agrees with the majority, median confidence among those
    clean = min(s["_v"] for s in agreeing)
    pool = sorted((s for s in agreeing if s["_v"] == clean), key=lambda s: s.get("confidence", 0))
    rep = pool[len(pool) // 2]
    feats = []
    for f in rep.get("key_features", []):
        fid = str(f.get("feature_id", "")).strip("[]")
        if fid not in ids:                                   # only evidence that was actually measured
            continue
        f["feature_id"] = fid.split(":", 1)[1] if fid.startswith("crit:") else fid
        f["cites"] = [c for c in f.get("cites", []) if c in valid]
        f["why"] = _clean_citations(f.get("why", ""), valid)
        f["strength"] = "weak"  # every measured item separates SSA from HP at AUC ~0.6 on its own
        feats.append(f)
    crit = conch.get("criteria", {})
    for f in feats:
        f["why"] = _readable(f["why"], crit)
    reasoning = _readable(_clean_citations(rep.get("reasoning", ""), valid), crit)
    check = [_readable(_clean_citations(c, valid), crit) for c in rep.get("check_by_eye", []) if isinstance(c, str)]
    alt = "HP" if top == "SSA" else "SSA"
    pnl_call = (conch.get("panel") or {}).get("top")
    return {
        "call": top,
        "consistency": round(100 * n_top / len(samples)),  # measured: how often it gave this answer
        "stated_confidence": int(rep.get("confidence", 0)),  # what it claims
        "alt": alt,
        "alt_consistency": round(100 * votes.get(alt, 0) / len(samples)),
        "alt_stated_confidence": int(rep.get("alt_confidence", 0)),
        "n_samples": len(samples),
        "agrees_with_panel": (top == pnl_call) if pnl_call else None,
        "rule_breaks": {"shown": clean, "all_samples": [s.get("_v") for s in agreeing]},
        "key_features": feats,
        "check_by_eye": check,
        "reasoning": reasoning,
        "caveats": _readable(_clean_citations(rep.get("caveats", ""), valid), crit),
        "citations": [p for p in papers if p["ref"] in set(re.findall(r"P\d+", json.dumps([reasoning, check, feats])))],
    }


# ---------- final verdict ----------
VERDICT_SYSTEM = """You write the final summary for a pathology research tool. You cannot see the image.
Compare the researcher's call and stated certainty with:
(a) the image panel: two fine-tuned image models combined. Its percentage is calibrated to the 7-pathologist majority
    ("how often the majority says SSA on MHIST-like fields") and it calls SSA above its stated cut-off, not above 50%.
    Compare the researcher's stated certainty with that percentage;
(b) what was measured on this field (crypt measurements are weak evidence on their own);
(c) a literature-grounded reasoning model (its "consistency" is how often it gave the same answer).
Rules: be direct and brief; point out disagreement, over- or under-confidence, and what to look at again. Never say the
evidence confirms or is diagnostic. Write percentages as whole numbers (e.g. 50%, or "under 1%") and never quote field
names such as p_ssl. If the field was flagged as unlike the training data, lead with that. Only name
features that were measured or that the literature model listed to check by eye. Research tool, not a diagnosis.
Respond with JSON only:
{"alignment": "agrees" | "partly agrees" | "disagrees", "headline": "max 10 words",
 "conclusion": "2-4 sentences", "look_again_at": "features or regions to re-check, or empty",
 "certainty_note": "one sentence comparing the researcher's certainty with the panel's percentage"}"""


def not_tissue_verdict() -> dict:
    """Fixed verdict for a field that is far from any colon tissue (photo, noise, screen chrome)."""
    return {"alignment": "partly agrees", "headline": "This doesn't look like colon polyp tissue",
            "conclusion": "The selected area is far from anything the image panel was trained on, so its call and any "
                          "measurements don't apply, and the literature comparison was skipped. Select one field of an H&E "
                          "colon polyp slide at about 5x.",
            "look_again_at": "", "certainty_note": "No comparison: the image evidence does not apply to this selection."}


def final_verdict(researcher: dict, panel: dict, judge: dict, flags: List[str], conch: Optional[dict] = None) -> dict:
    measured = _measured(conch)[0] if conch else None
    payload = json.dumps({"researcher": researcher, "image_panel": panel, "measured_on_this_field": measured, "literature_judge": {
        k: judge.get(k) for k in ("call", "consistency", "stated_confidence", "alt", "agrees_with_panel", "key_features", "check_by_eye", "reasoning")},
        "trust_flags": flags}, indent=1)
    try:
        r = client().chat.completions.create(
            model=VERDICT_MODEL, temperature=0.2, max_tokens=4000,
            messages=[{"role": "system", "content": VERDICT_SYSTEM}, {"role": "user", "content": payload}],
        )
        return _json_from(r.choices[0].message.content or "")
    except Exception as e:
        return {"error": f"verdict_failed: {type(e).__name__}"}
