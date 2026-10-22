"""Lumen (NCSA) side of gutfeeling: paper retrieval, the research LLM judge,
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
N_SAMPLES = int(os.environ.get("DEMHIST_JUDGE_SAMPLES", "3"))
# the judge's confidence never goes above this: experts don't agree that well either (on MHIST a single pathologist
# agrees with the 7-pathologist majority 83% of the time; two pathologists agree 73%; all seven on 34% of fields)
MAX_CONF = int(os.environ.get("DEMHIST_JUDGE_MAX_CONF", "92"))

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

    def search(self, query: str, k: int = 8, pool: int = 40, diversity: float = 0.6, per_topic: int = 1) -> List[dict]:
        """Top-k passages for the query, chosen for relevance AND variety (maximal marginal relevance, and at most
        per_topic passages from any one topic), so the judge gets several angles, not near-copies of one synthesis."""
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
            cand.remove(best)
            if sum(self.papers[j]["topic"] == self.papers[best]["topic"] for j in chosen) >= per_topic:
                continue
            chosen.append(best)
        hits = []
        for rank, i in enumerate(chosen):
            p = self.papers[i]
            title = p["topic"] if p["heading"].strip() == p["topic"].strip() else f"{p['topic']}: {p['heading']}"
            srcs = p.get("sources") if isinstance(p.get("sources"), list) else []
            hits.append({"ref": f"P{rank + 1}", "id": p["id"], "title": title, "topic": p["topic"],
                         "excerpt": p["text"][:1800], "sources": srcs, "papers": [_short_ref(s) for s in srcs[:3]]})
        return hits


def _short_ref(s: dict) -> dict:
    """A primary paper as pathologists cite it: 'Bettington et al. 2013', with its title and DOI. The library's
    citation strings vary, so the first author is found by its shape (Surname + initials)."""
    cit = s.get("citation", "") or ""
    m = re.search(r"\b([A-Z][a-zA-Z'\-]+(?: [a-z]{2,3})?) ([A-Z]{1,3})(,| et al|\.)", cit)
    year = str(s.get("year") or "")
    title = (s.get("title") or cit).strip()
    if m:
        title = title.split(m.group(0))[0].strip()           # some titles run on into the author list
        label = f"{m.group(1)}{' et al.' if (m.group(3) == ',' or 'et al' in cit[m.start():m.start() + 60]) else ''} {year}".strip()
    else:
        label = " ".join(title.split()[:6]) + ("..." if len(title.split()) > 6 else "") + (f" {year}" if year else "")
    return {"label": label, "title": title.rstrip(" ."), "doi": s.get("doi") or ""}


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
JUDGE_SYSTEM = """You support a gastrointestinal pathologist deciding between sessile serrated lesion (SSL) and hyperplastic
polyp (HP) for one field of a colon polyp. You CANNOT see the image. You get: the image panel's call (two validated
image models; its call is FINAL), what was measured on this field, and numbered literature excerpts [P1], [P2], ...
The reader works through 30-40 cases a session: be short and specific to THIS field, never recite textbook criteria.

Rules:
- Your "call" MUST be the image panel's call. Explain why it fits; never argue for the other class. If a measurement
  points the other way, give it once under "against" and lower your confidence.
- Only the MEASURED findings (cite them by their [id]) and the panel are known about this image. Never say a feature
  is present or absent unless it was measured. Do not speculate about the plane of section or whether crypt bases
  are visible.
- Write SSL and HP, never SSA.
- No generic checklists. Each "check" item must point to a PLACE in this field taken from the measurements (e.g. "the
  2 dilated crypts at the lower right") and say what to look for there.
- Never use: confirms, diagnostic, definitive, pathognomonic, highly specific, strongly supports, proves.
- If the field is flagged as unlike the training images, the summary starts with that.
- Cite an excerpt as [P#] only where it directly supports that sentence; at most 2 citations in the summary.
- Citations [P#] are only for general facts from the literature (what a feature means, which class it favours),
  never for a statement about this field. For each point cite the excerpt that addresses it MOST specifically; use
  the WHO criteria excerpt only for the definition itself, and draw on at least two different excerpts when they
  are relevant.
- Describe only what was measured. Never say something is lacking, absent, missing or not seen.
  Good: "Six round, regular crypts across the field fit HP; regular crypts are typical of HP [P1]."
  Bad: "The field shows regular architecture [P1]" (cites a paper for this field) and "lacking distortion" (an absence).
- "confidence" = how likely a 7-pathologist majority would agree with the panel's call, 50-92. Experts themselves
  match the majority only ~83% of the time, so never exceed 92. 55-70 if the evidence is mixed or the panel score is
  near 50%; 70-85 if the panel and the measurements agree; above 85 only if everything agrees and the field is typical.
Respond with JSON only:
{"call": "SSL" or "HP", "confidence": 50-92,
 "summary": "at most 45 words",
 "for": [{"id": "a measured [id]", "text": "at most 18 words, saying where"}] (0-2 items),
 "against": [{"id": "a measured [id]", "text": "at most 18 words"}] (0-1 items),
 "check": ["at most 18 words: a place in this field and what to look for there"] (1-2 items)}"""

# "confirms" is an overclaim; "verify to confirm" and "confirm whether" are instructions, not claims
OVERCLAIM = re.compile(r"\b(?<!to )(?<!would )(?<!can )(?<!help )(confirm(s|ed|ing)?\b(?!\s+(?:if|whether))|diagnostic of|definitive(ly)?|pathognomonic|"
                       r"strongly support(s|ed)?|clearly (shows|demonstrates)|proves?|certainly)\b", re.I)
ABSENCE = re.compile(r"\b(lack(s|ing|ed)?|absen(t|ce)|without|no (sign|evidence|features?)|not (seen|present|observed|visible))\b", re.I)
FIELD_TALK = re.compile(r"\b(this field|the field|measured|crypts? (at|in|on)|(upper|lower|middle|centre|left|right))\b", re.I)
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
        sc = lambda o: round(100 * o["score"]) if o.get("score") is not None else round(100 * o["p_ssl"])
        lines.append(f"- Panel: SSA score {sc(pnl)}% (the app's displayed score: above 50% means it calls SSA; underlying "
                     f"chance a 7-pathologist majority says SSA {round(100 * pnl['p_ssl'])}%), so its call is {pnl['top']}. "
                     "Member SSA scores: " + ", ".join(f"{j['id']} {sc(j)}%" for j in conch.get("judges", [])
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
        # measured with confidence: a learned outline with measurements, or a classical crypt with a clear nuclear ring
        sure = lambda g: g.get("gland_width_um") is not None or g.get("ring", 0) >= 0.85
        width = lambda g: g.get("gland_width_um") or g["width_um"]
        n_clear = sum(1 for g in allg if sure(g))
        for tag, key, what in (("dilated", "crypts:dilated", "dilated (the whole crypt wider than the field's other crypts; "
                                                             "whether the widening is at the BASE is not measured)"),
                               ("serrated", "crypts:serrated", "serrated lumen (saw-toothed border)"),
                               ("round regular", "crypts:round", "round, regular crypt (HP- or normal-like)")):
            n = sum(tag in g.get("tags", []) for g in glands)
            if n:
                widths = [width(g) for g in glands if tag in g.get("tags", [])]
                where = sorted({_where(*g["center"]) for g in glands if tag in g.get("tags", [])})
                size = f"width {widths[0]:.0f} um" if len(widths) == 1 else f"widths {min(widths):.0f}-{max(widths):.0f} um"
                evs = [g.get("evidence", 0) for g in glands if tag in g.get("tags", [])]
                push = sum(e > 0.1 for e in evs), sum(e < -0.1 for e in evs)
                lean = (f"; the image panel's own evidence map pushes toward SSA over {push[0]} and toward HP over {push[1]} of them"
                        if conch.get("heat", {}).get("kind") == "evidence" else "")
                crypt_lines.append(f"- [{key}] {n} of {n_clear} measured crypts {what}; {size}; at the {', '.join(where)}{lean}")
                ids[key] = what
        if not crypt_lines:
            crypt_lines.append(f"- {n_clear} measured crypts found; none clearly dilated or round-regular.")
    else:
        crypt_lines.append("- No crypts could be outlined on this field (cut lengthwise, open to the surface, or blurred).")
    crypt_lines += _evidence_regions(conch.get("heat"))
    # CONCH text-match "feature scores" are left out: they are guesses, not measurements, and they contradicted
    # each other (the judge used to argue from them)
    not_measured = ["crypt base shape (boot, L or inverted-T)", "serration depth", "horizontal growth along the muscularis mucosae",
                    "whether crypt bases are in the plane of section", "maturation", "goblet cell or microvesicular mucin type"]
    block = ("IMAGE PANEL\n" + "\n".join(lines) + "\n\nMEASURED CRYPTS (outlined and measured on this field)\n" + "\n".join(crypt_lines)
             + "\n\nNOT MEASURED on this field (never say present or absent): " + "; ".join(not_measured))
    return block, ids, not_measured


def judge_query(conch: dict) -> str:
    """Literature query from what this field shows and what the panel called (not a fixed WHO phrase: that made the
    WHO definitions win every search)."""
    call = (conch.get("panel") or {}).get("top")
    q = ["sessile serrated lesion distinguishing features specificity" if call == "SSA" else
         "hyperplastic polyp features distinguishing from sessile serrated lesion"]
    score = (conch.get("panel") or {}).get("score")
    if score is not None and 0.35 <= score <= 0.65:
        q.append("borderline cases interobserver variability")
    if (conch.get("ood") or {}).get("flag"):
        q.append("diagnostic pitfalls staining orientation")
    tags = {t for g in conch.get("glands", []) for t in g.get("tags", [])}
    if "dilated" in tags:
        q.append("dilated crypt base boot L-shaped inverted T horizontal growth muscularis mucosae")
    if "serrated" in tags:
        q.append("serration extending to crypt base versus upper third serration hyperplastic")
    if "round regular" in tags:
        q.append("regular straight crypts narrow base hyperplastic polyp microvesicular goblet")
    if not conch.get("glands"):
        q.append("tangential sectioning orientation crypt bases not visible small biopsy pitfalls")
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
    items = [i for i in s.get("for", []) + s.get("against", []) if isinstance(i, dict)]
    text = " ".join([str(s.get("summary", ""))] + [str(i.get("text", "")) for i in items]
                    + [c for c in s.get("check", []) if isinstance(c, str)])
    v = sum(1 for r in re.findall(r"P\d+", json.dumps(s)) if r not in valid)
    v += _overclaims(text)
    v += sum(1 for i in items if str(i.get("id", "")).strip("[]") not in ids)
    v += len(ABSENCE.findall(text))                          # claims that something is absent were never measured
    for sent in re.split(r"(?<=[.!?])\s+", text):           # a paper cited for something about THIS field
        if re.search(r"P\d+", sent) and FIELD_TALK.search(sent):
            v += 1
    v += len(re.findall(r"\bSSA\b", text))                 # the app says SSL
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
        for k in ("for", "against", "check"):          # tolerate a model that returns a string or null
            if not isinstance(out.get(k), list):
                out[k] = [out[k]] if out.get(k) else []
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
    # the panel's call is final (the image models are the measured, validated evidence): the judge explains it.
    # Answers that still argued for the other class are set aside; if all did, the majority answer is used.
    pnl_call = (conch.get("panel") or {}).get("top")
    votes = Counter(s["call"] for s in samples)
    top = pnl_call if pnl_call in CLASSES and votes.get(pnl_call) else votes.most_common(1)[0][0]
    n_top = votes.get(top, 0)
    against_all = pnl_call in CLASSES and top != pnl_call   # every answer argued for the other class
    agreeing = [s for s in samples if s["call"] == top]
    for s in agreeing:
        s["_v"] = _violations(s, valid, ids, not_measured)
    # shown explanation: the cleanest answer that agrees with the majority, median confidence among those
    clean = min(s["_v"] for s in agreeing)
    pool = sorted((s for s in agreeing if s["_v"] == clean), key=lambda s: s.get("confidence", 0))
    rep = pool[len(pool) // 2]
    crit = conch.get("criteria", {})
    ssl = lambda t: re.sub(r"\bSSAs?\b", "SSL", t) if isinstance(t, str) else t
    tidy = lambda t: ssl(_readable(_clean_citations(t, valid), crit))
    def items(key, n):
        out = []
        for i in rep.get(key, []):
            if isinstance(i, dict) and str(i.get("id", "")).strip("[]") in ids and i.get("text"):
                out.append({"id": str(i["id"]).strip("[]"), "text": tidy(i["text"])})
        return out[:n]
    summary = tidy(rep.get("summary", ""))
    support, against = items("for", 2), items("against", 1)
    check = [tidy(c) for c in rep.get("check", []) if isinstance(c, str) and c.strip()][:2]
    cited = set(re.findall(r"P\d+", json.dumps([summary, support, against, check])))
    alt = "HP" if top == "SSA" else "SSA"
    if pnl_call in CLASSES:
        top, alt = pnl_call, ("HP" if pnl_call == "SSA" else "SSA")
    return {
        "call": top,
        "alt": alt,
        "consistency": round(100 * n_top / len(samples)),  # measured: how often it gave this answer
        # what it claims, held to the expert-agreement ceiling (a coin toss if every answer argued the other way)
        "stated_confidence": 50 if against_all else max(50, min(MAX_CONF, int(float(rep.get("confidence", 0) or 0)))),
        "max_confidence": MAX_CONF,
        "n_samples": len(samples),
        "agrees_with_panel": (top == pnl_call) if pnl_call else None,
        "rule_breaks": {"shown": clean, "all_samples": [s.get("_v") for s in agreeing]},
        "summary": summary,
        "for": support,
        "against": against,
        "check": check,
        "key_features": [],
        "citations": [{k: p[k] for k in ("ref", "title", "topic", "papers")} for p in papers if p["ref"] in cited],
    }


# ---------- final verdict ----------
VERDICT_SYSTEM = """You write the final summary for a pathology research tool. You cannot see the image.
Compare the researcher's call and stated certainty with:
(a) the image panel: two fine-tuned image models combined. The app shows its "score" (top_conf): above 50% means it
    calls SSA, below 50% HP. When you quote a percentage for the panel, quote that displayed score, never p_ssl (the
    underlying calibrated chance that a 7-pathologist majority says SSA, which is lower). Compare the researcher's stated
    certainty with the displayed score;
(b) what was measured on this field (crypt measurements are weak evidence on their own);
(c) a literature-grounded reasoning model (its "consistency" is how often it gave the same answer).
Rules: be direct and brief; point out disagreement, over- or under-confidence, and what to look at again. Never say the
evidence confirms or is diagnostic. Write percentages as whole numbers (e.g. 50%, or "under 1%") and never quote field
names such as p_ssl. If the field was flagged as unlike the training data, lead with that. Only name
features that were measured or that the literature model listed to check by eye. Research tool, not a diagnosis.
If "researcher" is null, no call was given: compare the image panel with the literature model and the measurements
instead ("alignment" is whether they agree), and use certainty_note for how sure the image panel is.
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
    payload = json.dumps({"researcher": researcher if researcher.get("text", "").strip() else None, "image_panel": panel, "measured_on_this_field": measured, "literature_judge": {
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
