#!/usr/bin/env python3
"""Turn the OCR'd OpenEvidence answers into the literature judge's library.

1. Clean each answer with a Lumen model: fix OCR errors ONLY (spacing, broken words, stray
   email/browser text), keep wording, keep every [n] citation marker and the reference list.
   -> papers/openevidence/clean/Q{n}.clean.md
2. Split each cleaned answer into explanation passages (by section/paragraph), each linked to
   the original papers it cites via its [n] markers and that answer's reference list.
   -> papers/library.json  (what backend/llm.py searches)

The passages are OpenEvidence's evidence syntheses; the citations point at the primary papers.
The app shows both, so provenance stays honest.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "papers/openevidence/clean"
MODEL = os.environ.get("DEMHIST_CLEAN_MODEL", "gemma-4-31b-it")

TOPICS = {
    "Q1": "WHO 2019 criteria: sessile serrated lesion vs hyperplastic polyp",
    "Q2": "Ranking distinguishing features by specificity and reproducibility",
    "Q3": "Interobserver agreement and sources of disagreement",
    "Q4": "Diagnostic pitfalls: orientation, tangential sectioning, small samples",
    "Q5": "Hyperplastic polyp subtypes and mimics (TSA, mucosal prolapse)",
    "Q6": "Polyp size and location as diagnostic context",
    "Q7": "Sessile serrated lesion with dysplasia",
    "Q8": "Surveillance intervals and consequences of misclassification",
    "Q9": "Cancer risk and the serrated pathway",
    "Q10": "AI and digital pathology for serrated polyps",
}

CLEAN_PROMPT = """You are repairing OCR output of a medical evidence summary. Fix OCR errors ONLY:
- restore missing spaces and broken words, obvious character mistakes (e.g. "DOl" -> "DOI", "j-gastro" -> "j.gastro" inside DOIs), and hyphenation across line breaks;
- remove non-content text: email headers, timestamps, page numbers, URLs of the mail client, the original question text, model/category labels (e.g. "Guidelines · Snow"), meta-notes about the interface or formatting, and any closing offer such as "Would it be helpful to...".
Do NOT reword, summarise, add or drop facts. Keep every citation marker exactly as [n]. Keep numbers exactly.
Format as Markdown: section headings as "## Heading", paragraphs separated by blank lines, bullet points as "- ", tables as Markdown tables, figure captions as "Figure: ...".
End with "## References" followed by the numbered reference list, one reference per line, as given.
Output only the repaired Markdown."""


def clean_one(cli, path):
    out = path.with_suffix(".clean.md")
    if out.exists():
        return out
    raw = path.read_text(encoding="utf-8")
    # long answers fail upstream in one request: clean in ~6k-character pieces split at line breaks
    pieces, cur = [], ""
    for line in raw.splitlines(keepends=True):
        if len(cur) + len(line) > 6000 and cur:
            pieces.append(cur); cur = ""
        cur += line
    pieces.append(cur)
    cleaned = []
    for k, piece in enumerate(pieces):
        note = ("" if len(pieces) == 1 else
                f"\n(This is part {k + 1} of {len(pieces)} of one document: repair only this part; "
                "add the '## References' heading only if this part contains the reference list.)")
        for attempt in range(3):
            try:
                r = cli.chat.completions.create(model=MODEL, temperature=0, max_tokens=8000,
                                                messages=[{"role": "system", "content": CLEAN_PROMPT + note}, {"role": "user", "content": piece}])
                break
            except Exception:
                if attempt == 2:
                    raise
        text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        cleaned.append(re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text))
    out.write_text("\n\n".join(cleaned), encoding="utf-8")
    return out


def parse_refs(block):
    refs = {}
    for m in re.finditer(r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)", block, flags=re.M | re.S):
        n, body = int(m.group(1)), " ".join(m.group(2).split())
        doi = re.search(r"doi[:\s]*\s*(10\.\d{4,9}/[^\s;,]+?)\.?(?:\s|$)", body, flags=re.I)
        year = re.search(r"\b(19|20)\d{2}\b", body)
        title = re.split(r"(?<=[.?!])\s", body, maxsplit=1)[0]
        refs[n] = {"n": n, "citation": body, "title": title.strip(), "year": int(year.group(0)) if year else None,
                   "doi": doi.group(1) if doi else None}
    return refs


def passages(q, md, max_chars=1400):
    body, _, refblock = md.partition("## References")
    refs = parse_refs(refblock)
    out, heading, buf = [], TOPICS[q], []

    def flush():
        text = "\n".join(buf).strip()
        if len(text) > 80:
            nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
            out.append({"question": q, "topic": TOPICS[q], "heading": heading, "text": text,
                        "sources": [refs[n] for n in nums if n in refs]})
        buf.clear()

    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if para.startswith("## "):
            flush()
            heading = para[3:].strip()
            continue
        if sum(len(b) for b in buf) + len(para) > max_chars:
            flush()
        buf.append(para)
    flush()
    return out, refs


def main():
    cli = OpenAI(api_key=os.environ["LUMEN_API_KEY"], base_url="https://lumen.ncsa.illinois.edu/v1")
    files = sorted((p for p in CLEAN.glob("Q*.md") if ".clean" not in p.name), key=lambda p: int(p.stem[1:]))
    with ThreadPoolExecutor(5) as ex:
        cleaned = list(ex.map(lambda p: clean_one(cli, p), files))
    lib, n_refs = [], 0
    for path in cleaned:
        q = path.name.split(".")[0]
        ps, refs = passages(q, path.read_text(encoding="utf-8"))
        n_refs += len(refs)
        lib += ps
        print(f"{q}: {len(ps)} passages, {len(refs)} references, {sum(1 for p in ps if p['sources'])} passages with citations")
    for i, p in enumerate(lib):
        p["id"] = f"OE{i + 1:03d}"
    (ROOT / "papers/library.json").write_text(json.dumps(lib, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nlibrary: {len(lib)} passages, {n_refs} references -> papers/library.json")


if __name__ == "__main__":
    sys.exit(main())
