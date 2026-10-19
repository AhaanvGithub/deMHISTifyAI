#!/usr/bin/env python3
"""OCR the OpenEvidence answer PDFs (image-only) into clean per-question text.

Renders each page at high resolution, OCRs it, rebuilds lines in reading order, and strips
browser/email chrome (print header/footer, Outlook banner, From/Date/To, the echoed question
and model line). Output: papers/openevidence/clean/Q{n}.md
"""

import re
import sys
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "papers/openevidence"
OUT = SRC / "clean"
OUT.mkdir(exist_ok=True)
ocr = RapidOCR()

CHROME = [
    r"^\d{1,2}/\d{1,2}/\d{2},? \d{1,2}:\d{2} ?[AP]M", r"Mail - Verma", r"^https?://outlook", r"outlook\.office\.com",
    r"^\d+/\d+$", r"^Outlook$", r"^Re: Questions to ask OpenEvidence", r"^(From|Date|To)\b", r"@illinois\.edu",
]


def page_lines(img, step=700, pad=120):
    """OCR in horizontal strips: full tall pages get downscaled by the detector and lose lines."""
    a = np.asarray(img)
    boxes = []
    for y in range(0, a.shape[0], step):
        res, _ = ocr(a[y : y + step + pad])
        for box, text, conf in res or []:
            ys = [p[1] + y for p in box]; xs = [p[0] for p in box]
            mid = (min(ys) + max(ys)) / 2
            if y <= mid < y + step:  # each line belongs to exactly one strip
                boxes.append((min(ys), max(ys), min(xs), text.strip()))
    boxes.sort()
    lines, cur, cur_y = [], [], None
    for y0, y1, x0, t in boxes:
        mid = (y0 + y1) / 2
        if cur and abs(mid - cur_y) > (y1 - y0) * 0.6:
            lines.append(" ".join(t for _, t in sorted(cur)))
            cur = []
        cur.append((x0, t)); cur_y = mid if len(cur) == 1 else cur_y
    if cur:
        lines.append(" ".join(t for _, t in sorted(cur)))
    return lines


def main():
    pdfs = sorted(SRC.glob("Q*.pdf"), key=lambda p: int(p.stem[1:]))
    for pdf_path in pdfs:
        qn = {"Q33": "Q3"}.get(pdf_path.stem, pdf_path.stem)  # the file named Q33 is question 3
        pdf = pdfium.PdfDocument(str(pdf_path))
        lines = []
        for i in range(len(pdf)):
            lines += page_lines(pdf[i].render(scale=2.4).to_pil().convert("RGB"))
        lines = [l for l in lines if not any(re.search(p, l) for p in CHROME)]
        text = "\n".join(lines)
        # drop the echoed question block: everything before the first line after "...and a link."
        m = re.search(r"DOI or PMID,? and a link\.?", text)
        if m:
            text = text[m.end():].lstrip()
        (OUT / f"{qn}.md").write_text(text, encoding="utf-8")
        print(f"{pdf_path.name} -> {qn}.md  {len(pdf)} pages, {len(text)} chars", flush=True)


if __name__ == "__main__":
    main()
