# deMHISTifyAI

A desktop research tool for hard pathology calls. A researcher states their call and certainty, snips a slide region, and sees:
a heatmap of regions that support or contradict their call, a panel of 8 frozen CONCH judges, a literature-grounded LLM judge with citations (Lumen),
and a final verdict comparing the researcher's call with the evidence. First validated niche: sessile serrated lesion vs hyperplastic polyp (MHIST).

Research tool only, not for diagnosis.

- `research/`: embedding and evaluation scripts (run on NCSA Delta or a local GPU)
- `backend/`: local Python service (CONCH + Lumen)
- `app/`: Electron desktop app
- `papers/`: curated paper library (abstracts / open-access only)

Data (MHIST, model weights) is not included; each user must obtain access themselves.
