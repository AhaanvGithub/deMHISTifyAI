# Findings log (for the summary, slides and Q&A)

## Q: "Why don't researchers just use CONCH / MUSK directly?"

Short answer: on this exact problem, used directly (zero-shot, i.e. just giving them text prompts), they are close to a coin flip and very overconfident.

Evidence:
- Published zero-shot results on MHIST (SSL vs hyperplastic polyp), weighted F1: QuiltNet 57.2, CONCH 54.6, PLIP 45.1 with prompt ensembling (52.1 / 50.2 / 43.1 without). Source: MHIST tile-level leaderboard, https://www.sota2.com/research/sota/tile-level-classification-on-mhist
- A supervised ImageNet ResNet-18 trained on MHIST labels reaches 92.7% AUC (MHIST paper, Wei et al. 2021, arXiv 2101.12355). The gap is labelled data, not model size.
- Prompt wording alone swings zero-shot results by up to ~25% in pathology VLM benchmarks (HistoVL, arXiv 2503.12990; PathLang, arXiv 2610.11329).
- Our own first test (2026-10-18, 4 MHIST tiles, 8 prompt-only CONCH judges): the panel split 4-4, while individual judges reported 97-100% confidence for opposite answers. Confident and contradictory.

What deMHISTifyAI does about it:
- Judges keep the prompt as a starting point but are trained on labelled MHIST images (soft targets = share of the 7 pathologists who said SSL), then calibrated so stated confidence matches observed accuracy.
- The prompt-only panel is kept as a baseline in our results table so the improvement is measured, not claimed.
- The app flags "untrained judges" if trained heads are missing.

Results table (fill from docs/results/judges_eval.json once labels arrive):

| Setting | Balanced accuracy | AUC | Calibration error (ECE) |
|---|---|---|---|
| 8 prompt-only judges (zero-shot panel) | | | |
| 8 trained judges (panel) | | | |
| Panel accuracy when judges agree / split | | | |
| Stress: grayscale / colour jitter / JPEG (flip rate) | | | |
| AI judge votes vs pathologist votes (Spearman) | | | |

## Other notes for Q&A
- CONCH runs in 0.3 s per snip on a laptop RTX 5060; the local vision model finds the slide region in ~6 s using 2.6 GB of GPU memory; Lumen literature judge ~19 s (5 runs in parallel), verdict ~10 s.
- LLMs never see the image: they only get measured feature scores, judge votes and excerpts from our curated paper library, and may only cite those excerpts.
- Literature judge confidence is reported two ways: how often it gives the same answer over 5 runs (measured), and what it says its confidence is (self-reported, known to be unreliable).
