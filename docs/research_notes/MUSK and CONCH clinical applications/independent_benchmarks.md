# Independent benchmarking of pathology foundation models (2024–2026): where CONCH and MUSK land

Researcher notes, compiled 2026-10-17. About 20 tool calls. Several primary PDFs (THUNDER full text, the kaiko eva leaderboard SVG, and the MUSK and CONCH Nature papers) could not be machine-read. Gaps are flagged in each section.

## Q1. Which independent benchmarks exist, and where do CONCH and MUSK rank?

### Takeaway
On slide-level weakly supervised MIL using external clinical cohorts (Neidlinger/Kather, Nat. Biomed. Eng. 2025), CONCH ties Virchow2 for the best mean AUROC (0.71) out of 19 models. On tile-level frozen-feature benchmarks (THUNDER, NeurIPS 2025), CONCH, CONCH 1.5 and MUSK rank mid-to-low: well behind UNI2-h, Virchow2, H-optimus-0 and Prov-GigaPath, though ahead of Phikon-v2. On PathBench's private multi-centre clinical cohorts (2025), image-only models Virchow2 and H-optimus-1 rank highest. MUSK wins only isolated tasks there, such as breast PR status.

### Cited Findings

**Neidlinger et al., "Benchmarking foundation models as feature extractors for weakly supervised computational pathology"**
- Dates: arXiv 2408.15823, v1 28 Aug 2024, v2 8 Dec 2024. Published in *Nature Biomedical Engineering* in 2025, DOI 10.1038/s41551-025-01516-3. — [arXiv](https://arxiv.org/abs/2408.15823); [Springer ESM](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41551-025-01516-3/MediaObjects/41551_2025_1516_MOESM1_ESM.pdf)
- Scope of the final version: 19 histopathology foundation models plus 14 ensembles, tested on 13 patient cohorts (6,818 patients, 9,528 slides) and 31 weakly supervised tasks. The tasks cover morphology, biomarkers and prognosis in lung, colorectal, gastric and breast cancer. The cohorts are described as "truly external." The v1 preprint covered only 10 models (6,791 patients, 9,493 slides). — [arXiv](https://arxiv.org/abs/2408.15823)
- Averaged over all 31 tasks, CONCH and Virchow2 tie for the highest mean AUROC (0.71). Prov-GigaPath and DinoSSLPath follow at 0.69. — [arXiv PDF](https://arxiv.org/pdf/2408.15823). The search-tool summary attributed these values to the text; the published-version numbers were not checked against the journal page.
- By task family: CONCH has the highest mean AUROC on morphology (0.77, 5 tasks) and prognosis (0.63). Virchow2 and CONCH tie on biomarkers (0.73). — [arXiv PDF](https://arxiv.org/pdf/2408.15823)
- CONCH's lead is smaller in low-data settings and on low-prevalence tasks. A CONCH + Virchow2 ensemble beats the individual models in 55% of tasks. The authors conclude that "data diversity outweighs data volume." — [arXiv abstract](https://arxiv.org/abs/2408.15823)
- MUSK was not identified among the models in this benchmark from the pages I could read. I could not confirm whether MUSK is among the final 19.

**THUNDER (Tile-level Histopathology image UNDERstanding benchmark)**
- Marza, Fillioux, Boutaj, ..., Vakalopoulou. arXiv 2507.07860, v1 10 Jul 2025, v2 27 Oct 2025, v3 17 Feb 2026. NeurIPS 2025 Datasets & Benchmarks Track (Spotlight). It covers 23 foundation models on 16 datasets and evaluates KNN, linear probe, few-shot, segmentation, calibration and adversarial robustness. — [arXiv](https://arxiv.org/abs/2507.07860)
- THUNDER leaderboard values, as quoted in Filiot, Thaeter, Schmauch & Guillou, "Robustifying pathology foundation models via fine-tuning," arXiv 2607.22861 (24 Jul 2026), Appendix Table 5. The authors say they quote the official leaderboard values for previously released models. — [arXiv HTML](https://arxiv.org/html/2607.22861v1)

  Rank sum: lower is better. KNN, linear probe and few-shot: higher is better. Calibration and adversarial: lower is better. Per-metric rank in parentheses.

  | Model | Rank sum | KNN | Linear probe | Few-shot | Calib. | Adv. attack |
  |---|---|---|---|---|---|---|
  | UNI2-h (MP) | 35 | 83.3 (4) | 86.3 (1) | 79.8 (1) | 3.7 | 31.0 |
  | Virchow2 | 57 | 82.9 (5) | 84.8 (6) | 73.9 (20) | 3.9 | 31.1 |
  | Midnight-12k | 76 | 79.9 | 84.7 | 71.5 | 2.9 | 37.0 |
  | UNI | 81 | 80.8 | 83.5 | 78.1 | 3.8 | 40.3 |
  | KEEP (VLM) | 87 | 81.5 | 83.2 | 77.1 | 4.0 | 44.9 |
  | H-optimus-0 (MP) | 95 | 81.5 | 83.7 | 76.2 | 3.6 | 42.1 |
  | Hibou-B | 106 | 78.9 | 81.2 | 76.3 | 3.2 | 52.7 |
  | Prov-GigaPath | 110 | 79.5 | 82.9 | 75.5 | 3.4 | 42.1 |
  | Hibou-L | 110 | 78.6 | 83.7 | 73.8 | 4.7 | 39.5 |
  | **CONCH 1.5** | 112 | 79.9 (15) | 82.4 (20) | 75.0 (18) | 4.6 (23) | 75.8 (31) |
  | **CONCH** | 125 | 78.8 (21) | 81.9 (23) | 73.4 (24) | 4.1 (20) | 57.3 (29) |
  | Phikon-v2 | 152 | 73.9 (31) | 79.7 (31) | 71.8 (28) | 3.9 | 43.8 |
  | **MUSK** | 156 | 77.7 (26) | 81.1 (27) | 71.9 (27) | 4.0 (17) | 71.9 (30) |

  The source table also lists fine-tuned variants (UNI2-h† 31, Virchow2† 53, H-optimus-0† 50, Prov-GigaPath† 74), which are not shown here. PLIP, QuiltNet and H-optimus-1 do not appear in that table. — [arXiv 2607.22861](https://arxiv.org/html/2607.22861v1)

**PathBench (HKUST-led; Ma, Xu, Zhou, ..., Hao Chen)**
- arXiv 2505.20202v1, 26 May 2025. 19 PFMs, 64 tasks. All evaluation data come from private hospital cohorts that were kept out of pretraining. External validation uses separate hospitals. The project hosts an automated leaderboard ranked by average rank. — [arXiv HTML](https://arxiv.org/html/2505.20202v1); [project page](https://smartlab.cse.ust.hk/showcase/PathBench)
- Overall average rank: Virchow2 5.0, H-optimus-1 5.9, H-optimus-0 6.6, UNI2 7.1, mSTAR 7.4. The authors write that "Virchow2 and H-Optimus-1 are the most effective models overall." CONCH and MUSK are not in the top five; their exact average ranks were not in the extracted text. — [arXiv HTML](https://arxiv.org/html/2505.20202v1)
- MUSK has the top AUC on breast PR status prediction (0.8944). H-optimus-1 leads on ER (0.9185), H-optimus-0 on HER2 (0.8454) and UNI2 on CK5 (0.8481). — [arXiv HTML](https://arxiv.org/html/2505.20202v1)
- The paper is internally inconsistent: the text credits Virchow2 with 0.9865 on lung metastasis classification, while Table 3 assigns 0.9865 to H-optimus-1 and 0.9846 to Virchow2. — [arXiv HTML](https://arxiv.org/html/2505.20202v1)

**kaiko.ai eva**
- eva is an open evaluation framework with a frozen backbone. Patch tasks use a single-layer MLP head trained for 12,500 steps (batch 256, lr 3e-4, AdamW), averaged over 5 runs. Slide tasks use an ABMIL head averaged over 20 runs. Segmentation uses a conv decoder scored with Dice. Classification is scored with balanced accuracy. — [eva leaderboard docs](https://kaiko-ai.github.io/eva/main/leaderboards/)
- Leaderboard columns: BACH, BreakHis, CRC, Gleason, MHIST, PCam, Cam16Small, PANDASmall, CoNSeP, MoNuSAC, plus the average. The ranking excludes BACH. — [eva docs](https://kaiko-ai.github.io/eva/main/leaderboards/); [leaderboard SVG](https://raw.githubusercontent.com/kaiko-ai/eva/main/docs/images/leaderboards/pathology.svg)
- Third-party eva-protocol numbers from the MINT paper (arXiv 2603.07895, 2026): Virchow2 CRC 0.967 and MHIST 0.861; UNI2-h CRC 0.965 and MHIST 0.824. — [MINT arXiv](https://arxiv.org/pdf/2603.07895)

**Other 2025–2026 benchmarks**
- CellPath-Bench (arXiv 2608.21060, 2026) ranks CONCH first by win count (26), ahead of CONCH v1.5, SEAL (UNI2), H-optimus-0 and UNI2. This comes from a search snippet only; I did not read the full text. — [arXiv](https://arxiv.org/pdf/2608.21060)
- HistoVL / "How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark" (Al Majzoub et al., arXiv 2503.12990, 17 Mar 2025) compares histopathology VLMs (CLIP, PLIP, QuiltNet, BiomedCLIP, MI-Zero, CONCH) across 14 cohorts, 26 organs, 31 cancer types, more than 5M patches and more than 41K WSIs. Findings: most VLMs are highly sensitive to caption wording, with balanced accuracy dropping by up to 25% on metastasis detection. They also show low adversarial robustness and poor calibration (high ECE). — [arXiv](https://arxiv.org/abs/2503.12990)
- A single-institution digestive-pathology WSI study (arXiv 2505.00134, 2025) on 3,507 in-house WSIs found zero-shot average AUC of 0.876 for CONCH versus 0.666 for QuiltNet. — [arXiv](https://arxiv.org/abs/2505.00134v1)

### Inferences
- How CONCH ranks depends on the evaluation regime. It does best with slide-level MIL on external clinical cohorts (Neidlinger) and on morphology and prognosis tasks. Its relative standing drops on tile-level frozen-feature probing (THUNDER) and on large private multi-centre suites (PathBench). There, image-only models trained on far more data (Virchow2, UNI2-h, H-optimus-0/1) lead.
- MUSK has the weakest independent tile-level evidence. It ranks below CONCH on THUNDER (rank sum 156 vs 125) and close to Phikon-v2. Its standout results come mostly from its own paper, plus isolated wins such as PR status in PathBench.
- CONCH and MUSK both look fragile under adversarial attack in THUNDER (ranks 29–31 of the listed models). This fits HistoVL's finding of low adversarial robustness for VLMs.

### Gaps
- THUNDER's official per-dataset scores for CONCH and MUSK: the full-text PDF and HTML were too large to fetch.
- The kaiko eva leaderboard: the model names and full scores sit in an SVG that I could not parse completely, so CONCH and MUSK eva scores are unverified.
- Patho-Bench (Mahmood lab) and HEST-Benchmark rankings for CONCH and MUSK: not retrieved. The Filiot 2026 paper reportedly includes a HEST table, but I did not extract it.
- PathBench's exact average ranks for CONCH, MUSK, CHIEF, PLIP and Prov-GigaPath.
- Whether MUSK is among Neidlinger's final 19 models.

## Q2. Does vision-language pretraining (CONCH/MUSK) beat image-only SSL, and by how much?

### Takeaway
Not consistently. On slide-level MIL (Neidlinger), CONCH ties Virchow2 overall (both 0.71 mean AUROC) and edges ahead on morphology (0.77) and prognosis (0.63). On tile-level linear probe, KNN and few-shot (THUNDER), CONCH and MUSK trail UNI2-h and Virchow2 by roughly 3–5 points. Zero-shot and retrieval are the only capabilities unique to VLMs.

### Cited Findings
- **Slide-level MIL:** CONCH and Virchow2 tie at 0.71 mean AUROC over 31 tasks. CONCH's advantage shrinks in low-data and low-prevalence settings. The two model types are complementary: their ensemble wins 55% of tasks. — [Neidlinger arXiv](https://arxiv.org/abs/2408.15823)
- **Tile-level linear probe (THUNDER):** UNI2-h 86.3, Virchow2 84.8, CONCH 81.9, MUSK 81.1. That puts CONCH 4.4 points and MUSK 5.2 points behind UNI2-h. — [arXiv 2607.22861](https://arxiv.org/html/2607.22861v1)
- **Tile-level KNN (THUNDER):** UNI2-h 83.3, Virchow2 82.9, CONCH 1.5 79.9, CONCH 78.8, MUSK 77.7. — [arXiv 2607.22861](https://arxiv.org/html/2607.22861v1)
- **Few-shot (THUNDER):** UNI2-h 79.8, UNI 78.1, CONCH 1.5 75.0, CONCH 73.4, MUSK 71.9. Virchow2 scores only 73.9 here, so CONCH 1.5 beats it on few-shot. — [arXiv 2607.22861](https://arxiv.org/html/2607.22861v1)
- **Text-supervised alternative:** KEEP, a knowledge-enhanced VLM, scores linear probe 83.2, KNN 81.5 and few-shot 77.1 in THUNDER, ahead of both CONCH and MUSK. This shows that VLMs are not inherently worse at the tile level. — [arXiv 2607.22861](https://arxiv.org/html/2607.22861v1)
- **Weakly supervised AUPRC, balanced accuracy and F1:** one independent study reports CONCH highest on all three. This is a search snippet only, and I could not identify the source paper. — via [PathBench search context](https://arxiv.org/html/2505.20202v1); unverified

### Inferences
- Vision-language pretraining seems to add value mainly in slide-level aggregation of morphology-related signal, and through zero-shot and retrieval capabilities. It does not add value as a raw tile embedding. For tile linear probing, the largest image-only DINOv2-style models are the stronger choice.

### Gaps
- No independent head-to-head retrieval benchmark (image↔text) including both MUSK and CONCH was found. KEEP explicitly excluded CONCH and MUSK from its retrieval comparison over training-data leakage concerns; the comparison sits in its Table S6, which I did not read. — [KEEP arXiv 2412.13126](https://arxiv.org/pdf/2412.13126)

## Q3. Results on the datasets available to the team (NCT-CRC-HE-100K / CRC-VAL-HE-7K, PathMMU, BACH, MHIST, Osteo, ARCH, Quilt-1M, PathGen-1.6M)

### Takeaway
I did not retrieve verified, citable per-dataset zero-shot or linear-probe numbers for CONCH, MUSK, PLIP and QuiltNet on these datasets. The available independent numbers are CRC and MHIST for image-only models under the eva protocol, and qualitative claims from the Quilt-1M paper.

### Cited Findings
- **eva-protocol CRC (NCT-CRC-HE-100K train / CRC-VAL-HE-7K test) and MHIST:** Virchow2 0.967 and 0.861; UNI2-h 0.965 and 0.824 (balanced accuracy). — [MINT arXiv 2603.07895](https://arxiv.org/pdf/2603.07895)
- **BACH (4-class), from the OpenPath model card (not peer-reviewed):** UNI2-h 0.908, OpenMidnight 0.906. — [OpenPath HF card](https://huggingface.co/taejoon89/openpath)
- **BACH, CRC and MHIST linear-probe baseline (kaiko, "Towards Large-Scale Training of Pathology Foundation Models," arXiv 2404.15217, Apr 2024):** DINO ViT-S8 scores 0.834 on BACH, 0.946 on CRC and 0.832 on MHIST. — [arXiv](https://arxiv.org/pdf/2404.15217)
- **QuiltNet:** the Quilt-1M paper (arXiv 2306.11207, NeurIPS 2023) reports zero-shot results on 13 datasets, including PatchCamelyon, NCT-CRC-HE-100K, BACH and MHIST. It says QuiltNet outperforms CLIP, BiomedCLIP and PLIP on 12 of the 13. — [arXiv](https://arxiv.org/pdf/2306.11207)
- **CONCH on CRC100K:** the MR-PHE method claims 8.32% higher accuracy than the second-best model on CRC100K, compared against CONCH, PLIP and BiomedCLIP. Absolute numbers were not extracted. — [arXiv 2503.10731](https://arxiv.org/pdf/2503.10731)
- **MUSK (Xiang et al., *Nature* 638:769–778, Feb 2025, DOI 10.1038/s41586-024-08378-w):** pretrained on 50M pathology images from 11,577 patients and 1B text tokens. The paper evaluates 23 patch- and slide-level benchmarks covering retrieval, VQA, classification and biomarker prediction. — [RePEc record](https://ideas.repec.org/a/nat/nature/v638y2025i8051d10.1038_s41586-024-08378-w.html); [Stanford HAI](https://hai.stanford.edu/news/stanfords-multimodal-ai-model-advances-personalized-cancer-care)

### Inferences
- NCT-CRC is close to saturated for strong encoders (about 0.965 balanced accuracy), so it separates models poorly. MHIST shows more spread (0.82–0.86).

### Gaps
- Exact zero-shot and linear-probe numbers for CONCH, MUSK, PLIP and QuiltNet on CRC100K, BACH, MHIST, PathMMU, Osteo, ARCH and PathGen-1.6M were not retrieved; the Nature and Nat. Med. pages were paywalled or redirected. The primary sources to extract them from:
  - MUSK *Nature* paper, Extended Data tables. DOI 10.1038/s41586-024-08378-w.
  - CONCH, Lu et al., *Nat. Med.* 2024 (arXiv 2307.12914).
  - Quilt-1M (arXiv 2306.11207).
  - PathGen-CLIP paper.
  - PathMMU paper.
- I did not use any numbers recalled from memory.

## Q4. Critiques of benchmark practice

### Takeaway
The main critiques are:
- overlap between pretraining data and public evaluation data (TCGA, GTEx, PubMed-derived captions);
- medical-centre/site confounding in embeddings;
- task saturation on public tile datasets;
- sensitivity of VLM evaluation to prompt wording;
- inconsistent reporting.

Newer benchmarks respond by using private multi-centre cohorts, external validation and case-level splits.

### Cited Findings
- **Pretraining overlap:** the OpenPath model card argues that public benchmarks like CRC and BACH come from public repositories (TCGA, GTEx, etc.) used in FM pretraining, so apparent gains are "confounded by train/test leakage." This is a model-author claim. — [OpenPath HF](https://huggingface.co/taejoon89/openpath)
- **VLM retrieval leakage:** KEEP excluded MI-Zero, CONCH and MUSK from its retrieval comparisons. MI-Zero and CONCH do not release their image–text data, and MUSK's training data includes PubMed papers and textbooks, which risks leakage. — [KEEP arXiv 2412.13126](https://arxiv.org/pdf/2412.13126)
- **PathBench's leakage controls:**
  - It uses only private hospital data, excluded from pretraining.
  - It applies case-level splits for some tasks "to avoid potential data leakage."
  - It validates on external hospitals.
  - Classification uses 10 seeds with a 7:1:2 split; survival uses 5-fold cross-validation repeated 3 times; results carry bootstrapped 95% CIs.
  - Not all splits are explicitly patient-level: the lung H1 split is only label-stratified.
  — [arXiv 2505.20202](https://arxiv.org/html/2505.20202v1)
- **Neidlinger's design:** the authors stress "truly external" cohorts that were not used in FM pretraining. — [arXiv](https://arxiv.org/abs/2408.15823)
- **Site confounding (de Jong, Marcus & Teuwen, "Current Pathology Foundation Models are unrobust to Medical Center Differences," arXiv 2501.18055, v2 1 Feb 2025):** all 10+ tested FMs encode the medical centre strongly, and centre is predicted more accurately than tissue or cancer type.
  - Robustness Index: Virchow2 1.2 (the only value above 1), UNI2-h 0.93, Virchow 0.93, UNI 0.88, Phikon 0.84, Phikon-v2 0.74.
  - The evaluation used TCGA-2k: 5 cancer types × 5 centres.
  - CONCH and MUSK were not evaluated.
  — [arXiv HTML](https://arxiv.org/html/2501.18055v2)
- **PathoROB follow-up ("Towards robust foundation models for digital pathology," listed as Nature Communications):** robustness deficits across all 20 evaluated FMs. — [Paperity](https://paperity.org/p/371468284/towards-robust-foundation-models-for-digital-pathology)
- **Prompt sensitivity:** VLM zero-shot results swing by up to 25% in balanced accuracy with caption wording. — [HistoVL arXiv 2503.12990](https://arxiv.org/abs/2503.12990)
- **Reporting inconsistency:** PathBench's text and Table 3 disagree on its top lung metastasis result. — [arXiv 2505.20202](https://arxiv.org/html/2505.20202v1)
- **Leaderboard design choices that change results:** eva excludes BACH from its average ranking and subsamples Camelyon16 to 1,000 foreground patches per slide, which the eva docs say affects all models' results. — [eva docs](https://kaiko-ai.github.io/eva/main/leaderboards/)

### Inferences
- The independent tile-level rankings and zero-shot numbers that favour CONCH, MUSK, PLIP or QuiltNet on public datasets (CRC100K, BACH, PCam) should be treated as potentially optimistic. NCT-CRC is effectively saturated. External, patient-level, multi-centre evaluation is a better guide (Neidlinger, PathBench).

### Gaps
- I found no study that quantifies the actual TCGA or PubMed overlap specifically for CONCH or MUSK.
- I found no Robustness Index value for CONCH or MUSK.

## Q5. Compute and efficiency comparisons

### Takeaway
I retrieved little independently verified compute data within budget. The rankings above suggest that larger image-only models (ViT-H/ViT-g, such as UNI2-h, Virchow2 and H-optimus) buy their tile-level edge with substantially more compute than CONCH's ViT-B.

### Cited Findings
- MUSK pretraining used 50M images from 11,577 patients and 1B text tokens. — [RePEc/Nature](https://ideas.repec.org/a/nat/nature/v638y2025i8051d10.1038_s41586-024-08378-w.html)
- CONCH pretraining used more than 1.17M image–caption pairs. — [arXiv 2307.12914](https://arxiv.org/abs/2307.12914)
- Virchow2 and UNI2-h are run at 1280-D and 2560-D embedding settings in the robustness study. — [de Jong et al.](https://arxiv.org/html/2501.18055v2)
- The THUNDER table reports mixed-precision (MP) variants for UNI2-h and H-optimus-0. Precision choice moved H-optimus-0's rank sum (MP 95 vs fine-tuned 50). — [arXiv 2607.22861](https://arxiv.org/html/2607.22861v1)

### Inferences
- Using CONCH (ViT-B) as a slide-level MIL feature extractor matches Virchow2 at much lower inference cost. This is the efficiency case for CONCH. The exact parameter counts (commonly cited as CONCH ViT-B about 90M vs Virchow2 ViT-H about 632M) were not verified in this session.

### Gaps
- Verified parameter counts, embedding dimensions and throughput (tiles/s) for CONCH, MUSK, UNI2-h, Virchow2, H-optimus-0/1, Prov-GigaPath, Phikon-v2, CHIEF, PLIP and QuiltNet. The THUNDER paper reportedly includes a feature-extraction cost analysis, but it could not be fetched (file too large).
