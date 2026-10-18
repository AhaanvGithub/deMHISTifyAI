# CONCH and MUSK: Pathology Vision-Language Foundation Models and Their Clinical Applications

Research date: 2026-10-17. Nature.com article pages could not be fetched directly (login redirect), so the numbers below come from PMC author manuscripts, arXiv versions, and the official GitHub repos. Each source is dated.

## Q1. Training data, architecture, and headline benchmark results

### Takeaway
CONCH (Nature Medicine, March 2024) is a CoCa-style ViT-B/16 model trained on about 1.17M histopathology image-caption pairs. Its strengths are zero-shot slide and ROI classification, retrieval, and zero-shot segmentation. MUSK (Nature, January 2025) is a BEiT-3-style model with shared attention and modality experts. It was first trained on unpaired data (50M TCGA image patches plus 1B PubMed text tokens), then aligned on about 1M pairs. It leads on retrieval, VQA, biomarker, prognosis and immunotherapy-response tasks, and beats CONCH head-to-head in its own paper.

### Cited Findings

**CONCH: bibliographic details**
- Lu MY, ..., Mahmood F. "A visual-language foundation model for computational pathology." *Nature Medicine* 2024 Mar 19;30(3):863–874. DOI 10.1038/s41591-024-02856-4; PMID 38504017; PMCID PMC11384335. The arXiv preprint (2307.12914) was posted July 2023. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/); [arXiv](https://arxiv.org/abs/2307.12914)

**CONCH: training data**
- The team collected 1,786,362 image-caption pairs from educational sources ("EDU") and PubMed Central Open Access figures ("PMC-Path"). Removing nonhuman cases left 1,170,647 human pairs, the "1.17M" figure. An H&E classifier then selected an H&E subset of 457,372 pairs. — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)
- The model was not pretrained on large public slide collections (TCGA, PAIP, GTEx). The authors say this lowers the risk of benchmark contamination. — [GitHub mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH)

**CONCH: architecture**
- CoCa framework with three parts:
  - Image encoder: ViT-Base (12 layers, 12 heads, 768-dim, 16×16 patch tokens).
  - Text encoder: GPT-style, 12 layers.
  - Multimodal decoder: 12 layers. — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)
- Before alignment, each unimodal encoder was pretrained separately:
  - Image encoder: iBOT self-supervised training on 16M 256×256 tiles from 21,442 in-house WSIs.
  - Text model: 24-layer GPT-style model trained on more than 550k surgical pathology reports and more than 400k PubMed abstracts. Its first 12 layers initialize the text encoder. — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)
- The public release (`conch_ViT-B-16`, built on open_clip) leaves out the multimodal decoder weights, which are needed for captioning, to avoid leaking proprietary data or PHI. The authors state that the vision and text encoders, and therefore classification and retrieval results, are unaffected. — [GitHub](https://github.com/mahmoodlab/CONCH)

**CONCH: zero-shot classification (balanced accuracy unless noted; uses prompt ensembles)**
- Slide-level, using MI-Zero aggregation of tile scores:

| Dataset | CONCH | Comparison |
|---|---|---|
| TCGA NSCLC | 90.7% | +12.0 points over PLIP |
| TCGA RCC | 90.2% | +9.8 over PLIP |
| TCGA BRCA | 91.3% | other models 50.7% (PLIP) to 55.3% (BiomedCLIP) |
| DHMC LUAD pattern (Cohen's κ) | 0.200 | +0.12 over PLIP, P=0.055, not significant |

- ROI-level:

| Dataset | CONCH | Comparison |
|---|---|---|
| SICAP Gleason (quadratic κ) | 0.690 | +0.140 over BiomedCLIP |
| CRC100k | 79.1% | +11.7 over PLIP |
| WSSS4LUAD | 71.9% | +9.5 over PLIP |
| EBRAINS (30 classes, rare brain tumors) | 37.1% | chance 3.3%; +17.0 over BiomedCLIP |

  — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)

**CONCH: supervised and few-shot (frozen features)**
- Slide-level ABMIL: BRCA 86.7%, RCC 94.2%, NSCLC 93.3% balanced accuracy, and EBRAINS 68.2% (+6.8 over CTransPath). — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)
- ROI-level supervised: CRC100k 93.8% (ties CTransPath) and SICAP κ 0.833 (CTransPath 0.835). With full supervision, CONCH roughly matches a strong vision-only encoder at the ROI level. — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)
- Few-shot: on BRCA, CONCH with 8 labels per class outperformed PLIP, BiomedCLIP and OpenAI CLIP trained with 64 labels per class. — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)

**CONCH: zero-shot cross-modal retrieval (mean recall over Recall@1/5/10)**
- Average text-to-image mean recall was 44.0%, +17.3 over BiomedCLIP:
  - Source A (n=797): 68.8%
  - Source B (n=1,755): 39.0%
  - TCGA LUAD (n=165): 24.0% (+5.3, P=0.22, not significant)

  — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)

**CONCH: zero-shot segmentation (Dice)**
- SICAP: 0.601 (PLIP 0.549; BiomedCLIP 0.484).
- DigestPath: 0.615 (PLIP 0.426; BiomedCLIP 0.446). — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)

**CONCH: captioning**
- Captioning was evaluated with METEOR and ROUGE (Extended Data Fig. 9, Supplementary Table 30). I could not extract the exact values. The public release cannot caption because the decoder was withheld. — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/); [GitHub](https://github.com/mahmoodlab/CONCH)

**CONCH: later comparison against UNI posted in the repo (frozen features)**

| Task | Method / metric | CONCH | UNI |
|---|---|---|---|
| EBRAINS-C | ABMIL, balanced accuracy | 0.868 | 0.883 |
| EBRAINS-F | ABMIL, balanced accuracy | 0.689 | 0.675 |
| PANDA | ABMIL, quadratic κ | 0.934 | 0.946 |
| OncoTree-108 | ABMIL, balanced accuracy | 0.515 | 0.538 |
| IHC ER/PR | ABMIL, quadratic κ | 0.819 | 0.785 |
| CRC-100K-Raw | KNN (K=20), balanced accuracy | 0.941 | 0.925 |
| TCGA Uniform Tumor | KNN (K=20), balanced accuracy | 0.556 | 0.595 |

  — [GitHub](https://github.com/mahmoodlab/CONCH)

**MUSK: bibliographic details**
- Xiang J, Wang X, ..., Li R. "A vision-language foundation model for precision oncology." *Nature* 2025, published January 8, 2025; vol. 638, pp. 769–778; DOI 10.1038/s41586-024-08378-w; PMCID PMC12295649. — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649); [Stanford Health Care listing](https://stanfordhealthcare.org/publications/931/931479.html)

**MUSK: training data**
- Phase 1 used unpaired masked modeling:
  - Images: 50M patches from about 33,000 TCGA H&E WSIs, covering 11,577 patients and 33 tumor types.
  - Text: 1B tokens from 1,001,800 PubMed Central OA pathology articles.
- Phase 2 used contrastive alignment on about 1M image-text pairs: Quilt1M (802k) plus PathAsst (207k), with low-similarity pairs filtered out. — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)
- A pathology image tokenizer was trained on 5M TCGA images, using CTransPath as teacher. — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)
- Because MUSK was pretrained on TCGA, its TCGA-based downstream results (for example pan-cancer prognosis) involve images from the same source as pretraining. This is an inference from the data description.

**MUSK: architecture and training**
- Adapted from BEiT-3: shared self-attention with separate vision and language feed-forward experts. — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)
- Inputs and masking:
  - Images: 384×384, 16×16 patches.
  - Text: 64k SentencePiece vocabulary (the repo uses an XLM-RoBERTa tokenizer, max length 100).
  - Masking: 15% of text tokens and 40% of image patches.
- Phase 1 ran 1M steps at batch size 2048. Phase 2 ran 20 epochs at batch size 3072 with a peak learning rate of 1e-4.
- Phase 2 loss is contrastive plus 0.1 times an auxiliary MLM loss through a cross-attention decoder. — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649); [GitHub lilab-stanford/MUSK](https://github.com/lilab-stanford/MUSK)
- The release is `musk_large_patch16_384` (timm). The paper text I could access did not give a parameter count; it is in Extended Data Fig. 7 and Supplementary Table 16.
- The repo recommends different settings by use:
  - Zero-shot: `with_head=True, out_norm=True, ms_aug=False`.
  - Linear probe and MIL: `ms_aug=True`.

  — [GitHub](https://github.com/lilab-stanford/MUSK)

**MUSK: headline results (23 patch- and slide-level tasks)** — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)
- **Cross-modal retrieval (Recall@50, image-to-text)**
  - PathMMU: 34.4% vs CONCH 27.3%.
  - BookSet: 74.8% vs CONCH 71.3%.
  - Text-to-image: +4.0 and +7.5 points.
- **VQA (PathVQA):** 73.2% accuracy, vs 68.9% for K-PathVQA, the best task-specific model.
- **Image retrieval (BRACS, mMV@5):** +22.3% over CLIP, +8.6% over PLIP, +2.5% over CONCH.
- **Zero-shot classification (gain over the second-best model):** PatchCamelyon +10.5%, SkinCancer +27.5%, PanNuke +7.3%, UniToPatho +10.1%.
- **Few-shot:** at 10 shots, the largest gain was on UniToPatho (+9.8% over second-best).
- **Supervised / linear evaluation on 12 datasets (frozen features):** average accuracy 88.2%. Margins: CONCH +2.2, PLIP +9.1, Quilt1M +11.7, BiomedCLIP +11, CLIP +17.5.
- **Biomarkers:**
  - HER2 (BCNB): AUC 0.826 (0.813–0.839) vs GigaPath 0.786 and CONCH 0.771 (P=0.008).
  - ER, PR (BCNB) and IDH (MUV-IDH): significant gains reported; other AUCs not in the accessible text.
- **Melanoma 5-year relapse (VisioMel, n=1,342):** AUC 0.833 (0.818–0.847). Specificity at 90% sensitivity improved by about 12% (P=0.0079); an Extended Data caption says about 15%.
- **Pan-cancer prognosis (TCGA, 16 cancers, 6,602 patients, 7,927 slides)**
  - MUSK mean c-index 0.746 vs overall stage 0.645.
  - Other foundation models: CONCH 0.684, GigaPath 0.684, PLIP 0.667, Quilt1M 0.672, BiomedCLIP 0.668, UNI 0.649, Virchow 0.672.
  - MUSK ablations: image-only 0.673, text-only 0.654.
  - Best cancer: renal cell carcinoma, c-index 0.887. The prognosis model is multimodal, using slides plus pathology reports.
- **Immunotherapy, NSCLC (n=118)**
  - Response AUC 0.768 (0.724–0.812) vs PD-L1 0.606; other multimodal models 0.636–0.692.
  - PFS c-index 0.705 vs PD-L1 0.574.
  - PFS hazard ratio 2.54; median PFS 4.3 vs 13.4 months.
  - In the PD-L1 TPS=0 subgroup, hazard ratio 7.38.
- **Immunotherapy, gastro-esophageal (n=101)**
  - Response AUC 0.762 (0.718–0.805) vs MSI-H 0.616.
  - PFS hazard ratio 3.49; median PFS 3.6 vs 14.1 months.

**TITAN (CONCH's slide-level successor) and CONCH v1.5**
- Ding T, Wagner SJ, Song AH, Chen RJ, ..., Mahmood F. "A multimodal whole-slide foundation model for pathology." *Nature Medicine* 2025, DOI 10.1038/s41591-025-03982-3. The preprint (arXiv 2411.19666) appeared November 2024; the model was released December 2, 2024. — [arXiv](https://arxiv.org/html/2411.19666v1); [GitHub CONCH news](https://github.com/mahmoodlab/CONCH); [Broad](https://www.broadinstitute.org/publications/broad1370446)
- **CONCH v1.5:** an extended CONCH trained on 1.26M image-caption pairs (CoCa, 20 epochs), covering FFPE, frozen and IHC images. TITAN uses it as the patch encoder on 512×512 patches at 20×, producing 768-dim features. LazySlide lists CONCH v1.5 at 158.9M parameters under CC-BY-NC-ND-4.0, which suggests a ViT-L-scale model. I did not confirm the backbone size in the paper. — [arXiv 2411.19666](https://arxiv.org/html/2411.19666v1); [LazySlide docs](https://lazyslide.readthedocs.io/en/latest/api/_autogen/lazyslide_models.multimodal.Titan.html)
- **TITAN training:**
  - Data: Mass-340K, 335,645 WSIs from 20 organs (90.9% H&E, 9.1% IHC), drawn from in-house slides and GTEx.
  - Stage 1: iBOT on 8K×8K regions.
  - Stage 2: 423,122 ROI captions generated by PathChat.
  - Stage 3: 182,862 WSI-report pairs.
  - Model: 6-layer ViT with 2D ALiBi; 48.5M parameters.

  — [arXiv](https://arxiv.org/html/2411.19666v1)
- **TITAN results (relative to PRISM unless noted):**
  - Linear probe, morphological subtyping: about +8.4%.
  - Survival: +3.62% over CHIEF.
  - Few-shot, TCGA-UT-8K at 16 shots: +22.4% over CHIEF.
  - Zero-shot multi-class balanced accuracy: +56.52%.
  - Report generation (average of METEOR/ROUGE/BLEU): about +161%.
  - Rare-cancer retrieval Accuracy@K: +14.8%.
  - Slide-to-report retrieval: +20.5%.

  — [arXiv](https://arxiv.org/html/2411.19666v1)

### Inferences
- For a hackathon working with frozen weights at the tile level:
  - CONCH v1 (ViT-B/16, 224-px) is cheaper to run.
  - MUSK (large, 384-px) gives somewhat stronger linear-probe and retrieval results in its own paper (+2.2 points average linear accuracy over CONCH across 12 datasets).
  - Both are directly usable for zero-shot via their text encoders.
- MUSK's flagship clinical results (prognosis, immunotherapy) use multimodal slide+report or clinical inputs and MIL-style aggregation. They are not pure tile-level zero-shot results, so they should not be cited as evidence that tile features alone predict immunotherapy response.

### Gaps
- I did not extract exact CONCH captioning METEOR/ROUGE values, which are in the supplementary material.
- I did not find MUSK's parameter count or its exact ER/PR/IDH AUCs in the accessible text.
- I could not confirm CONCH v1.5's backbone size and initialization (some secondary sources suggest a UNI-initialized ViT-L) from a primary source.

## Q2. Follow-up papers (2024–2026) using CONCH or MUSK on clinical tasks

### Takeaway
Independent benchmarks consistently rank CONCH among the best frozen feature extractors for slide-level weakly supervised tasks. The Kather-group benchmark (2024–2025) ranked it first of 19 models. CONCH has also been extended into TITAN. MUSK is increasingly used as one backbone among several in 2025–2026 work (grading, fusion, rare-cancer prompt tuning, robustness), but I found few MUSK-only clinical validations.

### Cited Findings
- **Neidlinger et al., Kather group** (arXiv 2408.15823, August 2024; later published in *Nature Biomedical Engineering* 2025, DOI 10.1038/s41551-025-01516-3)
  - Benchmarked 19 foundation models on 13 cohorts (6,818 patients, 9,528 slides) covering lung, colorectal, gastric and breast cancer, across biomarker, morphology and prognosis tasks.
  - CONCH performed best overall, with Virchow2 a close second.
  - A CONCH+Virchow2 ensemble beat the individual models on 55% of tasks.
  - The authors conclude that "data diversity outweighs data volume."

  — [arXiv](https://arxiv.org/abs/2408.15823); [TU Dresden record](https://fis.tu-dresden.de/portal/en/publications/benchmarking-foundation-models-as-feature-extractors-for-weakly-supervised-computational-pathology(7db74965-b811-4cf6-9806-1d59f1c2031e).html)
- **TITAN** (Mahmood Lab, Nature Medicine 2025) builds on CONCH v1.5 patch features for slide-level zero-shot classification, rare-cancer retrieval and report generation. Figures are in Q1. — [arXiv 2411.19666](https://arxiv.org/html/2411.19666v1)
- **PathFMTools** (arXiv 2511.19751, November 24, 2025; ML4H 2025; also on PMC as PMC13197084)
  - Compared CONCH and MUSK for histological grading of cutaneous squamous cell carcinoma on 440 WSIs.
  - Benchmarked multiple adaptation strategies and validated training small specialist models on foundation-model embeddings.
  - I could not extract the numeric results. — [arXiv](https://arxiv.org/abs/2511.19751); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13197084)
- **PathPT** (He et al., arXiv 2508.15904, August 2025; listed by search as *Nature Communications* 2026, DOI 10.1038/s41467-026-71715-2)
  - Few-shot prompt tuning on VLM backbones (PLIP, CONCH, MUSK, KEEP).
  - Data: 8 rare-cancer datasets (4 adult, 4 pediatric; 56 subtypes; 2,910 WSIs) plus 3 common-cancer sets.
  - EBRAINS (30 subtypes) balanced accuracy, PathPT at 10 shots: KEEP 0.679, MUSK 0.519, CONCH 0.491, PLIP 0.251.
  - Zero-shot balanced accuracy, read from the first row of Supplementary Table S1 (an extraction I could not fully verify): KEEP 0.408, MUSK 0.253, CONCH 0.204, PLIP 0.111.
  - MUSK at 1 shot was near chance (0.033).

  — [arXiv html](https://arxiv.org/html/2508.15904v1)
- **KEEP** (arXiv 2412.13126; v1 December 17, 2024, v2 January 27, 2026)
  - A knowledge-graph-enhanced pathology VLM: 11,454 diseases, 139,143 attributes, 143k semantic groups.
  - Evaluated on 18 public benchmarks (more than 14,000 WSIs) and 4 institutional rare-cancer sets (926 cases).
  - Claims to beat existing VLMs, with "substantial gains for rare subtypes." It is a direct competitor to and successor of CONCH/MUSK for zero-shot use. — [arXiv](https://arxiv.org/abs/2412.13126)
- **Information-driven fusion** (arXiv 2512.11104, December 2025)
  - Fused tile-level FMs (CONCH v1.5, MUSK, Virchow2, H-Optimus1, Prov-GigaPath) and slide-level FMs (TITAN, CHIEF, MADELEINE) for tumor characterization.
  - Correlation-guided fusion focused attention more on tumor regions.

  — [arXiv](https://arxiv.org/html/2512.11104)
- **PathoROB robustness benchmark** (Kömen et al., arXiv 2507.17845, July 2025; *Nature Communications* 2026, DOI 10.1038/s41467-026-73923-2)
  - Scope: 20 FMs; 4 datasets; 28 biological classes; 34 medical centers.
  - All 20 models showed robustness deficits against non-biological (site/scanner/stain) variation.
  - More robust FMs, vision-language alignment, and post-hoc robustification reduced the risk of errors but did not eliminate it.
  - A search snippet says MUSK scored lower on representation robustness but, after robustification, matched Virchow2, Atlas and CONCH v1.5 on downstream performance. I did not verify this in the full text. — [arXiv](https://arxiv.org/abs/2507.17845); [search-surfaced Nature Comms link](https://www.nature.com/articles/s41467-026-73923-2)
- **Mitotic-figure benchmark** (arXiv 2508.04441, 2025) includes foundation models for mitosis classification. The search surfaced it, but I did not verify whether CONCH or MUSK results appear in it. — [arXiv](https://arxiv.org/pdf/2508.04441)
- **Whole-slide knowledge-enhanced model** (Nature Communications 2025, s41467-025-66220-x) uses CONCH as a vision-language baseline and reports +1.02% (P<0.001) over it on IHC biomarker prediction. This comes from a search snippet and was not verified. — [Nature Comms](https://www.nature.com/articles/s41467-025-66220-x)

### Inferences
- In practice, CONCH has become the default VLM backbone and baseline in 2024–2026 papers. MUSK usually appears as one of several comparators, not as the only backbone in clinical validations.
- Ensembling or fusing CONCH with a vision-only model (for example Virchow2) is a well-supported, cheap way to gain performance with frozen weights.

### Gaps
- I found no verified published study applying CONCH or MUSK specifically to **cytology**, **frozen sections**, or **IHC** as a primary task, apart from CONCH v1.5/TITAN including frozen/IHC data in pretraining and the CONCH repo's ER/PR IHC benchmark (κ 0.819).
- A search surfaced an intraoperative pathology FM preprint (arXiv 2510.04861), but I did not verify whether it uses CONCH.
- I found no independent external replication of MUSK's immunotherapy-response results.

## Q3. Where they excel vs. underperform; where zero-shot works vs. fails

### Takeaway
Zero-shot works well for coarse, morphologically distinct, few-class tasks: NSCLC, RCC and BRCA subtyping at about 90% balanced accuracy, CRC tissue types, and tumor vs. normal. It fails or degrades for many-class rare tumors, where EBRAINS 30-class zero-shot is about 0.2–0.4, ordinal or graded tasks (DHMC LUAD κ 0.200), and fine-grained segmentation. It is also prompt-sensitive and affected by stain and site shifts. With frozen features plus a small supervised head, both models are near the top of public benchmarks.

### Cited Findings
- **Strong zero-shot on coarse tasks:** CONCH TCGA NSCLC 90.7%, RCC 90.2%, BRCA 91.3%; CRC100k 79.1%. — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)
- **Weak zero-shot on hard tasks:** EBRAINS 30-class 37.1% (vs 68.2% with supervised ABMIL); DHMC LUAD pattern κ 0.200, not significant vs PLIP; TCGA LUAD retrieval 24.0%. — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)
- **Rare cancers:** zero-shot balanced accuracy on EBRAINS is about 0.1–0.4 across PLIP, CONCH, MUSK and KEEP. The PathPT authors attribute weak PathPT results with CONCH/MUSK backbones to weak tile-level zero-shot predictions. — [arXiv 2508.15904](https://arxiv.org/html/2508.15904v1)
- **Zero-shot grounding:** zero-shot Dice is low for all VLM backbones on CAMELYON16 (0.129–0.207) and PANDA (0.255–0.327). — [arXiv 2508.15904](https://arxiv.org/html/2508.15904v1)
- **Prompt sensitivity:** a study comparing Quilt-Net, Quilt-LLaVA and CONCH on digestive-tract WSIs (arXiv 2505.00134, May 2025) found that the best prompt varied by model and that near-synonymous terms changed AUC. CONCH was reported as the most robust of the three.
  - A search summary cites CONCH AUC moving from 0.935 to 0.736 across prompt sets. I did not verify this in the primary text. — [arXiv 2505.00134](https://arxiv.org/pdf/2505.00134)
- **Prompt length:** a PathLang benchmark preprint (arXiv 2610.11329, October 2026) reports, via a search summary, that CONCH does well with short prompts but degrades with long descriptions, while KEEP shows the opposite. Not verified in full text. — [arXiv](https://arxiv.org/html/2610.11329)
- **Supervised and linear probing with frozen weights:**
  - MUSK averages 88.2% across 12 datasets (+2.2 over CONCH).
  - CONCH ROI-supervised performance matches CTransPath (CRC100k 93.8%).
  - CONCH ranked first of 19 FMs in the Kather benchmark. — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649); [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/); [arXiv 2408.15823](https://arxiv.org/abs/2408.15823)
- **CONCH vs UNI (vision-only), frozen ABMIL/KNN** (full table in Q1):
  - CONCH ahead on IHC ER/PR (κ 0.819 vs 0.785) and CRC-100K-Raw KNN (0.941 vs 0.925).
  - CONCH behind on PANDA (0.934 vs 0.946), OncoTree-108 (0.515 vs 0.538) and TCGA Uniform Tumor KNN (0.556 vs 0.595). — [GitHub](https://github.com/mahmoodlab/CONCH)
- **MUSK zero-shot strengths:** large margins on PatchCamelyon (+10.5%), SkinCancer (+27.5%), PanNuke (+7.3%) and UniToPatho (+10.1%) over the second-best model. — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)

### Inferences
- A defensible hackathon design: use frozen CONCH or MUSK tile embeddings, a zero-shot baseline with prompt ensembles, and a linear-probe or few-shot head, with a clear comparison between them.
- Expect zero-shot to be competitive only on few-class, morphologically distinct problems. Report balanced accuracy and quadratic κ for graded tasks.
- Panels will likely ask about prompt sensitivity and stain or site shift. Prompt ensembles (as the CONCH paper used) and stain-shift evaluation partly address both.

### Gaps
- I found no verified head-to-head zero-shot results of CONCH vs MUSK on the same tile datasets from an independent group, other than the PathPT EBRAINS extraction.

## Q4. Limitations stated in the original papers

### Takeaway
CONCH's authors list modest pretraining scale, possible train/test overlap, untested stain and scanner robustness, unoptimized prompts, MI-Zero's unsuitability for non-exclusive or ordinal tasks, untested cell-level tasks, and weak zero-shot on rare many-class problems. MUSK's authors flag small single-center immunotherapy cohorts (about 220 patients), the need for prospective multi-site validation and regulatory approval, and the inherent difficulty of outcome prediction.

### Cited Findings
- **CONCH (from the Discussion):**
  - Pretraining data are far smaller than billion-scale general-domain sets.
  - Pretraining and test data may overlap, and duplicate detection is unexplored.
  - Robustness to staining, protocol and scanner variation was not investigated.
  - Prompts were not optimized on validation data.
  - MI-Zero suits only mutually exclusive morphological patterns, so it is a poor fit for Gleason scoring.
  - Mitosis detection, cell counting and fine-grained segmentation were not studied.
  - Zero-shot remains weak on rare, many-class problems.

  — [PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)
- **CONCH (release):** the captioning decoder is withheld from public weights, so captioning cannot be reproduced. — [GitHub](https://github.com/mahmoodlab/CONCH)
- **MUSK (from the Discussion):**
  - The immunotherapy cohorts are small (about 220 patients total, NSCLC n=118 plus gastro-esophageal n=101) and single-center.
  - Clinical use requires prospective trials in diverse populations and regulatory approval.
  - Outcome prediction is harder than diagnosis.

  — [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)
- **Press caveat:** Stanford Daily notes MUSK is not ready for clinical use (February 4, 2025). — [Stanford Daily](https://stanforddaily.com/2025/02/04/the-musk-model/)
- **TITAN's authors note:**
  - IHC reports were excluded from alignment, which likely weakens IHC performance.
  - Extrapolating from 8K crops to whole slides may miss context.
  - The comparison with CHIEF is confounded because CHIEF was pretrained on TCGA.

  — [arXiv 2411.19666](https://arxiv.org/html/2411.19666v1)

### Inferences
- MUSK's TCGA pretraining combined with TCGA-based prognosis evaluation is a contamination risk the panel may raise. CONCH explicitly avoided TCGA in pretraining.

### Gaps
- The full MUSK limitations paragraph in the Supplementary material was not reviewed.

## Q5. Licenses and access constraints

### Takeaway
Both models are CC-BY-NC-ND 4.0: non-commercial academic use only, with no redistribution and no commercial use of derivatives (including models trained on their outputs). Weights are gated on Hugging Face and require accepting the terms and authenticating with a token.

### Cited Findings
- **CONCH:**
  - License: CC-BY-NC-ND 4.0, non-commercial academic research with attribution.
  - Commercial use or monetization, including of derivatives and models trained on CONCH outputs, requires prior approval.
  - No redistribution; no re-identification of the deidentified data.
  - Weights are gated on HF (MahmoodLab/conch). Users must accept the terms individually and pass an HF token.
  - Commercial inquiries go to the corresponding author or the Mass General Brigham Innovation Office.

  — [GitHub](https://github.com/mahmoodlab/CONCH)
- **MUSK:**
  - License: CC-BY-NC-ND 4.0, non-commercial academic use with attribution.
  - Commercial use of the model and its derivatives (including models trained on MUSK outputs) requires approval.
  - Weights at HF `xiangjx/musk` are gated; users must accept the terms and log in with an HF token.
  - Immunotherapy cohort data is controlled-access under a data use agreement.

  — [GitHub](https://github.com/lilab-stanford/MUSK); [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)
- **TITAN and CONCH v1.5:** gated on HF; users agree not to distribute or reproduce. CONCH v1.5 is listed as CC-BY-NC-ND-4.0. — [HF TITAN README](https://huggingface.co/MahmoodLab/TITAN/resolve/main/README.md); [LazySlide](https://lazyslide.readthedocs.io/en/latest/api/_autogen/lazyslide_models.multimodal.Titan.html)

### Inferences
- A student hackathon (non-commercial) can use both models but should not redistribute the weights, for example by bundling them in a public repo or demo container. Each team member needs their own HF approval. Any commercialization of a hackathon product built on them would need permission.

### Gaps
- Turnaround times for HF access approval are not documented.
