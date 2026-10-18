# Open Research Gaps in Pathology Vision-Language Foundation Models (CONCH, MUSK, PLIP, QuiltNet): weekend-feasible ideas

Notes compiled 2026-10-17. Dates come from arXiv IDs (YYMM) or the publication date where known. Search snippets were cross-checked against abstract pages for the three most important items (PathLang 2610.11329, Huang & Razavian 2608.30390, Sharma et al. 2505.00134). The other items are based on search excerpts and should be read in full before anyone positions against them.

## Interpretability: concept bottlenecks, text explanations of embeddings, prototype retrieval, sparse autoencoders. What exists and what is open?

### Takeaway
Generic vision-language concept bottleneck models (LaBo, Z-CBM, VLG-CBM) and SAEs on pathology vision-only encoders (PLUTO, PICASSO, EXPOSE) exist. SpLiCE-style sparse text decomposition has reached pathology only through a single 2026 WSI preprint on PathGen-CLIP. I found no paper that does a tile-level zero-shot concept bottleneck with CONCH, QuiltNet or PLIP text concepts and also checks whether the concept scores are faithful. I also found no paper that maps SAE features from a pathology vision-language model to text through that model's own text encoder.

### Cited Findings
- LaBo (CVPR 2023) builds concept bottlenecks by having GPT-3 generate concepts and scoring them with CLIP. It is general-domain, not pathology. — [Yang et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Language_in_a_Bottle_Language_Model_Guided_Concept_Bottlenecks_for_CVPR_2023_paper.pdf)
- Zero-shot CBMs (Z-CBM, arXiv Feb 2025) predict concepts and labels with no training, using frozen off-the-shelf VLMs and a large concept bank. They are general CV and are not tested on pathology. — [arXiv 2502.09018](https://arxiv.org/abs/2502.09018)
- VLG-CBM (2024) and "Improving Concept Alignment in VL CBMs" (2024) show that CLIP concept scores can be poorly aligned with the concepts they name, which is a faithfulness problem. — [VLG-CBM arXiv 2408.01432](https://arxiv.org/pdf/2408.01432); [arXiv 2405.01825](https://arxiv.org/pdf/2405.01825)
- Medical CBM using VLM concept prediction plus test-time human intervention, in dermatology rather than pathology (2025). — [PMC11907460](https://pmc.ncbi.nlm.nih.gov/articles/PMC11907460/)
- SpLiCE (NeurIPS 2024) is post-hoc and training-free. It decomposes CLIP image embeddings into sparse, non-negative weights over a text-concept vocabulary and uses mean-centering to handle the modality gap. — [Bhalla et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/996bef37d8a638f37bdfcac2789e835d-Paper-Conference.pdf); [code](https://github.com/ai4life-group/splice)
- "Sparse concept attribution for histomorphological hypothesis generation from whole-slide classifiers" (arXiv Sep 2026) applies SpLiCE-style decomposition to WSI classifiers built on PathGen-CLIP-L tile embeddings. I saw only fragments and could not confirm whether it uses CONCH or QuiltNet. — [arXiv 2609.02985](https://arxiv.org/pdf/2609.02985)
- SAEs on a frozen PLUTO (ViT-S) pathology encoder find features for cell types, tissue structures and artifacts (arXiv Jul 2024; OpenReview). This is a vision-only encoder with no text side. — [arXiv 2407.10785](https://arxiv.org/html/2407.10785v1); [OpenReview](https://openreview.net/forum?id=daV16mhUBd)
- PICASSO uses an SAE to decompose pathology FM embeddings into an atlas of morphological concepts. — [PMC13307983, "Dissecting and directing pathology foundation models"](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307983/)
- EXPOSE (arXiv Aug 2026) uses an SAE bottleneck (ReLU encoder, expansion 5, L1) to find and suppress scanner- and stain-specific dimensions in pathology vision FMs. The search tool flagged its metadata as unusual, so verify it. — [arXiv 2608.28191](https://arxiv.org/pdf/2608.28191)
- Compositional histology with FMs plus SAEs, applied to TB and lung cancer (bioRxiv Jun 2026). — [bioRxiv 2026.06.03.725182](https://www.biorxiv.org/content/10.64898/2026.06.03.725182v1)
- GECKO (arXiv Apr 2025) does gigapixel vision-concept contrastive pretraining. It is slide-level pretraining, not a frozen tile-level CBM. — [arXiv 2504.01009](https://arxiv.org/pdf/2504.01009)
- The CONCH authors state that VL-pretrained models including CONCH still do poorly on hard zero-shot problems with many classes and rare diseases. — [Lu et al., Nat Med 2024, PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)

### Inferences
- **Open gap A (interpretability, strongest):** a faithfulness audit of text-concept bottlenecks on pathology VLMs. The idea is to build a concept vocabulary of roughly 50 to 150 morphology terms (from an LLM on Lumen, or mined from Quilt-1M captions) and score tiles with CONCH, QuiltNet and PLIP. Then measure (i) how accurate the concept-bottleneck linear head is compared with a linear probe on the full embedding (the "interpretability tax"), (ii) whether concept scores agree across the three models for the same tile, and (iii) intervention validity: zero out or replace a concept and check whether predictions change as a pathologist would expect. The general VL-CBM literature flags concept misalignment, but nobody has quantified it for pathology VLMs. Running SpLiCE directly on CONCH, QuiltNet and PLIP at tile level appears unpublished, apart from the PathGen-CLIP WSI preprint above.
- **Open gap B:** "text-grounded SAE features." Train an SAE on CONCH image embeddings (Quilt-1M plus NCT-CRC), then name each feature automatically by its nearest CONCH text embeddings, or by retrieving the Quilt-1M captions of its top-activating images and summarising them with a Lumen LLM. Prior pathology SAE papers (PLUTO, PICASSO, EXPOSE) work on vision-only encoders and label features by hand. A VLM lets you label features automatically and check those labels (caption-retrieval agreement), which is a differentiator.
- Prototype retrieval as explanation, meaning showing the k nearest Quilt-1M image–caption pairs for each prediction, is cheap. I found no paper that evaluates retrieved-caption explanations for faithfulness (for example, whether the retrieved captions mention the predicted class) at scale on pathology VLMs.

### Gaps
- I could not access the full text of 2609.02985 or PICASSO to confirm which encoders they use. A CONCH-based SpLiCE paper may exist that I did not find.
- I did not search ICLR 2025 "Interpretable Vision-Language Survival Analysis" in detail. It is slide-level survival, so its overlap is probably low.

## Reproducibility and reliability: prompt sensitivity, calibration, conformal prediction and selective classification, inter-model agreement, stain invariance. What exists and what is open?

### Takeaway
Prompt and paraphrase sensitivity is now well documented. PathLang (Oct 8, 2026) is a direct competitor on paraphrase robustness, and Sharma et al. 2025 did it on in-house WSIs. Calibration and conformal prediction for pathology VLM zero-shot outputs are much less studied. General papers exist (Conf-OT, WebCP, Huang & Razavian 2026, "Pitfalls of conformal prediction"), but I found no work on conformal or selective prediction for CONCH, QuiltNet or PLIP zero-shot or prompt-ensemble outputs on public tile benchmarks. Stain robustness has benchmarks (PathoROB), but they target vision FMs and linear probes, not zero-shot text classifiers.

### Cited Findings
- Sharma et al. (arXiv Apr 30, 2025) varied domain specificity, anatomical precision, instructional framing and output constraints for QuiltNet, Quilt-LLaVA and CONCH on an **in-house** set of 3,507 digestive WSIs. Prompt engineering significantly changed performance, and CONCH was the most robust. — [arXiv 2505.00134](https://arxiv.org/abs/2505.00134)
- PathLang (arXiv Oct 8, 2026; 18 authors, Cheng et al.) evaluates 9 VLMs on 5 public datasets across 4 organs. It covers zero-shot classification, retrieval, paraphrase robustness (semantic-equivalence paraphrases, length and reporting-style changes, prompt ensembling) and open-vocabulary retrieval. Key finding: "performance is highly sensitive to clinically equivalent paraphrases," and good alignment does not imply class separation. The abstract does not mention negation, hedging or calibration. — [arXiv 2610.11329](https://arxiv.org/abs/2610.11329)
- CONCH's own zero-shot protocol ensembles 21 templates with class-name synonyms. A later paper notes that zero-shot results depend on the text prompts. — [CONCH, PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/); [KEEP arXiv 2412.13126](https://arxiv.org/html/2412.13126v1)
- Calibration-aware prompt learning for medical VLMs (arXiv Sep 2025) reports severe miscalibration, with average ECE falling from about 8.49% to 1.85% under its method. It covers medical VLMs broadly; I did not confirm that CONCH, QuiltNet or PLIP were included. — [arXiv 2509.15226](https://arxiv.org/pdf/2509.15226)
- A CONCH-based renal study (arXiv Nov 2025) reports that low-supervision settings give confident but poorly calibrated predictions (ECE). — [arXiv 2511.11984](https://arxiv.org/html/2511.11984)
- Conservative calibration for test-time-adapted VLMs (arXiv Aug 2026) uses natural images. — [arXiv 2608.05945](https://arxiv.org/pdf/2608.05945)
- Conf-OT (arXiv May 2025) applies conformal prediction to zero-shot CLIP-style models. Domain shift inflates prediction sets, and a transductive optimal-transport step reduces set size (secondary summary: up to 20% smaller) while keeping coverage. It is not tested on pathology. — [arXiv 2505.24693](https://arxiv.org/pdf/2505.24693)
- WebCP (2023) calibrates zero-shot FMs with web-mined data and reports reaching target coverage on biomedical sets. — [arXiv 2310.09926](https://arxiv.org/pdf/2310.09926)
- "Pitfalls of Conformal Predictions for Medical Image Classification" (arXiv Jun 2025) uses histopathology and dermatology examples. It argues that conformal sets become unreliable under distribution shift, should not be used to select predictions for accuracy gains, and add little value with few classes. — [arXiv 2506.18162](https://arxiv.org/html/2506.18162)
- Huang & Razavian, "Uncertainty of Vision Medical Foundation Models" (arXiv Aug 31, 2026) covers retinal, histopathology and CXR. Domain-specific SSL pretraining gives better-calibrated point predictions and more efficient conformal sets, and re-calibration does not close the gaps between models. A search snippet adds that set size in histopathology showed no clear pattern by pretraining source. It studies **vision** FMs; the abstract does not name zero-shot VLMs. — [arXiv 2608.30390](https://arxiv.org/abs/2608.30390)
- Selective Conformal Risk Control (arXiv Dec 2025) and Conformal Alignment (2024) offer general selective-plus-conformal frameworks. — [arXiv 2512.12844](https://arxiv.org/pdf/2512.12844); [arXiv 2405.10301](https://arxiv.org/pdf/2405.10301)
- PathoROB (arXiv Jul 2025) is a robustness benchmark against non-biological confounders across 34 centers. Image-text models (CONCH, CONCH v1.5) rank as more robust than many TCGA-trained models. — [arXiv 2507.17845](https://arxiv.org/pdf/2507.17845)
- A protocol for evaluating robustness to H&E staining variation (arXiv Mar 2026). — [arXiv 2603.12886](https://arxiv.org/pdf/2603.12886)
- Stain and scanner variation in soft-tissue tumours, testing UNI-v2 and CONCH-v1.5 (bioRxiv Aug 2025). — [bioRxiv 2025.08.18.670932](https://www.biorxiv.org/content/biorxiv/early/2025/08/22/2025.08.18.670932.full.pdf)
- Fine-tuning improves PathoROB robustness (arXiv Jul 2026). — [arXiv 2607.22861](https://arxiv.org/pdf/2607.22861)
- **Important for the team's dataset:** Ignatov & Malivenko (arXiv Sep 2024) show that NCT-CRC-HE-100K has class-dependent color-normalization problems, JPEG artifacts and dynamic-range corruption. Mean RGB intensities alone give more than 50% 9-class accuracy, and ImageNet EfficientNet-B0 reaches more than 97.7%, beating pathology FMs. — [arXiv 2409.11546](https://arxiv.org/abs/2409.11546)
- A follow-up finds CRC-VAL-HE-7K more class-separable than the 100K training set (medRxiv Apr 2025). Normalization reduces but does not remove class-specific color signatures (JBCS 2026). — [medRxiv 2025.04.07.25325392](https://www.medrxiv.org/content/10.1101/2025.04.07.25325392v1.full); [JBCS](https://journals-sol.sbc.org.br/index.php/jbcs/article/view/7045)

### Inferences
- **Open gap C (reliability, strongest):** "conformal and selective zero-shot pathology." Apply split conformal prediction (APS/RAPS) and selective classification (risk–coverage curves, AURC) to the zero-shot and prompt-ensemble outputs of CONCH, QuiltNet and PLIP on PathMMU subsets (BACH, MHIST, Osteo, NCT-CRC). Add (i) **prompt-ensemble variance** as a nonconformity or abstention score and (ii) **cross-model disagreement** as a score. Huang & Razavian 2026 covers vision FMs with heads, Conf-OT is general-domain, and PathLang measures sensitivity but not set-valued guarantees. The "Pitfalls" paper must be cited and addressed, for example by reporting class-conditional coverage and testing under shift (train on NCT-CRC, test on CRC-VAL-HE-7K, or across PathMMU sources).
- **Open gap D:** turning prompt sensitivity into an uncertainty signal. PathLang documents that paraphrases change results. The open question is whether **per-image** disagreement across paraphrases predicts errors, so that it can serve as a free test-time uncertainty estimate. I found no paper evaluating this for pathology VLMs.
- **Open gap E (cheap, high-communication value):** a color-shortcut audit of zero-shot pathology VLMs on NCT-CRC. Ignatov & Malivenko show that linear models exploit color shortcuts. Nobody appears to have tested whether CONCH, QuiltNet or PLIP **zero-shot** (text-only) predictions also track color statistics, for example with grayscale, stain-augmentation or Macenko-normalization perturbations and by correlating per-class accuracy drops. Because this is a known dataset flaw, it is a credible "reproducibility" story for judges.

### Gaps
- I could not verify the full model lists in PathLang or Huang & Razavian. PathLang may already include prompt ensembling plus calibration figures in the full text. Read it before claiming novelty for gap D.
- I found no pathology-specific temperature-scaling or ECE table comparing CONCH, QuiltNet and PLIP zero-shot.

## Precision: few-shot / prompt learning (CoOp-style), LLM-generated description ensembles (CuPL-style), retrieval-augmented classification with Quilt-1M

### Takeaway
This area is crowded. CuPL-style LLM descriptions with CONCH already appear in a JBHI 2025 paper, GMAT, FG-PAN and ZSPMLG. CoOp, adapters and multi-backbone fusion appear in 2025–2026 papers, including Nature Communications 2026 (PathPT) and FFM-CP (Sep 2026). Retrieval-augmented zero-shot tile classification that uses Quilt-1M image–caption pairs as a non-parametric knowledge base did not come up in my searches and is the most plausible open slot here.

### Cited Findings
- "Leveraging Vision-Language Embeddings for Zero-Shot Learning in Histopathology Images" (arXiv Mar 2025; IEEE JBHI 2025) builds class embeddings from GPT-4-generated, curated prompts on frozen CONCH, with CuPL, CLIP-D and CLIP-E as baselines. — [arXiv 2503.10731](https://arxiv.org/html/2503.10731); [DOI 10.1109/jbhi.2025.3584802](https://doi.org/10.1109/jbhi.2025.3584802)
- GMAT (arXiv Aug 2025) uses multi-agent, grounded clinical descriptions with CONCH. Lists of descriptions improve AUC and F1 on TCGA-RCC and TCGA-Lung compared with a single description. — [arXiv 2508.01293](https://arxiv.org/pdf/2508.01293)
- FG-PAN (arXiv Aug 2025) builds LLM-generated class prototypes for brain tumor subtypes. — [arXiv 2508.01602](https://arxiv.org/html/2508.01602v2)
- ZSPMLG does zero-shot tissue segmentation using CONCH plus LLM descriptions. — [PubMed 42340907](https://pubmed.ncbi.nlm.nih.gov/42340907/)
- CuPL (original, 2022). — [arXiv 2209.03320](https://arxiv.org/pdf/2209.03320)
- PathPT (Nature Communications 2026) does few-shot prompt tuning over PLIP, CONCH, MUSK and KEEP for rare cancers. PLIP-, MUSK- and CONCH-based PathPT underperformed MIL baselines. — [Nat Commun s41467-026-71715-2](https://www.nature.com/articles/s41467-026-71715-2)
- FFM-CP (arXiv Sep 2026) fuses frozen pathology VLMs across backbones for few-shot learning, with CoOp, CLIP-Adapter and Tip-Adapter as baselines. — [arXiv 2609.27710](https://arxiv.org/pdf/2609.27710)
- Hierarchical VL few-shot WSI work (arXiv May 2025) evaluates PLIP, QuiltNet and CONCH at 16 shots. — [arXiv 2505.17982](https://arxiv.org/html/2505.17982)
- Annotation-free VLM specialization (arXiv Aug 2025) compares CoOp with TAPT on MHIST using QuiltNet. — [arXiv 2508.07835](https://arxiv.org/pdf/2508.07835)
- MGPATH (Feb 2025) does multi-granular prompt learning for WSI. — [arXiv 2502.07409](https://arxiv.org/html/2502.07409v3)
- Concept-Constrained Prompt Learning for few-shot CLIP (arXiv Jun 2026), general domain. — [arXiv 2606.22567](https://arxiv.org/pdf/2606.22567)
- Quilt-1M (NeurIPS 2023 D&B) reports pair counts that vary across versions (802,144, 768,826, 802,148). QuiltNet is evaluated on 13 patch datasets. — [arXiv 2306.11207](https://arxiv.org/html/2306.11207); [GitHub](https://github.com/wisdomikezogwo/quilt1m)
- HOMIE (arXiv Feb 2025) does pathology composed retrieval, which is a retrieval task, not retrieval-augmented classification. — [arXiv 2502.07221](https://arxiv.org/pdf/2502.07221)

### Inferences
- **Open gap F:** retrieval-augmented zero-shot classification with a caption vote ("Quilt-RAC"). For each test tile, retrieve the top-k Quilt-1M images by CONCH image–image similarity, then either (a) embed their captions and mix them with the class-prompt logits (Tip-Adapter-like, but with captions instead of labels), or (b) have a Lumen LLM read the k captions and output a class plus a rationale. The rationale is interpretable by construction because each prediction cites real captions. Evaluate on the PathMMU subsets. Because Quilt-1M overlaps QuiltNet's training data, report contamination by checking whether near-duplicate images exist (cosine above 0.95).
- CuPL-for-pathology on its own is **not novel** (JBHI 2025, GMAT). If used, it should be an ablation or a component, for example "do LLM descriptions reduce the prompt-induced variance found in gap D?"
- CoOp or linear probes on CONCH are baselines, not contributions.

### Gaps
- I did not find a paper doing retrieval-augmented zero-shot classification over Quilt-1M. Absence from about 2 searches is weak evidence. Search "retrieval-augmented CLIP pathology" and "SuRe / RA-CLIP histology" before claiming it.

## Text side: auditing report language, hedging terms, image–text consistency, caption noise in Quilt/PathGen

### Takeaway
A clinical literature shows that pathologists and clinicians interpret hedging phrases ("suspicious for", "consistent with", "cannot exclude") inconsistently. I found no work testing whether pathology VLM text encoders encode those certainty gradations, for example whether the embedding of "suspicious for adenocarcinoma" sits between "adenocarcinoma" and "benign". PathLang tests paraphrases but, per its abstract, not hedging or negation. Quilt-1M noise is acknowledged (57.9% precision for LLM word correction; a cleaning paper), but no independent audit of Quilt-1M against PathGen caption–image consistency using VLM scores turned up.

### Cited Findings
- Scientific Reports 2025 (Jordan, two-phase survey): clinicians read "suspicious for" and "cannot exclude" as lower certainty than pathologists. "Indefinite for", "suspicious for", "cannot exclude" and "cannot rule out" had the lowest mean certainty; "compatible with" and "consistent with" had the highest. — [Sci Rep s41598-025-11032-8](https://www.nature.com/articles/s41598-025-11032-8)
- Arch Pathol Lab Med 2022 (multicenter): "diagnostic of" and "consistent with" had the strongest agreement in meaning, and "suspicious for" and "compatible with" the weakest. Standardization is recommended. — [APLM 146(7):886](https://meridian.allenpress.com/aplm/article/146/7/886/472360/Communicating-Certainty-in-Pathology)
- Bracamonte et al. 2016 (Academic Pathology): wide variation in how phrases are interpreted; the authors suggest a certainty key. — [DOI 10.1177/2374289516659079](https://journals.sagepub.com/doi/10.1177/2374289516659079)
- Descriptive study of uncertainty language at a cancer center (Academic Pathology 2024). — [PubMed 38433775](https://pubmed.ncbi.nlm.nih.gov/38433775/)
- Quilt-1M's own pipeline reports 57.9% average precision for LLM-based word corrections, and 13.8% for unconditioned corrections. — [arXiv 2306.11207](https://arxiv.org/html/2306.11207v4)
- Model-based cleaning of Quilt-1M (arXiv Apr 2024) addresses image impurities (narrator, pointers, overlays) for text-to-image synthesis. — [arXiv 2404.07676](https://arxiv.org/pdf/2404.07676)
- PathGen-1.6M (arXiv Jul 2024) generates captions with a multi-agent pipeline that includes a revision step for hallucinations. The images are mostly from TCGA. This is the authors' claim and has not been independently audited. — [arXiv 2407.00203](https://arxiv.org/html/2407.00203); [GitHub](https://github.com/PathFoundation/PathGen-1.6M)
- Interobserver background: a classic 1985 study of colorectal adenoma dysplasia grading found intraobserver agreement of 70% and 67% and interobserver agreement of 59% and 66%. — [J Clin Pathol 38(2):174](https://jcp.bmj.com/content/38/2/174)
- A Brazilian four-pathologist study reported mean kappa of 0.420 for dysplasia degree. — [Rev Bras Cancerol](https://scielo.br/j/rbc/a/V9HL3xCBFXnYXBvM6VsSY6q/abstract/?lang=en)
- A Dutch nationwide study found considerable interlaboratory variation in grading dysplasia in colorectal adenomas. — [Erasmus repub 95880](https://repub.eur.nl/pub/95880)

### Inferences
- **Open gap G (novel, cheap, strong "communication" angle):** a hedging-certainty probe of pathology VLM text encoders. Build template sets such as "{hedge} {diagnosis}" with hedges taken from the survey papers ("diagnostic of", "consistent with", "compatible with", "suggestive of", "suspicious for", "cannot exclude", "negative for"), plus negation. Measure (i) whether the image-to-text score is monotone in clinician-rated certainty using the survey rankings as ground truth (Spearman rho), (ii) whether "negative for X" is confused with "X", and (iii) the effect on zero-shot accuracy when class prompts carry hedges. This needs **no GPU** beyond text encoding, plus image scoring on PathMMU. A plausible finding is that CLIP-style bag-of-words behaviour makes hedges invisible, which ties into the general literature (CLIP behaves like a bag-of-words cross-modally; [arXiv 2502.03566](https://arxiv.org/pdf/2502.03566), 2025).
- **Open gap H:** a caption–image consistency audit. Score all 653k Quilt-1M pairs and the PathGen sample with CONCH image–text cosine. Flag low-consistency pairs, have a Lumen LLM categorise the failure (non-morphologic chatter, wrong stain, mislabeled organ), and test whether filtering the bottom X% changes downstream retrieval or linear-probe results. The caveat is circularity, because QuiltNet was trained on Quilt. Use CONCH and PLIP as independent judges and report inter-judge agreement.

### Gaps
- I found no Gleason kappa figures because the search returned only colorectal results. The team should cite a dedicated Gleason study if it uses one.
- I did not verify whether PathLang's "reporting-style" paraphrases include hedges. Read Section 3 of 2610.11329.

## Cross-model agreement/disagreement as uncertainty; modality gap and embedding geometry in pathology CLIP models

### Takeaway
Cross-model disagreement as uncertainty exists at **slide level for vision FMs** (DICE, Jun 2026), and ELF documents low correlation between tile FMs. I found no tile-level study of disagreement **between VLM zero-shot predictions** (CONCH vs QuiltNet vs PLIP). The modality gap has been measured for PathGen-CLIP and MUSK (HOMIE: gap ≥ 0.571) but, as far as I found, not for CONCH, QuiltNet or PLIP, and nobody has tied it to zero-shot or calibration behaviour in pathology.

### Cited Findings
- DICE (arXiv Jun 2026) ensembles frozen pathology FMs with deep mutual learning. Disagreement serves as a slide-level uncertainty proxy that flags error-prone slides, and the consensus localizes abnormal patches without supervision. — [arXiv 2606.30020](https://arxiv.org/abs/2606.30020v1)
- ELF (Cancer Cell 2026; arXiv Aug 2025) ensembles five tile FMs and reports "extremely low correlations" between how models encode patches. — [arXiv 2508.16085](https://arxiv.org/html/2508.16085); [PMC13564370](https://pmc.ncbi.nlm.nih.gov/articles/PMC13564370)
- PICTURE (Nat Commun 2025) uses an uncertainty-aware FM ensemble for glioblastoma vs mimics, keeping the 50% lowest-uncertainty tiles. — [Nat Commun s41467-025-64249-6](https://www.nature.com/articles/s41467-025-64249-6)
- Benchmark of FMs as feature extractors (Nat Biomed Eng 2025). — [s41551-025-01516-3](https://www.nature.com/articles/s41551-025-01516-3)
- HOMIE (arXiv Feb 2025): pathology CLIP baselines (PathGen-CLIP, MUSK) show a large modality gap (≥ 0.571), and HOMIE reduces it to 0.325. — [arXiv 2502.07221](https://arxiv.org/pdf/2502.07221)
- Modality gap origins (Liang et al., NeurIPS 2022, cone effect plus contrastive optimisation). — [GitHub](https://github.com/Weixin-Liang/Modality-Gap); [ICLR 2023 related](https://iclr.cc/virtual/2023/13797)
- Double-ellipsoid geometry of CLIP (arXiv Nov 2024). — [arXiv 2411.14517](https://arxiv.org/pdf/2411.14517)
- "When Modality Gap Reduction Fails: Prediction-Level Hubness in CLIP" (arXiv Sep 2026) shows that gap correction shifts class scores unequally and can favour some class prototypes. — [arXiv 2609.01103](https://arxiv.org/html/2609.01103)
- Mitigate the gap (arXiv Jun 2024). — [arXiv 2406.17639](https://arxiv.org/html/2406.17639v2)

### Inferences
- **Open gap I:** tile-level tri-model disagreement among CONCH, QuiltNet and PLIP (and MUSK if approved) as a training-free abstention signal. Compare it against max-softmax, entropy and prompt-variance. Report AURC and accuracy at 80% coverage, and examine whether disagreement concentrates in specific class pairs (for example STR vs MUS, DEB vs NORM in NCT-CRC). This pairs naturally with gap C, so the two should be one project.
- **Open gap J:** a modality-gap and hubness census for CONCH, QuiltNet and PLIP. Measure the gap vector, check hubness of class-text prototypes (one class absorbing predictions), and test whether simple fixes (mean-centering à la SpLiCE, gap-vector subtraction) change zero-shot accuracy and calibration. 2609.01103 predicts unequal per-class effects, which can be tested on pathology classes. Compute is minimal.

### Gaps
- I did not read DICE in full, so I cannot rule out a tile-level VLM-disagreement experiment inside it.
- No CONCH, QuiltNet or PLIP modality-gap numbers were found. They would need to be computed (and that would be novel).

## For each candidate idea, is there already a paper doing it? Ranked weekend research questions with positioning

### Takeaway
The best-positioned project combines a reliability layer for zero-shot pathology VLMs (conformal plus selective prediction driven by prompt-paraphrase variance and cross-model disagreement) with a hedging and negation probe of the text encoders, run on public PathMMU tile subsets plus NCT-CRC with an explicit color-shortcut control. Each piece has a nearest neighbour (PathLang Oct 2026; Huang & Razavian Aug 2026; DICE Jun 2026; Ignatov 2024), but none does this combination at tile level with VLM zero-shot outputs.

### Cited Findings
- Nearest neighbours for positioning: PathLang ([2610.11329](https://arxiv.org/abs/2610.11329), Oct 2026); Sharma et al. ([2505.00134](https://arxiv.org/abs/2505.00134), Apr 2025); Huang & Razavian ([2608.30390](https://arxiv.org/abs/2608.30390), Aug 2026); Pitfalls of CP ([2506.18162](https://arxiv.org/html/2506.18162), Jun 2025); Conf-OT ([2505.24693](https://arxiv.org/pdf/2505.24693), May 2025); DICE ([2606.30020](https://arxiv.org/abs/2606.30020v1), Jun 2026); JBHI CONCH+LLM prompts ([2503.10731](https://arxiv.org/html/2503.10731), 2025); GMAT ([2508.01293](https://arxiv.org/pdf/2508.01293)); PathPT ([Nat Commun 2026](https://www.nature.com/articles/s41467-026-71715-2)); FFM-CP ([2609.27710](https://arxiv.org/pdf/2609.27710)); SpLiCE pathology ([2609.02985](https://arxiv.org/pdf/2609.02985)); PLUTO SAE ([2407.10785](https://arxiv.org/html/2407.10785v1)); EXPOSE ([2608.28191](https://arxiv.org/pdf/2608.28191)); HOMIE ([2502.07221](https://arxiv.org/pdf/2502.07221)); NCT-CRC bias ([2409.11546](https://arxiv.org/abs/2409.11546)); hedging surveys ([Sci Rep 2025](https://www.nature.com/articles/s41598-025-11032-8), [APLM 2022](https://meridian.allenpress.com/aplm/article/146/7/886/472360/Communicating-Certainty-in-Pathology)).

### Inferences
Compute estimates are my own rough inferences, not sourced. They assume ViT-B-class encoders (CONCH ViT-B/16, QuiltNet ViT-B/32, PLIP ViT-B/32) and that embedding one 224px tile per model costs well under 1 ms on an A100 at batch 256. Under those assumptions, 100k NCT-CRC tiles take minutes per model and 653k Quilt images take about 1 to 3 GPU-hours per model, with JPEG decoding and I/O likely the bottleneck. **Precompute and cache all embeddings once (about 6 to 10 GPU-hours total for 3 models across all data); everything else runs on CPU.**

**Rank 1: "Know when the VLM doesn't know": conformal and selective zero-shot pathology using prompt-paraphrase variance and cross-model disagreement** (gaps C, D, I).
- Question: do per-tile paraphrase variance and CONCH/QuiltNet/PLIP disagreement give better abstention (AURC) and tighter conformal sets at fixed coverage than softmax confidence, and do these guarantees hold under shift?
- Positioning: PathLang shows sensitivity but not its use as uncertainty. Huang & Razavian cover vision FMs with heads, not VLM zero-shot. DICE is slide-level vision FMs. Address the Pitfalls paper directly by reporting class-conditional coverage and shift.
- Data: PathMMU subsets (BACH 4-class, MHIST 2-class, Osteo, NCT-CRC 9-class). Calibrate on NCT-CRC-100K and test on CRC-VAL-HE-7K, if available, for the shift test.
- Prompts: 20 to 50 paraphrases per class generated by a Lumen LLM and pathologist-vetted where possible, plus CONCH's 21 templates.
- Metrics: accuracy, ECE (before and after temperature scaling), APS/RAPS empirical coverage and mean set size at α=0.1/0.05, class-conditional coverage, AURC, and accuracy at 80% coverage.
- Compute: about 3 to 5 GPU-hours (embeddings) plus CPU.
- Clinical hook: a triage rule that sends low-confidence tiles to a pathologist.

**Rank 2: "Does the model hear 'suspicious for'?": a hedging and negation certainty probe of pathology VLM text encoders** (gap G).
- Question: are pathology VLM text embeddings and image–text scores monotone in clinician-perceived certainty, and do they separate "negative for X" from "X"?
- Positioning: the clinical surveys (Sci Rep 2025, APLM 2022) provide the human certainty ordering. PathLang's abstract does not cover hedging or negation (verify). The general CLIP bag-of-words result (2502.03566) motivates the hypothesis.
- Data: text templates, plus PathMMU and NCT-CRC images for score shifts. Optionally mine hedge phrases from Quilt-1M and PathGen captions to report how often they occur (text only).
- Metrics: Spearman rho between model score and survey certainty rank, negation flip rate, and the zero-shot accuracy change when hedges are added.
- Compute: under 1 GPU-hour.
- Pairs with Rank 1 as "text-side reliability."

**Rank 3: color-shortcut audit of zero-shot pathology VLMs on NCT-CRC** (gap E).
- Question: does zero-shot or linear-probe accuracy of CONCH, QuiltNet and PLIP depend on the class-specific color artifacts documented by Ignatov & Malivenko?
- Experiment: compare a mean-RGB baseline with each model under perturbations (grayscale, Macenko or Reinhard normalization, HED jitter, JPEG re-encode). Report the per-class accuracy drop and the correlation of model logits with color statistics. Cross-check on BACH and MHIST.
- Metrics: accuracy drop, prediction flip rate under perturbation, and agreement between models.
- Compute: about 2 to 4 GPU-hours. A small but highly credible reproducibility contribution, which can be folded into Rank 1 as the shift condition.

**Rank 4: tile-level text-concept bottleneck faithfulness audit (CONCH vs QuiltNet vs PLIP), with SpLiCE decomposition** (gap A).
- Question: what accuracy does a text-concept bottleneck cost compared with a linear probe, and are concept scores consistent across models and responsive to intervention?
- Data: NCT-CRC (watch for the color shortcut) and BACH. The concept vocabulary comes from a Lumen LLM plus nouns mined from Quilt captions.
- Metrics: bottleneck vs probe accuracy, cross-model concept-score rank correlation, intervention success rate, and SpLiCE sparsity–cosine tradeoff.
- Positioning: LaBo, Z-CBM and VLG-CBM are general-domain; 2609.02985 is WSI-level on PathGen-CLIP.
- Compute: about 1 to 2 GPU-hours.

**Rank 5: retrieval-augmented zero-shot classification with Quilt-1M captions as evidence** (gap F).
- Question: does retrieving the k nearest Quilt-1M image–caption pairs (CONCH image space) and voting with caption-text similarity, or having a Lumen LLM read the captions, improve zero-shot accuracy and provide auditable rationales?
- Metrics: accuracy and balanced accuracy over k, the fraction of rationales whose cited captions mention the predicted class, and a near-duplicate contamination rate.
- Compute: about 2 to 3 GPU-hours to embed Quilt with CONCH, FAISS on CPU, and LLM calls on Lumen.
- Risk: noisy captions; Quilt's 57.9% correction precision suggests the knowledge base contains errors.

**Rank 6: modality-gap and hubness census for CONCH, QuiltNet and PLIP** (gap J). This is cheap and best used as a supporting figure for Rank 1: does mean-centering change calibration?

**Rank 7: text-grounded SAE on CONCH embeddings** (gap B). An SAE on about 750k cached embeddings trains in under 1 GPU-hour, and features are auto-labelled via the CONCH text encoder and retrieved Quilt captions. It is novel relative to the PLUTO, PICASSO and EXPOSE vision-only SAEs, but riskier to finish and evaluate in a weekend.

**Rank 8: caption–image consistency audit of Quilt-1M vs the PathGen sample** (gap H). It is useful but harder to show clinical impact. Circularity with QuiltNet must be managed.

**Lower priority, already crowded:** CuPL-style LLM prompt ensembles for CONCH (JBHI 2025, GMAT), CoOp and adapter few-shot (PathPT Nat Commun 2026, FFM-CP Sep 2026, 2508.07835). Use these only as baselines or components.

**Dataset caveats:** NCT-CRC-HE-100K has no patient IDs and known color and JPEG shortcuts (2409.11546), so do not report it alone. Make BACH, MHIST and Osteo co-primary. Quilt-1M overlaps QuiltNet's training data. Check whether the PathMMU subsets overlap with CONCH, QuiltNet or PLIP training sets; I did not verify this.

### Gaps
- I did not verify the full text of PathLang. It appeared 9 days before these notes and is the main novelty risk for Ranks 1 and 2.
- I could not confirm whether MUSK has published prompt-robustness or calibration numbers.
- I did not search MICCAI 2025 proceedings directly. A MICCAI paper on conformal pathology VLMs could exist undetected.
- The compute estimates are engineering inferences, not sourced benchmarks.
- Whether CRC-VAL-HE-7K is in the team's PathMMU NCT-CRC folder is unknown.
