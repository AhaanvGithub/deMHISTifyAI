# Clinical Validation, Real-World Helpfulness, and Failure Modes of Pathology Foundation Models (CONCH, MUSK, PLIP, QuiltNet, and similar)

Notes compiled 2026-10-17. "PR" = peer-reviewed; "preprint" = arXiv or similar, not peer-reviewed. Some numbers come from search-result summaries and not from full text; those are flagged "(summary)" and should be checked before quoting exactly.

## 1. Prospective, multi-center, and reader studies: do pathology FMs help pathologists in practice?

### Takeaway
I found no prospective or reader study that tests CONCH, MUSK, PLIP, or QuiltNet themselves as aids for pathologists. The reader-study evidence for FM-based assistance comes from organ-specific FMs, all from preprints: accuracy rose about 8 percentage points (pp) with AI help. A registered prospective randomized trial (NCT07157618) is still recruiting. For CONCH, the multi-site evidence is retrospective benchmarking, where it ranks at or near the top. For MUSK, the evidence is mostly the developers' own results.

### Cited Findings
- **PulmoFoundation (lung FM), preprint, May 2026 (arXiv 2605.25878).** Crossover randomized reader study with 8 pathologists and 4,928 case-reader pairs. Accuracy was 91.7% with AI vs 83.8% without (+7.9 pp). Median diagnostic time fell 19.6% and diagnostic confidence rose 8.7% (summary). — [arXiv 2605.25878](https://arxiv.org/pdf/2605.25878)
- **GRACE (gastric FM), preprint, about June 2026.** In a randomized crossover reader study, accuracy rose from 82.0% to 89.9% with assistance (summary from a listing; full text and peer-review status not verified). — [chatpaper listing](https://chatpaper.com/paper/295295)
- **NCT07157618, "Prospective Pathology Foundation Models" (Nanfang Hospital).** Slides are randomized 1:1:1 to model-only, pathologist-only, and model-assisted pathologist arms. Enrollment started August 2025 and the trial was listed as recruiting, so no outcomes are reported yet. — [ClinicalTrials.gov NCT07157618](https://clinicaltrials.gov/study/NCT07157618)
- **Independent multi-cohort benchmark, Nature Biomedical Engineering 2025 (PR), DOI 10.1038/s41551-025-01516-3 (Neidlinger et al., Kather group).** It evaluated 19 histopathology FMs on 13 patient cohorts (6,818 patients, 9,528 slides) covering lung, colorectal, gastric, and breast cancer. The vision-language FM CONCH had the highest overall performance compared with vision-only FMs. Its advantage was smaller in low-data and low-prevalence tasks. — [Nat Biomed Eng](https://rd.springer.com/article/10.1038/s41551-025-01516-3); [PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)
- **MUSK (Xiang et al., Nature 2025, PR).** The developers report better performance across 23 patch- and slide-level benchmarks, plus evaluation on clinical report and image data from more than 8,000 patients. These are developer-reported, retrospective results. — [search summary of MUSK paper / benchmark context](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)
- **The only independent CONCH vs MUSK head-to-head I found:** PathFMTools, PMLR v297 (2026, PR workshop/proceedings). It covers histological grading of cutaneous SCC on 440 H&E WSIs. I could not retrieve the head-to-head numbers. — [PMLR v297 diab26a](https://proceedings.mlr.press/v297/diab26a.html)
- **PathChat (Lu et al., Nature 2024, PR, DOI 10.1038/s41586-024-07618-3).** In the paper's human evaluation, pathologists preferred PathChat's answers and rated them more accurate than those of comparator MLLMs. This was an evaluation of answer quality, not a study of pathologists working with assistance. — [IDEAS/RePEc record](https://ideas.repec.org/a/nat/nature/v634y2024i8033d10.1038_s41586-024-07618-3.html)
- **DALPHIN benchmark, preprint, May 2026 (arXiv 2605.03544).** 1,236 images from 300 cases, 130 diagnoses, 14 subspecialties, 6 countries. The human reference is 31 pathologists (24 board-certified, 7 residents) on a 115-case subset. PathChat+ showed no significant difference from expert level on 4/6 tasks, vs 2/6 for Gemini 2.5 Pro and 1/6 for GPT-5. GPT-5 scored below every pathologist group on free-response diagnosis. Ground truth is held on a third-party platform (grand-challenge) to prevent leakage. — [arXiv 2605.03544](https://arxiv.org/pdf/2605.03544); [DALPHIN challenge](https://dalphin.grand-challenge.org/)
- **USCAP 2025 abstract (independent, DALPHIN precursor).** PathChat did better than generalist models on entry-level and advanced questions but scored lower on harder free-text questions (Q4, Q6). — [USCAP 2025 abstract PDF](https://impdiagnostics.com/wp/wp-content/uploads/2026/03/abstract-USCAP-2025-RadboudColab.cleaned.pdf)

### Inferences
- The best reader-study effect sizes so far (about +8 pp accuracy, about 20% time savings) come from task-specific FM-derived systems, not zero-shot CLIP-style models. They are single-group preprints with small reader panels (n=8).
- CONCH's advantage in independent benchmarks makes it a strong feature extractor. That is a retrospective, AUROC-level claim and not evidence of clinical benefit.

### Gaps
- No reader or prospective study evaluating CONCH, MUSK, PLIP, or QuiltNet as a pathologist aid was found.
- No independent external validation of MUSK's prognosis or immunotherapy-response claims was found.
- PulmoFoundation and GRACE peer-review status is unverified, and GRACE's sample size was not retrieved.

## 2. Robustness failures: stain, scanner, and site shift; compression; medical-center signatures; demographic bias

### Takeaway
Strong evidence shows that FM embeddings encode the medical center of origin, often more strongly than the biology. This leads to center-confounded errors. Vision-language models, CONCH in particular, are among the more scanner- and site-robust models but are not immune. Race-related performance gaps persist with FMs: foundation features shrink them but do not remove them.

### Cited Findings
- **de Jong, Marcus, Teuwen, "Current Pathology Foundation Models are unrobust to Medical Center Differences," arXiv 2501.18055 (preprint, Jan 29 2025).**
  - Ten public FMs were evaluated, and "all ... represent the medical center to a strong degree."
  - The medical center is predicted more accurately than tissue source and cancer type.
  - Only one model has a Robustness Index above 1 (biology dominating center), and only slightly.
  - Cancer-type errors trace to same-center confounders. — [arXiv 2501.18055](https://arxiv.org/abs/2501.18055)
  - A later review states that the model above 1 was Virchow2, with UNI and Phikon-v2 below 1. — [arXiv 2510.23807](https://arxiv.org/html/2510.23807v3)
- **PathoROB, Kömen, de Jong, Hense, ... Klauschen, Müller.** Preprint arXiv 2507.17845 (Jul 22 2025); peer-reviewed in Nature Communications 2026, DOI 10.1038/s41467-026-73923-2 (I could not access the journal page to confirm the exact date).
  - Four datasets, about 100k patches, 28 biological classes, 34 medical centers, 20 FMs.
  - All 20 FMs showed robustness deficits. Larger models trained on more diverse data, and vision-language models, did best.
  - In a documented example, a model confidently labeled a clearly malignant patch as normal because it came from a hospital the model associated with healthy tissue.
  - Post-hoc robustification "considerably reduced (but did not yet eliminate)" such errors. — [arXiv 2507.17845](https://arxiv.org/abs/2507.17845); [BIFOLD news](https://www.bifold.berlin/news-events/news/view/news-detail/making-pathology-ai-robust-across-hospitals); [DKFZ record](https://inrepo02.dkfz.de/record/313290)
- **Thiringer, Gustafsson, Ledesma Eriksson, Rantalainen, "Scanner-Induced Domain Shifts Undermine the Robustness of Pathology FMs," arXiv 2601.04163 (preprint, Jan 7 2026).**
  - Setup: 384 breast cancer slides, each scanned on 5 scanners, with 14 feature extractors.
  - Embedding consistency: the 1-NN cross-scanner match rate for the G20X–PHIL pair was 81.8% for CONCHv1.5 vs 25.8% for Phikon-v2.
  - Prediction consistency: Fleiss' kappa ranged from about 0.4 (Phikon-v2, HER2) to 0.85 (CONCHv1.5, ER).
  - Ranking: CONCH and CONCHv1.5 were the most robust. Phikon-v2, Virchow, and Virchow2 had the lowest neighbourhood consistency, and UNI/UNI2-h ranked worst on cosine distance.
  - AUCs mostly stayed within bootstrap CIs, but NHG grading was consistently lower on G20X.
  - Predicted probabilities were poorly calibrated across scanners for most models.
  - MUSK, PLIP, and QuiltNet were not tested. — [arXiv 2601.04163](https://arxiv.org/html/2601.04163v1)
- **Vaidya et al., "Demographic bias in misdiagnosis by computational pathology models," Nature Medicine, Apr 19 2024 (PR), DOI 10.1038/s41591-024-02885-z.**
  - AUROC gaps between white and Black patients were 3.0% for breast subtyping, 10.9% for lung subtyping, and 16.0% for IDH1 mutation prediction in glioma.
  - FM features (UNI, CONCH) reduced but did not remove the disparities.
  - Small subgroup sizes were a limitation. — [PubMed 38641744](https://pubmed.ncbi.nlm.nih.gov/38641744/); [MGB press release](https://www.massgeneralbrigham.org/en/about/newsroom/press-releases/researchers-reduce-bias-in-pathology-ai-algorithms-using-foundation-models)
- **QERA framework, arXiv 2502.16889 (preprint, 2025).** It tests whether gender, age, race, and hospital can be inferred from pathology FM embeddings and evaluates group fairness across demographic and institutional subgroups. — [arXiv 2502.16889](https://arxiv.org/html/2502.16889v2)
- **SIIM 2025 abstract, "Foundation Models Fail to Close [the gap]" (Khosravi).** It reports that FMs do not close demographic performance gaps. Only the title and summary were seen. — [SIIM PDF](https://siim.org/wp-content/uploads/2025/08/Foundation-Models-Fail-to-Close_Khosravi-1.pdf)
- **Compression.** A March 2025 paper notes that standard JPEG-type compression "often fail[s] to preserve fine-grained phenotypic details." Autoencoder compression degraded downstream FM tasks only minimally. — [arXiv 2503.11591](https://web3.arxiv.org/abs/2503.11591?context=cs)

### Inferences
- The site signature is the most consistently documented failure mode, and it has a concrete clinical-error mechanism (center-label shortcut). Vision-language pretraining (CONCH) appears to give relatively better invariance, possibly because text supervision anchors the embedding to biology.
- Calibration under scanner shift is poor even when AUROC holds. Fixed thresholds deployed at a new site are a risk.

### Gaps
- Per-model Robustness Index values from PathoROB (including CONCH, MUSK, PLIP, QuiltNet) could not be extracted because the full text was not accessible.
- No benchmark isolating JPEG quality or magnification perturbations for CONCH, MUSK, PLIP, or QuiltNet embeddings was found.
- Sex-specific disparity numbers for UNI and CONCH were not retrieved. The Vaidya supplement and github.com/mahmoodlab/cpath_demographics may contain them.

## 3. Zero-shot fragility: prompt sensitivity, calibration, OOD detection, uncertainty

### Takeaway
Zero-shot CLIP-style pathology performance depends heavily on prompt wording. AUC can swing by about 0.2 between reasonable prompts, and the best wording differs by model. Pathology VLMs, PLIP and QuiltNet especially, show overconfidence. I found no systematic calibration or OOD study specific to CONCH or MUSK.

### Cited Findings
- **Sharma et al., arXiv 2505.00134 (preprint, Apr 2025).** Tested CONCH, QuiltNet, and Quilt-LLaVA on 3,507 in-house digestive-pathology WSIs, varying prompt specificity, anatomical precision, framing, and output constraints. The best term differed by model: "Precancerous" gave CONCH its top dysplasia AUC of 0.904, while "Atypia" was best for Quilt-LLaVA. For invasiveness, a concise anatomically specific prompt gave CONCH an AUC of 0.935 vs 0.736 for an elaborate expert-framed prompt (summary; verify in the paper). — [arXiv 2505.00134](https://arxiv.org/abs/2505.00134v1); [Emergent Mind summary](https://www.emergentmind.com/topics/conch)
- **MVC-Bench, arXiv 2608.27004 (preprint, Aug 2026).** A benchmark of calibration in medical VLMs. Overconfidence was most pronounced in histopathology for PLIP and QuiltNet, and calibration-method gains were less consistent there (summary). — [arXiv 2608.27004](https://arxiv.org/pdf/2608.27004)
- **Medical VQA overconfidence, arXiv 2604.02543 (preprint, Apr 2026; general medical VLMs, not pathology-specific).** Overconfidence persisted across scale and prompting. Platt scaling reduced calibration error but left AUROC unchanged. — [arXiv 2604.02543](https://arxiv.org/pdf/2604.02543)
- **Predictive entropy, arXiv 2604.08941 (preprint, 2026).** Single-pass predictive entropy flags both errors and paraphrase-unstable answers. A LoRA ensemble's calibration collapsed under dataset shift (ECE 42.9, CXR domain). — [arXiv 2604.08941](https://arxiv.org/abs/2604.08941)
- **Scanner-shift miscalibration in FM-based classifiers.** See Section 2. — [arXiv 2601.04163](https://arxiv.org/html/2601.04163v1)

### Inferences
- Prompt ensembling, as used in the CONCH paper, is a partial mitigation. The evidence suggests prompts must be validated per task and per model. Zero-shot scores should not be read as probabilities without site-specific calibration.

### Gaps
- No pathology-specific zero-shot OOD-detection benchmark for CONCH or MUSK was found.
- No ECE values specific to CONCH or MUSK were found.

## 4. Shortcut learning and data leakage in benchmarks

### Takeaway
NCT-CRC-HE-100K, a staple zero-shot and linear-probe benchmark for CONCH, PLIP, QuiltNet, and MUSK, contains strong class-specific color and JPEG artifacts. Color statistics alone reach more than 82% on this 9-class task, so high scores on it overstate morphological understanding. Center-confounding (Section 2) is a related shortcut at the representation level.

### Cited Findings
- **Ignatov & Malivenko, arXiv 2409.11546 (preprint, Sep 2024).**
  - The dataset has inconsistent color normalization, class-dependent JPEG artifacts, and patches corrupted by bad dynamic-range handling.
  - A model using only per-image mean RGB intensities exceeds 50% accuracy, and a color-histogram model exceeds 82%. — [arXiv 2409.11546](https://arxiv.org/abs/2409.11546)
- **Rodrigues et al., Journal of the Brazilian Computer Society, 2026 (PR).** Normalization reduces color-based class separability but no method fully removes tissue-specific color signatures. Distortion artifacts disproportionately affect one class. The analysis focuses on the NONORM variant. — [JBCS article 7045](https://journals-sol.sbc.org.br/index.php/jbcs/article/view/7045); [City Univ. PDF](https://openaccess.city.ac.uk/id/eprint/37639/1/7045-Article%20Text-39742-1-10-20260507.pdf)
- The CRC-VAL-HE-7K test set comes from 50 separate patients. — [HF dataset script](https://huggingface.co/datasets/1aurent/NCT-CRC-HE/blob/7730e526210509fb83dd87c6fb3e4e89457a1b76/NCT-CRC-HE.py)
- **PathoROB / de Jong.** Errors are attributable to same-center confounders, so benchmarks whose classes are unevenly distributed across centers reward center shortcuts. — [arXiv 2501.18055](https://arxiv.org/abs/2501.18055)
- **DALPHIN** sequesters ground truth specifically to prevent the training-data contamination common to public VLM benchmarks. — [DALPHIN](https://dalphin.grand-challenge.org/)

### Inferences
- Reported accuracies of 90% or more on CRC-100K for zero-shot or linear-probe use should be discounted. External, multi-center, patient-split cohorts (as in the Nat Biomed Eng benchmark) are better evidence.
- Pretraining corpora scraped from the web (Twitter for PLIP, YouTube for QuiltNet, PubMed for CONCH) may overlap with public evaluation images. I found no formal contamination audit.

### Gaps
- No published audit of patient-level leakage within the NCT-CRC-HE-100K training split was found. The dataset ships without per-patch patient IDs, which I could not confirm from a primary source in this session.
- No pretraining-to-benchmark contamination audit for PLIP, QuiltNet, or CONCH was found.

## 5. Hallucination and reliability of pathology LLMs/VLMs (PathChat, Quilt-LLaVA, MedGemma)

### Takeaway
Generative pathology copilots reach near-expert performance on some structured tasks (PathChat+ on DALPHIN). They remain weaker on free-text diagnosis, struggle to express uncertainty, and lose coherence at whole-slide scale. I found no dedicated pathology-report hallucination-rate study with pathologist adjudication.

### Cited Findings
- **DALPHIN (preprint, May 2026).** Models struggled to express uncertainty in ambiguous cases. Gemini tended to overcall and GPT tended to undercall (podcast summary of the paper). — [arXiv 2605.03544](https://arxiv.org/pdf/2605.03544); [Digital Pathology Podcast ep. 239](https://digitalpathologypodcast.buzzsprout.com/410071/episodes/19288089-239-can-ai-copilots-keep-up-with-pathologists)
- **PathGLS, arXiv 2603.16113 (preprint, Mar 2026).** Ground-truth-free consistency evaluation of pathology VLMs.
  - On PathMMU, MedGemma-4B-it had the top patch-level PathGLS (0.80 vs 0.69 for Quilt-LLaVA).
  - On TCGA at WSI scale, Quilt-LLaVA was highest overall (0.83) but its logic consistency fell to 0.78, while MedGemma dropped to 0.56 overall.
  - The authors conclude that coherence across separate WSI regions remains hard. — [arXiv 2603.16113](https://arxiv.org/pdf/2603.16113)
- **Quilt-LLaVA (CVPR 2024, PR).** Trained on about 107K instruction pairs from YouTube educational videos. It was evaluated on 985 images and 1,283 QA pairs with GPT-4-scored responses, and no hallucination-specific evaluation was done. — [arXiv 2312.04746](https://arxiv.org/abs/2312.04746v3); [CVPR anthology](https://mlanthology.org/cvpr/2024/seyfioglu2024cvpr-quiltllava)
- **MedHEval, arXiv 2503.02157 (preprint, 2025).** A medical LVLM hallucination benchmark with report-generation case studies, which are radiology, not pathology. — [arXiv 2503.02157](https://arxiv.org/pdf/2503.02157)
- **Independent USCAP 2025 evaluation of PathChat.** Strong on entry-level and multiple-choice questions, weaker on harder free-text questions. — [USCAP abstract](https://impdiagnostics.com/wp/wp-content/uploads/2026/03/abstract-USCAP-2025-RadboudColab.cleaned.pdf)

### Inferences
- The gap between multiple-choice and free-text performance, plus poor uncertainty expression, suggests that unverified generated narrative is the main hallucination risk. These systems are better suited to bounded tasks (organ ID, MCQ triage) than to autonomous report drafting.

### Gaps
- No pathologist-adjudicated hallucination rate (for example, fabricated findings per report) for PathChat, Quilt-LLaVA, or MedGemma in pathology report generation was found.

## 6. Regulatory status: FDA-cleared or CE-marked products built on these models

### Takeaway
I found no FDA-cleared or CE-IVD-marked diagnostic product built on CONCH, MUSK, PLIP, or QuiltNet. The closest is Paige PanCancer Detect, built on the Virchow2 FM, which holds FDA Breakthrough Device designation (April 2025), not clearance. Paige's actual FDA-authorized AI, Prostate Detect, predates the FM era.

### Cited Findings
- **Paige PanCancer Detect.** FDA Breakthrough Device designation was announced Apr 3 2025, described as the first such designation for an AI tool detecting common and rare cancers across anatomic sites. The tool is powered by Virchow2 (built with Microsoft) and covers more than 40 tissue types. One report describes it as research use only. — [BusinessWire Apr 2025](https://www.businesswire.com/news/home/20250403363714/en); [BioPharmaTrend](https://biopharmatrend.com/post/1191-paiges-ai-tool-for-multi-tissue-cancer-detection-gets-fdas-breakthrough-device-designation); [Medical Device Developments](https://www.medicaldevice-developments.com/news/paige-expands-pancancer-detect-ai-application-to-over-40-tissue-types/)
- Paige Prostate (Detect) was the first FDA-authorized AI application in pathology. Paige also holds a 510(k) for its FullFocus viewer, which is not a diagnostic AI. — [BusinessWire Jan 2025](https://www.businesswire.com/news/home/20250114113728/en)
- **H-optimus-0 (Bioptimus).** Integrated into Proscia Concentriq Embeddings (Oct 2024) as a research and development platform with no regulatory clearance. — [BusinessWire Oct 2024](https://www.businesswire.com/news/home/20241023399717/en)
- **Atlas (Mayo/Charité/Aignostics)** is a research preprint with no stated regulatory status. — [arXiv 2501.05409](https://arxiv.org/html/2501.05409v2)

### Inferences
- The academic VLMs carry non-commercial licenses: CONCH and UNI are CC-BY-NC-ND, and QuiltNet is research-only. This probably blocks direct productization, so regulated FM-based products will come from companies' proprietary FMs (Virchow, Atlas, H-optimus).

### Gaps
- I did not query the FDA AI-enabled device list or EUDAMED directly, so the absence of clearance is based on press and literature searches only.
- The current (Oct 2026) status of PanCancer Detect clearance was not confirmed.
- I did not verify whether any 2025–2026 510(k)/De Novo clearance (for example, from Paige, Ibex, Aignostics, or Owkin) explicitly cites an FM backbone.
