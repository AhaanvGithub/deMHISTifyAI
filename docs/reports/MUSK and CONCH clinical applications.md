# Teach pathology VLMs to say "I don't know"

CONCH (Nature Medicine, 2024) and MUSK (Nature, 2025) are the leading open pathology vision-language foundation models. Their real clinical value today is as **strong frozen feature extractors and zero-shot classifiers for coarse, few-class problems**. They are not validated diagnostic aids. CONCH has the best independent record: it **tied Virchow2 for the top mean AUROC (0.71) of 19 models on 13 external cohorts**. MUSK's headline results are mostly developer-reported. Those include immunotherapy-response prediction, pan-cancer prognosis and HER2 status, and none has been independently replicated. On tile-level frozen-feature benchmarks, both models trail large image-only encoders such as UNI2-h and Virchow2 by roughly 3–5 points. No prospective or reader study has tested either model as a pathologist aid, and no regulated product is built on them. The documented failure modes are consistent: prompt wording swings zero-shot results by up to 25%, embeddings encode the hospital of origin, probabilities are poorly calibrated, and popular benchmarks such as NCT-CRC-HE-100K carry color shortcuts. Those failures point to the best opening for a three-person weekend team. Build and rigorously evaluate a **training-free reliability layer** for zero-shot pathology VLMs. It would use prompt-paraphrase variance and CONCH/QuiltNet/PLIP disagreement as uncertainty signals, wrapped in conformal prediction and selective abstention, and stress-tested under the known NCT-CRC color shift. A small text-side probe would test whether the models can tell "suspicious for carcinoma" from "carcinoma". The idea fits the "precise, reproducible, interpretable" challenge directly. It needs well under 10 of the 30 GPU-hours. Its nearest competitors (PathLang, Oct 2026; Huang & Razavian, Aug 2026; DICE, Jun 2026) each cover only one piece of it.

## Two models, two different bets on how to learn pathology language

CONCH is a CoCa-style ViT-B/16 trained on **1.17M human histopathology image-caption pairs** from educational sources and PubMed Central figures. Its image and text encoders were each pretrained first on in-house slides and on more than 550k surgical pathology reports ([Lu et al., PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)). It deliberately left out TCGA, PAIP and GTEx to reduce benchmark contamination. The public release omits the captioning decoder, so only classification and retrieval can be reproduced ([GitHub mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH)). MUSK follows a different recipe. It is a BEiT-3-style model with shared attention and modality-specific experts. It was first trained without pairs on **50M TCGA patches and 1B PubMed tokens**, then aligned on about 1M pairs from Quilt-1M and PathAsst, and it takes 384-px input ([Xiang et al., PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)). One consequence follows from that data description: MUSK's TCGA-based prognosis results are evaluated on the same source it was pretrained on. A research-oriented judge will notice that.

In zero-shot use, CONCH reaches about **90% balanced accuracy on TCGA NSCLC, RCC and BRCA subtyping** and 79.1% on CRC100k. It also gives a usable zero-shot segmentation Dice of about 0.6 on SICAP and DigestPath ([PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)). MUSK claims wins over CONCH in its own paper:

- **Linear-probe accuracy:** +2.2 points averaged over 12 datasets.
- **PathVQA:** 73.2%.
- **HER2 (BCNB):** AUC 0.826 vs 0.771.
- **Melanoma relapse:** AUC 0.833.
- **Pan-cancer prognosis:** mean c-index **0.746 vs 0.684 for CONCH**.
- **NSCLC immunotherapy response:** AUC **0.768 vs 0.606 for PD-L1** ([PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)).

The flagship clinical results are multimodal (slide plus report or clinical data) and aggregated with MIL. They are therefore not evidence that tile embeddings alone predict therapy response. The immunotherapy cohorts are small and single-center: 118 NSCLC and 101 gastro-esophageal patients, which the authors themselves flag ([PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649)).

Follow-up work in 2024–2026 treats CONCH as the default VLM backbone and MUSK as one comparator among several. Examples:

- TITAN builds a whole-slide model on CONCH v1.5 patch features ([arXiv 2411.19666](https://arxiv.org/html/2411.19666v1)).
- PathFMTools compares CONCH and MUSK for grading cutaneous squamous cell carcinoma on 440 slides ([arXiv 2511.19751](https://arxiv.org/abs/2511.19751)).
- Fusion studies combine CONCH v1.5, MUSK and Virchow2 ([arXiv 2512.11104](https://arxiv.org/html/2512.11104)).
- PathPT does rare-cancer prompt tuning across PLIP, CONCH, MUSK and KEEP ([arXiv 2508.15904](https://arxiv.org/html/2508.15904v1)).

The applications are real, but almost all of them are retrospective research uses. Both models are licensed CC-BY-NC-ND 4.0 with gated Hugging Face weights. Each teammate needs individual approval, and the weights must not be redistributed in a public repo or demo ([CONCH GitHub](https://github.com/mahmoodlab/CONCH); [MUSK GitHub](https://github.com/lilab-stanford/MUSK)).

## Independent benchmarks rank CONCH first on slides, mid-pack on tiles

How useful these models look depends on the evaluation regime, and the independent evidence splits sharply by regime.

| Benchmark (independent) | Regime | Where CONCH / MUSK land |
|---|---|---|
| Neidlinger et al., Nat Biomed Eng 2025: 19 FMs, 13 external cohorts, 31 tasks | Slide-level MIL | **CONCH ties Virchow2 at 0.71 mean AUROC**; CONCH leads on morphology (0.77) and prognosis (0.63) ([arXiv 2408.15823](https://arxiv.org/abs/2408.15823)) |
| THUNDER, NeurIPS 2025 D&B: 23 FMs, 16 datasets | Tile-level frozen features | Linear probe: UNI2-h 86.3, Virchow2 84.8, **CONCH 81.9, MUSK 81.1**. Rank sum: CONCH 125, **MUSK 156** (near Phikon-v2) ([arXiv 2607.22861, quoting leaderboard](https://arxiv.org/html/2607.22861v1)) |
| PathBench 2025: 19 FMs, 64 tasks, private hospital cohorts | Mixed, external | Virchow2 and H-optimus-1 lead overall; MUSK wins only isolated tasks such as breast PR (AUC 0.894) ([arXiv 2505.20202](https://arxiv.org/html/2505.20202v1)) |
| HistoVL 2025: 14 cohorts, >5M patches | VLM zero-shot | Balanced accuracy drops by **up to 25%** with caption wording; low adversarial robustness, high ECE ([arXiv 2503.12990](https://arxiv.org/abs/2503.12990)) |

The pattern is consistent. Vision-language pretraining adds value through **slide-level aggregation of morphology signal and through zero-shot and retrieval, which only VLMs can do**. It does not produce a better raw tile embedding. KEEP shows that a knowledge-enhanced VLM can match image-only models at the tile level (THUNDER linear probe 83.2) ([arXiv 2607.22861](https://arxiv.org/html/2607.22861v1)). So the gap is not inherent to text supervision. In a weekend project, that means a raw linear probe on CONCH should be presented as a baseline. A team that claims a CONCH linear-probe win on a public tile dataset will be outgunned by any UNI2-h number.

Zero-shot works on few-class tasks with distinct morphology and breaks down elsewhere. On 30-class rare brain tumors (EBRAINS), CONCH's zero-shot score is 37.1%, against 68.2% with supervised ABMIL. LUAD growth-pattern κ is 0.200, which is not significantly better than PLIP. TCGA LUAD text-to-image retrieval is 24.0% ([PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)). In PathPT's extraction, EBRAINS zero-shot balanced accuracy was 0.20 for CONCH and 0.25 for MUSK. MUSK at one shot was near chance ([arXiv 2508.15904](https://arxiv.org/html/2508.15904v1)). Zero-shot grounding Dice on CAMELYON16 is only 0.13–0.21 across VLM backbones ([arXiv 2508.15904](https://arxiv.org/html/2508.15904v1)).

## No reader study, no clearance, and five well-documented failure modes

The clinical-validation gap is stark. These notes found **no prospective or reader study testing CONCH, MUSK, PLIP or QuiltNet as a pathologist aid**. The best reader-study effect sizes come from organ-specific foundation-model systems, and all of them are preprints:

- PulmoFoundation raised accuracy from 83.8% to 91.7% in a crossover study with 8 pathologists ([arXiv 2605.25878](https://arxiv.org/pdf/2605.25878)).
- A registered randomized prospective trial (NCT07157618) is still recruiting ([ClinicalTrials.gov](https://clinicaltrials.gov/study/NCT07157618)).

No FDA-cleared or CE-marked product is built on these academic VLMs. The closest commercial analogue is Paige PanCancer Detect, which runs on Virchow2 and holds Breakthrough Device designation, not clearance ([BusinessWire](https://www.businesswire.com/news/home/20250403363714/en)). MUSK's own press coverage says it is not ready for clinical use ([Stanford Daily](https://stanforddaily.com/2025/02/04/the-musk-model/)).

The failure modes are better documented than the successes, and each is a potential hackathon handle.

**Site signatures.** All tested foundation models encode the medical center more strongly than the biology. The hospital of origin is predicted more accurately than cancer type ([de Jong et al., arXiv 2501.18055](https://arxiv.org/abs/2501.18055)). PathoROB found robustness deficits in all 20 models it tested, and documents a malignant patch labeled "normal" because of the hospital it came from ([arXiv 2507.17845](https://arxiv.org/abs/2507.17845)). VLMs fare relatively well. Across 5 scanners, CONCH v1.5 matched the same tile across scanners 81.8% of the time, versus 25.8% for Phikon-v2. Even so, calibration broke across scanners for most models ([arXiv 2601.04163](https://arxiv.org/html/2601.04163v1)).

**Prompt fragility.** For CONCH, one dysplasia task's AUC moved from 0.935 with a concise prompt to 0.736 with an elaborate expert-framed one, and the best term differed between models ([Sharma et al., arXiv 2505.00134](https://arxiv.org/abs/2505.00134)). PathLang (October 2026) reports that results are "highly sensitive to clinically equivalent paraphrases" ([arXiv 2610.11329](https://arxiv.org/abs/2610.11329)).

**Miscalibration.** Overconfidence is most pronounced for PLIP and QuiltNet in histopathology ([MVC-Bench, arXiv 2608.27004](https://arxiv.org/pdf/2608.27004)). These notes found no CONCH- or MUSK-specific ECE table.

**Benchmark shortcuts.** In NCT-CRC-HE-100K, mean RGB alone gives more than 50% 9-class accuracy and a color histogram more than 82% ([Ignatov & Malivenko, arXiv 2409.11546](https://arxiv.org/abs/2409.11546)). Stain normalization does not remove these signatures ([JBCS 2026](https://journals-sol.sbc.org.br/index.php/jbcs/article/view/7045)).

**Demographic gaps.** Foundation-model features shrink, but do not eliminate, AUROC gaps between white and Black patients. Without them the gaps reach 16.0% for IDH1 prediction ([Vaidya et al., Nat Med 2024](https://pubmed.ncbi.nlm.nih.gov/38641744/)).

Taken together, the evidence says the bottleneck to clinical use is no longer raw accuracy. It is **knowing when a prediction is trustworthy**, and that is a reproducibility problem.

## Ten candidate gaps, ranked for a 30-GPU-hour weekend

The constraints shape the ranking. Weights are frozen and work is tile-level. CONCH is in hand, QuiltNet and PLIP are on servers, and MUSK is pending. GPU work ends Sunday at 6 pm, and the rubric rewards innovation, progress, clinical impact and communication. Interpretability and reliability layers on top of cached embeddings are therefore much stronger choices than any new training. Few-shot prompt learning and LLM-written class descriptions are already crowded. CuPL-style GPT-4 prompts on CONCH appear in IEEE JBHI 2025 ([arXiv 2503.10731](https://arxiv.org/html/2503.10731)) and GMAT ([arXiv 2508.01293](https://arxiv.org/pdf/2508.01293)). Prompt tuning appears in PathPT (Nat Commun 2026) and FFM-CP ([arXiv 2609.27710](https://arxiv.org/pdf/2609.27710)). Those methods belong in the baseline column.

| Rank | Idea | Nearest prior work | Novel slot | GPU-h (est.) | Rubric fit |
|---|---|---|---|---|---|
| **1** | **Conformal and selective zero-shot pathology driven by paraphrase variance and CONCH/QuiltNet/PLIP disagreement, tested under color shift** | PathLang ([2610.11329](https://arxiv.org/abs/2610.11329)) measures sensitivity but does not use it as uncertainty. Huang & Razavian ([2608.30390](https://arxiv.org/abs/2608.30390)) cover vision FMs with heads. DICE ([2606.30020](https://arxiv.org/abs/2606.30020v1)) is slide-level, vision-only. Conf-OT ([2505.24693](https://arxiv.org/pdf/2505.24693)) is not pathology. | Tile-level, VLM zero-shot, set-valued guarantees plus abstention | 3–6 | Reproducible, precise, a clear triage story |
| 2 | Hedging and negation probe: does the text encoder rank "suspicious for" below "consistent with", and separate "negative for X" from "X"? | Clinician certainty surveys ([Sci Rep 2025](https://www.nature.com/articles/s41598-025-11032-8); [APLM 2022](https://meridian.allenpress.com/aplm/article/146/7/886/472360/Communicating-Certainty-in-Pathology)); CLIP bag-of-words ([2502.03566](https://arxiv.org/pdf/2502.03566)) | No pathology-VLM hedging test found | <1 | Very high communication value |
| 3 | Color-shortcut audit of zero-shot VLMs on NCT-CRC | Ignatov & Malivenko ([2409.11546](https://arxiv.org/abs/2409.11546)) covered linear models only | Do text-only zero-shot predictions also follow color? | 2–4 | Reproducibility; best folded into Rank 1 |
| 4 | Tile-level text-concept bottleneck faithfulness audit (SpLiCE on CONCH/QuiltNet/PLIP) | SpLiCE ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/996bef37d8a638f37bdfcac2789e835d-Paper-Conference.pdf)); WSI-level PathGen-CLIP version ([2609.02985](https://arxiv.org/pdf/2609.02985)); VLG-CBM misalignment ([2408.01432](https://arxiv.org/pdf/2408.01432)) | Cross-model concept agreement and intervention validity | 1–2 | Interpretability; good stretch goal |
| 5 | Retrieval-augmented zero-shot classification with Quilt-1M captions as cited evidence | HOMIE is composed retrieval only ([2502.07221](https://arxiv.org/pdf/2502.07221)) | Absent from about 2 searches (weak evidence) | 2–3 | Interpretable rationales; noisy captions are a risk |
| 6 | Modality-gap and hubness census for CONCH, QuiltNet and PLIP | MUSK and PathGen gap ≥0.571 (HOMIE); hubness ([2609.01103](https://arxiv.org/html/2609.01103)) | No CONCH, QuiltNet or PLIP numbers found | <1 | A supporting figure |
| 7 | Text-grounded SAE on CONCH embeddings | PLUTO SAE ([2407.10785](https://arxiv.org/html/2407.10785v1)); PICASSO; EXPOSE (all vision-only) | Automatic feature naming through CONCH's own text encoder | ~1 | Novel but hard to evaluate in 48 h |
| 8 | Quilt-1M vs PathGen caption-image consistency audit | Quilt correction precision 57.9% ([2306.11207](https://arxiv.org/html/2306.11207v4)) | No independent audit found | 3–5 | Weak clinical hook; circular with QuiltNet |

Rank 1 wins because it scores on every rubric line. On innovation, it turns a documented weakness (paraphrase sensitivity) into a free uncertainty signal, which nobody has published for tile-level pathology VLMs. On progress, it is all cached embeddings plus CPU statistics, so a working pipeline by Saturday night is realistic. On clinical impact, it makes the triage rule concrete: "auto-label X% of tiles at a guaranteed error rate and route the rest to a pathologist." On communication, risk-coverage curves and coverage tables read cleanly. Interobserver agreement on colorectal dysplasia grading is modest (mean κ 0.420 in a four-pathologist study, [Rev Bras Cancerol](https://scielo.br/j/rbc/a/V9HL3xCBFXnYXBvM6VsSY6q/abstract/?lang=en)). That gives a framing: the goal is a tool that knows which cases are hard, not to replace pathologists. Rank 2 costs almost nothing, suits the biomed teammate, and adds a memorable result to the writeup, so the plan includes it as a parallel track. Rank 3 becomes the shift condition inside Rank 1, not a separate project.

Two novelty risks need checking before anything is written. **PathLang appeared nine days ago.** Its full text could already contain calibration or hedging experiments, so read Section 3 before claiming novelty on Ranks 1–2. A MICCAI 2025 paper on conformal pathology VLMs could also exist and not have surfaced in these searches. The "Pitfalls of conformal prediction" paper ([arXiv 2506.18162](https://arxiv.org/html/2506.18162)) argues that conformal sets break under shift and add little with few classes. Treat it as a reviewer you must answer, not one to ignore: report class-conditional coverage, run an explicit shift test, and use the 9-class task as the main conformal testbed.

## Weekend plan: "Know when the VLM doesn't know"

**Research question.** Two candidate signals need no training. One is per-tile variance across paraphrased class prompts. The other is disagreement among CONCH, QuiltNet and PLIP. Do these signals give better abstention (lower AURC) and tighter conformal prediction sets at a fixed coverage than softmax confidence? And do the coverage guarantees survive a documented stain/color shift?

**Data.** The NCT-CRC-HE-100K → CRC-VAL-HE-7K pair is the main testbed. The 7K test set comes from 50 separate patients ([HF dataset script](https://huggingface.co/datasets/1aurent/NCT-CRC-HE/blob/7730e526210509fb83dd87c6fb3e4e89457a1b76/NCT-CRC-HE.py)), and the two halves differ in class separability ([medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.04.07.25325392v1.full)), which gives a natural shift test. Then add whatever tile-classification subsets the team's PathMMU copy contains, targeting BACH, MHIST and Osteo, as co-primary endpoints. Do not report NCT-CRC alone. Confirm on Saturday morning which subsets are actually present. Use Quilt-1M only for prompt mining, not evaluation, because QuiltNet was trained on it.

**Timeline and division of labor.**

| Window | CS-1 (pipeline) | CS-2 (uncertainty/stats) | Biomed |
|---|---|---|---|
| Sat AM | Load CONCH locally; test QuiltNet/PLIP server throughput; embed NCT-CRC 100K+7K for all 3 models; cache to disk | Write split-conformal (APS/RAPS), MSP, entropy and AURC code on toy logits | Read PathLang §3 and the Pitfalls paper; write the class-synonym list; draft hedge templates from the survey papers |
| Sat PM | Embed the PathMMU subsets; build perturbed copies of the 7K set (grayscale, Macenko, HED jitter, JPEG q=30) and embed them | Generate 30 paraphrases per class with a Lumen LLM plus CONCH's 21 templates; embed all text | Vet the paraphrases for clinical equivalence (drop wrong ones); log rejection reasons as a data point |
| Sat night | Compute zero-shot logits: model × paraphrase × tile | First risk-coverage curves on NCT-CRC | Run the hedging/negation text probe (Rank 2) |
| Sun by 6 pm | Embed MUSK too if access lands; **all GPU work stops** | Calibrate on 100K, test on 7K and perturbed 7K; bootstrap CIs | Error case review: pull 30 abstained and 30 confidently wrong tiles and annotate the morphology |
| Mon–Wed | Figures, repo, README | Ablations and statistics | Clinical framing, limitations, and most of the written submission |

Compute is not the binding constraint. ViT-B encoders on roughly 120k NCT tiles plus perturbations plus PathMMU subsets should take a few GPU-hours across the three models; that is an engineering estimate, not a benchmarked number. That leaves 20+ hours of headroom for MUSK at 384 px and for a Rank 4 stretch. Precompute and cache every embedding once, then run all conformal analysis on CPU. That keeps the Sunday 6 pm deadline safe and makes Monday–Wednesday iteration free.

**Methods compared.** Six uncertainty scores:

1. Max softmax probability (MSP).
2. Entropy.
3. Temperature-scaled MSP, with the temperature fit on the calibration split.
4. **Paraphrase variance**: the variance or mean pairwise flip rate of the predicted class across paraphrase subsets.
5. **Cross-model disagreement**: vote entropy over CONCH, QuiltNet and PLIP, plus KL divergence between their probability vectors.
6. A simple logistic combination of 4 and 5, fit on the calibration split only.

Use each score both for selective classification and as a nonconformity score inside split conformal (APS and RAPS at α = 0.1 and 0.05). Baselines are the single-prompt zero-shot model and CONCH's 21-template ensemble. Report a CONCH linear probe only as an accuracy ceiling.

**Metrics.**

- Accuracy and balanced accuracy.
- ECE before and after temperature scaling.
- Conformal coverage: marginal and **class-conditional** empirical coverage, and mean and median set size.
- Selective classification: **AURC**, plus accuracy at 80% and 90% coverage.
- Shift: the coverage gap from calibration (100K) to test (7K) and to each perturbed set. This tests whether the guarantee holds and answers the Pitfalls paper.
- Color dependence: Spearman correlation between per-tile logit changes and mean-RGB change under the perturbations. This tests whether VLM zero-shot inherits the Ignatov shortcut.

Report everything with 1,000-sample bootstrap 95% CIs over tiles, and repeat over 10 random calibration/test splits for the within-distribution numbers.

**Rank 2 track (biomed-led).** Build the template grid "{hedge} {diagnosis}". Take the hedges from the survey rankings: diagnostic of, consistent with, compatible with, suggestive of, suspicious for, cannot exclude, negative for. Take the diagnoses from NCT-CRC and PathMMU class names. Then measure three things:

- **Monotonicity:** Spearman ρ between the image-text score and the clinician certainty rank ([Sci Rep 2025](https://www.nature.com/articles/s41598-025-11032-8)).
- **Negation:** the flip rate when "negative for X" replaces "X" as a class prompt.
- **Accuracy:** the change in zero-shot accuracy when class prompts carry hedges.

A null result ("hedges are invisible to all three encoders") is still a strong, communicable finding. It also links back to Rank 1: if the text encoders ignore uncertainty language, uncertainty has to come from outside the text, which is exactly what the reliability layer supplies.

**What success looks like.** The headline claim would be: "At 90% coverage, tri-model disagreement plus paraphrase variance raises CONCH's selective accuracy on CRC-VAL-HE-7K from A% to B% (95% CI) versus MSP. Conformal sets keep class-conditional coverage within C points of nominal under stain perturbation, while MSP-based sets do not." Whatever the numbers turn out to be, the triage figure is the clinical-impact slide: the fraction of tiles auto-labeled vs. routed to a pathologist at a fixed error budget. If MUSK access arrives, add it as a fourth voter. Whether a stronger fourth model reduces disagreement-driven abstention is itself a useful ablation.

**Papers to position against.** Primary:

- PathLang ([arXiv 2610.11329](https://arxiv.org/abs/2610.11329))
- Sharma et al. ([arXiv 2505.00134](https://arxiv.org/abs/2505.00134))
- Huang & Razavian ([arXiv 2608.30390](https://arxiv.org/abs/2608.30390))
- Pitfalls of CP ([arXiv 2506.18162](https://arxiv.org/html/2506.18162))
- Conf-OT ([arXiv 2505.24693](https://arxiv.org/pdf/2505.24693))
- DICE ([arXiv 2606.30020](https://arxiv.org/abs/2606.30020v1))
- Ignatov & Malivenko ([arXiv 2409.11546](https://arxiv.org/abs/2409.11546))

Context:

- CONCH ([PMC11384335](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/)) and MUSK ([PMC12295649](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649))
- Neidlinger benchmark ([arXiv 2408.15823](https://arxiv.org/abs/2408.15823))
- HistoVL ([arXiv 2503.12990](https://arxiv.org/abs/2503.12990))
- PathoROB ([arXiv 2507.17845](https://arxiv.org/abs/2507.17845))
- MVC-Bench ([arXiv 2608.27004](https://arxiv.org/pdf/2608.27004))

**Threats to state openly.**

- Results are tile-level and do not transfer to slide-level clinical decisions.
- NCT-CRC has no patient IDs and has known artifacts.
- The PathMMU subsets may overlap with QuiltNet or PLIP pretraining data, because those were scraped from the web.
- Conformal guarantees assume exchangeability, which the shift experiment deliberately breaks.
- The paraphrases were checked by one biomedical student, not a board-certified pathologist.

## Conclusion

The literature quietly reverses the usual framing for pathology VLMs. CONCH and MUSK are accurate enough for many coarse tasks, and independent benchmarks show that more scale mostly buys tile-level points that image-only models already hold. What no one has shown is that their zero-shot outputs come with trustworthy confidence. Under scanner, site or prompt variation, the evidence points the other way. A team that turns these models' best-documented weakness, prompt and model disagreement, into a calibrated abstention signal is working on the actual barrier between foundation models and the sign-out desk, not on a leaderboard.

Two strategic points follow for the team. First, the strongest contribution here is negative-space work: rigorous measurement of when and how these models fail, with honest shift tests. Research-oriented judges reward that more than a marginal accuracy gain on a saturated, color-confounded dataset. Second, MUSK access is not on the critical path. Its independent tile-level standing is weaker than CONCH's (THUNDER rank sum 156 vs 125), so its best role is as an optional fourth voter that tests whether a stronger model makes the uncertainty signal redundant.
