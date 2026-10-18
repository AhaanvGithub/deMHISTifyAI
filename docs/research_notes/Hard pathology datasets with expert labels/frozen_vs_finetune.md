# Frozen vs fine-tuned pathology foundation models (CONCH, MUSK, UNI, Virchow, QuiltNet, PLIP) for small, hard, noisy-label tile tasks

Research date: 2026-10-18. Tool budget: about 20 calls. Several primary PDFs (THUNDER, MUSK Nature, CONCH Nat Med) were too large or paywalled to fetch, so some specific numbers are marked as gaps or unverified.

## Q1. Published comparisons: linear probe vs full fine-tune vs LoRA on pathology FMs for tile classification

### Takeaway
The best controlled benchmark (Lee et al. 2024, 4 FMs x 14 datasets) finds LoRA/PEFT is the most consistent winner. Its margin over linear probing on small, hard tile sets is usually a few accuracy points, and on the noisy-label Chaoyang set linear probing actually beat LoRA for UNI. Full fine-tuning is rarely the best option and is the most commonly reported overfitting/forgetting risk. I found no head-to-head published for CONCH or MUSK specifically on MHIST.

### Cited Findings
- **Lee et al., "Benchmarking Pathology Foundation Models: Adaptation Strategies and Scenarios" (arXiv 2410.16038, Oct 2024).** It covers CTransPath, Lunit (ViT-S DINO), Phikon (ViT-B iBOT) and UNI (ViT-L DINOv2, 303.4M params). The methods compared are training from scratch, linear probe, full fine-tune, partial fine-tune (bottom 50% of layers) and LoRA (rank 8 on self-attention). — [arXiv](https://arxiv.org/html/2410.16038v1)
  - Chaoyang (CRC, 4 classes, noisy labels): LoRA was best for CTransPath (77.01% acc / 0.768 F1) and Lunit (70.41 / 0.704). **Linear probing was best for Phikon (74.42 / 0.743) and UNI (70.28 / 0.701).** — [arXiv](https://arxiv.org/html/2410.16038v1)
  - BACH (breast, 4 classes, 400 images): LoRA was best for every model. UNI reached 85.48 / 0.849 and Lunit 83.95 / 0.830. UNI trained from scratch reached only 48.21 / 0.444. — [arXiv](https://arxiv.org/html/2410.16038v1)
  - BRACS (7 classes): LoRA gave UNI 69.29 / 0.662, Phikon 73.30 / 0.720 and CTransPath 68.23 / 0.647. Full fine-tuning gave CTransPath a slightly higher F1 (0.664). — [arXiv](https://arxiv.org/html/2410.16038v1)
  - PEFT had a 75–100% win rate against the other fine-tuning methods across datasets. Partial fine-tuning was competitive in-domain (PANDA, KBSMC). — [arXiv](https://arxiv.org/html/2410.16038v1)
  - Few-shot results for UNI on an out-of-domain breast task: kNN 63.16% at 1 shot, Baseline++ 78.22% at 5 shots and 82.94% at 10 shots. kNN sometimes beat a trained linear head at 1 shot. — [arXiv](https://arxiv.org/html/2410.16038v1)
- **Mitotic figure classification (arXiv 2508.04441, 2025/2026).** LoRA-adapted FMs beat linear probing and reached near full-data performance with 10% of the data, especially H-optimus-0. Full fine-tuning of conventional CNNs stayed competitive. — [arXiv](https://arxiv.org/pdf/2508.04441)
- **Brain tumour transfer-learning study (arXiv 2501.11014, 2025).** Linear probing beat fine-tuning on external test sets, which the authors attribute to catastrophic forgetting. — [arXiv](https://arxiv.org/pdf/2501.11014)
- **Position paper (arXiv 2510.23807).** It argues that full fine-tuning of pathology FMs often degrades accuracy relative to linear probing because of overfitting and forgetting, and that the models are "too large, memory-intensive, and unstable" to fine-tune on moderate-sized data. This is an argument, not a controlled experiment. — [arXiv](https://arxiv.org/html/2510.23807v1)
- **Al Mamun et al., breast microsegmentation (IEEE SPMB 2026 preprint, unpublished).** Frozen linear probes of GigaPath, UNI and CONCH all beat EfficientNet-B0, and LoRA added further gains. The best system was CONCH+LoRA (mDice 0.646, Cohen's κ 0.544, +11.2% relative over the baseline). This is a segmentation-style task, not standard tile classification. Unverified because I could not read the tables. — [preprint PDF](https://isip.piconepress.com/publications/unpublished/conferences/2026/ieee_spmb/fine_tuning/paper_v10.pdf)
- **MUSK-LoRA on a local Chilean cohort (Research Square preprint, not peer reviewed, 100 patients).** It reports ROC-AUC 0.95 and MCC 0.81, against a linear-probe MCC of about 0.71. — [Research Square](https://www.researchsquare.com/article/rs-10194651) (UNVERIFIED, from a search snippet only)
- **Sharma et al., NeurIPS 2025 benchmark of PEFT on pathology VLMs.** This is the first PEFT benchmark for pathology VLMs and includes CONCH (+MMRL) on prostate, CRC and a new neurofibrillary-tangle dataset with annotator variability. With full data, PEFT-adapted generalist VLMs rival adapted pathology FMs, but they fall short in few-shot settings. No numbers appear on the abstract page. — [NeurIPS](https://neurips.cc/virtual/2025/128772), [OpenReview](https://openreview.net/forum?id=N7NtYkZfdx)

### Inferences
- For hundreds to a few thousand tiles, the realistic gain of LoRA over a linear/MLP probe on a strong FM is about 0–5 accuracy points. The gain is larger on out-of-domain or fine-grained tasks such as BACH and mitosis, and can be negative on noisy-label tasks such as Chaoyang with UNI. Full fine-tuning is the weakest risk/benefit choice.
- On a noisy, multi-rater set like MHIST, the Chaoyang result (linear probing best for UNI and Phikon) is the closest analogue. It suggests that frozen features plus a well-regularized head is a strong default.

### Gaps
- I found no published linear probe vs LoRA vs full fine-tune comparison for **CONCH or MUSK on MHIST, BRACS, Chaoyang, BACH or SICAPv2** with exact numbers. The MUSK Nature paper (DOI 10.1038/s41586-024-08378-w, Nature 638, Feb 2025) and the CONCH Nature Medicine paper (2024) were paywalled or unfetchable here.

## Q2. Known performance numbers on MHIST, BRACS, Chaoyang, BACH and SICAPv2 (frozen vs fine-tuned)

### Takeaway
MHIST is small (2,175 train / 977 test tiles at 224px) with moderate inter-rater agreement (κ about 0.45). A fully trained ImageNet ResNet-18 already reaches 92.7% AUC. Frozen-FM leaderboards (kaiko eva, THUNDER) report MHIST balanced accuracy, but I could not extract exact per-model MHIST values for CONCH or UNI from them.

### Cited Findings
- **MHIST dataset** (Wei et al., arXiv 2101.12355, AIME 2021):
  - 3,152 tiles at 224×224 from 328 WSIs, labelled HP vs SSA.
  - Train split: 1,545 HP / 630 SSA. Test split: 617 HP / 360 SSA.
  - Labels are the majority of 7 GI pathologists. Mean pairwise agreement is 72.9% and mean per-pathologist κ is 0.450. About 16.7% of images have only 4/7 agreement.
  - — [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.12355)
- **MHIST baselines** (same paper):
  - ResNet-18 reaches 84.5 ± 1.1% AUC from random init and **92.7 ± 0.4% AUC** with ImageNet pretraining.
  - It trains in about 6 minutes on an RTX 3090 with 3.5 GB of memory.
  - Deeper ResNets did not help.
  - — [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.12355)
- **MHIST training on agreement subsets** (random-init ResNet-18, AUC): very easy only 79.9, +easy 84.6, +hard (5/7) 85.1, all images 84.5. Dropping only the 4/7 images was slightly better than using all images (85.1 vs 84.5). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.12355)
- **kaiko eva frozen leaderboard protocol.**
  - Setup: frozen backbone, single-layer MLP head, cross-entropy, 12,500 steps, batch 256, lr 3e-4, AdamW, early stopping. Metric is balanced accuracy, averaged over 5 runs.
  - MHIST is one of the columns, alongside BACH, BreakHis, CRC, Gleason, PCam, Cam16Small, PANDASmall, CoNSeP and MoNuSAC.
  - The top rows are Virchow2, UNI2-h and H-optimus-0.
  - — [eva leaderboard](https://kaiko-ai.github.io/eva/main/leaderboards/), [chart SVG](https://kaiko-ai.github.io/eva/main/images/leaderboards/pathology.svg)
  - I could not reliably map per-model MHIST values from the SVG, so those numbers are a gap. The BACH column ranges about 0.76–0.92 across the top rows.
- **THUNDER benchmark** (arXiv 2507.07860, NeurIPS 2025 D&B Spotlight, v3 17 Feb 2026). It covers 23 FMs on 16 tile-level datasets including MHIST, with linear probing plus robustness and uncertainty analyses. A search snippet showed MHIST linear-probe balanced accuracy of about 75.8 and 81.2 for two HIBOU models. I could not fetch the CONCH, MUSK or UNI rows because the PDF exceeded the fetch limit. — [arXiv](https://arxiv.org/abs/2507.07860)
- **Chaoyang dataset.**
  - About 6,160 colon tiles, 4 classes, 3 pathologists.
  - The test set contains only the patches where all 3 pathologists agreed. For roughly 40% of training patches with disagreement, one pathologist's label was chosen at random, so training labels are hard and noisy and no vote distribution is released.
  - — [HSA-NRL arXiv 2112.03694](https://arxiv.org/pdf/2112.03694), [PapersWithCode](https://astro.paperswithcode.com/dataset/chaoyang)
  - Best reference result is about 83.4% accuracy. — [arXiv 2303.01099](https://arxiv.org/pdf/2303.01099)
  - FM numbers are under Q1: UNI linear probe 70.28%.
- **BACH and BRACS FM numbers** (Lee et al.): see Q1. — [arXiv](https://arxiv.org/html/2410.16038v1)

### Inferences
- Because about 17% of MHIST tiles are 4/7 split decisions, the label ceiling is well below 100%. Leaderboard gaps of 1–2 points between frozen FMs on MHIST are probably within noise from label ambiguity and seeds.
- An ImageNet ResNet-18 at 92.7% AUC sets a strong bar. A frozen-FM probe should be compared against it on the same AUC metric, not only balanced accuracy.

### Gaps
- I did not obtain exact MHIST, BRACS or SICAPv2 balanced accuracy for CONCH, MUSK, UNI, Virchow2, PLIP or QuiltNet, whether from eva, THUNDER, Patho-Bench, the CONCH paper or the MUSK paper. The report writer should either flag this or have someone read the eva SVG or THUNDER tables directly.
- I found no published fine-tuned (LoRA or full) FM result on MHIST.

## Q3. Compute and time cost of LoRA or full fine-tuning on one A100 for a few thousand 224px tiles

### Takeaway
At this data size, cost is not the binding constraint. Embedding about 3k tiles once and training heads takes minutes. LoRA on a ViT-L costs roughly 2.6x linear probing per step and about 6.7x the memory, which still fits easily on an A100. The real cost is that every hyperparameter or seed trial needs another full backbone pass.

### Cited Findings
- **UNI (ViT-L, 303M params) per-batch costs**, from Lee et al. The batch size and GPU are not shown in my extract.

  | Method | Memory per batch | Time per batch |
  |---|---|---|
  | Linear probe | 2,331 MB | 523 ms |
  | LoRA | 15,561 MB | 1,384 ms |
  | Partial fine-tune | 21,555 MB | not extracted |
  | Full fine-tune | 27,039 MB | 1,425 ms |

  - LoRA adds about 0.19M parameters to UNI.
  - — [arXiv](https://arxiv.org/html/2410.16038v1)
- LoRA on Virchow2 trains about 2M parameters, 0.3% of 633M. — [arXiv 2510.04861](https://arxiv.org/pdf/2510.04861)
- MHIST ResNet-18 full training takes about 6 minutes on an RTX 3090. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.12355)
- MUSK's native input is 384×384 (`musk_large_patch16_384`), not 224. — [MUSK GitHub](https://github.com/lilab-stanford/MUSK)

### Inferences (not directly sourced; back-of-envelope)
- Embedding MHIST once (3,152 tiles) with a ViT-L should take roughly 1–3 minutes on an A100. MUSK at 384px costs about 3x the tokens.
- MLP/linear heads on cached embeddings train in seconds on GPU, or even on CPU.
- A LoRA run on about 2,200 train tiles for 20 epochs is about 1,400 forward/backward passes of batch 32. At about 1–1.5 s per step that is about 20–35 minutes per run including evaluation.
- With about 10 GPU-hours per person, a small LoRA sweep is feasible: about 3 seeds x 3 configurations ≈ 3–5 h. But that compute comes at the expense of the frozen-feature experiments, which are far cheaper and allow many seeds, cross-validation and ensembling.

### Gaps
- I found no published wall-clock numbers specifically for CONCH or MUSK LoRA on an A100.

## Q4. Learning from soft labels and rater-vote distributions, label-noise robustness, and calibration

### Takeaway
On MHIST-style colorectal polyp data, using per-image annotator agreement as soft or smoothed targets cut calibration error by almost 70% while maintaining or improving accuracy (Wei et al. 2022). This is the most directly relevant, cheap and frozen-compatible intervention. Curriculum ordering by agreement also helps. I found no published study that measures calibration (ECE) for frozen-probe vs LoRA vs full fine-tuning of pathology FMs.

### Cited Findings
- **Wei, Torresani, Wei & Hassanpour, "Calibrating Histopathology Image Classifiers using Label Smoothing"** (arXiv 2201.11866, Jan 2022; Dartmouth, the MHIST group).
  - Agreement-aware label smoothing on colorectal polyp classification reduced calibration error by almost 70% while maintaining or improving accuracy.
  - A confidence-aware variant, using model confidence as a proxy for agreement, also helps when only one annotator is available.
  - — [arXiv](https://arxiv.org/abs/2201.11866v1), [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11866)
  - I did not read the exact ECE numbers or confirm that MHIST itself was the dataset, so treat both as unverified.
- **Wei et al., "Learn like a Pathologist: Curriculum Learning by Annotator Agreement"** (WACV 2021). Ordering training from high to low annotator agreement improves histopathology classification. — [CVF](https://openaccess.thecvf.com/content/WACV2021/html/Wei_Learn_Like_a_Pathologist_Curriculum_Learning_by_Annotator_Agreement_for_WACV_2021_paper.html), [arXiv](https://arxiv.org/pdf/2009.13698.pdf)
- **MHIST annotation format.** `annotations.csv` gives the majority label plus the number of the 7 annotators who voted SSA. This converts directly to a soft target p(SSA) = votes/7. — [MHIST page](https://backiee.wasmer.app/https_bmirds_github_io/MHIST); confirmed by [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.12355)
- **Soft labels in general (Wu et al., Findings of EMNLP 2023).** Annotator disagreement and confidence can be used to generate soft labels that improve single-label classifiers. — [ACL Anthology](https://preview.aclanthology.org/new-sigs/2023.findings-emnlp.355/)
- **HSA-NRL (Zhu et al., arXiv 2112.03694).**
  - Introduces Chaoyang and a noise-robust method: it detects easy, hard and noisy samples from training history, then self-trains with label correction.
  - Its core message is that hard examples should not be confused with noisy ones.
  - — [arXiv](https://arxiv.org/pdf/2112.03694)
- **Multi-Head Multi-Loss calibration (arXiv 2303.01099).** Evaluates calibration on Chaoyang and highlights its high aleatoric uncertainty from label ambiguity. — [arXiv](https://arxiv.org/pdf/2303.01099)
- **Sharma et al. (NeurIPS 2025).** Their new NFT dataset deliberately captures annotator variability to assess "reliability and alignment" of PEFT-adapted CONCH. No numbers were extracted. — [NeurIPS](https://neurips.cc/virtual/2025/128772)

### Inferences
- With frozen embeddings, soft-label training is trivial: train the head with cross-entropy or KL against [1−v/7, v/7]. It can be combined with temperature scaling on a validation split. This is the highest expected-value, lowest-cost calibration intervention.
- Fine-tuning (especially full) on small, noisy data tends to produce overconfident models. This is general deep-learning knowledge, not pathology-FM-specific evidence found here. Frozen probes with soft labels are likely better calibrated, but this is unverified for these specific models.
- Evaluate against both the majority label and the vote distribution: balanced accuracy and AUC on majority labels, plus ECE, Brier score and NLL against soft labels, possibly stratified by agreement level (4/7 to 7/7).

### Gaps
- I found no paper reporting ECE or Brier score for CONCH, MUSK or UNI under linear probe vs LoRA vs full fine-tuning.
- I found no published FM result trained on MHIST soft votes.

## Q5. Prompt learning (CoOp/CoCoOp) and text-guided heads for CONCH as a middle ground

### Takeaway
Prompt learning on CONCH or QuiltNet is cheap and works in few-shot regimes, but the evidence suggests it mainly helps at very low shot counts. With hundreds to thousands of labelled tiles, a linear or MLP head on image embeddings usually matches or beats it. Zero-shot CONCH or QuiltNet prompts are weak baselines on some tasks; for example, QuiltNet zero-shot SICAP had weighted κ of only 0.02.

### Cited Findings
- **"Effortless VLM specialization in histopathology without annotation" (arXiv 2508.07835, Aug 2025).**
  - Continued pretraining of CONCH or QuiltNet on retrieved image-caption pairs improved QuiltNet zero-shot by 15.22% (TAPT) and 10.41% (DAPT) on MHIST, and by 79.62% and 55.34% on BACH.
  - SICAP weighted κ for QuiltNet went from 0.02 to 0.50.
  - On MHIST, TAPT with 16 shots matched CoOp with 8 shots, and TAPT with 100 shots matched CoOp with 16 shots.
  - LoRA showed similar patterns to full updates, but rank 2 underperformed with more data.
  - DAPT gave larger, more consistent gains for CONCH.
  - — [arXiv](https://arxiv.org/html/2508.07835v1)
- **PathFiT supplementary table** (via search snippet). CONCH with text prompt learning on pan-cancer TCGA reached about 58% balanced accuracy at 16 shots. — [arXiv 2412.20430](https://arxiv.org/pdf/2412.20430) (UNVERIFIED snippet)
- **Slide-level prompt learning study (arXiv 2503.17238).** CoOp trailed zero-shot at low shot counts on CLIP/BiomedCLIP, which shows prompt learning can be unstable with very few shots. — [arXiv](https://arxiv.org/pdf/2503.17238)
- **Sharma et al. (NeurIPS 2025).** PEFT-adapted VLMs degrade in few-shot settings due to "label scarcity, terminology mismatch, and modality-specific biases". — [NeurIPS](https://neurips.cc/virtual/2025/128772)

### Inferences
- A practical middle ground for CONCH or MUSK: initialise a linear head from text embeddings of class prompts (zero-shot weights), then fine-tune the head on cached image embeddings (a WiSE-FT / "LP-init-from-text" style approach). This costs nothing extra in GPU time and stays within the "frozen weights" recommendation. It is not directly benchmarked in the sources above, so it remains an inference.

### Gaps
- I found no CoOp/CoCoOp numbers for CONCH or MUSK on MHIST, BRACS or Chaoyang with exact values.

## Q6. Constraints: hackathon guidance, GPU budget, and licensing

### Takeaway
The evidence supports the slides' "freeze, embed once, train small heads" recommendation for this setting. LoRA has a modest expected upside (0–5 points, sometimes negative on noisy labels), a real time cost in a one-day window, and adds licensing friction for MUSK.

### Cited Findings
- **MUSK license.** CC-BY-NC-ND 4.0, for non-commercial academic use only. It prohibits commercial use or monetization of "the MUSK model and its derivatives", including "models trained on outputs from the MUSK model". The repo does not explicitly address fine-tuning, though it notes that pytorch-lightning was used for downstream fine-tuning. — [MUSK GitHub](https://github.com/lilab-stanford/MUSK)

### Inferences
- Under CC licences, ND restricts *sharing* adapted material, not private creation or use. Fine-tuning MUSK for internal hackathon evaluation is probably fine for non-commercial research, but **publishing or sharing fine-tuned MUSK weights or LoRA adapters would likely violate ND**. A head trained on MUSK embeddings may also count as a "model trained on outputs" for commercial-use purposes. This is not legal advice.
- CONCH and UNI are believed to carry similar CC-BY-NC-ND terms with gated access. That belief is UNVERIFIED in this session; check the Hugging Face model cards.
- **Recommended plan given about 10 A100-hours and a 6pm Central cutoff today (2026-10-18):**
  1. Embed all tiles once with CONCH, MUSK (384px) and UNI/Virchow2, at under 15 minutes each. Cache the embeddings to disk so work can continue on CPU after GPU access ends.
  2. Train logistic-regression and MLP heads with hard labels vs soft vote labels, using 5+ seeds or cross-validation. Apply temperature scaling. Report balanced accuracy, AUC, ECE and Brier score.
  3. Optionally ensemble heads across FMs.
  4. Only if time remains, run a single LoRA config on the best FM (about 30 min per run) as an ablation. Do not share the resulting MUSK weights.

### Gaps
- I did not verify the licence texts for CONCH, UNI or Virchow2 in this session.
