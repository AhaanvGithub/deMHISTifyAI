# Hard pathology datasets with expert labels (multi-rater / agreement focus) — catalogue as of 18 Oct 2026

Legend: [AVAILABLE] = already on team cluster. [V] = verified in this session from a cited page. [UNV] = from prior knowledge / secondary source, not verified this session; check before relying on it.

## Which datasets include per-pathologist labels or agreement levels at tile level?

### Takeaway
Few public tile-level datasets have per-rater labels or agreement scores. MHIST [AVAILABLE] ships an SSA vote count out of 7. Chaoyang is described as having 3 pathologist labels per patch, with a consensus-only test set, but it is unclear whether the public JSON contains all three labels. Gleason2019 has 6 per-pathologist pixel maps on TMA cores (tile-sized images), so tile-level per-rater labels can be derived from it. MITOS-ATYPIA-14 has 2 senior pathologists' atypia scores per x20 frame, with some recorded disagreements. NuCLS has a 53-FOV multi-rater subset with 7 pathologists at the nucleus level. Most other datasets (BRACS, MIDOG++, AMi-Br, PANDA, SICAPv2, BreakHis, Camelyon) give only consensus or single-rater labels.

### Cited Findings

**Catalogue: datasets with per-rater or agreement information**

- **MHIST [AVAILABLE]**: colorectal polyps, binary hyperplastic polyp (HP) vs sessile serrated adenoma (SSA). It has 3,152 H&E images at 224x224 px (magnification not stated on the page). annotations.csv holds the majority-vote label and the number of the 7 annotators who voted SSA. Individual per-annotator votes are not described. Access is a Research Use Agreement form, with links sent by email that expire after 4 h ("wait a few hours and resubmit" if no email). images.zip is 333 MB. — [MHIST page](https://bmirds.github.io/MHIST/) [V]. An official train/test partition column exists [UNV, not stated on the page fetched].
- **Chaoyang**: colon patches in 4 classes: normal, serrated, adenocarcinoma, adenoma. Train has 4,021 patches (1,111 / 842 / 1,404 / 664) and test has 2,139 (705 / 321 / 840 / 273), 6,160 in total. — [HSA-NRL GitHub](https://github.com/bupt-ai-cz/HSA-NRL) [V]
  - Patches are 512x512 and come from WSIs scanned at 20x. — [Project page](https://bupt-ai-cz.github.io/HSA-NRL/) [V for 20x]; [Systematic review, J Digit Imaging 2026](https://link.springer.com/article/10.1007/s10278-026-01899-y) [512x512, secondary]
  - "Each patch has three noisy labels annotated by three pathologists", and the test set contains only patches all experts agreed on. Another description says that for the ~40% of train patches with disagreement, one doctor's label was chosen at random. — via [search snippets of arXiv 2112.03694 / 2303.01099](https://arxiv.org/pdf/2112.03694) [secondary; it is unresolved whether the released JSON contains all 3 labels or one sampled label]
  - Access is a web form (name, email, institution, country) plus agreement to a non-commercial license. Redistribution is not allowed; derivative models are allowed. — [Project page](https://bupt-ai-cz.github.io/HSA-NRL/), [GitHub README](https://github.com/bupt-ai-cz/HSA-NRL) [V]. Approval latency was not stated. In past practice a link usually arrived quickly by email [UNV].
- **Gleason2019 (MICCAI 2019)**: prostate TMA cores with pixel-level Gleason annotations. Maps 1–6 are the labels of 6 pathologists, and each image is annotated by 4–6 experts. 244 public labelled TMAs out of 333 in total (some sources say 250). Test labels are not public. — [Gleason2019 site](https://gleason2019.grand-challenge.org/), [Register page](https://gleason2019.grand-challenge.org/Register) [V via snippets]
  - A Mendeley Data mirror exists, [DOI 10.17632/s384w7kv78.1](https://data.mendeley.com/datasets/s384w7kv78/1) [V via snippet]. It is likely a direct download without approval [UNV]. The original requires agreeing to a Data Use and Confidentiality Agreement. TMA images are about 5120x5120 px at 40x [UNV], so they can be tiled into patches that carry per-rater labels.
- **MITOS-ATYPIA-14**: breast cancer frames. Nuclear atypia is scored 1–3 at x20, given independently by 2 senior pathologists, and some frames have disagreements. Training has 284 x20 frames and 1,136 x40 frames, as TIFF. — [Challenge dataset page](https://mitos-atypia-14.grand-challenge.org/Dataset/) [V via snippet]. A secondary review claims 3 junior pathologists, which conflicts with the official page. The download page is [here](https://mitos-atypia-14.grand-challenge.org/Donwload/). It was not opened, so login requirements and size are unverified. Mitosis labels in the same frames use multi-pathologist consensus [UNV].
- **NuCLS**: breast cancer (TCGA) nuclei, for classification, detection and segmentation. There is a single-rater set of 1,744 FOVs with 59,485 nuclei, and a multi-rater set of 53 FOVs annotated separately by 7 pathologists (per arXiv 2206.08182). Multi-rater data comes with and without algorithmic suggestions. — [NuCLS site](https://nucls.grand-challenge.org/), [PMC9112766](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9112766/), [HF mirror](https://huggingface.co/datasets/minhanhto09/NuCLS_dataset) [V via snippets]. This is nucleus-level, not a lesion diagnosis. It is direct download, CC0 [UNV].
- **Arvaniti et al. 2018 prostate TMA (Gleason)**: discovery cohort of 641 patients, test cohort of 245 patients annotated by 2 pathologists. Inter-pathologist kappa = 0.71, and model vs each pathologist = 0.75 and 0.71. Disease-specific survival data is available for the test cohort. — [Sci Rep 2018, PMC6089889](https://pmc.ncbi.nlm.nih.gov/articles/PMC6089889) [V via snippet]. It is hosted on Harvard Dataverse (doi:10.7910/DVN/OCYCMP), a direct download of roughly tens of GB [UNV; Dataverse page not confirmed this session].
- **Haggenmüller/Brinker 2025, melanocytic tumours (Nat Commun 16:789)**: 792 melanoma-suspicious lesions from 8 German hospitals, each reviewed by 8 expert pathologists. Complete agreement was 53.5%; majority ≥5 was 90.9%; non-invasive melanoma had complete agreement in only 10 of 73 cases. The paper says the authors "provide access" to a panel-validated dataset with dermoscopic and histopathologic images plus metadata. — [DOAJ record](https://doaj.org/article/db350769110441bebee694db13fbb5b0), [IDEAS](https://ideas.repec.org/a/nat/natcom/v16y2025i1d10.1038_s41467-025-56160-x.html) [V via snippet]. The access route, license and whether per-pathologist diagnoses are included were not verified. Likely request-based [UNV].
- **SICAPv2 [AVAILABLE]**: prostate patches with a majority-vote patch label. A 2024 paper treats its label inconsistency with multi-label ensembles. — [PMC11226137](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11226137/). Per-annotator labels were not confirmed to be released.

**Consensus-only or single-rater datasets (hard lesions, but no per-rater labels)**

- **BRACS**: breast, 189 patients, 547 WSIs and 4,539 ROIs. Classes are normal, benign (PB, UDH), atypical (FEA, ADH), DCIS and invasive, i.e. 7 ROI classes, labelled by consensus of 3 board-certified pathologists. — [BRACS site](https://www.bracs.icar.cnr.it/), [arXiv 2111.04740](https://arxiv.org/abs/2111.04740v1) [V; class names UNV]. The site has Register/Login and a Download page. Access mechanics and license were not stated on the homepage. Official train/val/test splits are reported to exist [UNV]. ROIs are variable-size at 40x, roughly tens of GB for ROIs and TB-scale for WSIs [UNV].
- **MIDOG++**: 503 cases across 7 tumour types (human and canine). 11,937 mitotic figures and 14,351 imposters, labelled by a blinded consensus of 2 pathologists with a third adjudicating. Hosted on a Springer Nature figshare collection. — [figshare collection](https://springernature.figshare.com/collections/MIDOG_A_Comprehensive_Multi-Domain_Dataset_for_Mitotic_Figure_Detection/6615571) [V]. It is a direct download; the size is unverified (the WSI-region TIFFs are likely tens of GB [UNV]). MIDOG 2021 and 2022 training sets are on Zenodo ([4643381](https://zenodo.org/record/4643381), [6547151 PNG](https://zenodo.org/record/6547151)).
- **AMi-Br (2025)**: atypical vs normal mitoses in human breast cancer, built from TUPAC and MIDOG 2021. 3,720 figures from 223 cases (832 atypical, 22.4%). Uses a 3-annotator consensus vote. Balanced accuracy is 0.806 with a patch-level split and 0.713 with a patient-level split. Code and data: DeepMicroscopy/AMi-Br_Benchmark on GitHub. Pathologists show low inter-rater agreement on atypical mitoses. — [arXiv 2501.04467](https://arxiv.org/abs/2501.04467v2), [arXiv 2506.21444](https://arxiv.org/html/2506.21444v1) [V via snippets]. Whether the individual votes are released is unverified. It is small and patch-based, so plausibly a quick download [UNV].
- **PANDA**: prostate biopsies, 10,616 WSIs from Radboud and Karolinska, labelled with ISUP grade. Radboud has pixel masks (stroma, benign, Gleason 3/4/5). Karolinska labels came from a single experienced pathologist without a second-pathologist check. A Kaggle competitor reported removing about 20% of Radboud labels as noisy. PANDA-PLUS adds 546 slides with refined pixel annotations. — [Zenodo 3715938](https://zenodo.org/records/3715938), [Google Research blog](https://research.google/blog/an-international-scientific-challenge-for-the-diagnosis-and-gleason-grading-of-prostate-cancer/), [PMC8799467](https://pmc.ncbi.nlm.nih.gov/articles/PMC8799467), [UroToday PANDA-PLUS](https://www.urotoday.com/recent-abstracts/urologic-oncology/prostate-cancer/170126-panda-plus-improved-dataset-of-prostate-whole-slide-images-from-panda-challenge-with-pixel-level-expert-annotations-beyond-the-abstract.html?acm=_15613). Access is via Kaggle, which requires accepting the competition rules. It is about 400 GB, has a data_provider column and no per-rater labels [UNV].
- **LUAD-HistoSeg**: ambiguous patches with disagreement among 3 pathologists were dropped, so it is consensus-only. — [systematic review snippet / arXiv 2110.08048](https://arxiv.org/pdf/2110.08048)
- **IMP-cervix 2024 (INESC TEC)**: cervical LEEP and surgical specimens, 600 WSIs at 40x (Leica GT450, .svs). Classes are non-neoplastic, LSIL, HSIL and non-representative. Total size is 775 GB, DOI 10.25747/9028-C140. The portal has a "download and usage" document. — [INESC TEC RDM](https://rdm.inesctec.pt/es/dataset/nis-2024-003) [V via snippet]. It is single-label WSI and too large for today.
- **Oral epithelial dysplasia**: no public human dataset with multi-pathologist grades was found. Inter-observer studies exist (Peradeniya 200 slides with 6 raters; Manchester 68 slides with 4 raters; Aga Khan 80 slides with 3 raters), but no data release was found. The public "OralEpitheliumDB" is 456 mouse-tongue images ([GitHub LIPAIGroup/OralEpitheliumDB](https://github.com/LIPAIGroup/OralEpitheliumDB)). The Sheffield ODYN cohort (358 OED and 105 control WSIs) does not appear to be released. — [PMC11589032](https://pmc.ncbi.nlm.nih.gov/articles/PMC11589032), [Manchester](https://research.manchester.ac.uk/en/publications/why-oral-histopathology-suffers-inter-observer-variability-on-gra/)
- **Barrett's oesophagus dysplasia**: BOLERO has biopsies reviewed by a 4-expert panel plus 51 test pathologists. 2,805 H&E diagnoses gave >70% concordance for NDBE and HGD, 42% for LGD and 23% for IND. No public download was found. The JAMIA 2019 set (doi 10.1093/jamia/ocz089) has 429 labelled images with very few dysplasia examples. — [BOLERO, UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10090488/1/Bolero%20Paper_complete.pdf), [arXiv 2303.12711](https://arxiv.org/pdf/2303.12711), [PMC6798571](https://pmc.ncbi.nlm.nih.gov/articles/PMC6798571/table/ocz089-T2)
- **Thyroid follicular lesions**: no public multi-class dataset (FA / FTC / NIFTP) was found. There are only individual ZMDL-GUMED DICOM slides (CC BY-NC-SA), e.g. [NIFTP slide](https://doi.org/10.34808/46zz-yp25) and [minimally invasive FTC](https://doi.org/10.34808/rcht-ew57).
- **Meta-resource**: "Learning from Disagreement in Digital Pathology" (Zenodo, Sept 2026) codes 345 studies on disagreement. It is a literature resource, not images, but useful for finding more multi-rater datasets. — [Zenodo 23049803](https://zenodo.org/records/23049803)

**Other candidates from the brief, not verified this session** (prior knowledge only; mark [UNV])
- BreakHis: breast benign/malignant with 8 subtypes, 7,909 images at 700x460 px and 40/100/200/400x, 82 patients. Patient ID is embedded in filenames. Direct download from UFPR, about 4 GB. Single label.
- Camelyon16/17: lymph-node metastasis WSIs, open on AWS/GigaDB, TB-scale. No per-rater labels.
- DigestPath 2019: colonoscopy tissue and signet-ring cells. Required registration; current availability is uncertain.
- BACH, Osteo, SkinCancer, WSSS4LUAD, LC25000, NCT-CRC-HE-100K, ARCH, Quilt-1M, PathGen [AVAILABLE]: all single or consensus labels.
- PathMMU: a VQA benchmark with expert-validated questions. No per-rater diagnosis labels. HF download, possibly gated.
- CATCH (canine cutaneous, TCIA): 350 WSIs, single or consensus annotation, large.

### Inferences
- The best per-rater or agreement data that can be used immediately is MHIST (7-vote count, already on the cluster). It can be paired with Gleason2019 (6 per-pathologist maps, which can be tiled into patches with per-rater labels and agreement) and MITOS-ATYPIA-14 (2-rater atypia scores with disagreements).
- Chaoyang is the closest analogue to MHIST in difficulty (serrated vs adenoma vs normal colon). Its value for multi-rater work depends on whether the release has all 3 labels. Even if not, the train/test design (test = full agreement, train includes ~40% disagreement) defines "hard" vs "easy" samples implicitly.
- Agreement levels can also be proxied by lesion class: BRACS atypical classes (FEA/ADH) and Barrett's LGD/IND are documented low-concordance categories, but no public per-rater labels exist for them.

### Gaps
- Whether Chaoyang's released files include all three pathologists' labels could not be confirmed (sources conflict: "three noisy labels per patch" vs "one randomly chosen label").
- Whether MHIST annotations.csv carries per-annotator identities: the page says only a vote count.
- Access and license terms for the Haggenmüller/Brinker 2025 melanocytic dataset, and whether per-pathologist diagnoses are included.
- No 2023–2026 tile-level dataset built explicitly around inter-observer variability (besides AMi-Br and the 2025 melanocytic set) was found. The search returned mostly older resources.

## Which are tile-level and small enough to download and process within a few hours today?

### Takeaway
These are realistic today, assuming the download path is quick:
- MHIST [AVAILABLE, 333 MB]
- Gleason2019, via the Mendeley mirror: 244 TMAs with 6 maps; the size of a few GB is [UNV]
- MITOS-ATYPIA-14: small number of frames; size and login [UNV]
- NuCLS: small, HF mirror
- AMi-Br: small patch set, GitHub
- Chaoyang: 6,160 patches at 512x512, likely ~1–2 GB, if the form returns a link quickly
- BreakHis: about 4 GB [UNV]
- Arvaniti TMA: Dataverse direct download; the size of tens of GB is [UNV]

WSI datasets (PANDA ~400 GB, IMP-cervix 775 GB, Camelyon, BRACS WSIs, MIDOG++ full) are not realistic before the 6pm Central GPU cutoff.

### Cited Findings
- MHIST is 3,152 images at 224x224 and 333 MB — [MHIST](https://bmirds.github.io/MHIST/)
- Chaoyang is 6,160 patches at 512x512, from 20x scans — [GitHub](https://github.com/bupt-ai-cz/HSA-NRL), [project page](https://bupt-ai-cz.github.io/HSA-NRL/)
- Gleason2019 has 244 public labelled TMAs and a Mendeley mirror — [Mendeley](https://data.mendeley.com/datasets/s384w7kv78/1)
- MITOS-ATYPIA-14 has 284 x20 and 1,136 x40 TIFF frames — [challenge](https://mitos-atypia-14.grand-challenge.org/Dataset/)
- NuCLS has 1,744 single-rater FOVs and 53 multi-rater FOVs; HF mirror available — [HF](https://huggingface.co/datasets/minhanhto09/NuCLS_dataset)
- AMi-Br has 3,720 mitotic-figure patches — [arXiv 2501.04467](https://arxiv.org/abs/2501.04467v2)
- IMP-cervix is 775 GB — [INESC TEC](https://rdm.inesctec.pt/es/dataset/nis-2024-003)
- PANDA has 10,616 WSIs — [Zenodo](https://zenodo.org/records/3715938)

### Inferences
- Within a few hours, the best "hard lesion + rater signal" plan is: MHIST (on hand), plus Chaoyang (submit the form now), plus Gleason2019 tiles with per-pathologist maps, plus MITOS-ATYPIA-14 x20 frames with 2-rater atypia scores.
- BRACS ROI-only (not WSI) could be feasible if registration is instant, but its size and access were unverified.

### Gaps
- Exact download sizes for Chaoyang, Gleason2019 mirror, MITOS-ATYPIA-14, MIDOG++, BRACS ROIs and the Arvaniti Dataverse were not verified.

## Which require registration/approval that would not arrive in time?

### Takeaway
Form-based but usually quick (email link): MHIST (links expire after 4 h) and Chaoyang. Approval latency was unverified for both. Grand-challenge downloads (Gleason2019, MITOS-ATYPIA-14) need an account and may need challenge-join approval by organisers, which can take days; the Gleason2019 Mendeley mirror avoids this. BRACS requires site registration (latency unknown). PANDA needs a Kaggle account and rules acceptance (instant, but data is huge). The Haggenmüller/Brinker melanocytic data, BOLERO (Barrett's) and the oral dysplasia cohorts are request-only or unreleased, so treat them as not available today.

### Cited Findings
- MHIST: form plus Research Use Agreement; links are emailed and expire after 4 h — [MHIST](https://bmirds.github.io/MHIST/)
- Chaoyang: form (name, email, institution, country) plus non-commercial license — [project page](https://bupt-ai-cz.github.io/HSA-NRL/)
- Gleason2019: Data Use and Confidentiality Agreement on grand-challenge; Mendeley mirror available — [Register](https://gleason2019.grand-challenge.org/Register), [Mendeley](https://data.mendeley.com/datasets/s384w7kv78/1)
- BRACS: Register/Login and Download pages on the official site — [BRACS](https://www.bracs.icar.cnr.it/)
- MIDOG++: public figshare collection (no approval) — [figshare](https://springernature.figshare.com/collections/MIDOG_A_Comprehensive_Multi-Domain_Dataset_for_Mitotic_Figure_Detection/6615571)
- IMP-cervix: DOI landing page with a usage document — [INESC TEC](https://rdm.inesctec.pt/es/dataset/nis-2024-003)

### Inferences
- Submit the Chaoyang and BRACS forms immediately, in parallel with grabbing the Gleason2019 Mendeley mirror and NuCLS/AMi-Br, so that a slow approval doesn't block the day.

### Gaps
- Actual approval turnaround for Chaoyang, BRACS and grand-challenge (Gleason2019, MITOS-ATYPIA-14) is not documented on their pages.
- Arvaniti Dataverse URL and size were not confirmed this session.
