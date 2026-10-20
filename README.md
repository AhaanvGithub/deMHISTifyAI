# deMHISTifyAI

A desktop tool for one of pathology's hardest calls: **sessile serrated lesion (SSL) vs hyperplastic
polyp (HP)** on H&E. You type your call and how sure you are, snip the slide region on screen, and get:

- an image panel's call and confidence, trained to match a 7-pathologist consensus
- a map of which parts of the field drove that call, sharpening while you read
- the crypts in the field outlined and measured (dilated, serrated, round and regular)
- a literature judge that reasons over published evidence, with citations
- a verdict comparing your call with the evidence, and what to look at again

Ashby Hackathon 2026 submission by Aiden Wang, Alice Yu and Ahaan Verma. Research tool, not for diagnosis.

---

## Using it

1. Start the app: `cd app` then `npm start`. It runs quietly in the background (the Python backend starts
   with it) and opens a demo image with two MHIST-sized fields, A and B, to try it on.
2. Press **Ctrl+Shift+Space**, type your call (e.g. "sessile serrated lesion, 70%") and press Enter.
3. Drag a box around the tissue. The app places a square field on it:
   - if the viewer shows a scale bar (e.g. "100 µm"), the square is exactly **448 µm**, one MHIST field; you
     can move it, and the app says when to zoom in or out;
   - otherwise it is the largest square that is mostly tissue; you can move and resize it.
4. Press Enter and read the panel beside your selection. Toggle the map between **AI's call** (what pushed the
   call toward SSL, red, or HP, blue) and **Your call** (what supports your call). Hover for detail.
5. Optional: the **gear** sets the decision cut-off. It shows how SSL and HP training images score, with
   presets from "catch at least 95% of SSLs" to "clear at least 99% of HPs". The default is the balanced
   cut-off, which the reported results use.

| Folder | What's in it |
|---|---|
| `app/` | Electron desktop app: launcher, snip screen, results panel, cut-off settings |
| `backend/` | Local service on port 8765: `engine.py` (image panel, map), `glands.py` (crypts), `scalebar.py`, `region.py` (field placement), `llm.py` (literature judge), `server.py` |
| `research/` | Training and evaluation scripts |
| `docs/results/` | Result files |
| `papers/` | Literature library. `papers/library.json` is not distributed; build it from your own OpenEvidence answers with `research/build_library.py` |

Models and data are not included; each user obtains access themselves (see Reproducing).

---

## How it works

| Stage | What happens |
|---|---|
| Capture | The screen is captured in true colour (Windows GDI, through the backend) when the snip screen opens |
| Field | The scale bar is read with Windows' built-in OCR to size a 448 µm square; without one, tissue is separated from glass and viewer chrome by colour |
| Image panel | ConvNeXt-Base and MUSK each score the field; their weighted, calibrated combination gives P(SSL) and the call |
| Unlike-training-data check | The field's ConvNeXt features are compared with the 2,175 training images; beyond what 99% of training images reach, the panel warns that its call may not apply |
| Evidence map | Each region of the field is blurred out in turn and the change in ConvNeXt's score is recorded: 16 x 16 regions first, then 32, 64 and 128 while the literature judge runs |
| Crypts | Crypts are found by their pale interior and ring of nuclei, outlined to the edge of the nuclear band, and measured: width (dilated), lumen edge roughness (serrated), shape (round and regular) |
| Feature scores | CONCH compares the field with text descriptions of WHO-2019 features; only features whose direction on training images matches the literature are passed on |
| Literature judge | An LLM (Lumen) reasons over the panel's result, the usable feature scores and a curated passage library, citing only those passages. It never sees the image |
| Verdict | Compares your call with the image panel and the literature judge |

### Design decisions

| Decision | Why |
|---|---|
| Score one square field, the size MHIST tiles show (448 µm) | The models learned SSL vs HP at that field size; a different size or shape changes what they see |
| Read the scale bar to size the field | Viewers zoom freely; a known scale is the direct way to show the models exactly one training-sized field |
| Panel of an ImageNet CNN and a pathology model | They make different mistakes; together they beat any single model on training data |
| Fine-tune instead of frozen features | Frozen features reach ~0.80 balanced accuracy on MHIST; fine-tuning reaches 0.89 (results below) |
| Two calibrations: the call and the displayed percentage | The call balances missed SSLs and over-called HPs; the percentage reads as "how often the pathologist majority says SSL" |
| Cut-off set on training data, adjustable in the app | A clinic may prefer catching SSLs over clearing HPs; the trade-off is shown, not hidden |
| No stain normalisation | Every method tested scored slightly worse than none, on MHIST and on simulated labs; training with colour variation already makes the panel robust |
| Evidence map by blurring regions out | It measures what actually moved this call, at the field size the model was trained on |
| Crypts measured, not matched to text | Matching image regions to WHO phrases located features poorly; measured outlines follow real glands, and their thresholds come from MHIST training crypts |
| Warn on unfamiliar input | Trained on two classes only, the panel would otherwise call anything (glass, photos, other stains) HP |
| Every number from training data; test scored once | So the reported results show what to expect on new slides |

---

## The image panel

### Data

MHIST: 3,152 H&E tiles of colorectal polyps (224 x 224 px, Dartmouth-Hitchcock), each labelled with how
many of 7 GI pathologists called it SSL.

| Split | HP | SSL | Total |
|---|---|---|---|
| Train | 1,545 | 630 | 2,175 |
| Test (official, fixed) | 617 | 360 | 977 |

The goal is to match the pathologists' consensus. The headline metric is **balanced accuracy** (SSL and
HP count equally), so the panel cannot score well by leaning toward the more common class.

### Members

Four candidates were trained; the panel-building rule (below) chose **ConvNeXt-Base + MUSK**.

| Candidate | Type | Training | Learning rate | Input | Cross-validated AUC (train) | In the panel |
|---|---|---|---|---|---|---|
| ConvNeXt-Base | ImageNet CNN | Classifier first, then all layers | 1e-4 | 224 px | 0.962 | **Yes** (weight 1.13) |
| MUSK | Pathology vision-language model (ViT-L) | Classifier first, then LoRA (rank 8) on the attention of all 24 layers | 1e-4 | 384 px | 0.946 | **Yes** (weight 0.64) |
| PLIP | Pathology vision-language model | Classifier first, then all layers | 1e-5 | 224 px | 0.951 | No |
| CONCH | Pathology vision-language model | Classifier first, then all layers | 1e-5 | 224 px | 0.949 | No |

CONCH (original weights) still powers the feature scores and their outlines on the image.

### Training recipe (all members)

| Choice | Setting | Why |
|---|---|---|
| Target | Share of the 7 pathologists who called SSL (e.g. 5/7 = 0.71) | Learns the consensus, including how split it was |
| Class balance | Each SSL image weighted 2.45 (= 1,545 HP / 630 SSL) | Both classes carry equal total weight, matching balanced accuracy |
| Classifier first | 1 epoch training only the classifier, then up to 8 fine-tuning epochs, best epoch kept | Fine-tuning starts from a sensible classifier, so the model's existing knowledge is adjusted gently |
| Learning rate | 1e-5 for full fine-tuning of pathology models, 1e-4 for the CNN and for LoRA | Standard rates for each kind of fine-tuning on small data |
| Image variation (training only) | Flips, 90-degree rotations, mild brightness/contrast/saturation/hue jitter | Robust to lighting and stain differences; predictions use the image as-is, so a snip always gets the same score |
| Precision | bf16 activations, fp32 weights | Full fine-tuning fits on an 8 GB laptop GPU |

### Building the panel

| Step | What happens |
|---|---|
| Cross-validation | Each member is trained 5 times on 4/5 of the training images (same folds for all), so every training image gets a prediction from a model that never saw it |
| Every combination | All 15 groups of members are scored, each with one learned weight per member |
| Selection | The best group by AUC; among groups within 0.003 AUC of it, the smallest wins. All four scored 0.9662; ConvNeXt-Base + MUSK scored 0.9655 and was chosen (ConvNeXt-Base alone, 0.9622, was not close enough) |
| Calibration | The displayed percentage is calibrated to the 7-pathologist majority call, so "80%" means the majority says SSL about 80% of the time on MHIST-like fields |
| Cut-off | Set to maximise balanced accuracy on the training images: the app calls SSL above 27% |
| Final models | Each chosen member is retrained once on all 2,175 training images; the app runs these |
| Evaluation | The 977 test images are scored once, on the exact version the app runs |

All choices are made on training images; the test split is used only for the final score.

---

## Performance

On the 977 MHIST test images, scored once with the exact models, weights and cut-off the app runs
(`docs/results/panel.json`):

| | Balanced accuracy | AUC | Accuracy | SSL caught | HP caught | Calibration error | Correlation with vote count |
|---|---|---|---|---|---|---|---|
| **Panel (ConvNeXt-Base + MUSK)** | **0.888** | **0.956** | **0.887** | **89.2%** | **88.5%** | **2.2%** | **0.858** |
| Earlier panel "CLIK" (ConvNeXt-Tiny + PLIP + MUSK, last 2 layers) [c] | 0.877 | 0.959 | 0.863 | 93.1% | 82.3% | 4.7% | 0.860 |
| ConvNeXt-Base alone | 0.853 | 0.948 | 0.827 | 95.3% | 75.4% | 10.0% | 0.840 |
| PLIP alone | 0.862 | 0.946 | 0.858 | 87.8% | 84.6% | 11.2% | 0.838 |
| MUSK alone | 0.820 | 0.936 | 0.781 | 96.7% | 67.3% | 5.4% | 0.825 |
| CONCH alone | 0.831 | 0.932 | 0.804 | 93.3% | 72.8% | 8.4% | 0.816 |
| Fine-tuned CONCH, last 2 blocks (earlier release) | 0.819 | 0.908 | 0.826 | 79.2% | 84.6% | 3.4% | 0.775 |

[c] CLIK was the best group before ConvNeXt-Base and MUSK with LoRA were trained: the average of its 5
cross-validation models, with weights, calibration and cut-off fit on training images
(`docs/results/preliminary_test_check.json`). The final panel replaced it by the same training-data rule.
Single members use a plain 50% cut-off, so they lean toward SSL; the panel's calibrated cut-off balances
the two classes. About 1 in 6 test tiles is a 4-3 split among the pathologists, which caps any model: even
a perfect model would agree with a fresh 7-pathologist majority only about 89-92% of the time. The panel's
0.887 accuracy is at that ceiling. On the 4-3 splits it agrees with the majority 65% of the time.

### Compared with published results on MHIST

No method here trains on test images. What differs is whether the test split was used to *choose* the
model (which epoch or checkpoint to keep), which makes a score look better than it would on new data. The
first table holds only methods that never used the test split, the fair comparison for deMHISTifyAI;
the second holds methods that used it or do not say. Both use the official MHIST test split (977 tiles),
sorted by balanced accuracy; rows that do not report it follow. Blank = not reported.

**Never used the test split to choose the model**

| Method | How the model is adapted | Balanced acc. | Accuracy | AUC | F1 | ECE | Test images used to choose the model | Source |
|---|---|---|---|---|---|---|---|---|
| **deMHISTifyAI panel** (ConvNeXt-Base + MUSK) | Fine-tuned (section above), group chosen on training data | **0.888** | **0.887** | **0.956** | **0.888** | **2.2%** | None | this repo |
| deMHISTifyAI, earlier panel "CLIK" (ConvNeXt-Tiny + PLIP + MUSK) | Fine-tuned; average of 5 cross-validation models | 0.877 | 0.863 | 0.959 | 0.865 | 4.7% | None | this repo [c] |
| Virchow2, LoRA fine-tuned | LoRA | | | | ~0.875 (derived) | | None (validation carved from train) | THUNDER [2] |
| Virchow2 | Frozen + linear | 0.848 | | | 0.836 | 0.6% | None (validation carved from train) | THUNDER [2] |
| H-optimus-0 | Frozen + linear | 0.837 | | | 0.828 | 1.6% | None (validation carved from train) | THUNDER [2] |
| Virchow | Frozen + linear | 0.827 | | | 0.823 | 3.5% | None (validation carved from train) | THUNDER [2] |
| UNI | Frozen + linear | 0.825 | | | 0.824 | 3.1% | None (validation carved from train) | THUNDER [2] |
| deMHISTifyAI, earlier release (fine-tuned CONCH x4) | Last 2 blocks fine-tuned, soft labels, calibrated | 0.819 | 0.826 | 0.908 | 0.827 | 3.4% | None | this repo |
| DINOv2-B (natural images) | Frozen + linear | 0.819 | | | 0.826 | 0.6% | None (validation carved from train) | THUNDER [2] |
| CONCH | Frozen + linear | 0.800 | | | 0.797 | 3.2% | None (validation carved from train) | THUNDER [2] |
| MUSK | Frozen + linear | 0.792 | | | 0.800 | 7.0% | None (validation carved from train) | THUNDER [2] |
| PLIP | Frozen + linear | 0.790 | | | 0.785 | 1.7% | None (validation carved from train) | THUNDER [2] |
| QuiltNet | Frozen + linear | 0.735 | | | 0.735 | 1.8% | None (validation carved from train) | THUNDER [2] |
| CONCH, LoRA fine-tuned | LoRA | | | | ~0.851 (derived) | | None (validation carved from train) | THUNDER [2] |
| UNI, LoRA fine-tuned | LoRA | | | | ~0.847 (derived) | | None (validation carved from train) | THUNDER [2] |

**Used the test split to choose the model, or did not say**

| Method | How the model is adapted | Balanced acc. | Accuracy | AUC | F1 | ECE | Test images used to choose the model | Source |
|---|---|---|---|---|---|---|---|---|
| Virchow2 | Frozen + MLP head | 0.861 | | | | | All 977 (checkpoint selection) | EVA protocol [3] |
| UNI / H-optimus-0 | Frozen + MLP head | 0.843 | | | | | All 977 (checkpoint selection) | EVA protocol [3] |
| CONCH | Frozen + MLP head | 0.832 | | | | | All 977 (checkpoint selection) | EVA protocol [3] |
| Virchow | Frozen + linear | 0.830 | 0.834 | | 0.835 | | Not stated | Virchow paper [4] |
| ImageNet ViT (natural images) | Frozen + linear | 0.821 | 0.826 | | 0.827 | | Not stated | Virchow paper [4] |
| CTransPath | Frozen + linear | 0.801 | 0.817 | | 0.816 | | Not stated | Virchow paper [4] |
| PolarHE (self-supervised) | Frozen + linear |  | 0.891 | 0.941 | 0.890 |  | Not stated | PolarHE [5] |
| UNI (as reported by [5]) | Frozen + linear |  | 0.856 | 0.920 | 0.856 |  | Not stated | PolarHE [5] |
| DiRL (as reported by [5]) | Not stated |  | 0.836 | 0.950 |  |  | Not stated | PolarHE [5] |
| DINO (as reported by [5]) | Frozen + linear |  | 0.831 | 0.898 | 0.816 |  | Not stated | PolarHE [5] |
| ResNet-18, ImageNet-pretrained | Fully fine-tuned | | | 0.927 | | | All 977 (best epochs kept) | MHIST paper [1] |
| ResNet-18, from scratch | Fully trained | | | 0.845 | | | All 977 (best epochs kept) | MHIST paper [1] |

How the protocols differ:
- **deMHISTifyAI**: every choice (members, epochs, weights, calibration, cut-off) is made on training images;
  the test split is scored once. F1 is weighted F1, as in the other rows. The panel has the highest balanced
  accuracy and AUC of any comparable result we found, using only openly available models; accuracy and F1
  are level with PolarHE [5], which does not say how its model was selected.
- **[5] PolarHE**: frozen features with a linear classifier; reports the official split sizes (2,175 / 977) but
  not how the model was selected.
- **Not comparable**: a 2026 preprint reporting 90.2% accuracy and 0.955 AUC with vision transformers (CaiT) used
  5-fold cross-validation over all 3,152 images instead of the official test split [6].
- **[1] MHIST paper**: AUC is the mean of each run's 5 best test-set epochs over 10 seeds, so the epoch is
  chosen on the test split. It reports AUC only.
- **[2] THUNDER** (NeurIPS 2025, 23 models): frozen features with a linear classifier, 95% confidence
  intervals of about +-2.5 points on MHIST. LoRA rows are derived as THUNDER's frozen F1 plus its reported
  LoRA gain on MHIST (Virchow2 +3.9, CONCH +5.4, UNI +2.3 points).
- **[3] EVA** (kaiko.ai protocol, values from the MINT paper): frozen features with an MLP head, 5 runs;
  EVA uses the MHIST test split for early stopping and checkpoint selection.
- **[4] Virchow paper**: frozen features with a linear classifier (appendix table A4).

Sources:
[1] Wei et al., *A Petri Dish for Histopathology Image Analysis*, 2021 ([arXiv 2101.12355](https://arxiv.org/abs/2101.12355)).
[2] *THUNDER: Tile-level Histopathology image UNDERstanding benchmark*, NeurIPS 2025, tables S38, S39, S52 and table 3 ([arXiv 2507.07860](https://arxiv.org/abs/2507.07860)).
[3] *MINT: Molecularly Informed Training with Spatial Transcriptomics Supervision for Pathology Foundation Models*, table 2 ([arXiv 2603.07895](https://arxiv.org/abs/2603.07895)); protocol: [kaiko.ai eva](https://kaiko-ai.github.io/eva/main/leaderboards/).
[4] Vorontsov et al., *Virchow: A Million-Slide Digital Pathology Foundation Model*, table A4 ([arXiv 2309.07778](https://arxiv.org/abs/2309.07778)).
[5] *Beyond H&E: Unlocking Pathological Insights with Polarization via Self-supervised Learning* (PolarHE), 2025 ([arXiv 2503.05933](https://arxiv.org/abs/2503.05933)).
[6] *Vision Transformers Based AI Models For Predicting Colorectal Cancer from Digital Pathology WSI: Use Case Of MHIST dataset*, medRxiv 2026 ([link](https://www.medrxiv.org/content/10.64898/2026.02.03.26345516v1.full)).

Supporting analyses:

| Question | Finding | File |
|---|---|---|
| Does weighting SSL vs HP differently change how well cases are ranked? | No: AUC is the same for every weight from 30/70 to 70/30; only the cut-off moves | `docs/results/weight_sweep_learning_curve.json` |
| Does more training data help? | Yes: AUC 0.878 / 0.896 / 0.906 with 25 / 50 / 100% of train | `docs/results/finetune_learning_curve.json` |
| Do brightness/contrast/zoom adjustments at prediction time help? | No: all within +-0.002 AUC of the plain image | `research/image_tta_sweep.py` |
| Do the CONCH feature scores match the visible tissue? | Only weakly; only the 3 of 14 whose direction on training images matches the literature are passed to the literature judge, as hints | `docs/results/feature_reality_check.md`, `docs/results/feature_calibration.json` |
| Does averaging several phrasings per feature help? | No: 5 phrasings per WHO feature gave 2 usable features instead of 3, and flipped the direction of several | `docs/results/feature_calibration.json` |
| Do the measured crypt labels agree with the pathologists? | Yes, modestly, on 1,575 training images not used to set the thresholds: dilated and serrated crypts are more common the more pathologists called SSL (Spearman +0.23, +0.27), round regular crypts less (-0.27). Each alone separates SSL from HP at AUC ~0.6, so they explain, they don't decide | `docs/results/gland_check.json` |
| Is the scale bar read correctly? | On 120 generated viewer screens (two random sets; six fonts, 11-22 pt, black, white and yellow labels above, below or beside the bar, µm and mm, 125% and 150% display scaling): 84 of 102 bars read within 3%, none read wrongly, and all 18 screens without a bar left alone. Labels must be round scale values (1, 2, 2.5 or 5 times a power of ten), which turns OCR slips into misses instead of wrong scales; when the bar can't be read, the app falls back to the tissue square | `docs/results/scale_stress.json`, `docs/results/scale_stress_seed1.json` |
| Does the app reproduce the reported test result? | Yes: the 977 test tiles sent through the app's own path as screen snips give balanced accuracy 0.886 (0.892 through a 150% display-scaling round trip) vs 0.888 reported, with no errors; ~1% are flagged as unlike the training data | `docs/results/app_panel_check.json` |
| Does stain normalisation help? | No: Reinhard, Macenko, Vahadane and a derived method all score slightly below no normalisation, on MHIST and on four simulated labs (ConvNeXt-Base, held-out training folds). The app uses images as they are | `docs/results/stain_compare.json` |
| Does it hold up on another hospital's slides? | UniToPatho (Turin; different scanner and stain), scaled to MHIST's field size: ConvNeXt-Base calls 99-100% of 545 hyperplastic polyps (41 slides) HP, depending on how each patch is cropped to one field. Normal colon and adenomas score lower still (median P(SSL) 0.15 and 0.18 vs 0.21 for HP), matching their lack of serration. UniToPatho has no SSLs, so outside SSL detection is untested | `docs/results/unitopatho_check.json` |
| Does the unlike-training-data check work? | It flags all 16 non-tissue and wrong-stain test images (photos, noise, a brown immunostain, a photo recoloured in H&E colours) and 1% of MHIST training images by design. On UniToPatho it flags 14% of HP, 9% of adenoma and 20% of normal colon fields. Frozen CONCH features flagged 51% of the outside HP and were not used | `research/export_panel.py` |

Scope: trained on MHIST (one lab, about 5x). Slides that look very different, e.g. much higher power or a
very different stain, are outside what the models have learned; the unlike-training-data check warns about
many of these. The crypt finder is a measurement, not a trained detector: it finds clearly ringed crypts and
can miss crypts cut lengthwise or open to the surface.

---

## Reproducing

Use `.venv\Scripts\python.exe` (Python 3.12, CUDA) with `requirements.txt` (versions as used; see the comments
in it for PyTorch, CONCH and MUSK). The app itself needs Node.js (`cd app`, `npm install`). Requires Hugging Face access to CONCH
(`MahmoodLab/CONCH`) and MUSK (`xiangjx/musk`), and the MHIST dataset in `data/mhist/`. MUSK's code is
loaded from a clone of github.com/lilab-stanford/MUSK via `MUSK_SRC`.

```bash
# 5-fold cross-validation per candidate
python research/kfold_cv.py --backbone conch         --blocks -1 --lr 1e-5 --batch 8 --accum 2 --tag conch_full
python research/kfold_cv.py --backbone convnext_base --blocks -1 --lr 1e-4 --batch 8 --accum 2 --tag convnext_base
python research/kfold_cv.py --backbone plip          --blocks -1 --lr 1e-5 --tag plip_full
MUSK_SRC=<MUSK clone> python research/kfold_cv.py --backbone musk_lora --lr 1e-4 --batch 2 --accum 8 --tag musk_lora

# final model per candidate on all training images (N = median best epoch from its folds), e.g.
python research/kfold_cv.py --backbone convnext_base --blocks -1 --lr 1e-4 --batch 8 --accum 2 --tag convnext_base --final-epochs N

# choose the panel on training data, then score the test split once
python research/panel.py --members conch_full convnext_base plip_full musk_lora

# export what the app needs (standardisation, calibrations, cut-off chart, unlike-training-data reference)
python research/export_panel.py

# crypt label thresholds from MHIST training crypts, and the held-out check against pathologist votes
python research/gland_thresholds.py
python research/gland_check.py

# feature-score calibration for the literature judge (training images only)
python research/feature_calibration.py

# demo image opened at start-up (needs the UniToPatho HP patches in data/unitopatho/)
python research/make_demo_image.py
```

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free to use, change and share for any non-commercial purpose
(study, teaching, research, hackathon judging and demos); commercial use needs permission.

- Built for a hackathon and for research; not part of a publication.
- A research tool, not a medical device: not for diagnosing or treating patients.
- Third-party models and data (CONCH, MUSK, PLIP, ConvNeXt weights, MHIST, OpenEvidence content) are not
  included and keep their own terms; several forbid commercial use and sharing derivatives.
