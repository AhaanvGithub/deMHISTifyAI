# gutfeeling

A second read for one of pathology's least reproducible calls: **sessile serrated lesion (SSL) vs hyperplastic
polyp (HP)** on H&E. Press a hotkey over any slide viewer, snip one field, and gutfeeling shows:

- **the call**, from image models fine-tuned and calibrated on a 7-pathologist consensus, with a score where 50% is
  the decision line and how often the panel catches SSL
- **where the call comes from**: a map of which parts of the field pushed it toward SSL or HP
- **the crypts**, outlined and measured in microns (dilated, round and regular), by a segmentation network
- **similar cases**: the most similar labelled SSL and HP fields from other labs, side by side
- **a literature reading**: a short, located explanation of the call with citations, plus links to the references
  pathologists use (PathologyOutlines, WHO Blue Books, ExpertPath)

Built for pathologists who read 30-40 cases a session: it opens instantly, stays quiet, and every number on screen
is measured. Ashby Hackathon 2026, by Aiden Wang, Alice Yu and Ahaan Verma. Research tool, not for diagnosis.

**Demo v2:** [watch on YouTube](https://youtu.be/q3kmF1mwB1Q)

[![gutfeeling demo v2](https://img.youtube.com/vi/q3kmF1mwB1Q/hqdefault.jpg)](https://youtu.be/q3kmF1mwB1Q)

---

## Using it

1. Start the app: `cd app` then `npm start`. It runs in the background (the Python backend starts with it).
2. Over any slide viewer (QuPath, a web viewer, an image), press **Ctrl+Shift+Space**. The screen darkens into the
   snip view.
3. Click or drag where the field should go:
   - if the viewer shows a scale bar (QuPath's, bottom left, or a "100 um" bar anywhere), the field is exactly
     **448 um**, one MHIST field, and can be moved; the app says when to zoom in or out;
   - otherwise it is the largest square that is mostly tissue, and can be moved and resized.
4. Press **Enter**. The results open beside the field; **F** enlarges the field into a focused view.
5. Read the panel, top to bottom: the image call, the two models and their sizes, similar cases (open; click a thumbnail to enlarge),
   the literature reading, and the references.

| Key | What it does |
|---|---|
| Ctrl+Shift+Space | Snip the screen |
| Enter | Analyse the field |
| F | Focused view on or off (a fade, nothing moves across the screen) |
| Space (hold) | Hide everything the AI drew, panels included: just the tissue |
| Scroll / drag | Zoom and pan the field (up to 8x, from the screenshot's full resolution) |
| E / C / S | Exposure, contrast, saturation views (they change only the display, never the call) |
| 0 | Reset zoom and views |
| Esc | Close |

The **gear** sets the decision cut-off: it shows how SSL and HP training images score, with presets from "catch at
least 95% of SSLs" to "clear at least 99% of HPs". The default is the balanced cut-off used in every reported result.

| Folder | What's in it |
|---|---|
| `app/` | Electron app: snip screen, results panel, cut-off settings |
| `backend/` | Local service on port 8765: `engine.py` (image panel, evidence map, similar cases), `gland_net.py` + `glands.py` (crypts), `scalebar.py`, `region.py` (field placement), `llm.py` (literature reading), `server.py` |
| `research/` | Training and evaluation scripts |
| `docs/results/` | Result files |
| `papers/` | Literature library. `papers/library.json` is not distributed; build it from your own OpenEvidence answers with `research/build_library.py` |

Models and data are not included; each user obtains access themselves (see Reproducing).

---

## How it works

| Stage | What happens |
|---|---|
| Capture | The screen is captured in true colour (Windows GDI, through the backend) the moment the hotkey is pressed; the snip view shows that screenshot, so it looks like the live screen |
| Field size | The viewer's scale bar is read first in the bottom-left corner (QuPath draws a see-through bar there), then anywhere: the bar is found as a thin line, its label with Windows' built-in OCR. A label must be a round scale value and must sit centred on its bar, so a misread gives no scale rather than a wrong one. The field is then exactly 448 um; without a bar, tissue is separated from glass and viewer chrome by colour |
| Image panel | ConvNeXt-Base (88M parameters) and MUSK (305M, LoRA) each score the field; a learned, calibrated combination gives the probability that a 7-pathologist majority would call SSL. The displayed score puts the cut-off at 50%, so an SSL call always reads above 50%; the underlying probability is one click away |
| Unlike-training-data check | The field's ConvNeXt features are compared with the 2,175 training images; beyond what 99% of training images reach, the field is flagged. Far beyond anything real tissue reached (photos, noise, screen content), the crypts and the literature reading are skipped |
| Evidence map | Each region of the field is blurred out in turn and the change in ConvNeXt's score is recorded: 16 x 16 regions first, refined to 32, 64 and 128 while the rest runs |
| Crypts | A U-Net (ConvNeXt-Base encoder) trained on GlaS expert gland outlines finds the crypts; each is measured in um and labelled dilated (wider than the field's other crypts and above the MHIST threshold) or round and regular |
| Similar cases | The field's ConvNeXt features are compared with 454 labelled reference fields from other labs: 214 SSL (pathologist-labelled SPIDER fields from 71 slides, plus HunCRC) and 240 HP (UniToPatho). The three most similar of each class are shown, with which class is closer |
| Literature reading | An LLM (Gemma-4 31B on Lumen) explains the panel's call from what was measured on this field (crypts, their positions, the evidence map, the unfamiliar-slide flag) and from five excerpts retrieved from a curated library. It never sees the image and never overrules the panel: it gives a confidence (capped at 92%), at most two measured points for the call and one against, and at most two located things to check by eye. Three runs; the answer shown is the cleanest, checked in code for invented citations, overclaiming, unmeasured evidence and claims of absence |

### Design decisions

| Decision | Why |
|---|---|
| Score one square field, the size MHIST tiles show (448 um) | The models learned SSL vs HP at that field; a different size or shape changes what they see (a bigger field was tested and did not help, below) |
| Read the scale bar to size the field | Viewers zoom freely; a known scale is the direct way to show the models exactly one training-sized field |
| Panel of an ImageNet CNN and a pathology model | They make different mistakes; together they beat any single model on training data |
| Fine-tune instead of frozen features | Frozen features reach ~0.80 balanced accuracy on MHIST; fine-tuning reaches 0.89 (results below) |
| Calibrate to the 7-pathologist consensus | There is no clean ground truth for SSL vs HP: on MHIST two pathologists agree 73% of the time and all seven on 34% of fields. The panel learns the consensus, including how split it was |
| Display score with the cut-off at 50% | The balanced cut-off is a calibrated 27%; showing "SSL 30%" for an SSL call reads as a contradiction |
| The image models decide; the LLM explains | The image models are trained and validated; a general vision LLM given the same images scores 55% balanced accuracy (below). The literature reading always takes the panel's call |
| Confidence capped at 92% | Experts don't agree more than that: a single pathologist matches the MHIST majority 83% of the time |
| Similar cases from other labs, per class | Real labelled precedent instead of an opinion; per-class nearest neighbours, because the pools differ in size |
| Fast and quiet | Pathologists read ~100 slides a session: no splash, no counting-up numbers, no streaming text; transitions are short fades |
| No stain normalisation | Every method tested scored slightly worse than none, on MHIST and on simulated labs |
| Evidence map by blurring regions out | It measures what actually moved this call, at the field size the model was trained on |
| Crypts measured, not matched to text | Text-matched "feature scores" (CONCH) located features poorly and contradicted each other; they were dropped. Measured outlines follow real glands |
| Warn on unfamiliar input | Trained on two classes only, the panel would otherwise call anything (glass, photos, other stains) HP |
| Every number from training data; the test split scored once | So the reported results show what to expect on new slides |

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
| Cut-off | Set to maximise balanced accuracy on the training images: SSL above a calibrated 27%, which the app displays as a score of 50% |
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
first table holds only methods that never used the test split, the fair comparison for gutfeeling;
the second holds methods that used it or do not say. Both use the official MHIST test split (977 tiles),
sorted by balanced accuracy; rows that do not report it follow. Blank = not reported.

**Never used the test split to choose the model**

| Method | How the model is adapted | Balanced acc. | Accuracy | AUC | F1 | ECE | Test images used to choose the model | Source |
|---|---|---|---|---|---|---|---|---|
| **gutfeeling panel** (ConvNeXt-Base + MUSK) | Fine-tuned (section above), group chosen on training data | **0.888** | **0.887** | **0.956** | **0.888** | **2.2%** | None | this repo |
| gutfeeling, earlier panel "CLIK" (ConvNeXt-Tiny + PLIP + MUSK) | Fine-tuned; average of 5 cross-validation models | 0.877 | 0.863 | 0.959 | 0.865 | 4.7% | None | this repo [c] |
| Virchow2, LoRA fine-tuned | LoRA | | | | ~0.875 (derived) | | None (validation carved from train) | THUNDER [2] |
| Virchow2 | Frozen + linear | 0.848 | | | 0.836 | 0.6% | None (validation carved from train) | THUNDER [2] |
| H-optimus-0 | Frozen + linear | 0.837 | | | 0.828 | 1.6% | None (validation carved from train) | THUNDER [2] |
| Virchow | Frozen + linear | 0.827 | | | 0.823 | 3.5% | None (validation carved from train) | THUNDER [2] |
| UNI | Frozen + linear | 0.825 | | | 0.824 | 3.1% | None (validation carved from train) | THUNDER [2] |
| gutfeeling, earlier release (fine-tuned CONCH x4) | Last 2 blocks fine-tuned, soft labels, calibrated | 0.819 | 0.826 | 0.908 | 0.827 | 3.4% | None | this repo |
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
- **gutfeeling**: every choice (members, epochs, weights, calibration, cut-off) is made on training images;
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

### Could a general vision LLM do this?

The same 977 MHIST test images were given to **Gemma-4 31B**, a general multimodal LLM, with an expert prompt that
states the WHO 2019 criteria, one image at a time (`research/vlm_baseline.py`, `docs/results/vlm_baseline.json`):

| On the 977 MHIST test images | Balanced accuracy | SSL caught | HP cleared | Called SSL |
|---|---|---|---|---|
| **gutfeeling panel** | **0.888** | 89.2% | 88.5% | |
| Gemma-4 31B, expert prompt | 0.554 | 96.4% | 14.4% | 89.6% of images |
| Chance | 0.500 | | | |

The general model calls nearly every field SSL: it describes H&E correctly but cannot separate the two lesions. The
decision here needs models trained on pathologists' calls.

---

## Other labs' slides

None of these were used for training. They test how far the MHIST-trained panel carries to other hospitals.

| Dataset | What it is | Result |
|---|---|---|
| UniToPatho (Turin) | 545 HP crops, 41 slides; also adenoma and normal colon | 99.6% of HP called HP, and every adenoma and normal field; crypt measurements follow the biology (serrated crypts 27% of labelled crypts in HP, 6% in adenoma, 3% in normal colon); 11-21% flagged unfamiliar. No SSL, so SSL detection is untested here (`unitopatho_panel_check.json`) |
| SPIDER-colorectal | 10,886 fields from many slides, SSL and HP labelled by region | HP cleared 99.8% (5,893 fields), SSL caught 6.0% (4,993 fields); 9.5% flagged unfamiliar. Its SSL fields measure the same as its HP fields on every crypt measure (in MHIST and HunCRC, SSL halves crypt density), consistent with region-level labels (`spider_panel_rows.jsonl`, `scale_audit.md`) |
| HunCRC (Budapest) | 24 HP slides and one SSL slide (lesion area not marked) | HP slide fields 93% called HP; 4% of the SSL slide's fields called SSL, partly because much of that slide is normal or unusable tissue (`field_size_sweep.md`) |

The honest picture: the panel is strong where it was trained and conservative elsewhere: it keeps HP right but
under-calls SSL on other labs' slides. Two causes were tested:

- **Scale.** MHIST's tiles are likely about 720-860 um, not the nominal 448 um: crypt geometry and lymphocyte size,
  two independent rulers, both read about 1.9x against UniToPatho's known scale (`scale_audit.md`). But scoring
  outside fields at 650 and 790 um did not bring out SSL (`field_size_sweep.md`), so scale is not the reason.
- **Labels and labs.** Each outside set is labelled by its own lab, sometimes by region; pathologists themselves
  agree on SSL vs HP only about three-quarters of the time. Shared consensus labels, or molecular markers such as
  BRAF, are what the field needs to measure this properly.

---

## Supporting analyses

| Question | Finding | File |
|---|---|---|
| How much do pathologists agree? | On MHIST, a single pathologist agrees with the 7-pathologist majority 83% of the time, two pathologists agree 73%, all seven on 34% of fields. About 1 in 6 test tiles is a 4-3 split, which caps any model at roughly 89-92% agreement with a fresh majority; the panel's 0.887 accuracy is at that ceiling | `data/mhist/annotations.csv` |
| Does weighting SSL vs HP differently change how well cases are ranked? | No: AUC is the same for every weight from 30/70 to 70/30; only the cut-off moves | `docs/results/weight_sweep_learning_curve.json` |
| Does more training data help? | Yes: AUC 0.878 / 0.896 / 0.906 with 25 / 50 / 100% of train | `docs/results/finetune_learning_curve.json` |
| Do brightness/contrast/zoom adjustments at prediction time help? | No: all within +-0.002 AUC of the plain image | `research/image_tta_sweep.py` |
| Does the crypt finder outline crypts correctly? | On expert GlaS outlines (patients never seen in training), benign glands are found at F1 0.91, against 0.36 for the earlier rule-based finder, and a raw tile and a screen snip of it give nearly the same crypts (0.94) | `docs/results/crypt_finder.md` |
| Do the measured crypt labels agree with the pathologists? | Yes, modestly: dilated crypts track the pathologists' SSL votes (Spearman 0.45 on 1,575 held-out training images). Each alone separates SSL from HP weakly, so they explain, they don't decide | `docs/results/gland_check.json` |
| Is the scale bar read correctly? | Generated viewer screens (fonts 11-22 pt, black/white/yellow labels, um and mm, 125-150% display scaling): no wrong scales; QuPath's see-through corner bar is read on real QuPath screens (0.64 and 0.81 um/px), and on a demo sheet at four display sizes within 1%. When a bar can't be read, the app falls back to the tissue square | `docs/results/scale_stress.json` |
| Does the app reproduce the reported test result? | Yes: the 977 test tiles sent through the app's own path as screen snips give balanced accuracy 0.886 (0.892 through a 150% display-scaling round trip) vs 0.888 reported | `docs/results/app_panel_check.json` |
| Is it fast and stable? | The panel's call is on screen in about 2 s; the literature reading follows in about 7-10 s; 25 analyses in a row plus overlapping re-snips complete with flat GPU memory | `docs/results/stability_check.json` |
| Does the evidence map show what drove the call? | Yes, by the deletion test on 120 held-out images: blurring the 15% the map ranks highest moves the score 4.5x more than a random 15% (98% of images) | `docs/results/map_faithfulness.json` |
| Do the map and the crypt measurements agree? | In SSL fields, dilated crypts sit on SSL-pushing evidence about 3x more than round regular crypts (p = 0.009) | `docs/results/evidence_vs_crypts.json` |
| Does stain normalisation help? | No: Reinhard, Macenko, Vahadane and a derived method all score slightly below none | `docs/results/stain_compare.json` |
| Does the unlike-training-data check work? | It flags all 16 non-tissue and wrong-stain test images and about 1% of MHIST fields; on other labs 9-21% of fields | `research/export_panel.py` |
| Is the literature reading well-behaved? | Rule breaks (invented citations, overclaiming, unmeasured evidence, absence claims, papers cited for the field itself) 0.80 per answer across 5 test fields x 3 runs, down from 1.13; about 7 s per field | `docs/results/literature_reading_eval.md` |
| Could a general vision LLM make the call? | No: Gemma-4 31B scores 0.554 balanced accuracy on the MHIST test images, calling 90% of them SSL | `docs/results/vlm_baseline.json` |
| Is MHIST at its nominal scale? | Likely not: ~3.2-3.8 um/px (tiles ~720-860 um), from two independent rulers that agree | `docs/results/scale_audit.md` |
| Would a bigger field help on other labs? | No: 448, 650 and 790 um give the same outside results | `docs/results/field_size_sweep.md` |

Scope: trained on MHIST (one lab). It is strongest on MHIST-like fields and conservative on other labs' slides
(above); slides that look very different, e.g. much higher power or another stain, are flagged as unfamiliar. The
crypt finder can miss crypts cut lengthwise or open to the surface. Future work: predicting the mutation profile
(e.g. BRAF) from the image, which needs SSL slides with genotype labels (none are public yet), larger models, and
shared multi-lab consensus labels.

---

## Reproducing

Use `.venv\Scripts\python.exe` (Python 3.12, CUDA) with `requirements.txt`. The app needs Node.js (`cd app`,
`npm install`). Requires Hugging Face access to CONCH (`MahmoodLab/CONCH`) and MUSK (`xiangjx/musk`), the MHIST
dataset in `data/mhist/`, GlaS for the crypt finder, and for the outside checks and similar cases UniToPatho, SPIDER
and HunCRC under `data/`. MUSK's code is loaded from a clone of github.com/lilab-stanford/MUSK via `MUSK_SRC`. The
literature reading needs a Lumen key (`LUMEN_API_KEY`).

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

# crypt finder (U-Net on GlaS) and its label thresholds from MHIST training crypts
python research/train_gland_net.py
python research/gland_thresholds.py

# similar-cases reference set (other labs' labelled fields at the app's field size)
python research/build_reference_cases.py --field-um 448 --gpu

# demo sheet: six other-lab fields with a 100 um bar
python research/make_outside_demo.py

# checks of the app as built (nothing is chosen from these)
python research/eval_app_panel.py            # test tiles through the app's own path
python research/unitopatho_panel_check.py    # another hospital's slides
python research/scale_stress.py 0            # scale-bar reading on generated viewer screens
python research/map_faithfulness.py          # deletion test for the evidence map
python research/evidence_vs_crypts.py        # evidence map vs crypt measurements
python research/vlm_baseline.py              # a general vision LLM on the MHIST test images (uses Lumen)
python research/scale_audit.py measure       # dataset scale audit (then: python research/scale_audit.py summary)
python research/field_size_panel_sweep.py    # outside results by field size
python research/stability_check.py           # repeated and overlapping analyses against a running backend
```

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free to use, change and share for any non-commercial purpose
(study, teaching, research, hackathon judging and demos); commercial use needs permission.

- Built for a hackathon and for research; not part of a publication.
- A research tool, not a medical device: not for diagnosing or treating patients.
- Third-party models and data (CONCH, MUSK, PLIP, ConvNeXt weights, MHIST, GlaS, UniToPatho, SPIDER, HunCRC,
  OpenEvidence content) are not included and keep their own terms; several forbid commercial use and sharing
  derivatives.
