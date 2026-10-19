# deMHISTifyAI

A desktop research tool for a hard pathology call: **sessile serrated lesion (SSL) vs hyperplastic polyp
(HP)** on H&E. A researcher types their call and certainty, snips a slide region, and sees an AI panel's
call, a heatmap, feature scores, a literature-grounded LLM judge with citations (Lumen), and a final
verdict comparing their call with the evidence.

Goal: match the consensus of the 7 MHIST pathologists as closely as possible. The headline metric is
**balanced accuracy** (SSL and HP count equally), because the thesis is helping pathologists decide,
not winning by leaning toward the more common class.

Research tool only, not for diagnosis.

| Folder | What's in it |
|---|---|
| `app/` | Electron desktop app (launcher, snip overlay, results panel) |
| `backend/` | Local Python service on port 8765: `engine.py` (image models), `llm.py` (Lumen), `server.py` |
| `research/` | Training, evaluation and analysis scripts |
| `docs/results/` | Every result file referenced below |
| `docs/research_notes/` | Planning notes, including `benchmark_job_plan.md` |
| `papers/` | Literature library for the literature judge. `papers/library.json` (OpenEvidence passages) is not in git; rebuild it from your own OpenEvidence answers with `research/build_library.py` |

Data (MHIST) and model weights are not included; each user must obtain access themselves.

---

## 1. Data

**MHIST** (Wei et al. 2021): 3,152 H&E tiles of colorectal polyps, 224 x 224 px, from Dartmouth-Hitchcock.
Each tile has the number of 7 GI pathologists who called it SSL.

| Split | HP | SSL | Total |
|---|---|---|---|
| Train | 1,545 | 630 | 2,175 |
| Test (fixed, official) | 617 | 360 | 977 |

- The test split is never used to make any choice; it is scored once on the final version.
- About 1 in 6 test tiles is a 4-3 split among the pathologists, which caps how accurate any model can be.
- The paper does not say whether tiles from one slide are kept within one split, and no slide IDs are
  published. Everyone uses the same fixed split, so results compare fairly with other MHIST work.
- **Class imbalance**: kept as is (it is realistic: HP is more common). Instead of dropping or repeating
  images, each SSL image counts **2.45x** in training (1,545 / 630), so both classes carry equal total
  weight. A weighting sweep (section 5) showed the weight only moves where the model switches its call;
  the final cut-off is tuned for balanced accuracy anyway.
- **HunCRC** was considered as extra data and rejected: its slide table lists 1 sessile serrated slide out
  of 200 (24 hyperplastic), so it would add almost only HP.

---

## 2. How the app works

| Stage | What happens | Code |
|---|---|---|
| Launcher | Ctrl+Shift+Space opens a bar; the researcher types their call (e.g. "SSL, 70%") | `app/launcher.html`, `app/main.js` |
| Snip | Screen captured with Windows GDI (true colours); researcher drags a box; a vision model proposes the tissue box | `app/capture.js`, `app/overlay.js`, `backend/region.py` |
| Image models | The **whole snip is one field** (scored like one MHIST image; long snips split into squares). The voting judges give P(SSL) | `backend/engine.py` |
| Heatmap | Sliding windows over the snip; AI view is a fixed P(SSL) scale (HP blue, SSL red) | `backend/engine.py`, `app/overlay.js` |
| Feature scores | Original CONCH compares the image with text descriptions of WHO-2019 features; only features with measured reliability are marked usable | `backend/engine.py`, `backend/config/calibration.json` |
| Literature judge | Lumen LLM reasons over the feature scores and a curated passage library; it never sees the image | `backend/llm.py`, `papers/library.json` |
| Verdict | Compares the researcher's call with the image panel and the literature judge | `backend/llm.py` |

### Pipeline problems found and fixed (2026-10-19)

The app was scoring worse than the model did in training. Each cause was measured, then fixed:

| Problem | Effect | Fix |
|---|---|---|
| 4 of the 8 judges were fed colour-normalised (Macenko) images the models never saw in training | Panel caught 54% of SSL vs 79% for the raw-image judges | Only raw-image judges vote; normalised ones shown as a colour check |
| Nucleus-size "auto zoom" mis-sized the analysis windows (read anything up to ~1.5x as 1x, multi-tile snips up to 3x too zoomed) | SSL tiles pushed toward HP | Removed: the whole snip is one field, as a pathologist frames one field |
| Electron's screen capture stretched contrast (dark grey 21 -> 6, highlights 119 -> 153) | Over-exposed look; model saw different-looking images | GDI capture via a warm PowerShell helper (`app/capture.js`), ~140 ms |
| numpy's OpenBLAS ran multi-threaded inside the threaded colour normalisation | Backend crashed (access violation) on ~444 px windows | `OPENBLAS_NUM_THREADS=1` before numpy loads (`backend/server.py`, `engine.py`) |

Result on all 977 test images through the real app pipeline (`docs/results/app_pipeline_eval_v2.json`):

| | Balanced acc. | AUC | ECE | Spearman vs votes | SSL caught | HP caught |
|---|---|---|---|---|---|---|
| Training-time score of the model | 0.819 | 0.908 | 0.034 | 0.774 | | |
| App **before** fixes | 0.742 | 0.893 | 0.038 | 0.756 | 54% | 94% |
| App **after** fixes, image as stored | **0.819** | **0.908** | 0.034 | 0.774 | 79% | 85% |
| App after fixes, 150% PNG screenshot | **0.825** | **0.910** | 0.036 | 0.778 | 81% | 84% |

---

## 3. The current model in the app

4 partly fine-tuned versions of CONCH (seeds 0-3), averaged (`research/finetune_conch.py`):
CONCH blocks 0-9 frozen and shared; blocks 10-11, pooling and projection fine-tuned on MHIST train
(minus a fixed 20% validation slice); a linear classifier on top; temperature-calibrated on validation.
Soft labels (share of SSL votes), flips and rotations, learning rate 1e-5, early stopping on validation AUC.

Test: AUC 0.908, balanced accuracy 0.819, plain accuracy 0.826 (`docs/results/finetune_seeds.json`).

---

## 4. Final architecture (agreed 2026-10-19; training in progress)

### Members

| Member | Type | How it is adapted | Learning rate | Input |
|---|---|---|---|---|
| CONCH | Pathology model | Classifier first (model frozen, 1 epoch), then all layers | 1e-5 | 224 px |
| PLIP | Pathology model | Classifier first, then all layers | 1e-5 | 224 px |
| MUSK | Pathology model (ViT-L) | Classifier first, then last 2 layers (too large to train fully on 8 GB) | 1e-5 | 384 px |
| ConvNeXt-Tiny | ImageNet CNN (not gated) | Classifier first, then all layers | 1e-4 | 224 px |

QuiltNet was dropped (weakest frozen model, overlaps PLIP). Stronger gated models (UNI2, Virchow2,
H-optimus) are outside the hackathon's allowed models and are listed as a future extension.

### Training (same for every member, `research/kfold_cv.py`)

| Choice | Setting | Why |
|---|---|---|
| Target | Share of the 7 pathologists who voted SSL (e.g. 5/7 = 0.71) | Learns the consensus, including how split it was |
| Class weight | SSL x 2.45 (n_HP / n_SSL), fixed per training set | Equal total weight for both classes; matches balanced accuracy. (The old loss re-estimated it per batch of 16: median 1.87, swinging 1.2-3.0) |
| Classifier first, then fine-tune | 1 head-only epoch, then up to 8 fine-tune epochs, best epoch kept | A random classifier at the start sends random corrections back through the model and can damage what it already knows; training it first avoids that |
| Learning rate | 1e-5 for the pathology transformers, 1e-4 for the CNN | 1e-5 is standard for full fine-tuning of this size; the partial fine-tune at 1e-5 was still improving at its last epoch. 5e-6 would need ~2x the epochs. CNNs are usually tuned ~10x higher |
| Image variation (training only) | Flips, 90-degree rotations, mild colour jitter (brightness/contrast/saturation +-10%, hue +-0.02) | Teaches the model to ignore lighting/stain shifts; mild because MHIST is one lab. Prediction uses images as-is, so the same snip always gets the same score |
| Precision | bf16 autocast (weights and optimiser fp32) | Full fine-tuning fits in 8 GB |

Speed on a laptop RTX 5060 (8 GB): CONCH ~34 img/s (7.4 GB peak), PLIP ~92 img/s, ConvNeXt ~94 img/s,
MUSK ~20 img/s (4.4 GB).

### Choosing the panel (training images only, `research/panel.py`)

| Step | What happens |
|---|---|
| 5 folds | Each member is trained 5 times on 4/5 of the 2,175 training images (same folds for all members), so every training image gets a prediction from a model that never saw it |
| Every group | All 15 groups of the 4 members; each gets one learned weight per member from those out-of-fold predictions, scored by AUC on a second split |
| Luck check | Among groups within 0.003 AUC of the best, the one with the fewest members wins |
| Calibration | Platt scaling on out-of-fold scores, so "80%" means about 80% |
| Cut-off | Set to maximise balanced accuracy on the training images |
| Final models | Each member retrained once on all 2,175 training images at its median best epoch; the app runs one copy per chosen member |
| Test | The 977 test images scored once, on the version the app will run |

Why these choices: combining helps only when members are good *and* wrong on different images; with
~1,700 training images, a combiner with few numbers (one weight per member) overfits least. Joining
raw features gives the classifier thousands of numbers to fit; voting throws away confidence.

---

## 5. Experiments and what they showed

| Question | Result | File |
|---|---|---|
| Does weighting SSL vs HP differently in training help? (30/70 ... 70/30) | No. AUC 0.848-0.853 for every weight; balanced acc. 0.777-0.784 once the cut-off is tuned. Weighting only moves the cut-off | `docs/results/weight_sweep_learning_curve.json` |
| Does more MHIST-like data help? (fine-tuned CONCH) | Yes, still climbing: test AUC 0.878 / 0.896 / 0.906 at 25 / 50 / 100% of train | `docs/results/finetune_learning_curve.json` |
| ... frozen CONCH + linear classifier | Plateaus: 0.829 / 0.846 / 0.850 / 0.850 at 25-100% | `docs/results/weight_sweep_learning_curve.json` |
| Do image adjustments at prediction time help? | No. Validation AUC: plain 0.921; 8 orientations averaged 0.923; brightness/contrast/gamma/saturation/zoomed crops 0.920-0.922 | `research/image_tta_sweep.py` (validation pass) |
| Do other models add information? (frozen features, test AUC) | CONCH 0.873, PLIP 0.872, QuiltNet 0.852; CONCH+PLIP joined 0.889; all three 0.886 | ad hoc, frozen embeddings in `data/emb/mhist/` |
| Does combining with the fine-tuned model help? | Fine-tuned CONCH averaged with frozen CONCH+PLIP: AUC 0.912, balanced acc. 0.835 (vs 0.908 / 0.819) | ad hoc |
| Do CONCH's zero-shot feature scores match the visible tissue? | No, mostly not (non-pathologist visual read of 8 tiles) | `docs/results/feature_reality_check.md` |
| Does the model work on images unlike MHIST? | No: Wikipedia's SSL image (higher power, much pinker stain) scores P(SSL) 0.10-0.23 at every window size. Fix would be more varied datasets | ad hoc |
| Reference point | The MHIST paper's fully trained ImageNet ResNet-18 reached AUC 0.927 (their setup), above our partly fine-tuned CONCH; motivated full fine-tuning and adding a CNN | Wei et al. 2021 |

---

## 6. Reproducing

Use `.venv\Scripts\python.exe` (Python 3.12, CUDA). MUSK's code is loaded from a clone of
github.com/lilab-stanford/MUSK via `MUSK_SRC` (not pip-installed; it pins old torch/timm); `einops` and
`fairscale` were added with `--no-deps`. MUSK and CONCH need Hugging Face access.

```bash
# app pipeline evaluation (real backend code on rendered snips)
python research/eval_app_pipeline.py 977 native,screenshot

# final architecture: 5-fold CV per member, then final retrain at the median best epoch
python research/kfold_cv.py --backbone conch    --blocks -1 --lr 1e-5 --tag conch_full
python research/kfold_cv.py --backbone plip     --blocks -1 --lr 1e-5 --tag plip_full
python research/kfold_cv.py --backbone convnext --blocks -1 --lr 1e-4 --tag convnext_full
MUSK_SRC=<MUSK clone> python research/kfold_cv.py --backbone musk --blocks 2 --lr 1e-5 --batch 8 --tag musk_b2
python research/kfold_cv.py --backbone conch --blocks -1 --lr 1e-5 --tag conch_full --final-epochs <median best epoch>
# (same --final-epochs step for each member)

# choose the panel on training data, then score the test split once
python research/panel.py --members conch_full plip_full convnext_full musk_b2
```

Cross-validation outputs go to `data/models/cv/` (never `data/models/`, which the app loads as judges).

## 7. Caveats

- During development some test-set numbers were looked at (e.g. the CONCH+PLIP combination); every
  remaining choice is made on training data only, and the final version is scored once.
- If tiles from one slide sit in different folds, fold scores are slightly optimistic; this affects all
  members alike, so choices stay fair, but fold scores are never reported as results.
- The literature judge's claims about features can overstate the (weak) feature scores; a guard rail is
  planned but untested.

## License

Built for a hackathon and for research; not part of a publication. Free for non-commercial study,
teaching, research and hackathon use with attribution; not for clinical use. Third-party models and data
(CONCH, MUSK, PLIP, ConvNeXt weights, MHIST, OpenEvidence content) keep their own terms and are not
included. See [LICENSE](LICENSE).
