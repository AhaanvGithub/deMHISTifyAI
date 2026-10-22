# deMHISTifyAI: handoff (written 2026-10-19 ~20:30 CDT)

Everything needed to continue without the original conversation: who decides what, the principles, every
decision with its reason, every result, what is running, and what is next.

---

## 1. Project and people

- **Ashby Hackathon 2026 submission** by Aiden Wang, Alice Yu and Ahaan Verma (the user). Public repo:
  github.com/AhaanvGithub/deMHISTifyAI (license: PolyForm Noncommercial 1.0.0; GitHub description starts
  "Ashby Hackathon 2026 Submission - Aiden Wang, Alice Yu, Ahaan Verma").
- **Product**: Electron desktop app (`app/`) + Python backend on port 8765 (`backend/`). A researcher types
  their call, snips a slide region, and gets an image-model panel call, heatmap, feature scores, a literature
  judge (Lumen LLM over an OpenEvidence passage library) and a verdict. Task: sessile serrated lesion (SSL)
  vs hyperplastic polyp (HP) on H&E, trained on MHIST.
- **Deadlines**: fine-tuning done by **Wed 2026-10-21 00:00**; UI polish Tue night; final deadline
  **Thu 2026-10-22 07:00**; live presentations.
- **Hackathon rules** (from the user + `docs/reports/`): use some of the **4 provided models** (CONCH, PLIP,
  QuiltNet, MUSK) - strictly enforced; an ImageNet CNN is allowed (not one of the "models", not gated).
  Rubric: innovation, progress, clinical impact, communication. The hackathon slides recommended frozen
  weights (we fine-tune; be ready to justify with numbers). Stronger gated models (UNI2, Virchow2,
  H-optimus) are NOT allowed - mention only as a future extension.

## 2. The user's principles and preferences (follow these)

- **#1 goal: replicate the 7-pathologist MHIST consensus as closely as possible**; optimise for the benchmark.
- **Headline metric: balanced accuracy** (thesis: help pathologists decide; must not win by leaning toward HP).
  Also report AUC, accuracy, F1, ECE, Spearman vs vote count.
- **First principles**: derive designs from how the tool is actually used and what the model was trained on;
  prefer the direct rule over clever heuristics (e.g. "a pathologist snips one field -> whole snip = one
  field", replacing a nucleus-size zoom estimator).
- **Consult the user before decisions**; give a recommendation, pros/cons, plain language.
- **Talk less, less jargon**; tables are welcome. Be honest about uncertainty and mistakes.
- **Test set is scored once** on the final version; every choice made on training data. A preliminary test
  peek happened (section 5) and is recorded; it must not influence any choice.
- **GPU/battery**: no GPU jobs on battery unless allowed; laptop sleeps after 3 min on battery (never when
  plugged in). Lock, don't close the lid. The user gives explicit time windows for compute.
- Likes live progress in a terminal (`Get-Content "$env:TEMP\gutfeeling\queue.log" -Wait -Tail 40`).
- README must read like a **product** (final choices, no "fixes/changes" history).
- Commit messages: plain, no AI attribution lines (has been the practice in this repo). Push when asked
  (has been pushing to main routinely after README/plan edits).

## 3. Data facts

- MHIST: 3,152 tiles, 224 x 224 px, Dartmouth-Hitchcock, Aperio AT2 at 40x downsampled 8x to **5x,
  ~2.0 um/px -> one tile = 448 x 448 um (~0.45 mm)** (MHIST paper, Virchow paper, kaiko eva docs).
- Split: train 2,175 (1,545 HP / 630 SSL), test 977 (617 HP / 360 SSL). Slide IDs unknown.
- ~1 in 6 test tiles is a 4-3 split. **Ceiling**: a perfect model agrees with a fresh 7-pathologist majority
  only ~89-92% (computed from vote counts).
- HunCRC rejected (1 SSL slide of 200). Real HP/SSL slides for a demo: HunCRC has 24 HP + 1 SSL (TCIA,
  1-2 GB each; test first - Hungarian staining).

## 4. App pipeline (current state, all committed)

- Panel = whole snip scored as one field (long snips split into squares); only raw-image judges vote;
  true-colour GDI screen capture (`app/capture.js`); OpenBLAS single-threaded (Macenko thread crash fix).
- UI: sunset + light liquid-glass theme, launcher pill glow, AI-first diverging P(SSL) heatmap (opacity 45%),
  panel grows as results arrive, word-by-word LLM text, Esc/Alt+Tab window fixes, citation rendering.
- **The app still runs the OLD model** (4 partly fine-tuned CONCH, `data/models/conch_ft_seed*.pt`):
  test balanced acc 0.819, AUC 0.908, acc 0.826, ECE 3.4%. Deploying the new panel is the top task.

## 5. The new image panel (final architecture, agreed)

**Members**: CONCH, PLIP, MUSK (pathology models) + ConvNeXt-Tiny (ImageNet CNN). QuiltNet dropped.

**Training recipe** (`research/kfold_cv.py`, all members):
- Soft labels = SSL votes / 7. **Fixed class weight: SSL x 2.45** (= 1,545/630), computed per training set
  (the old loss re-estimated it per batch of 16: median 1.87, range 1.2-3.0).
- **Classifier first** (1 head-only epoch, model frozen) **then fine-tune** up to 8 epochs, keep best epoch.
- CONCH/PLIP/ConvNeXt: **all layers**; MUSK: last 2 layers (too big for 8 GB). LR 1e-5 (pathology models),
  1e-4 (CNN). Reasoning for 1e-5 over 5e-6: standard for full FT, the 2-block FT at 1e-5 was still improving,
  classifier-first removes the main risk of the higher rate, 5e-6 needs ~2x epochs.
- Training-only image variation: flips, 90-degree rotations, mild colour jitter (+-10% brightness/contrast/
  saturation, hue +-0.02). Prediction uses images as-is (deterministic).
- bf16 autocast; CONCH uses batch 8 x grad accumulation 2 (OOM at 16). MUSK input 384 x 384 (bicubic
  upscale + centre crop, Inception mean/std, as in MUSK README).
- 5-fold StratifiedKFold(seed 0) over the 2,175 train images, same folds for all members. Final model per
  member retrained on all 2,175 at the median best epoch (`--final-epochs`).

**Panel** (`research/panel.py`): all 15 groups of members, one learned weight per member (balanced logistic
regression on standardised OOF logits), scored by AUC on a second 5-fold split; **smallest group within
0.003 AUC of the best wins**; call = balanced calibration + cut-off maximising balanced accuracy on train;
**displayed percentage = Platt calibration to the majority call at real frequency** (ECE 6.8% -> 1.3% on
train, cross-fitted); records `shown_cut_off` so the app can say "calls SSL above X%". Panel refuses to run
while `data/models/cv/HOLD_PANEL` exists (test scored once).

**Cross-validated results (OOF AUC on train)**: CONCH 0.949, ConvNeXt-Tiny **0.957**, PLIP 0.951,
MUSK(2 layers) 0.941. Best groups (train only): all four 0.964; ConvNeXt+PLIP+MUSK 0.963 (bal acc 0.900*);
ConvNeXt+PLIP / ConvNeXt+MUSK 0.962 (*optimistic: cut-off on same data).

**Preliminary test check** (`research/preliminary_test_check.py`, `docs/results/preliminary_test_check.json`;
fold-model averages, recorded, not used for choices): every new group 0.862-0.882 balanced acc,
0.938-0.961 AUC vs app 0.819 / 0.908. **"CLIK" = ConvNeXt + PLIP + MUSK**: bal acc 0.877, AUC 0.959,
acc 0.863, weighted F1 0.865, ECE 4.7% (before the display calibration fix). With a cut-off tuned for plain
accuracy on train: acc 0.892, F1 0.892, bal acc 0.890 (observed after the rule was fixed -> app keeps the
balanced-accuracy rule). Groups lean SSL on test (SSL caught 93-96%, HP 78-83%).

## 6. Literature comparison (README has the full tables, sorted by balanced accuracy, split into "never used
the test split to choose the model" vs "used it or did not say")

- Fair (THUNDER, frozen + linear, NeurIPS 2025): Virchow2 0.848, H-optimus-0 0.837, UNI 0.825, CONCH 0.800,
  MUSK 0.792, PLIP 0.790, QuiltNet 0.735. LoRA (derived F1): Virchow2 ~0.875, CONCH ~0.851, UNI ~0.847.
- Used test / not stated: EVA (test used for checkpoint selection) Virchow2 0.861, CONCH 0.832; Virchow
  paper; MHIST paper ResNet-18 AUC 0.927 (best test epochs kept); PolarHE acc 0.891 / F1 0.890 / AUC 0.941.
- Not comparable: CaiT preprint 90.2% acc (5-fold CV over all 3,152).
- **Conclusion so far**: no comparable published result clearly beats CLIK; its AUC 0.959 is the highest
  comparable one found; accuracy/F1 level with PolarHE depending on cut-off rule.

## 7. Experiments and findings (all in README "Supporting analyses" or docs/results)

- Class weighting 30/70..70/30 doesn't change AUC (only moves the cut-off). Keep 50/50-equivalent (2.45x).
- Learning curve (fine-tuned): AUC 0.878/0.896/0.906 at 25/50/100% of train -> more data helps.
- Test-time image adjustments (brightness/contrast/gamma/zoom crops/orientations): within +-0.002 AUC -> no.
- CONCH zero-shot feature scores don't match visible tissue (feature reality check) -> shown as weak hints.
- Out-of-distribution: Wikipedia SSL image (high power, pink stain) scores HP at every scale.

## 8. Running now (overnight queue)

- Allowed hours file: `%TEMP%\gutfeeling\windows.json` = until 2026-10-20 09:00 (runners re-read it each check).
- Main runner (`research/overnight/queue_runner.py`, terminal tab "app accuracy 977"): all CV done; final
  retrains in progress (CONCH and ConvNeXt done by 20:30, PLIP running, MUSK next).
- Follow-up (`research/overnight/queue2.py`, background): after the main queue -> ConvNeXt-Base 5-fold
  (replaces Tiny if OOF AUC higher) -> MUSK + LoRA (rank 8 on image-expert q/k/v/out of all 24 layers,
  lr 1e-4, batch 4 x accum 4, retry batch 2 x accum 8 if OOM; replaces 2-layer MUSK if OOF AUC higher) ->
  delete HOLD_PANEL -> `research/panel.py` + **the single test evaluation** -> `docs/results/panel.json`.
- Log: `%TEMP%\gutfeeling\queue.log`. Restart safely: kill python processes matching queue2/queue_runner/kfold_cv,
  then rerun both runners (they skip "DONE" jobs; CV resumes from the last finished fold).
- Note: `queue_runner.py` sleeps in 60-s naps (a single long sleep stalled when the laptop slept).
- **Progress 21:20**: all final retrains done. **ConvNeXt-Base won** (OOF AUC 0.962 vs Tiny 0.957; final model
  trained) -> panel uses `convnext_base`. MUSK-LoRA ran out of memory at batch 4 and is running at batch 2 x
  accum 8 (fold 1 epoch 1 val AUC 0.928; ~28 min/fold -> done ~00:30), then the panel + single test run.

## 9. Stain normalisation (decided: none)

- **Result (per-image source, MHIST averaged target; `docs/results/stain_compare.json`, `unitopatho_check.json`)**:
  on held-out training images under 5 simulated labs, no normalisation was best almost everywhere (AUC 0.960-0.962
  whatever the colours); every method slightly hurt clean images (derived least, -0.003 bal acc; Vahadane worst,
  -0.026). UniToPatho (545 real outside HP patches): 100% called HP under every method (mean P(SSL) ~0.20), but it
  has no SSL, so it cannot separate "recognises HP" from "defaults to HP on outside images". Weights verified
  unchanged by SHA-256. **App uses no normalisation**; `backend/stain.py` stays as an option.
- Earlier notes:

- User's idea: shift a new slide's colours toward MHIST's, measured per slide (whole screen), possibly
  regionally. Implemented as Macenko dye separation + **MHIST's own reference** (`backend/stain.py`,
  `backend/config/stain_mhist.json` from 2,175 train images, `research/stain_reference.py`).
- Self-check: MHIST tiles change 7.6/255 per-tile, **3.9/255 with one estimate per screen** -> per-slide.
- Wikipedia SSL image re-coloured looks MHIST-like; ConvNeXt (5 fold models) P(SSL) 0.21 -> 0.26 (20x) and
  0.22 -> 0.29 (40x): moves toward SSL but still HP (key crypt bases mostly at the image edge).
- Not "better than Macenko" by itself: plan to compare **none vs Reinhard averaging vs Macenko-MHIST** on
  held-out fold models over (a) MHIST validation images and (b) artificially re-stained versions; adopt the
  winner only if no harm on (a) and recovery on (b). Needs GPU (after the queue). No test split involved.

## 10. To do (order)

1. Check overnight results (`docs/results/panel.json`, queue.log): Tiny vs Base, MUSK vs LoRA, chosen group,
   official test numbers. Update README performance + comparison rows (replace "preliminary").
2. **Deploy the chosen panel into the app** (`backend/engine.py` loaders for the members' `final.pt` in
   `data/models/cv/<tag>/`, standardisation + weights + both calibrations + cut-off from panel.json; judges
   list = each member; show "calls SSL above X%"); re-run `research/eval_app_pipeline.py` (in-app numbers).
3. Stain normalisation comparison (section 9); deploy if it wins.
4. **Scale-aware snip square**: read the viewer's scale bar automatically (local vision model for the label +
   bar detection), size a draggable **448 um** square, prompt "zoom out" if it does not fit or "zoom in" if
   under 224 screen px, manual fallback (drag along the scale bar + type its label). Demo in QuPath.
5. "Unlike training data" warning (out-of-distribution flag).
5b. **"Don't miss SSL" screening mode** (toggle): a second cut-off chosen on training OOF predictions to catch ~95% of
   SSL, with its HP false-alarm rate shown. Evidence (ConvNeXt-Base OOF, train): raw 0.5 cut-off misses 17% of SSL
   vs 6% of HP (errors lean HP); the best-balanced-accuracy cut-off (0.41) gives 90% / 89% and bal acc 0.896 (vs
   0.884), and the panel already uses that rule; a 95%-SSL cut-off (~0.30) costs ~3 points of balanced accuracy
   (HP caught 78%), so it is an opt-in mode, not the default.
6. Literature judge guard rail (stop "strongly supported by features" / claims about non-measurable features;
   needs Lumen to test - ask the user).
7. Optional: second seed for the best member; disagreement-aware UI (flag likely 4-3 cases).
8. Demo: QuPath with a real slide if the panel handles it (test HunCRC SSL+HP offline first), else an MHIST
   mosaic opened at 2.0 um/px (known answers). Check slide licences before showing publicly.

## 11. Key files

| Path | What |
|---|---|
| `research/kfold_cv.py` | CV + final retrain for all members (conch, plip, musk, musk_lora, convnext, convnext_base) |
| `research/panel.py` | group search, calibrations, cut-off, single test evaluation |
| `research/preliminary_test_check.py` | recorded test peek |
| `research/overnight/` | queue runners (copies) + OOF preview |
| `data/models/cv/<tag>/` | fold weights, OOF/test logits, `final.pt` (not in git) |
| `docs/results/cv/<tag>.json` | per-member CV summaries |
| `backend/stain.py` | stain normalisation |
| `docs/research_notes/benchmark_job_plan.md` | job list, findings, to-do |
| `README.md` | product description, performance, literature tables |
| MUSK code | clone of github.com/lilab-stanford/MUSK at `%TEMP%\musk_src\MUSK` (`MUSK_SRC`); einops + fairscale installed `--no-deps` |

## 12. Addendum (2026-10-19 evening): scale check and outside datasets

### MHIST field size re-checked: ~2.0 um/px (~448 um per tile) holds

- The MHIST paper's wording is ambiguous: "scanned by an Aperio AT2 scanner at 40x resolution; to increase the
  field of view, we compress the slides with 8x magnification." Read as "shrunk **by** 8x" -> ~2.0 um/px,
  448 um per tile; read as "shrunk **to** 8x" -> ~1.25 um/px, 280 um per tile. HistoKT (arXiv 2201.11246,
  Table 1) lists MHIST at 1.25 um.
- Measured on 12 training tiles by eye (20-px grid; no test images):

| Structure | Real size | Expected at 2.0 um/px | Expected at 1.25 um/px | Seen |
|---|---|---|---|---|
| Lymphocyte nucleus | ~7 um | ~3.5 px | ~5.6 px | 1-3 px dark dots |
| Colon crypt, cross-section | ~50-60 um | ~25-30 px | ~40-48 px | mostly ~22-30 px |

- **Conclusion**: 1.25 um/px does not fit; ~2.0 um/px does. Precise to roughly 1.8-2.2 um/px (crypt size
  varies), which is enough for the 448 um snip square. An automatic nucleus-size measurement was unreliable at
  this resolution (nuclei are a few pixels) and was not used. Definitive answer, if ever needed: ask the MHIST
  authors (Jerry Wei / Saeed Hassanpour, Dartmouth).

### Outside datasets (none used for training)

| Dataset | What it is | HP / SSL labels? | Use for us |
|---|---|---|---|
| **ADPv2** (Hosseini et al., arXiv 2507.05656, Zenodo) | 20,004 patches from 461 slides of **healthy** colon (Kingston General Hospital, St. Michael's, Sunnybrook, TCGA); 32 tissue types, several per patch; 544 x 544 um patches at 0.2-0.5 um/px; KGH slides stained HPS, the rest H&E | No | Don't train on it. Possible extra stain test or "unlike training data" demo (shrink to ~2.0 um/px). Its paper found a model trained on healthy tissue was clearly less confident on SSL+HP regions than normal tissue (weaker for adenomas) - citable outside support that serrated polyps look distinct. Licence not stated. |
| **UniToPatho** (Univ. of Turin, arXiv 2101.09991, IEEE DataPort, Zenodo) | 9,536 H&E patches from 292 slides; Hamamatsu NanoZoomer S210 at 20x (~0.44 um/px); classes: normal, HP, tubular / tubulo-villous adenoma (low / high grade) | HP only (~545 regions), **no SSL** | Outside test of HP calls from another hospital, scanner and stain (shrink to ~2.0 um/px; the app should call HP); good real-stain case for the stain normalisation comparison (section 9). Adenoma patches can demo the out-of-distribution warning. Check licence before showing publicly. |

- "pKGH" (from another agent's roadmap) is most likely ADPv2's Kingston General Hospital source - HPS stain,
  healthy tissue; not usable for training.
- The DHMC colorectal polyp slide archive was **rejected for training**: same hospital as MHIST, possibly the
  same patients, and MHIST publishes no slide IDs, so overlap with the test split cannot be ruled out. Also no
  time before the Wed 00:00 training deadline. Good "future work" line: more institutions and stains, checked
  for patient overlap.

No pathologist study is planned (decided by the user, 2026-10-19).
