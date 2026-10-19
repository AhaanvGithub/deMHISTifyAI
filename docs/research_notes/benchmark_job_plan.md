# Benchmark push: job list and findings so far

Goal: highest accuracy on the MHIST test split (977 images, majority vote of 7 pathologists),
with every choice made on training data and the test split scored once at the end.
Status 2026-10-19 ~15:00: queue stopped by request, to replan from the ground up. Nothing is running.

## Baseline to beat (the app today)

4 fine-tuned CONCH versions (blocks 0-9 frozen, last 2 blocks + pooling + linear head trained), averaged.
Test: AUC 0.908, balanced accuracy 0.819, plain accuracy 0.826, SSL caught 79%, HP caught 85%,
ECE 0.034, Spearman vs SSA votes 0.774. The in-app pipeline now reproduces this exactly
(`docs/results/app_pipeline_eval_v2.json`).

## What we already know

| Finding | Evidence | File |
|---|---|---|
| Class weighting does not improve the model | AUC 0.848-0.853 for every SSL:HP weight 30/70..70/30; balanced acc 0.777-0.784 once the cut-off is tuned on validation | `docs/results/weight_sweep_learning_curve.json` |
| More MHIST-like data still helps the fine-tuned model | test AUC 0.878 / 0.896 / 0.906 at 25 / 50 / 100% of train | `docs/results/finetune_learning_curve.json` |
| Frozen-feature classifier plateaus | AUC 0.829 / 0.846 / 0.850 / 0.850 at 25-100% | same as row 1 |
| Image adjustments at test time do not help | validation AUC: original 0.9209, 8-orientation average 0.9231, brightness/contrast/gamma/saturation/multi-scale 0.920-0.922 | queue log (test pass not run) |
| A second model adds signal | frozen CONCH 0.873, PLIP 0.872, QuiltNet 0.852, CONCH+PLIP joined 0.889, all three 0.886 (test AUC) | ad hoc |
| Combining with the app model helps a little | fine-tuned CONCH averaged 50/50 with frozen CONCH+PLIP: AUC 0.912, balanced acc 0.835, plain acc 0.826 | ad hoc |
| Fully trained small CNN beats partial CONCH fine-tune | original MHIST paper: ImageNet ResNet-18 AUC 0.927 (their setup) vs our 0.908 | Wei et al. 2021 |
| HunCRC is not useful | 1 sessile serrated slide out of 200 (24 hyperplastic) | `slide_level_annotations.csv` |

## Panel members (decided 2026-10-19)

MUSK, CONCH, PLIP, and an ImageNet CNN (ConvNeXt-Tiny) as the 4th member (different architecture,
so less correlated mistakes; the MHIST paper's fully trained ResNet-18 reached AUC 0.927).
QuiltNet dropped (weakest, overlaps PLIP).

Design space per member: frozen + linear/MLP head | partial fine-tune | full fine-tune | LoRA |
linear-probe-then-fine-tune. Combination: feature join | weighted average | stacking | greedy selection.
Training: soft labels, seeds, CV ensemble vs retrain on 100%. Decision: calibration + cut-off on OOF.

## Jobs (not yet run unless marked)

| # | Job | Command | Est. time | Why |
|---|---|---|---|---|
| 1 | Image adjustment test, test pass | `python research/image_tta_sweep.py` | 15 min | validation pass done (above); only worth finishing to confirm |
| 2 | MUSK frozen features | `MUSK_SRC=<MUSK clone> python research/musk_embed.py` | 15 min | needed for any MUSK member |
| 3 | Frozen-feature CV heads: CONCH, PLIP, QuiltNet, MUSK + joins CONCH+PLIP, CONCH+MUSK, PLIP+MUSK, CONCH+PLIP+MUSK | `python research/frozen_cv_heads.py` | 20 min (CPU) | cheap panel members |
| 4 | CV fine-tune CONCH, 2 blocks, lr 1e-5 (= today's recipe, on 5 folds) | `python research/kfold_cv.py --backbone conch --blocks 2 --lr 1e-5 --tag conch_b2_lr1e-5` | ~2.5 h | baseline member trained on all of train |
| 5 | CV fine-tune PLIP, 2 blocks | `... --backbone plip --blocks 2 --lr 1e-5 --tag plip_b2_lr1e-5` | ~1.5-2 h | different model, cheap |
| 6 | CV fine-tune CONCH, all 12 blocks, lr 5e-6 (proposed) | `... --backbone conch --blocks 12 --lr 5e-6 --tag conch_b12_lr5e-6` | ~3 h | full fine-tuning: the ResNet-18 hint |
| 7 | CV fine-tune ConvNeXt-Tiny (ImageNet), full (proposed; needs a `timm` backbone in kfold_cv.py) | - | ~1.5 h | a CNN: different mistakes from the transformer models |
| 8 | CV fine-tune MUSK, 2 blocks, bf16, batch 8 | `MUSK_SRC=... python research/kfold_cv.py --backbone musk --blocks 2 --lr 1e-5 --batch 8 --tag musk_b2_lr1e-5` | ~6.5 h | strong, different backbone; slow |
| 9 | CV fine-tune CONCH, 4 blocks / lr 3e-5 | `... --blocks 4 ...` / `... --lr 3e-5 ...` | ~2.5 h each | small tweaks, lowest priority |
| 10 | Panel: choose combination on out-of-fold predictions only, calibrate, score test once | `python research/panel.py` | 10 min | final number |
| 11 | Optional: stronger gated models (UNI2, Virchow2, H-optimus) | needs Hugging Face access | - | possibly a large lever |

All CV jobs share the same 5 folds (StratifiedKFold, seed 0, over the 2,175 training images), save to
`data/models/cv/<tag>/` (never `data/models/`, which the app loads), and resume from the last finished
fold. The overnight runner (scratchpad `queue_runner.py`) handled the allowed hours
(19 Oct until 20:00, then 22:00 to 09:00) and pausing/resuming.

## Notes

- MUSK code is loaded from a clone of github.com/lilab-stanford/MUSK (not pip-installed; it pins old
  torch/timm). `einops` and `fairscale` were added to `.venv` with `--no-deps`. MUSK peaks at 5.2 GB
  GPU in fine-tuning at batch 8; CONCH 3.5 GB, PLIP 0.7 GB.
- `research/finetune_conch.run()` saves to `data/models/conch_ft_seed{N}.pt`, which the app loads as
  judges; scripts that reuse it must block that save (see `research/finetune_learning_curve.py`).
- Report AUC, plain accuracy and balanced accuracy; headline metric still to be decided.

## To do

- [ ] **MUSK + LoRA** (Tuesday, only if MUSK earns a place in the panel): MUSK is too large to fine-tune
      fully on 8 GB, so tonight it trains only its last 2 of 24 layers. LoRA adds small trainable pieces to
      every layer within the same memory; THUNDER reports +2 to +8 F1 points from LoRA on MHIST over frozen
      features. Needs the `peft` library; ~2 h for 5 folds + final retrain. Keep it only if its out-of-fold
      AUC beats the 2-layer version.

