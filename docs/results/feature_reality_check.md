# Feature reality check: CONCH feature scores vs what is visible

**This is a non-pathologist visual read.** It was made by eye from 224 px MHIST tiles (viewed upscaled x3), with the label already known. Treat it as a sanity check, not ground truth.

Tiles: 8 held-out MHIST test images. 4 are clear SSA (7/7 pathologists voted SSA) and 4 are clear HP (0/7 voted SSA).
Scores: `Engine.analyze()` on the native image (from `docs/results/app_pipeline_eval.json`, `per_image.native`).
Each score is 0..1, where 0.5 is the MHIST-train average. *usable* is the flag in `backend/config/calibration.json`.

Visible-feature key: **Y** = clearly seen, **~** = possible or partial, **N** = not seen, **?** = can't judge (crypt base not in the field, or crypts cut in cross-section).

| Tile (votes) | P(SSA) panel | Serration to base (usable) | Serration upper only (usable) | Narrow straight bases (usable) | Dilated bases | Boot-shaped | Horizontal growth |
|---|---|---|---|---|---|---|---|
| agb SSA 7/7 | 0.82 | ? / **0.75** | N / 0.71 | N / 0.76 | ~ (mucin-filled dilated crypts, lower right) / 0.74 | N / 0.68 | ? (crypts cut lengthwise) / 0.82 |
| agd SSA 7/7 | 0.61 | Y (serrated lumens deep into crypts) / 0.64 | N / 0.49 | N / 0.37 | Y (rounded, dilated bases) / **0.45** | ~ (bottom-left crypt) / 0.60 | ~ / 0.25 |
| ais SSA 7/7 | 0.44 | ~ (serrated, dilated; base not in field) / 0.77 | N / 0.34 | ? / 0.38 | Y (dilated mucinous crypts) / 0.56 | ? / 0.60 | ? / 0.33 |
| ajn SSA 7/7 | 0.76 | ~ / 0.74 | N / 0.61 | N / 0.68 | Y (irregular dilated crypt with mucin) / 0.81 | N / 0.33 | ~ (crypt lying along the muscularis) / 0.75 |
| acb HP 0/7 | 0.08 | N / **0.62** | ~ (mild, surface only) / 0.57 | Y (narrow round bases) / 0.71 | N / 0.66 | N / 0.57 | N / 0.57 |
| acw HP 0/7 | 0.06 | ? (only the upper crypt is in the field) / 0.58 | Y / 0.55 | ? / 0.52 | ? / 0.67 | ? / 0.48 | ? / 0.47 |
| aeb HP 0/7 | 0.01 | ? (cross-sections, regular) / **0.73** | ? / 0.51 | ~ (uniform small crypts) / 0.44 | N / 0.46 | N / 0.64 | N / **0.69** |
| aee HP 0/7 | 0.12 | ~ (stellate lumens, mid-crypt plane) / 0.49 | ? / 0.51 | ? / 0.60 | N / 0.49 | N / **0.79** | N / 0.32 |

Each feature cell reads "visible / CONCH score".

## What this shows

- **The image judges are right on all 8 tiles** at the 0.5 cut, except ais (0.44, a surface fragment with no crypt bases in view). The diagnosis does not depend on the feature scores.
- **Serration to base** (usable) is somewhat higher on SSA (mean 0.72) than on HP (0.61). However, it gives 0.73 to aeb, an HP tile cut in cross-section where no crypt base is visible. That is a direction signal, not a reliable read of the feature.
- **Serration upper only** and **narrow crypt base** (both usable) do not separate these tiles. The highest values for both are on SSA tile agb. Their flags rest on a train AUC of only 0.42–0.44 (i.e. 0.56–0.58 when flipped), so this is consistent with that weak signal.
- **The unusable features don't match the tissue:** the dilated-base score misses agd's obvious dilated bases (0.45), boot-shaped scores 0.79 on an HP tile with no boot shapes, and horizontal growth scores 0.69 on an HP cross-section. This supports keeping them flagged unusable.
- Bottom line: at this field of view (224 px MHIST tiles, often without the crypt base), CONCH's zero-shot feature scores do not reliably mark the WHO-2019 features that are visibly present. The app should keep presenting them as weak hints, with the check-by-eye list doing the real work.
