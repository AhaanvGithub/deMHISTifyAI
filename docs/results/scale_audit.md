# Scale audit: is it scale, or procedure?

`research/scale_audit.py` → `scale_audit.json`. 3,975 fields from four datasets: MHIST (training images only), UniToPatho, SPIDER and HunCRC, with HP and SSL measured separately. Every field goes through the same pipeline: a 448 µm square under the dataset's assumed µm/px is resized to 224 px, then crypts are found by the app's U-Net crypt finder and nuclei by haematoxylin blobs. All of it ran on the CPU.

## Verdict

**MHIST is not at 2.0 µm/px. Within HP, its differences are mostly one scale factor.** Every measure that can detect scale gives a ratio between 1.7 and 2.2 against UniToPatho, the only dataset whose µm/px is known. This holds for both crypt geometry (width, length, area, spacing, density) and lymphocyte size. Those two rulers are independent of each other, and they agree. The pooled factor is **×1.92 (95% CI 1.79–2.08)**, which puts MHIST at **~3.8 µm/px (field ~860 µm)** if UniToPatho is the ruler.

The ruler itself is uncertain by about 20%. HunCRC differs from UniToPatho by a clean ×1.19 on every measure, including nuclei, with zero heterogeneity (I² = 0). So either HunCRC's stated 0.484 µm/px or UniToPatho's crops are off by that much. Against HunCRC, MHIST comes out at **~1.6× (≈3.2 µm/px, field ~720 µm)**.

**Best honest range: MHIST ≈ 3.2–3.8 µm/px, so a tile covers ~720–860 µm, not 448 µm.**

Scale is not the whole story:
- MHIST's epithelial walls are relatively thin: ×1.70, against ~1.9 for the other measures.
- **SPIDER** shows procedural differences that no scale factor can fix:
  - Its crypts cover more of the tissue (scale-free crypt share ×1.18).
  - Its walls are thicker (0.80) and its lumens narrower (1.70).
  - **Its SSL fields measure the same as its HP fields on every crypt measure** (SSL/HP ratios 0.99–1.02). In MHIST and HunCRC, SSL halves the crypt density (0.48 and 0.49). This fits SPIDER's SSL labels being region-level rather than lesion-level, as concluded on 2026-10-20 when SPIDER training was dropped.

## How to read the ratios

**k** is the factor that a dataset's assumed µm/px must be multiplied by to match UniToPatho HP. k > 1 means its images cover more tissue than assumed.

**Control:** the measures don't read scale perfectly. To test them, UniToPatho HP fields were deliberately drawn c times too small (c = 0.8, 1.3, 1.6, 1.75). On those fields:
- Crypt measures read about 90% of the true error (sensitivity b ≈ 0.9). The crypt finder is close to a true ruler.
- Lymphocytes read about 57% (b = 0.57), because at 2 µm/px a nucleus is only 2–3 px wide.
- Goblet vacuoles (b = 0.02) and epithelial nuclei (b = 0.43) cannot see scale at all, so they are left out.

The calibrated k inverts each measure's own control curve. This is why the earlier nucleus-only estimate (×1.4) undershot: the raw nucleus reading is compressed toward 1.

| Measure (control sensitivity b) | MHIST HP | MHIST SSL* | SPIDER HP | HunCRC HP | HunCRC SSL* | UniToPatho NORM |
|---|---|---|---|---|---|---|
| Crypt width, short axis (0.90) | **1.96** [1.83–2.09] | 1.86 | 1.01 [0.94–1.10] | 1.18 [1.05–1.30] | 1.39 | 1.20 |
| Crypt long axis (0.91) | **1.76** [1.61–1.95] | 1.35 | 0.94 | 1.17 | 1.03 | 1.21 |
| Crypt area (0.91) | **1.84** [1.71–2.00] | 1.60 | 0.97 | 1.16 | 1.20 | 1.21 |
| Crypt interior width (0.92) | **2.24** [1.90–2.72] | 2.56 | 1.70 | 1.45 | 1.77 | 2.59 |
| Epithelial wall thickness (0.92) | **1.70** [1.58–1.92] | 1.10 | 0.80 | 1.09 | 1.04 | 0.92 |
| Nearest-neighbour crypt spacing (0.81) | **1.90** [1.77–2.08] | 1.49 | 0.99 | 1.18 | 1.21 | 1.10 |
| Crypts per mm² (0.67) | **1.95** [1.64–2.30] | 1.14 | 0.87 | 1.17 | 0.69 | 0.94 |
| Lymphocyte-like nucleus (0.57) | **2.10** [2.01–2.22] | **2.14** | 1.02† | 1.17 [1.09–1.25] | 1.15 | 1.08 |
| Stromal nucleus (0.49, not pooled) | 1.98 | 2.03 | 0.99† | 1.18 | 1.09 | 1.03 |
| **Pooled k** [95% CI] | **1.92 [1.79–2.08]** | 1.59 [1.45–1.74] | 1.01 [0.95–1.07] | **1.19 [1.09–1.29]** | 1.15 [1.05–1.24] | 1.21 [1.08–1.33] |
| Spread max/min · I² | 1.31 · 0.70 | 2.33 · 0.96 | 2.13 · 0.90 | **1.32 · 0.00** | 2.56 · 0.92 | 2.81 · 0.88 |
| *Scale-free (should be 1):* crypt share of tissue | 0.90 [0.82–1.00] | 0.84 | **1.18** [1.05–1.30] | 1.03 | 1.20 | 0.76 |
| *Scale-free:* crypt width / spacing | 1.08 | 1.26 | 1.01 | 1.01 | 1.15 | 1.09 |

\* **The SSL crypt ratios are bridged through SPIDER SSL**, because UniToPatho has no SSL. SPIDER's SSL looks like HP, so for crypts that bridge is not trustworthy. For MHIST SSL, read the nuclei instead: nuclei don't depend on the diagnosis, and they give **2.14**, the same as MHIST HP. That is as it should be for one dataset.

† **SPIDER's 0.48 µm/px was itself set from UniToPatho nuclei**, so ≈1.0 for SPIDER's nuclei is circular. Its crypt measures are an independent check, and they also give ≈1.0.

The crypt-interior width is the least trustworthy of the "usable" measures. Lumen and mucin depend on the tissue: UniToPatho's own normal mucosa reads 2.59 against its own HP. The interior width pulls the pooled spread up, and without it the MHIST HP measures range from 1.70 to 2.10.

## Medians [IQR] under each dataset's assumed scale

| Measure | MHIST HP | MHIST SSL | UniToPatho HP | UniToPatho NORM | SPIDER HP | SPIDER SSL | HunCRC HP | HunCRC SSL |
|---|---|---|---|---|---|---|---|---|
| Fields used / measured (slides) | 364/400 (–) | 346/373 (–) | 499/545 (41) | 94/100 (21) | 364/400 (364) | 345/400 (243) | 353/357 (24) | 366/400 (1) |
| Crypt width, µm | 45 [37–55] | 47 [36–66] | 81 [65–99] | 70 [56–86] | 82 [63–102] | 81 [59–104] | 71 [56–86] | 61 [44–85] |
| Crypt long axis, µm | 67 [53–87] | 85 [57–129] | 109 [87–138] | 93 [74–120] | 118 [88–160] | 118 [86–161] | 97 [78–122] | 109 [85–149] |
| Crypt area, µm² | 2372 | 3024 | 7000 | 5096 | 7570 | 7464 | 5484 | 5084 |
| Crypt interior width, µm | 14 [9–22] | 16 [8–28] | 28 [17–42] | 12 [8–19] | 18 [11–28] | 24 [12–40] | 21 [14–30] | 23 [12–39] |
| Epithelial wall, µm | 19 [16–24] | 26 [20–34] | 31 [25–40] | 34 [29–44] | 39 [30–50] | 34 [26–45] | 29 [23–36] | 27 [18–41] |
| Crypt spacing, µm | 52 [45–62] | 65 [52–80] | 86 [72–102] | 82 [68–95] | 89 [73–106] | 90 [74–110] | 77 [65–91] | 77 [61–99] |
| Crypts per mm² | 146 [99–198] | 70 [53–94] | 59 [40–81] | 55 [35–75] | 50 [35–68] | 48 [35–66] | 73 [50–106] | 36 [25–55] |
| Lymphocyte-like nucleus, µm | 3.6 [3.2–4.2] | 3.6 [3.2–4.1] | 5.3 [4.7–6.0] | 5.2 [4.7–5.8] | 5.4 [4.8–6.1] | 5.4 [4.8–6.2] | 5.0 [4.4–5.6] | 5.1 [4.5–5.8] |
| Crypt share of tissue | 0.47 | 0.41 | 0.52 | 0.40 | 0.61 | 0.57 | 0.54 | 0.58 |

Nucleus sizes are blob sizes at 2 µm/px, not true diameters: they read small everywhere (real lymphocytes are ~7 µm). Only their ratios mean anything.

**SSL ÷ HP within each dataset** (scale cancels out): crypt density is 0.48 in MHIST, 0.49 in HunCRC and 0.98 in SPIDER. Crypt long axis is 1.27, 1.13 and 1.00. In MHIST and HunCRC, SSL has fewer, longer crypts; in SPIDER, SSL looks like HP.

## Caveats

- **MHIST has no slide IDs**, so its intervals come from resampling images and are too narrow. HunCRC SSL is a single slide, and its intervals are also too narrow.
- **The crypt finder was trained on GlaS shrunk to 2.0 µm/px.** On a dataset that is really ~3.5 µm/px, it sees crypts about half the size it learned on, and its 600 µm² minimum drops small crypts. The positive control measures this bias directly (crypt b ≈ 0.9, so it is small) and corrects for it. The control can only vary scale on UniToPatho tissue, though, so it cannot test whether the finder behaves differently on another lab's staining.
- **Fields need ≥ 50% tissue and ≥ 3 crypts.** The same rule applies everywhere, but it favours crypt-rich areas.
- **The finder averaged 2 flip/rotation views, not the app's 8**, to save CPU time.
- **The goblet-vacuole and epithelial-nucleus measures failed the control and are not used.** Their calibrated values in the JSON are meaningless.
- **"k" assumes UniToPatho's 0.4415 µm/px is right.** HunCRC sitting at a clean ×1.19 from it is unexplained. It is the main source of uncertainty in MHIST's true µm/px: the range is 3.2–3.8.
