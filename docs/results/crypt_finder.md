# Crypt finder: classical vs learned (2026-10-20)

The app outlines crypts in each field and tags them **dilated** (leans SSL) or **round regular** (leans HP) to
explain the image call. The tags explain the call; they don't make it. This compares the old classical finder with a
learned one, against ground truth only (expert outlines, the 7-pathologist MHIST votes, slide labels).

**Result: the learned finder replaces the classical one in the app** (`backend/glands.py` `find()`,
`backend/gland_net.py`). The "serrated" tag is dropped.

## Ground truth

Hand-tracing MHIST fields was planned (`research/trace_crypts.py`, a local tracing page for 32 stratified training
fields; `research/crypt_gt.py` holds the selection and scoring) but not done. Outlines are scored instead against
**GlaS** (Warwick, MICCAI 2015): 165 colon H&E images, every gland outlined by a pathologist, 0.62 um/px. Each image
is shrunk to MHIST's 2.0 um/px and placed in a 448 um field exactly as an MHIST tile is (`research/score_crypt_finders.py`).
Detection = one-to-one matching at IoU >= 0.5. GlaS has healthy colon, adenoma and cancer; benign (healthy + adenoma)
is what matters for serrated polyps.

## 1. Outlines vs expert outlines (GlaS benign, 909 glands)

| Finder | Precision | Recall | F1 | Outline IoU (matched) |
|---|---|---|---|---|
| Classical (pale interior grown to the nuclear band) | 0.37 | 0.34 | 0.36 | 0.71 |
| SAM ViT-base, prompted at classical crypt centres | 0.38 | 0.15 | 0.22 | 0.72 |
| **U-Net, trained on GlaS at 2 um/px (out-of-patient)** | **0.94** | **0.85** | **0.89** | **0.88** |

Same picture with crypts cut by the field edge treated as don't-care (the classical finder skips them by design):
classical F1 0.42, U-Net 0.90. Healthy colon only: classical 0.39, U-Net 0.90. Cancer (not our task): 0.09 vs 0.67.
The U-Net is scored out-of-patient: GlaS split into two patient-disjoint halves, each image scored by the model
trained on the other half. The app uses a model trained on all 165 images.

- SAM gives either the whole tissue or a single goblet-cell blob per prompt, never a crypt-sized mask at this
  resolution, so it was dropped.
- Shown outlines skip shapes under 600 um^2 or with solidity < 0.8 (on GlaS: precision 0.91 -> 0.94, recall
  0.88 -> 0.85). On MHIST this mainly hides two touching crypts merged into one U or W shape. Fewer, right outlines
  beat many shaky ones.

U-Net details: ConvNeXt encoder (ImageNet), U-Net decoder, three classes (stroma / gland / gland border, so
touching glands come apart). Training images are shrunk to 2 um/px and enlarged 2x, so the network sees what an
MHIST tile shows. Random scale 0.85-1.18 (MHIST is ~1.8-2.2 um/px), flips, rotations, stain jitter. 80 epochs.
Inference averages 8 flips/rotations in one batch. `research/train_gland_net.py`.

**Bigger encoder.** ConvNeXt-Base (88M parameters) vs Tiny (28M), both trained on the same patient half and scored
on the other half's images (`research/compare_gland_nets.py`, `docs/results/gland_net_tiny_vs_base.json`):

| GlaS, unseen patients | Tiny F1 | Base F1 |
|---|---|---|
| All benign glands (282) | 0.889 | **0.908** |
| Healthy colon (36) | 0.909 | **0.957** |
| Adenoma (246) | 0.887 | **0.901** |
| Benign glands not cut by the edge (87) | 0.908 | 0.904 |
| Cancer, all (354) | 0.728 | **0.768** |

The gain is mostly on glands cut by the field edge. It's modest and from one patient half, but in the same direction
in every group. The app uses Base (`data/models/gland_net_base.pt`) and falls back to Tiny (`gland_net.pt`), then to
the classical finder. More expert-outlined data (CRAG, Warwick) would be the next step, but it's behind a login.

**Speed.** Per field on an RTX 5060 laptop GPU: the network (8 views, Tiny) takes 0.14 s, and 1 view 0.04 s. Most of
the time was the measuring on the CPU, which now skips what labels from the outline don't use (nuclear-ring share,
lumen roughness): same labels on 30/30 fields, 2.4x faster. So fewer views would save almost nothing, and a bigger
network is cheap.

## 2. Same field, same crypts: raw tile vs screen snip (100 MHIST training fields)

A snip is the tile drawn at a larger size in a viewer and captured (`research/crypt_consistency.py`). F1 between the
two crypt sets (1.0 = identical):

| Snip | Classical | U-Net |
|---|---|---|
| Zoom 1.5x | 0.77 | **0.94** |
| Zoom 2.37x | 0.77 | **0.94** |
| Zoom 3.3x | 0.78 | **0.94** |
| Zoom 2.37x, box shifted 4% | 0.66 | **0.92** (Tiny 0.90) |

Matched outlines overlap at IoU 0.94 (classical 0.83). The U-Net first brings any field to MHIST's own 224 px, so a
snip and a tile carry the same detail. Matched crypts keep the same label 91-92% of the time when zoomed
(classical 89%) and 83% with the shifted box (classical 87%; the field median that "dilated" is compared with changes
when part of the field is lost).

## 3. Labels vs the 7 pathologists (1,575 held-out MHIST training images)

Per image: share of labelled crypts tagged dilated / round regular, against the number of 7 pathologists who called
it SSL (`research/gland_check.py`; none of the 600 images that set the thresholds).

| | Dilated: Spearman vs SSL votes | Dilated: AUC SSL vs HP | Round regular: Spearman | Images with a label |
|---|---|---|---|---|
| Classical | 0.18 | 0.58 | -0.27 | 1,222 |
| U-Net outlines, labels from the pale interior | 0.32 | 0.61 | -0.32 | 1,170 |
| U-Net (Tiny) outlines, labels from the crypt outline | 0.45 | 0.68 | -0.45 | 1,405 |
| **U-Net (Base) outlines, labels from the crypt outline (shipped)** | **0.47** | **0.69** | **-0.47** | **1,414** |

Labels from the outline use the same rules as before, recomputed on the shipped finder's crypts in the same 600
training images (`backend/config/gland_thresholds_unet_g.json`): dilated = outline width at or above the 90th
percentile (76.9 um) and >= 1.5x the field's median; round regular = solidity >= 60th percentile (0.965),
eccentricity <= 0.75, no larger than the median crypt (56.7 um). Measuring the pale interior made sense when the outlines were unreliable.
Now that they're reliable, the crypt itself is the better thing to measure.

## 4. Can "serrated" be seen at ~2 um/px? No. Dropped.

UniToPatho (Turin, 0.44 um/px): hyperplastic polyps (serrated lumens) vs normal colon (smooth lumens), same lab and
scanner. Crypts are found once at 2 um/px; each lumen's roughness (perimeter / convex-hull perimeter, as in
`glands.py`) is measured inside the same outline at 0.44, 1.0 and 2.0 um/px (`research/serration_resolution.py`).

| Scale | AUC HP vs normal, per crypt | per field |
|---|---|---|
| 0.44 um/px (native) | 0.60 | 0.62 |
| 1.0 um/px | 0.60 | 0.59 |
| 2.0 um/px (MHIST) | 0.60 | 0.55 |

(Classical crypts: 0.57 / 0.59 / 0.59 per crypt.) Roughness barely separates serrated from smooth crypts even at full
resolution, and per field it is near chance at MHIST's scale. A serration tooth is ~5-10 um, i.e. 2-5 pixels at
2 um/px: at the limit of what the image resolves. Serration is also common to HP and SSL, so it couldn't separate
them anyway. The tag is off (`glands.SHOW_SERRATED = False`). The app showed it on clean round crypts, which hurt
trust more than it helped.

## 5. Tried and not shipped: boot / L / inverted-T crypts

Shape of each outline's centre line (`research/crypt_shape.py`): a sharp bend (>= 60 deg) or a long side arm. On 300
held-out MHIST fields the share of such crypts does not separate SSL from HP (AUC 0.50). Looking at them, most are
two neighbouring crypts merged into a U, not real boot-shaped bases. A 448 um field rarely shows a whole crypt down
to its base. Revisit on larger fields (whole slides in a viewer) with a model that splits touching crypts better.

## Limits

- GlaS is healthy colon, adenoma and cancer from one UK lab, not serrated polyps. The U-Net still merges touching
  crypts on some MHIST polyp fields and misses crypts cut lengthwise at the field edge.
- No hand-traced MHIST ground truth yet; `research/trace_crypts.py` is ready for it.
- Label agreement (AUC 0.68 for dilated) is per image and modest. The tags explain the call; the image panel makes it.
