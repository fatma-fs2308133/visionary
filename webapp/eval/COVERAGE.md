# Garment coverage: showing the selected garment's own shape and length

*Visionary · CMAI480 · 2026-11-03. Follows `DEPTH.md`. All runs local on the M1 Pro, identical inputs
for before and after, CatVTON standard (512×384, 25 steps, seed 42). Scripts: `eval/coverage.py`,
`eval/calibrate_length.py`, `eval/coverage_figure.py`, `eval/run_studio.py --coverage`.*

## Short answer

The studio render showed "a plausible outfit" instead of the selected garment for two reasons:

1. **Our length estimate was wrong.** The automatic "shoulder" anchors sit near the neckline, at about
   0.73 of the seam-to-seam width. Scaling by them made every garment about 35% too long, so a crop tee
   reached the hips and the velvet top became a dress.
2. **CatVTON fills its whole mask with garment.** This is a known property of mask-based try-on (FitDiT
   names it and fixes it by retraining with relaxed masks). Our mask was "old top + new footprint", so
   the new garment always covered at least the old top. On a cropped garment it invented a
   garment-coloured skirt below the real hem (yellow cami in `grid_coverage_cropped_before.jpg`).

The fix keeps CatVTON and changes what we give it:
- **Calibrated length.** The scale now comes from the garment's chest width, fitted on 30 VITON-HD
  test models wearing their paired product.
- **Separate coverage from regeneration.** CatVTON's mask is the garment's own target plus the neck and
  arms. Old clothing outside it (the exposed region) is repainted too, but is pinned to the person's own
  trousers below the waistband for most of the denoising. This uses blended latent diffusion, so the
  garment cannot grow past that point.

## 1. Length: ground truth instead of a guess

The VITON-HD test set pairs each model photo with the product photo of the garment the model is wearing
(`eval/vitonhd_gt/`, 33 pairs). For each pair we compare our predicted hem with the hem actually worn,
both in shoulder widths below the shoulder line:

| | median worn / predicted | median abs. error | corr(worn, predicted) |
|---|---|---|---|
| scale by shoulder anchors (before) | ≈ 0.6 | > 0.5 | – |
| scale by chest width ×0.91 (first fit, 1 cropped pair) | 0.81 | 0.25 | +0.16 |
| **scale by chest width ×0.77 (fitted on 30 pairs)** | **0.94** | 0.27 | **+0.35** |

**What the ground truth also shows:**
- **How a top is worn moves the hem a lot.** Tucked tops and high-waisted bottoms are common: worn hems
  sit a median 0.18 shoulder widths *above* the hip joints, and almost always end where the trousers
  start. From a product photo alone, the worn length is uncertain by about ±0.25 shoulder widths. No
  scale constant removes that.
- **The cost of a length error is asymmetric.** A target that is too long makes CatVTON paint a skirt
  over the trousers (it did: polo on 00035, gingham on 00035, polo on 00034 in our first run). One that
  is too short only looks slightly cropped. So the garment may overlap visible trousers by at most 0.25
  shoulder widths, and never by a sliver (< 0.12) that would become a band.
- **One pair (00654) failed at garment preparation**, not geometry: a white tee on a white background,
  where rembg cut out only the print.

## 2. Masks: what the garment covers vs what has to change

`hybrid/pipeline.py: coverage_masks`:
- **target T:** the product silhouette placed by our proportional homography warp (+ sleeves, + TPS)
- **garment region M:** T grown 4% + neck below the chin + arm corridors. Old sleeves stay in M: CatVTON
  already turns them into skin or new sleeves correctly, as on 00035 and stock_man.
- **exposed region E:** old clothing outside M

| How E is rebuilt (tested on the same pairs) | Result |
|---|---|
| inside CatVTON's mask, free (= old method) | garment-coloured skirt or garment extended to the old hem |
| separate SD-1.5 inpainting pass, prompt "bare skin…", negative "shirt, fabric…" | paints fabric (a grey under-layer, a khaki band) and continues the old shirt. **Rejected** |
| CatVTON, E pinned to a flat skin-coloured fill for 50% of steps | a translucent beige "garment". **Rejected** |
| **CatVTON, E below the waistband pinned to the person's own trousers (mirrored upward) for 70% of steps; above it free** | trousers continue upward, and the model paints the midriff itself under a cropped hem. **Adopted** |

The last row works because CatVTON *can* draw a bare midriff under a cropped hem (it did in the old
run). It only went wrong where it had to invent something for the hip area, which in its training data
is always trousers or garment.

## 3. Results

### Cropped garments over longer tops (4 garments × 6 people, `eval/garments_short/`)

Garments: cropped colour-block FILA tee, yellow crop cami, red off-shoulder crop, black wrap crop (VITON-HD
/ OOTDiffusion examples, eval only). 14627 is the VITON-HD model actually wearing the black wrap crop.

| | length error (abs., sw) | exposed region still garment-like ↓ | exposed region parsed as top ↓ | visible trousers repainted ↓ | trousers SSIM outside garment ↑ | colour match ↑ |
|---|---|---|---|---|---|---|
| before | 0.338 | 0.651 | 0.569 | 0.110 | 0.932 | 0.587 |
| **after** | **0.282** | **0.536** | **0.407** | **0.034** | **0.991** | **0.597** |

**Visually** (`coverage_cropped.jpg`, grids `grid_coverage_cropped_before.jpg` / `grid_coverage_cropped.jpg`):
- **Correct now:**
  - 00035: cami, black wrap and off-shoulder top are cropped with a natural midriff and her jeans; the
    FILA tee ends at the waistband.
  - 00034 + cami: a cropped cami over high-waisted trousers, where before there was a yellow skirt.
  - 14627: all four are correct.
- **Still wrong:**
  - The black wrap on 00034, 01992 and stock_man still renders long. Black on black trousers is the
    hardest case, and the model joins them.
  - The off-shoulder top on 00034 and 01992 still has a long peplum.
  - The cami on 00121 keeps a yellow band at the shorts.
  - The cami on 01992 and stock_man shows a pale patch at the waistband, from the mirrored-trouser
    anchor.

### All 24 benchmark pairs (regular-length garments)

| | visible trousers repainted ↓ | trousers SSIM outside garment ↑ | rest-of-image SSIM ↑ | face SSIM ↑ | colour match ↑ | old collar zone left ↓ |
|---|---|---|---|---|---|---|
| hip-mapped (`DEPTH.md` before) | 0.046 | 0.999 | 0.996 | 0.999 | 0.688 | 1.7% |
| proportional (current app before this change) | 0.385 | 0.797 | 0.966 | 0.999 | 0.632 | 2.4% |
| **garment coverage (after)** | **0.168** | **0.999** | **0.988** | 0.999 | **0.645** | **1.4%** |

**What the numbers support:**
- **The washed-out trouser strip is gone.** Before, 38.5% of the visible trousers were repainted;
  outside the garment, nothing is now (SSIM 0.999).
- **Neckline and colour are slightly better**, so the change generalises beyond the worst pairs.

**Trade-offs, visible in `coverage_24pairs.jpg`:**
- **Hanging design elements are cut.** The velvet's side tie and the red tee's tie hang below the hem
  over the trousers, and the 0.25 overlap cap now removes most of them (00055 velvet, 00034 red tee).
  This partly undoes a visible win from `DEPTH.md`.
- **Invented belts.** On 00055 (velvet, gingham), CatVTON adds a belt where the garment meets her
  high-waisted jeans. That is a change to the trousers that the input does not have.
- **Garments end near the trousers.** That matches how VITON-HD models wear tops, but a long gingham
  shirt now reads as tucked in.

### Instant (classical) mode

It uses the same corrected length.
- **Rest-of-image SSIM** goes from 0.966 to 0.993: less garment over the trousers.
- **More of the old top shows below a shorter garment** (old top left 4.7% → 6.9%, colour 0.557 →
  0.482). An overlay cannot remove clothing, which is the classical ceiling.

## 4. Would another pretrained model do this better?

- **FitDiT** (Jiang et al. 2024) addresses exactly this. It trains with a *dilated-relaxed mask*, a
  coarse rectangle whose size is randomised, so the model learns the garment's own length instead of
  filling the mask. It is a large DiT that needs a CUDA GPU, and its hosted Space was out of quota for
  us.
- **Mask-free models** (BooW-VTON, CVPR 2025; MFTON, IJCAI 2025; "All Parts Matter", ICCV 2025) avoid
  the problem by design. We found no small, Mac-runnable release of them.
- **IDM-VTON, StableVITON and Leffa** are mask-based like CatVTON and fill their mask the same way.
- **Conclusion:** the reliable fix in the literature needs retraining or a CUDA-sized model. On this Mac
  the practical route is what we did: give CatVTON a garment-shaped mask and constrain the rest.

Sources: FitDiT arXiv 2411.10499 (https://arxiv.org/abs/2411.10499); BooW-VTON CVPR 2025
(https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_BooW-VTON_Boosting_In-the-Wild_Virtual_Try-On_via_Mask-Free_Pseudo_Data_Training_CVPR_2025_paper.html);
MFTON IJCAI 2025 (https://www.ijcai.org/proceedings/2025/105).

## 5. Reproduce

```bash
.venv/bin/python -m eval.calibrate_length                       # length ground truth (33 VITON-HD pairs)
SG="--garment-dir eval/garments_short --garments fila_crop_tee,yellow_crop_cami,red_offshoulder_crop,black_wrap_crop"
P="--persons 00034_00,00035_00,01992_00,stock_man,00121_00,14627_00"
.venv/bin/python -m eval.run_studio standard $SG $P --coverage mask --tag _short_base   # before
.venv/bin/python -m eval.run_studio standard $SG $P --coverage garment --tag _short_cov # after
.venv/bin/python -m eval.run_studio standard --coverage garment --tag _cov              # 24 pairs
.venv/bin/python -m eval.coverage studio_standard_short_base studio_standard_short_cov --garment-dir eval/garments_short
.venv/bin/python -m eval.coverage studio_standard studio_standard_cov
.venv/bin/python -m eval.neckline studio_standard studio_standard_cov
```

Note: `studio_standard_short_base` was rendered before the length recalibration, with the mask method
of the app as it was. The coverage metrics compute the target T and the exposed region E from the
inputs with the *current* calibration, the same for every method.
