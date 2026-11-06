# Does the pipeline lack depth? Audit of 3D information

*Visionary · CMAI480 · 2026-11-03. Same 6 people × 4 garments as `RESULTS.md`, on the M1 Pro, all local.
Scripts: `eval/proportions.py`, `eval/run_studio.py`, `eval/make_figures.py`.*

**Short answer.** The pipeline was entirely 2D, but **the missing depth was not the main cause of the
wrong proportions**. The biggest error was a 2D one: the garment's length came from the person's
**old top** (through our mask) and from the **hip joints** (through our homography), never from the
**product**. Fixing that is pure geometry, and it moved the rendered length from tracking the old top
(r = +0.63) to tracking the product (r = +0.81). Real 3D information helped in exactly one small,
free way: torso rotation from MediaPipe's 3D landmarks, which the pipeline had been discarding. A
monocular depth model or SMPL would add little to the studio render, because CatVTON cannot take depth
as an input. Tested in the instant render, cylinder curvature gave a marginal gain and new seams.

## 1. What spatial information we extract from the person

| Source | What we used | What it offers that we ignored |
|---|---|---|
| MediaPipe Pose Landmarker | x, y and visibility of 13 points (shoulders, elbows, wrists, hips, eyes, mouth) | **z per point** and **3D world landmarks in metres** (hip-centred), which give torso rotation |
| MediaPipe person mask | 2D silhouette (fit, fallback) | – |
| SegFormer / MediaPipe multiclass | 2D labels: top, arms, face, hair, background, lower body | – |
| Our geometry | DLT homography (planar torso), TPS (2D edge fit), affine sleeves | – |

**Depth, surface normals, body shape and 3D pose: none of these were used anywhere.** The torso was a
plane through four points. The studio render received a 2D mask, the person photo and the flat
product photo.

Measured torso rotation from the unused 3D landmarks: 00055 **+16°**, 01992 **+19°**, 00034 −7°,
00121 +2°, stock_man −3°.

## 2. Which failures are really caused by missing 3D information?

| Failure seen | Evidence | Real cause | 3D needed? |
|---|---|---|---|
| Garment length copies the old top (cropped tee → every garment cropped) | 24 studio pairs: corr(render, old top) **+0.54**, corr(render, product) **−0.17** | **Masking**: the mask was the old top, clipped at the old hem. **2D geometry**: the homography sent the hem anchors to the hip joints | **No** |
| Cropped garments stretched, long ones squashed (also in MATLAB) | `_scaled_torso` maps the garment hem to the hips, whatever the garment's length | **2D geometry** | **No** |
| Garment slightly too small when the person is turned | The 2D shoulder width shrinks by cos(yaw): −4% at 16°, −5% at 19° | Missing **torso rotation** | **Yes, but MediaPipe's 3D landmarks suffice** |
| Instant render looks flat and pasted on | No folds or roundness beyond copied luminance | Partly curvature (3D). Mostly that a classical overlay cannot synthesise cloth | Partly (see §4) |
| Fine patterns become texture (gingham) | Same at all lengths and poses | **Model resolution** (512×384, SD VAE) | No |
| Pale haze on a bare shoulder (01992 + polo) | Mask edge between a new sleeve and bare skin | **Model limitation** at the mask/skin boundary | No |
| Soft results on the stock photo | Torso ≈ 200 px wide in the input | **Input resolution** | No |
| Hands, hair and face over the garment | Already handled by 2D layering from parsing | Occlusion order in a front-facing photo is 2D | No |

## 3. How established try-on models handle body geometry

Verified in their code during the earlier research (`AUDIT.md`):
- **No surveyed model uses monocular depth or SMPL at inference.**
- **IDM-VTON** and **StableVITON** feed a **DensePose** map (a dense body-surface coordinate image, i.e. "2.5D" correspondence) as extra UNet input channels, together with an agnostic mask.
- **Leffa** builds its mask from DensePose + SCHP.
- **CatVTON** (what we run) takes only the person, the garment and a mask. DensePose is used only inside its optional auto-masker, never as model input.
- **OOTDiffusion** uses OpenPose + parsing only.

How geometry is handled instead:
- **Folds, shading and body roundness** are implicit: the models learn them from ~10k–70k real photos (VITON-HD, DressCode). That is why the studio render already shows folds and curvature that the classical pipeline cannot.
- **Occlusion** is solved by the mask: what is not masked stays in front. With a parser, we do the same.
- **True 3D try-on** (SMPL-based draping) needs a 3D garment mesh. That does not exist for a product photo, so it is not applicable here [A: not re-verified in code].

## 4. Would depth / SMPL improve our actual pipeline? Tested

**Depth Anything V2 Small** (25M params, `depth-anything/Depth-Anything-V2-Small-hf`): 0.55 s per photo on the M1 CPU.
- It gives a clean body silhouette. Across the chest it shows the expected bulge on 3 of 4 people (centre 0.83–1.00 against edges 0.59–0.66, relative units). On the small stock photo it is flat.
- **Limit 1: it measures the old clothing's surface, not the body** (a boxy tee reads as a slab).
- **Limit 2: its depth is relative.** Bending a garment around the torso still needs an *assumed* torso depth, at which point a simple elliptical cross-section does the same job.
- **Studio render: no possible effect.** CatVTON has no depth input. Using one would mean fine-tuning a model, which we cannot do on this hardware.

**Elliptical-cylinder torso wrap** (implemented: `pipeline.cylinder_wrap`, option `curvature`):
- It compresses the pattern toward the sides and adds cos(θ) shading, with the cylinder centre shifted by the 3D torso yaw.
- In the instant render it gave a slightly rounder look on the gingham and velvet. It also added visible seams where the curved band meets the shoulders and sleeves, and it did not address the dominant instant-mode problem (the old print bleeding through).
- **Kept as an option, off by default.**

**SMPL-based reconstruction** (not run):
- It would give body shape and 3D orientation, which MediaPipe's world landmarks already give us coarsely, for free.
- The studio model cannot use it, and the instant pipeline has no 3D garment to drape.
- Typical SMPL regressors are larger, GPU-oriented models [A].
- **More complexity than benefit here.**

## 5. What we changed (justified by the measurements)

1. **Proportion-aware garment mapping** (`pipeline._proportional_torso`):
   - One scale factor, body shoulder width / garment shoulder width, sets both width and length.
   - The hem lands at the shoulder line + the product's own shoulder-to-hem distance × scale, along the torso axis, instead of on the hip joints.
   - The old mapping remains as `proportional=False`.
2. **Rotation-corrected size:** the body shoulder width is divided by cos(yaw), with yaw from MediaPipe's 3D world landmarks (clamped at 60°). This is the one 3D quantity that measurably matters for proportions, and it costs nothing.
3. **Garment-aware studio mask:** the mask may extend down to the *new* garment's predicted hem (from the proportional footprint) and over the top of the trousers, instead of stopping at the old hem.

## 6. Results on the worst cases (identical inputs)

15 pairs: persons 00055, 00121, 00035, 01992, stock_man × velvet wrap, colour-block polo, red tie-waist tee.

| | corr(render length, old top) | corr(render length, product) | mean length (shoulder widths) |
|---|---|---|---|
| before (mask = old top, hem → hips) | **+0.63** | **0.00** | 1.38 |
| after (proportional + garment-aware mask) | −0.09 | **+0.81** | 1.81 |

**All 24 pairs** (`results/studio_standard_hipmapped/` before, `results/studio_standard/` after; 32.5 s per render):

| | corr(length, old top) | corr(length, product) | colour match | old collar zone left | rest-of-image SSIM |
|---|---|---|---|---|---|
| before | +0.54 | −0.17 | **0.689** | **1.7%** (1 pair > 10%) | **0.996** |
| after | −0.03 | **+0.68** | 0.632 | 2.4% (2 pairs > 10%, both 00035) | 0.966 |

**This is a trade-off.** Length and design features now follow the product. Colour match drops,
because the garment region now includes the strip over the trousers, where the repaint can wash out.
The rest-of-image SSIM falls as expected, because more of the trousers is now (legitimately)
covered. We keep it, because wrong garment length and missing design features were the complaint.
The trouser strip is the next thing to fix.

Figures: `results/figures/proportions_before_after.jpg` (also `results/studio_standard_hipmapped/` vs `results/studio_standard/`).

**Visible wins:**
- The red tee's tie-waist, a defining design feature, now appears (00055, 00035, stock_man). Before, it was cut off at the old hem.
- The velvet wrap gets its side tie and real length.

**New or remaining problems:**
- Where a longer garment now covers the top of the trousers, the repainted strip is sometimes washed out (00055 + velvet).
- The polo stays short on 00055.
- The stock_man renders are still soft (input resolution).
- A garment **shorter** than the old top still comes out long. The model fills the whole mask, and the old top has to be inside the mask to be removed (a model limitation; it would need a "skin/trousers below the hem" prior that CatVTON does not have).

## 7. Recommendation

- **Do not add a depth model or SMPL.** On this pipeline they cannot reach the part that makes results realistic (the studio render), and in the instant render the tested benefit is marginal.
- **Keep the proportion-aware mapping and the garment-aware mask.** They address the measured cause of the poor fit and are explainable 2D/3D geometry: DLT homography, rotation-corrected scale, TPS.
- **If more geometry is wanted later,** the established route is a **DensePose-conditioned** try-on model (IDM-VTON, Leffa), not depth. That means building detectron2 for CPU on macOS (see `AUDIT.md`), which is outside the deadline.
