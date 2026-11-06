# 3D-aware try-on: feasibility and proof of concept

*Visionary · CMAI480 · 2026-11-03. Experimental only: nothing here is used by the app or the GUI.
Code: `experiments/body3d/` (`body.py`, `reconstruct.py`, `wrap.py`, `run_poc.py`, `capture.py`).
Run: `.venv/bin/python -m experiments.body3d.run_poc` (about 25 s on the M1 Pro, CPU only).
Outputs: `experiments/body3d/out/`.*

## 1. The short answer

| Question | Answer |
|---|---|
| Can we reconstruct body geometry from front + side + back photos on this Mac? | **Yes, for the torso**, with our own geometry (silhouettes → per-height cross-sections), in under a second. Girth error below the armpits ≤ 2.6 cm on the test bodies. Not yet tested on real photos (we have none). |
| Does it measure things a single photo cannot? | **Yes: torso depth, and therefore girth.** On a body with a deep torso, one front photo misses the waist girth by 8.1 cm (−10%); multi-view misses it by 0.6 cm. |
| Does it improve garment *placement*? | **Geometrically, yes:** print scale, the centre-front line on a turned body, and garment size. **Visually, no:** a flat product photo wrapped on a 3D torso looks like body paint, with no folds, sleeves or fabric behaviour. It looks worse than CatVTON. |
| Physically simulated draping? | **No.** It needs a 3D garment mesh with sewing patterns and fabric parameters, which a product photo does not contain. |
| SMPL/SMPL-X, SAM 3D Body, 3DFit? | **SMPL/SMPL-X:** need an account (registration-gated). **SAM 3D Body:** gated (Meta approval). **Anny** (Apache-2.0 body model): installable, but its install was blocked in this session. **"3DFit":** no paper or repository found under that name. |
| Recommendation | Do **not** replace the CatVTON pipeline. Multi-view capture is worth keeping as a **measurement and fit** module, and as a **better target for CatVTON**: centre-front line and side compression on a turned body, garment size vs body. Test it on real captures of the team first. |

## 2. What we built (all local, no new dependencies)

1. **Test bodies with known geometry** (`body.py`). Two synthetic bodies are stacks of superellipse
   cross-sections, with head, A-pose arms and legs.
   - They have the **same frontal widths**, but body B's torso is much deeper. A single front photo
     cannot tell them apart.
   - They are rendered front / side / back by a small z-buffer renderer.
   - We have no multi-view photos of real people with tape measurements, so this is the only ground
     truth available here.
2. **Multi-view reconstruction** (`reconstruct.py`, our own):
   - The person's height scales each silhouette to cm, so rows of different photos line up.
   - Front/back give the width per height; the side gives the depth and how far the chest sits forward.
   - Each height becomes an ellipse, the rounded visual hull of two orthogonal views.
   - Output: torso mesh (OBJ), chest/waist/hip girths, widths and depths.
3. **Orientation of a new photo:** from the shoulder keypoints, since the multi-view capture tells us
   the true shoulder width.
4. **Garment placement** (`wrap.py`) uses our homography, inverse warp, remap and blend:
   - **Rectify:** our DLT homography + inverse bilinear warp turn the product photo into a **metric
     panel**. The photo gives the shape; a size chart's flat chest width gives the cm.
   - **Wrap:** the garment is a loop of circumference 2 × flat width around each cross-section. It
     stands off the body by the ease, or stretches if the garment is too small. The front panel runs
     from the centre front to the side seams.
   - **Render:** for each output pixel, find the visible loop point (after the body's turn) and its arc
     length from the centre front, then sample the panel with our bilinear remap. Lambert shading from
     the loop's normal, then our feathered blend.
   - **The reference:** the same wrap on the *true* body. It has no drape either; it only tells whether
     the placement is geometrically right.

## 3. Results

### Body measurements (`out/reconstruction.jpg`, `out/girth_profiles.png`)

| body | landmark | true girth | multi-view (front+side+back) | single front view (depth assumed 0.70 × width) |
|---|---|---|---|---|
| A | chest (124 cm) | 85.3 | 89.9 (+4.6) | 90.7 (+5.4) |
| A | waist | 73.7 | 73.9 (+0.2) | 75.2 (+1.5) |
| A | hip | 89.9 | 92.1 (+2.2) | 96.4 (+6.5) |
| B | chest (122 cm) | 88.4 | 90.4 (+2.0) | 84.4 (−4.0) |
| B | waist | 83.3 | 83.9 (+0.6) | **75.2 (−8.1)** |
| B | hip | 96.7 | 99.3 (+2.6) | 96.4 (−0.3) |

**Torso depth** (what a single photo cannot see):
- **Waist:** body B is 25.0 cm deep. Multi-view measures 25.4; the single view assumes 19.6.
- **Chest:** 25.4 true, 26.1 multi-view, 22.0 single view.

**Where it fails, and why:**
- **Arms at the chest.** Between 124 and 140 cm the A-pose arms touch the chest in the front
  silhouette, and girth is overestimated by up to ~30 cm there (`girth_profiles.png`). Hip width is
  likewise inflated by the tops of the legs.
- **The fix for real photos:** remove the arms from the silhouette with our body parsing, which labels
  arms, or require arms further from the body. A silhouette-only method has no other way to separate
  them.

### Orientation of a later photo, body A

| true yaw | silhouette width change | yaw from silhouette widths | yaw from shoulder keypoints + known shoulder width | error per 1% keypoint error |
|---|---|---|---|---|
| 10° | −0.4% | 7.5° | 10° | ±3.3° |
| 20° | −2.2% | 16.5° | 20° | ±1.7° |
| 30° | +5.5% (an arm merges) | **0° (fails)** | 30° | ±1.1° |

- **Silhouette width is a poor rotation cue.** The torso silhouette barely narrows (−2% at 20°), so
  that estimate is ill-conditioned.
- **The shoulder keypoint ratio works,** because the capture tells us the true shoulder width. It is
  noise-limited at small angles, where it also matters least.
- **MediaPipe's 3D landmarks**, already used in the app, remain the practical yaw source for a single
  photo.

### Garment placement on the same body render

Figures: `out/placement_fila_crop_tee.jpg`, `out/placement_colour_block_polo.jpg`. Columns: body · 2D
(current app) · 3D size S (84 cm girth) · 3D size L (100 cm). Rows: facing the camera, turned 25°.
Numbers below are for the FILA tee; the polo gives the same picture.

| | 2D (current) | 3D (reconstructed torso) |
|---|---|---|
| Print size at the centre front vs the geometric reference | **0.78** (S), **0.66** (L): printed smaller than the real garment, and identical for every size | 1.03 (S, stretched 2.7%), 1.00 (L) |
| Centre-front line, body turned 25° | **5.3 cm off.** It sits midway between the keypoints, not where the turned chest faces the camera | 0.2–1.2 cm off |
| Garment size | ignored: S and L look the same | S: −2.5 cm ease (2.7% stretch); L: +13.5 cm ease (stands off the body) |
| Side compression on a turned body | uniform squash (homography) | stronger on the far side, weaker on the near side, as on a real body |
| Folds, drape, sleeves, collar | none (shading copied from the photo) | **none.** Torso panel only, the back panel is guessed, and it looks painted on |

## 4. Geometry vs physically simulated draping

| | What it needs | What it gives | Feasible here? |
|---|---|---|---|
| **Estimated 3D body geometry** (this PoC; SMPL-X / Anny / SAM 3D Body if obtained) | photos (+ height) | girths, depth, shoulder orientation, a body mesh | **Yes.** Silhouette version done (CPU, < 1 s). Learned single-image versions are gated or blocked here. |
| **Geometric garment wrap** (this PoC) | body geometry + product photo + garment size | correct print scale, centre line, side compression, tight/loose verdict | **Yes,** but visually flat |
| **Learned 3D garment deformation** (GarNet, TailorNet-style, vto-garment-collisions) | a 3D garment *template* per garment type, trained models (often TensorFlow 1/2, CUDA) | drape-like folds for that template | Templates only; it can't map an arbitrary product photo's design onto a correct 3D pattern |
| **Physically simulated draping** (cloth simulation) | 3D garment mesh **with sewing patterns and fabric parameters** + body mesh | real folds, gravity, collisions, fit | **No:** a product photo has no patterns or fabric parameters |
| **CatVTON (current)** | one photo + product photo + our mask | photoreal folds and shading learned from 10k+ photos | Yes (running) |

**What we can actually achieve:** accurate *body* geometry from multi-view capture, and *geometrically
correct placement*. **What we cannot:** true drape. The realism of the current results comes from
CatVTON's learned prior, not from geometry.

## 5. What 3D would and would not improve in our actual results

| Problem seen in our CatVTON results | Would multi-view 3D help? |
|---|---|
| Wrong garment length (`COVERAGE.md`) | **Partly.** With a measured torso and the garment's size-chart length, the hem height is physical, not a ±0.25 shoulder-width guess. That only holds if the size-chart length is known, and how the garment is worn (tucked) still isn't. |
| Print or logo too small, or centred wrongly on a turned person | **Yes** (print scale 0.66–0.78 → ~1.0; centre front 5.3 → ≤ 1.2 cm). This could feed CatVTON as a better target mask or warm start. |
| "Does size S fit me?" | **Yes, a new capability.** Girth vs garment girth gives ease and strain; 2D cannot. |
| Fine patterns become texture, small labels lost, bare-shoulder haze, invented belts | **No:** these are CatVTON's resolution and learned prior |
| Folds and drape | **No:** needs simulation with data we do not have |
| One-photo convenience | **Worse:** the user must take 3 photos in a set pose |

## 6. Proposed next steps (only if the team wants to continue)

1. **Validate on real people (the missing evidence).** Capture 2–3 team members front / side / back
   (protocol in `capture.py`) and tape-measure chest, waist and hip. Run `capture.py`, compare girths,
   and add arm removal via our SegFormer parsing.
2. **If girth error stays ≤ 3 cm:** use the 3D placement only as a **target for CatVTON** (centre-front
   line and side compression on turned bodies, hem from size-chart length), plus a fit indicator in the
   UI. Keep CatVTON for the appearance.
3. **For a learned body model:** request SMPL-X (registration) or SAM 3D Body (Meta approval) access
   with a team account, or allow installing Anny (Apache-2.0). These would give a full body mesh from a
   single photo, to compare against the silhouette method on the same captures.

## Sources

- Anny body model, Apache-2.0: https://pypi.org/project/anny/ · https://arxiv.org/abs/2511.03589
- SAM 3D Body (gated): https://huggingface.co/facebook/sam-3d-body-dinov3
- Learned 3D garment deformation: https://github.com/isantesteban/vto-garment-collisions · GarNet (EPFL), https://infoscience.epfl.ch/record/279074
- 3D try-on research: DreamVTON https://arxiv.org/abs/2407.16511 · VTON 360 https://arxiv.org/abs/2503.12165 · Gaussian Wardrobe https://arxiv.org/abs/2603.04290
