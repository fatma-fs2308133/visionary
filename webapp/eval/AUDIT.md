# Virtual try-on: technical audit and recommendation

*Visionary · CMAI480 · 2026-11-02. Everything here is reproducible with the scripts in `webapp/eval/`. Figures are in `eval/results/figures/`.*

**Short version.** Our results look fake mainly because of the overall approach, not because of a bug in the homography or the blending. A flat product photo, warped with 8 degrees of freedom and pasted on top of the old clothes, cannot remove the old garment, generate skin, follow the body outline, or create folds. Better landmarks will not fix that. We built and measured three alternatives on identical inputs:
- a classical upgrade (thin-plate spline (TPS) warp + occlusion handling);
- a **hybrid**, where our geometry builds the mask and a pretrained CatVTON model renders inside it, **locally on the M1 Pro in ~29 s**;
- hosted IDM-VTON.

**Recommendation: adopt the hybrid as the "Studio render".** Keep our DLT homography, inverse bilinear warp and feathered blend as the demonstrable core. Also port the classical fixes into live mode.

---

## 1. Test setup (same inputs for every method)

| | |
|---|---|
| People | 5 VITON-HD test models (from the IDM-VTON examples) + 1 stock photo: `eval/persons/` |
| Garments | 4 VITON-HD product photos: velvet wrap top, colour-block polo, gingham shirt, red tie-waist tee |
| Pairs | 6 × 4 = 24 for local methods. The hosted subset is 4 pairs (`eval/pairs.json`) |
| Hardware | Apple M1 Pro, 16 GB unified memory, PyTorch MPS. No NVIDIA GPU |
| Measurement parser | SegFormer-B2 clothes parser (`mattmdjaga/segformer_b2_clothes`, 18 ATR labels, verified from the model config) |

**Metrics** (`eval/metrics.py`):
- **Coverage metrics** (methods with a known garment alpha):
  - `old_visible`: share of the original top not covered.
  - `spill_bg`, `spill_face`: share of garment pixels on background, or on face/hair.
  - `forearm_covered`: forearm/hand pixels hidden by the garment.
  - `iou_upper`: overlap with the person's clothing region.
- **Output metrics** (any method, including black boxes):
  - `ssim_face`, `ssim_keep`: identity/background preservation outside the clothing area (SSIM).
  - `garment_hist`: colour-histogram match between product photo and output top.
  - `old_unchanged`: share of old-top pixels left as they were.

Two metrics turned out unreliable, and we say so instead of using them:
- **`edge_seam`** scores real, unedited photos at 3–14 (mean 8), so it cannot tell artefacts from real garment edges. It is excluded from the conclusions.
- **`garment_hist`** is noisy for small patterns (gingham), so treat it as indicative only.

## 2. Root-cause analysis of the current pipeline

Baseline: the current Python port, which has the same algorithm as MATLAB. All 24 pairs (`eval/run_classical.py`; diagnostic overlays in `results/classical/*_diag.jpg`: red = old top visible, blue = spill, yellow = forearms covered).

| Symptom | Measured | Root cause | Stage at fault |
|---|---|---|---|
| Old shirt shows around and through the garment (long sleeves under a tee, logo bleed) | `old_visible` 9% mean (up to 25%). The VANS logo shows through | Nothing removes the original garment. Shading transfer copies its print | **Approach** (no "agnostic" step) |
| Collar over the chin, garment on the background | `spill_face` 4.4%, `spill_bg` 5.9%. Face SSIM 0.90 | The garment keeps its flat-lay shape and no layer order is enforced | **Body alignment / compositing** |
| Sleeves cover forearms and hands, and look like detached shards | `forearm_covered` 27% mean, 96% worst (long sleeve) | One rigid affine per sleeve along the upper arm only. No elbow bend. No occlusion | **Geometric model** |
| Shape does not follow the body | `iou_upper` 0.64. Width ratio 0.73–1.08 | A 4-point homography is a plane-to-plane map (8 DoF). The torso is not planar, and flat-lay ≠ worn shape | **Geometric model** |
| Looks flat, with no folds or drape | Visual | 2-D image warping cannot create 3-D cloth appearance. Shading transfer copies the *old* shirt's folds | **Approach** |

**Landmarks are not the main problem.** MediaPipe found every torso with confidence ≈ 1.0, and the torso quads are plausible on all 6 people. Garment anchor errors mattered more, and were fixed earlier. A better pose model would not change the conclusion.

**The homography and bilinear warp are mathematically correct:**
- equal to `cv2.findHomography` within 2e-5 px;
- equal to `cv2.warpPerspective` to a mean difference of 1e-7 (`tests/test_core.py`).

They are just the wrong model for the whole problem on their own.

## 3. What the open-source state of the art does

Research was done at code level: the actual `app.py`, pipeline and config files, plus model cards and papers.

**[V]** = verified in code, paper or hub API. **[A]** = assumed or estimated.

| Model | Base / size | How it warps | Inputs / masking | VITON-HD (paper) | License | On our M1 Pro |
|---|---|---|---|---|---|---|
| **IDM-VTON** (ECCV'24) | SDXL-inpaint TryonNet + SDXL GarmentNet + IP-Adapter, ~7B params, ~29 GB download [V] | Implicit: GarmentNet features are fed into self-attention every step [V] | OpenPose + SCHP parsing (ONNX, CPU) + DensePose (detectron2, `cuda` hard-coded) [V] | FID 6.29, SSIM 0.870, LPIPS 0.102 [V] | CC BY-NC-SA [V] | No locally (~14 GB fp16 weights). Hosted: ~4 runs/day anonymous [V docs]; Replicate ~$0.024/run [V] |
| **CatVTON** (ICLR'25) | SD-1.5-inpainting, 899M total, **49.6M trained** (self-attention only), no text encoder [V] | Implicit: person and garment latents concatenated, self-attention matches them [V] | **Person + garment + mask only** [V]. Its auto-masker needs DensePose + SCHP, but the pipeline accepts any mask [V] | FID 5.43, SSIM 0.870, LPIPS 0.057 [V] | CC BY-NC-SA [V] | **Yes: we run it.** 512×384, fp16, **~29 s/image** (measured) |
| **Leffa** (CVPR'25) | SD-1.5-inpaint + reference UNet, ~3.6 GB fp16 [V/A] | Flow-field learning in attention [V] | DensePose + SCHP + OpenPose auto-mask [V] | **FID 4.54**, SSIM 0.899, LPIPS 0.048 (best paired) [V] | MIT [V] | Possible but DensePose is fiddly. Hosted Space works [V]; our run hit the quota |
| **FitDiT** | SD3 MM-DiT, 19.5 GB at 1024 (6 GB with offload) [V] | Implicit + dilated-relaxed mask [V] | DWPose + parsing [V] | FID 4.73, best unpaired FID 8.20 [V] | Gated, non-commercial [V] | Too slow locally. Space errored on every call (upstream) |
| **OOTDiffusion** | Two full SD-1.5 UNets [V] | Outfitting fusion in self-attention [V] | OpenPose + SCHP (`torch.cuda.set_device` in code) [V] | FID 8.81 (own), 9.31 (CatVTON's re-eval) [V] | CC BY-NC-SA [V] | Patchable, ~2× CatVTON compute [A]. Space running [V] |
| **StableVITON** (CVPR'24) | SD-1.4 / Paint-by-Example + encoder copy [V] | Zero cross-attention + attention total-variation loss [V] | You build agnostic + DensePose yourself, no demo [V] | FID 8.70 [V] | CC BY-NC-SA, no LICENSE file [V] | Weights only on SharePoint. Marginal [A] |
| **FASHN v1.5** (2026) | ~1B pixel-space MMDiT [V] | Mask-free [V] | DWPose + own parser [V] | none published [V] | Apache-2.0 [V] | Documented CPU path [V]. Hosted run hit the quota |

**Classical and learned warping** (CP-VTON GMM, ACGPN, PF-AFN, GP-VTON) all need a trained network, and most have CUDA-only parts:
- **PF-AFN** needs cupy/CUDA 9 [V].
- **GP-VTON** scripts assume 8 GPUs [V].

The **classical** precedent we can use without training is VITON's own `shape_context_warp.m`: shape-context matching + TPS between the product mask and the person's clothing region [V]. That is what `tryon_core/tps.py` now implements, from scratch.

**What all the strong methods share:**
1. An **agnostic mask** that erases the old garment and the arms, so they can be regenerated.
2. **Implicit, attention-based warping** instead of a parametric warp.
3. A **generative model** that synthesises skin, folds and shading.

Item 1 is classical computer vision, which is where our code can contribute.

## 4. What we built and measured

| Method | What it is | Runs where |
|---|---|---|
| `classical` | Current pipeline (pose → our DLT → our inverse bilinear warp → sleeves → lighting → shading → our feathered blend) | CPU, ~0.2 s |
| `classical_plus` | + **TPS edge fit** (ours) + occlusion (face, hair, forearms in front) + no background spill + Telea inpainting of uncovered old top | CPU, ~0.5 s |
| `hybrid` | **Our pose + DLT warp footprint + parsing → try-on mask**. CatVTON renders inside it. **Our feathered blend** pastes it back at full resolution | Local MPS, ~29 s |
| `hybrid_warm70` | Same, but denoising starts from our classical composite (SDEdit, strength 0.7) | Local MPS, ~24 s |
| `idm` | IDM-VTON, hosted Space | Cloud. Only 2 of 4 pairs ran before the anonymous quota ran out |

**All 24 pairs:**

| | old_visible ↓ | spill_bg ↓ | spill_face ↓ | forearm_covered ↓ | iou_upper ↑ | ssim_face ↑ | old_unchanged ↓ | time |
|---|---|---|---|---|---|---|---|---|
| classical | 0.089 | 0.059 | 0.044 | 0.266 | 0.639 | 0.897 | 0.092 | 0.2 s |
| classical_plus | 0.075* | **0.000** | **0.000** | **0.000** | **0.757** | 0.992 | 0.036 | 0.5 s |
| hybrid | n/a | n/a | n/a | n/a | n/a | **1.000** | **0.035** | 29.8 s |

\*Pixels counted as "visible" here are inpainted, not old fabric. That is why `old_unchanged` falls to 0.036.

**4-pair subset with the hosted model:**

| | ssim_keep ↑ | ssim_face ↑ | garment_hist ↑ | old_unchanged ↓ |
|---|---|---|---|---|
| classical | 0.997 | 0.916 | 0.390 | 0.079 |
| classical_plus | 0.997 | 0.993 | 0.415 | 0.011 |
| **hybrid** | **0.997** | **1.000** | 0.457 | 0.013 |
| hybrid_warm70 | 0.996 | 1.000 | 0.387 | 0.022 |
| IDM-VTON (n=2) | 0.934 | 0.784 | **0.781** | **0.009** |

Figure: `results/figures/comparison_subset.jpg`. Full grids: `grid_classical.jpg`, `grid_classical_plus.jpg`, `grid_hybrid.jpg`.

**Reading the results:**
- **classical_plus** removes every *geometric* error the audit measured: spill, face coverage, forearm coverage, and +0.12 IoU. It is still visibly a flat overlay. Where the old top is not covered (long sleeves under a tee), Telea inpainting smears background colour into the arm. **A classical method cannot synthesise an arm.** That is the ceiling of the geometric approach.
- **hybrid** is the first result that looks like a photo:
  - real bare arms under short sleeves, collars on the neck, folds, no logo bleed;
  - identity and background are *exactly* preserved (face SSIM 1.000), because our feathered blend only replaces pixels inside our mask.
  - **Weaknesses at 512×384:** fine patterns are lost (the gingham check becomes plain pink) and small details disappear (the tie-waist knot).
  - **Failure 1, old necklines leak** (about 10 of 24 pairs; see `grid_hybrid.jpg`). A band of the old collar next to the neck stays outside our mask: the parser has no neck label, and subtracting the dilated face/hair removes that band. The model then keeps the old collar, for example the black band on 00035/00121 or the orange V edge on 01992. This is a mask bug, not a model limit (plan step 3).
  - **Failure 2, small full-body subjects** (stock_man): the torso fills only a small part of the 512×384 crop, so colours drift (the red tee comes out brown-orange, pulled by the old mustard sweater).
- **Warm start is worse:** it inherits classical artefacts, such as the old sleeves on 00035. Our geometry helps most as a **placement and mask prior**, not as an image prior. This is a useful negative result for the report.
- **IDM-VTON** keeps the most garment detail (the knot, highest colour match), but it **regenerates the whole picture**: face SSIM 0.78, and the background changes. It also needs the cloud and runs out of free quota after ~4 images a day.

## 5. Recommended architecture

```
                 ┌──────────── live mode (≥ 20 FPS) ─────────────┐
camera/photo ─► MediaPipe pose ─► OUR DLT homography ─► OUR inverse bilinear warp ─► TPS edge fit (ours)
                     │                                       │
                     ▼                                       ▼
            parsing (SegFormer / MediaPipe            occlusion + spill clip ─► lighting/shading ─► OUR feathered blend
             multiclass segmenter)                           (instant preview)
                     │
                     ▼  studio render (~30 s, local)
            OUR try-on mask = old top ∪ warped footprint ∪ arm corridors − face/hair − lower body
                     │
                     ▼
            CatVTON (pretrained, 512×384, MPS) inside the mask ─► OUR feathered blend paste-back
```

| Component | Decision | Why |
|---|---|---|
| DLT homography, inverse bilinear warp, feathered blend | **Keep** (core, tested against OpenCV) | Demonstrable course algorithms. Used in both modes |
| MediaPipe pose | **Keep** | Not the bottleneck (confidence ≈ 1.0) |
| Sleeve affine | **Improve**: clip to the arm region, put forearms in front | Removes the shards and forearm coverage |
| TPS edge fit, occlusion, spill clip | **Add** to live mode | Measured: spill and forearm errors → 0, IoU +0.12, ~0.3 s |
| Human parsing | **Add**: SegFormer for photos; MediaPipe multiclass segmenter for live (faster, already a dependency) [A] | Needed for occlusion and the agnostic mask |
| Studio render | **Replace** hosted IDM-VTON with **local hybrid CatVTON** | No quota or account, ~30 s, perfect identity preservation. Our geometry decides where the model may draw |
| Hosted Leffa/IDM | **Keep as optional** "max detail" with an `HF_TOKEN` | Best detail, but cloud-only and changes identity |
| DensePose, training, StableVITON/OOTDiffusion locally | **Don't** | CUDA builds, swapping on 16 GB, deadline |

**MATLAB:** stays the reference implementation of the core algorithms (DLT, inverse warp, blend), with its own tests. R2023b's Python bridge supports Python 3.9–3.11 [A: verify], and our venv is 3.12. If the MATLAB app needs the studio render, call the FastAPI server over HTTP (`webwrite`) rather than embedding PyTorch.

## 6. Implementation plan

1. **Classical fixes into the app** (½ day): move occlusion, spill clipping and TPS from `hybrid/classical_plus.py` into `tryon_core/pipeline.py` behind options. Use MediaPipe's multiclass segmenter for live speed. Add unit tests (TPS exact at control points; occlusion keeps forearms).
2. **Studio render → local hybrid** (½–1 day):
   - load CatVTON once at server start (~4 GB);
   - run it in a worker thread with a progress callback;
   - keep the hosted model as a fallback.
3. **Mask quality** (½ day). This is the biggest remaining quality issue in the hybrid:
   - add the old garment's neckline to the mask (subtract only the face, not a dilated face/hair band; or add a neck label from MediaPipe's multiclass segmenter or SCHP-LIP);
   - crop tighter around the torso for full-body photos;
   - take the old garment's hem from parsing;
   - add a per-garment-type mask (sleeveless vs long sleeve).
   - Re-run `eval/run_hybrid.py` after each change.
4. **Detail at higher resolution** (experiment, ½ day): try CatVTON's 1024×768 `mix` checkpoint with attention slicing. Measure time and memory. [A: ~2–4 min/image; possible memory pressure]
5. **Report** (Week 13): use the tables and figures above.
   - Root-cause analysis → Methodology.
   - Comparison → Related Work.
   - Warm-start result → Discussion.
   - Hosted vs local → Limitations.

## 7. Reproduce

```bash
cd webapp
.venv/bin/python -m eval.run_classical          # baseline, 24 pairs, metrics + overlays
.venv/bin/python -m eval.run_plus               # classical + TPS/occlusion
.venv/bin/python -m eval.run_hybrid 1.0         # local CatVTON hybrid (~12 min)
.venv/bin/python -m eval.run_hybrid 0.7 --subset
HF_TOKEN=hf_... .venv/bin/python -m eval.run_hosted idm leffa fashn   # hosted models (needs quota)
.venv/bin/python -m eval.score_outputs classical classical_plus hybrid idm --subset
.venv/bin/python -m eval.make_figures
```

**What was not done:**
- Leffa, FASHN and IDM (2 of 4 pairs) were not run: the anonymous ZeroGPU quota ran out. The FitDiT Space failed upstream on every call.
- With an HF token, `eval/run_hosted.py` resumes where it stopped.

**Licensing:** CatVTON and IDM-VTON weights, the SegFormer parser and the VITON-HD images are all non-commercial or research-only. That is fine for the course; cite them.
