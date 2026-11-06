# Visionary Try-On: web app (Python)

Two ways to dress a person, both running locally on the Mac. No hosted service is needed.

| Mode | How | Speed (M1 Pro) | Looks |
|---|---|---|---|
| **Instant** (photo + live camera) | MediaPipe pose → **our DLT homography** → **our inverse bilinear warp** → sleeves follow arms → **our TPS edge fit** onto the person's clothing outline → occlusion (face, hair, forearms stay in front) → lighting, shading → **our feathered blend** | live, see `eval/RESULTS.md` for FPS | aligned overlay |
| **Studio render** (photo; in live mode it takes the current frame) | **Hybrid**: our pose, homography/TPS footprint and parsing give the garment's own coverage (length calibrated on VITON-HD ground truth) and the old clothing it exposes (see `eval/COVERAGE.md`). **CatVTON** (pretrained diffusion, run locally on the Apple GPU) renders the garment inside it. **Our feathered blend** pastes it back, so pixels outside the garment area stay the original photo | ~30–45 s per render (512×384, median 40 s in the benchmark) | photorealistic |

The MATLAB code in `../tryon` stays the course implementation of the core algorithms. Its app can
request a studio render from this server (see *MATLAB* below).

## Run
```bash
cd webapp
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn server:app --port 8000
```
Open http://localhost:8000.

First-start downloads (cached afterwards):
- MediaPipe pose and segmentation models (~15 MB, `models/`)
- the SegFormer clothes parser (~110 MB)
- rembg (~170 MB, on the first garment upload)
- the first studio render: SD-1.5-inpainting UNet fp16 (~1.7 GB), the VAE (~330 MB) and the CatVTON attention weights (~200 MB). Peak GPU memory is about 5.5 GB

After that, nothing leaves the machine.

## Garments
`data/catalog/` holds 9 front-facing product photos from the VITON-HD dataset (Choi et al., CVPR 2021,
the Zalando set linked in the proposal), taken from the IDM-VTON examples. They are for research and
non-commercial use, so cite VITON-HD in the report.
- Branded and cartoon prints and bodysuits were left out.
- To add more, put photos in `data/catalog/`, add their names to `catalog.json`, and delete `data/garments/` so it is rebuilt.
- Or use *Upload* in the app.

## Using it
- **Photo**: drop a photo of yourself (facing the camera, shoulders to hips visible), then pick a
  garment. The instant render appears at once. The sliders and switches update it, including
  *Edge fit (thin-plate spline)* and *Face, hair & hands in front*.
- **Studio render**: the progress shown is the real progress of the diffusion steps. The first render
  also loads the model (a few seconds). An experimental 768×576 "detail" level exists in the API only
  (`quality: "detail"`, ~2.2 min). In our benchmark it shifted colours, so the app does not offer it.
- **Live mirror**: *Start camera* runs the instant pipeline on the webcam. Body parsing is recomputed
  every 3rd frame. *Studio render* takes the current frame.
- **Pipeline view**: the output of every stage (pose and torso, homography warp, occlusion mask,
  light and folds, feathered alpha, composite), for the report and the demo.
- **Add your shirt**: any front-facing product photo or transparent PNG. The background is removed
  with rembg (IS-Net). The original photo is kept for the studio render.

## MATLAB
MATLAB R2023b on this Mac is an Intel (x86_64) app running under Rosetta, and this venv's PyTorch is
arm64-only. The Apple-GPU back end needs a native arm64 process, so the model cannot run inside
MATLAB through `py.*`.

Instead, `tryon/studioRender.m` sends the photo and garment to this server with `webwrite`, as base64
JSON to `POST /api/studio/render`. In `TryOnApp`, the *Studio render* button does this. Start the
server first.

## Hosted alternative (off by default)
`TRYON_HOSTED=1` re-enables `/api/hd-tryon` (IDM-VTON on Hugging Face or Replicate; see `ai_tryon.py`).
It sends photos to a third party and is limited by quota, so the app does not use it.

## Evaluation
`eval/` holds a fixed benchmark: 6 people × 4 garments, with the same inputs for every method. It has
metric scripts and figure generation. See `eval/AUDIT.md` (research and root-cause analysis),
`eval/RESULTS.md` (the integrated pipeline), `eval/DEPTH.md` (3D information) and `eval/COVERAGE.md`
(garment length and coverage, cropped-garment test set, VITON-HD ground truth in `eval/vitonhd_gt/`).

## Files
| File | Purpose |
|---|---|
| `server.py` | FastAPI: REST endpoints, studio render jobs, live WebSocket, static files |
| `tryon_core/homography.py` | **From scratch**: normalised DLT homography (SVD) |
| `tryon_core/warp.py` | **From scratch**: inverse warping with bilinear interpolation |
| `tryon_core/tps.py` | **From scratch**: thin-plate spline (fit, evaluate) and bilinear remap |
| `tryon_core/blend.py` | **From scratch**: disk erosion, separable Gaussian convolution, alpha compositing |
| `tryon_core/pipeline.py` | All instant-render stages for one frame (port of `tryOnPipeline.m`, plus TPS and occlusion) |
| `tryon_core/parsing.py` | Body/clothing parsing: SegFormer (photos), MediaPipe multiclass segmenter (live) |
| `tryon_core/pose.py` | MediaPipe Pose Landmarker: body and face keypoints, person mask |
| `tryon_core/garment.py` | Garment cut-out and automatic anchors |
| `hybrid/pipeline.py` | Studio render: try-on mask, crop, colour correction, paste-back |
| `hybrid/catvton.py` | Minimal CatVTON inference for Apple MPS (from the CatVTON code, CC BY-NC-SA) |
| `ai_tryon.py` | Optional hosted IDM-VTON |
| `static/` | The web page (HTML, CSS, JS; no build step) |
| `tests/test_core.py` | From-scratch parts vs OpenCV, TPS, occlusion and studio mask checks: `.venv/bin/python -m tests.test_core` |
| `eval/` | Benchmark, metrics, figures, reports |

## Licences
CatVTON (code and weights), IDM-VTON, the SegFormer clothes parser and the VITON-HD images are for
research and non-commercial use only. SD-1.5 is CreativeML OpenRAIL-M. All of this is fine for the
course; cite them.
