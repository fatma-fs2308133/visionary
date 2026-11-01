# Virtual try-on (MATLAB)

Upload a photo of yourself (or take one with the webcam) and a picture of a garment: the system
finds your shoulders, hips and elbows and dresses you in the garment. A live-video mode runs the
same processing on the webcam stream. Each photo / frame goes through:

**preprocess → pose (HRNet + KLT tracking) → person mask → homography warp (from scratch) →
body-shape fit → sleeves follow arms → lighting match → shading transfer → feathered blend (from scratch)**

## Quick start
```matlab
cd tryon
TryOnApp            % the GUI
test_tryOn          % 37 automated checks, prints PASS/FAIL
demo_tryOn          % figures 1-7, one per stage (no camera needed)
runExperiments      % all numbers + charts for the report (results/)
liveTryOn           % lightweight live version with keyboard controls
```

### Using the app
1. **Your photo**: *Upload photo...* (facing the camera, shoulders to hips visible), or
   *Take photo (webcam)*. That takes ONE snapshot after a countdown; with *Also capture empty
   background* ticked it first photographs the scene without you, which enables the body-shape fit.
   For an uploaded photo you can add such a picture with *Background photo (optional)...*.
2. **Garment**: pick one, or *Upload garment picture...*: a flat, front-facing product photo on a
   plain background, or a transparent PNG. The background is removed and the garment points are
   found automatically (*Fix garment points...* if it sits badly).
3. **Body points**: found automatically with HRNet (if installed); otherwise a window opens and you
   click 6 points: shoulders, hips, elbows.
4. Adjust the warp model / stages / sliders, then *Save result* (saved in `results/tryon/`).

### Add-ons to install (Home → Add-Ons)
| Add-on | Needed for |
|---|---|
| **MATLAB Support Package for USB Webcams** | *Take photo* and live video (`webcam`, `snapshot`) |
| **Computer Vision Toolbox Model for Object Keypoint Detection** | automatic pose (`hrnetObjectKeypointDetector`) |

Without them the app still works with uploaded photos: you click the 6 body points yourself.

On an Apple-silicon Mac, install the **Apple silicon** build of MATLAB. The Intel build runs under
Rosetta emulation and is noticeably slower.

## Files
| File | Purpose |
|---|---|
| **Core algorithms** | |
| `estimateHomographyDLT.m` | **From scratch.** Normalised Direct Linear Transform: builds the 2N×9 matrix A and solves Ah = 0 with the SVD. |
| `inverseWarpBilinear.m` | **From scratch.** Inverse mapping through H⁻¹ plus hand-written bilinear interpolation. |
| `applyHomography.m` | Maps points through a homography (homogeneous coordinates). |
| `blendGarmentOntoFrame.m` | **From scratch.** Erodes the mask, feathers it with a hand-built Gaussian kernel, then composites with manual alpha blending. |
| `warpGarment.m` | Places the garment in the frame: `homography` (our DLT, default), `piecewise` affine or `affine`. |
| `tryOnPipeline.m` | Processes one frame through every stage in order. Each stage can be switched off in the config. |
| `fitGarmentToBody.m`, `segmentPersonBackground.m` | Body-shape fit using a person mask from background subtraction. |
| `splitGarmentSleeves.m`, `warpSleeve.m` | Cut off the sleeves and rotate each one with its upper arm. |
| `matchGarmentLighting.m`, `transferBodyShading.m` | Match the garment's lighting to the scene, and copy the folds of the real clothes onto it. |
| `getTorsoKeypoints.m` | Reads pose output in COCO-17, OpenPose or MediaPipe format, or a struct. |
| `defaultTryOnConfig.m` | Every tunable parameter, with its assumption documented. |
| **Input / real time** | |
| `preprocessFrame.m` | Resizes to 640 px, denoises, then for dark or flat frames applies a percentile stretch plus CLAHE on luma. |
| `PoseTracker.m` | Runs HRNet every N frames (person box from ACF or the previous keypoints) and tracks with KLT in between. Manual mode is available. |
| `clickTorsoKeypoints.m` | Manual keypoints (6 clicks). |
| `TryOnSession.m` | The per-frame processing shared by the GUI, the live script and the experiments. |
| `TryOnApp.m` | The GUI: upload or take your photo, upload or pick a garment, automatic or clicked body points, photo and result side by side, person mask, status, warp model, stage toggles, sliders, save, optional live video. |
| `liveTryOn.m` | Keyboard version. Can log per-frame timings to a CSV file. |
| **Garments / dataset** | |
| `prepareGarment.m` | Turns a dataset garment photo (or a whole folder) into a transparent PNG with automatic anchors. Uses VITON-HD `cloth-mask` files when present. |
| `autoGarmentAnchors.m` | Estimates the 8 anchor points from the garment's silhouette. |
| `pickGarmentAnchors.m` | Click the 8 anchors yourself. This overrides the automatic ones. |
| `loadGarmentLibrary.m` | Loads every garment in `data/garments`. |
| `makeSampleGarments.m` | Creates 5 synthetic placeholder garments, so the app works before the dataset is ready. |
| `splitDataset.m` | Reproducible train/val/test split of the downloaded garment images. |
| **Evaluation** | |
| `captureTestFrames.m` | Records a test session: background, then N frames, as `data/test_frames/<person>_<condition>`. |
| `annotateTorso.m` | Hand-mark the torso area in every frame (ground truth). |
| `runExperiments.m` | E1 from-scratch validation, E2 detection rate, E3 torso IoU per garment and warp model, E4 speed, E5 failure cases. |
| `test_tryOn.m`, `demo_tryOn.m`, `makeSyntheticTestData.m`, `makeTestPose.m` | Tests, stage-by-stage figures, synthetic test data. |

## Workflow for the report
```matlab
% 1. Garments from the dataset (VITON-HD layout: .../cloth and .../cloth-mask)
splitDataset('downloads/zalando-hd/test/cloth', 'data/garment_split');
prepareGarment('data/garment_split/test/cloth', 'data/garments');   % then pick >= 5
pickGarmentAnchors('data/garments/00006_00.png');                  % optional, better fit

% 2. Test sessions: each member, in different conditions (with consent)
captureTestFrames('habiba_bright');   captureTestFrames('habiba_dim');
annotateTorso('data/test_frames/habiba_bright');
annotateTorso('data/test_frames/habiba_dim');

% 3. All numbers and charts
runExperiments          % -> results/summary.txt, *.csv, fig_*.png
demo_tryOn              % set saveFigures = true for the stage figures
```
`runExperiments` compares the results against the Phase 1 objectives: at least 10 fps, at least
90 % torso detection, mean torso IoU of at least 0.75, and at least 5 garments. Without annotated
sessions it runs a **synthetic smoke test** instead, which is not a result for the report.

## Live controls
**GUI (`TryOnApp`):** everything is a button, a checkbox or a slider. In live video, *Background
photo* starts a 3-second countdown: step out of view.

**`liveTryOn` keys:** B background · G next garment · W next warp model · F fit · S shading ·
A sleeves · L lighting · P preprocessing · M mask · K click keypoints again · Q quit.

## Key judgement calls
All of these are documented in the code, so you can change them.
- **Homography as the default warp.** It is one smooth projective map of the torso plane, so it
  handles turning and tilting with no seam. Piecewise affine has a kink along the diagonal, and a
  single affine can't narrow one side of the torso. `runExperiments` compares all three.
- **Detect + track.** HRNet is too slow on a CPU to run on every frame, so it runs every 5 frames
  and KLT tracks the keypoints in between. If tracking loses a torso point, HRNet runs again on
  the next frame.
- **Blend defaults:** erosion radius 3 px, Gaussian sigma 2 px, tuned for 640×480. Frames are
  resized to 640 px wide, so these hold for any camera.
- **Lighting strength 0.6:** the torso region measured in the frame also contains the colour of
  the shirt the user is really wearing.
- **Shape fit:** strength 0.8 plus 4 px "ease", starting at armpit height.
- **Shading:** strength 0.8. Works best when the real shirt underneath is plain and light.
- **Garment anchors:** stored as normalised positions. They are estimated automatically, and
  `pickGarmentAnchors` overrides them.
