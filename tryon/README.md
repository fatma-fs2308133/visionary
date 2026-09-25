# Virtual try-on: Phase 2 core algorithms (MATLAB)

These are standalone functions that you can test one at a time and then call from the App Designer UI.
Each frame goes through: **pose keypoints → warp → light-match → feathered blend**.

| File | Purpose |
|---|---|
| `tryOnPipeline.m` | Top-level function for one frame. Calls the stages below in order. |
| `warpGarment.m` | Affine warp (3 points) or piecewise-affine warp (4 points split into 2 triangles), using `fitgeotrans` + `imwarp`. |
| `blendGarmentOntoFrame.m` | **From-scratch algorithm.** Erodes the mask, feathers it with a hand-built Gaussian kernel, then does manual alpha compositing. |
| `matchGarmentLighting.m` | Compares HSV V-channel statistics of the torso and the garment, then applies `imadjust` to the garment RGB. |
| `getTorsoKeypoints.m` | Adapter for pose-model output: a struct, COCO-17, OpenPose-18/25 or MediaPipe-33 (pixel or normalised coordinates). |
| `defaultTryOnConfig.m` | Holds every tunable parameter, each with a comment explaining the assumption behind it. |
| `liveTryOn.m` | Live webcam loop: camera → pose (HRNet or manual clicks) → pipeline → window. |
| `demo_tryOn.m` | Visual demo that runs stage by stage on a synthetic frame. You can also switch it to your own photo and click the keypoints with `ginput`. |
| `test_tryOn.m` | 18 automated checks that print PASS/FAIL. |
| `makeSyntheticTestData.m`, `makeTestPose.m` | Build the fake frame, keypoints, garment and pose variants (tilt / lean / turn / closer). |

## Quick start
```matlab
cd tryon
liveTryOn               % LIVE webcam try-on (synthetic shirt); press Q to stop
liveTryOn('shirt.png')  % live, with your own transparent garment PNG
test_tryOn              % numeric checks
demo_tryOn              % static figures 1-6, one per stage (no camera)
```
`liveTryOn` needs the **MATLAB Support Package for USB Webcams**. For automatic pose it
uses `hrnetObjectKeypointDetector` (Computer Vision + Deep Learning Toolbox, R2023b+,
"Computer Vision Toolbox Model for Object Keypoint Detection" add-on); without it, it falls
back to clicking your shoulders/hips once. Stand far enough back that your **hips are in view**.

## Using it in App Designer
```matlab
% startupFcn
app.Cfg = defaultTryOnConfig();
[app.GarmentRGB, ~, app.GarmentAlpha] = imread('shirt.png');
% timer callback
frame = snapshot(app.Cam);
kp    = myPoseModel(frame);                 % external pretrained model
out   = tryOnPipeline(frame, kp, app.GarmentRGB, app.GarmentAlpha, app.Cfg);
app.Img.CData = out;
% slider callbacks, e.g.
app.Cfg.blend.gaussSigma = app.SigmaSlider.Value;
```

## Key judgement calls
All of these are documented in the code, so you can change them.
- **Piecewise split:** the quad is cut along the diagonal from the left shoulder to the right hip. Each triangle's affine transform then covers its whole half-plane, so the sleeves and collar outside the quad still get warped. The two transforms agree along the diagonal, so there is no seam.
- **Blend defaults:** erosion radius 3 px, Gaussian sigma 2 px (kernel 13×13), mask threshold 0.5. These are tuned for 640×480 input; roughly double them at 720p.
- **Lighting strength 0.6:** the torso region also includes the colour of the shirt the user is really wearing. Matching 100% would turn a white garment grey on someone dressed in black.
- **Garment anchors:** stored as normalised positions on the garment PNG (for a front-facing shirt, the wearer's left is on the image right). Adjust them for each garment; figure 1 of the demo shows where they sit.
