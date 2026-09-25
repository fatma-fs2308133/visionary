function [outFrame, dbg] = tryOnPipeline(webcamFrame, poseKeypoints, garmentImg, garmentAlpha, cfg)
%TRYONPIPELINE  One frame of the virtual try-on: warp -> light-match -> blend.
%
%   outFrame        = tryOnPipeline(webcamFrame, poseKeypoints, garmentImg, garmentAlpha)
%   [outFrame, dbg] = tryOnPipeline(..., cfg)
%
%   INPUTS
%     webcamFrame    H x W x 3 frame from the camera (uint8 typically).
%     poseKeypoints  Output of the (external, pretrained) pose estimator; see
%                    getTorsoKeypoints for the accepted formats (struct with
%                    leftShoulder/rightShoulder/leftHip/rightHip, or a COCO /
%                    OpenPose / MediaPipe keypoint array).
%     garmentImg     garment RGB image (e.g. from [img,~,alpha] = imread(png)).
%     garmentAlpha   garment alpha channel, same height/width as garmentImg.
%     cfg            (optional) parameter struct, see defaultTryOnConfig.
%
%   OUTPUTS
%     outFrame   composited frame, same size/class as webcamFrame. If the
%                torso is not detected (or the pose is degenerate) the
%                original frame is returned unchanged.
%     dbg        struct with all intermediate results (keypoints used, warped
%                garment, masks, lighting statistics, ...) for the demo and
%                for debug views in App Designer.
%
%   APP DESIGNER USAGE (e.g. inside a timer callback)
%     frame = snapshot(app.Cam);
%     kp    = <your pose model>(frame);
%     out   = tryOnPipeline(frame, kp, app.GarmentRGB, app.GarmentAlpha, app.Cfg);
%     app.ImageHandle.CData = out;          % faster than imshow every frame
%
%   See also defaultTryOnConfig, warpGarment, matchGarmentLighting,
%   blendGarmentOntoFrame, getTorsoKeypoints.

if nargin < 5 || isempty(cfg)
    cfg = defaultTryOnConfig();
end

outFrame = webcamFrame;               % fallback result: frame untouched
dbg = struct('ok', false, 'reason', '');

%% ---- 0. Torso keypoints -------------------------------------------------------
torso = getTorsoKeypoints(poseKeypoints, size(webcamFrame), cfg.pose.minScore);
dbg.torso = torso;
if ~torso.valid
    dbg.reason = torso.reason;
    return
end

%% ---- 1. Point correspondences garment <-> body ------------------------------
[srcPts, dstPts] = buildCorrespondences(torso, size(garmentImg), cfg);
dbg.srcPts = srcPts;
dbg.dstPts = dstPts;

%% ---- 2. Warp the garment into frame coordinates -----------------------------
try
    [warpedRGB, warpedAlpha, warpInfo] = warpGarment(garmentImg, garmentAlpha, ...
                                                     srcPts, dstPts, size(webcamFrame));
catch err
    % e.g. a degenerate triangle when the person stands exactly side-on:
    % skip the overlay for this frame instead of crashing the live app.
    dbg.reason = err.message;
    return
end

%% ---- 3. Match the garment's lighting to the person's torso ------------------
torsoRoi = torsoBoundingBox(dstPts, size(webcamFrame), cfg.lighting.roiShrink);
if cfg.lighting.enable
    torsoCrop = webcamFrame(torsoRoi(1):torsoRoi(2), torsoRoi(3):torsoRoi(4), :);
    [litRGB, lightStats] = matchGarmentLighting(warpedRGB, torsoCrop, warpedAlpha, cfg.lighting);
else
    litRGB = warpedRGB;
    lightStats = [];
end

%% ---- 4. Feathered alpha blend onto the frame --------------------------------
if nargout > 1
    [outFrame, finalAlpha, blendDbg] = blendGarmentOntoFrame(webcamFrame, litRGB, warpedAlpha, cfg.blend);
else
    outFrame = blendGarmentOntoFrame(webcamFrame, litRGB, warpedAlpha, cfg.blend);
end

%% ---- Debug output -------------------------------------------------------------
if nargout > 1
    dbg.ok          = true;
    dbg.warpedRGB   = warpedRGB;
    dbg.warpedAlpha = warpedAlpha;
    dbg.warpInfo    = warpInfo;
    dbg.litRGB      = litRGB;
    dbg.torsoRoi    = torsoRoi;          % [rowStart rowEnd colStart colEnd]
    dbg.lightStats  = lightStats;
    dbg.finalAlpha  = finalAlpha;
    dbg.blend       = blendDbg;
end
end

% ==========================================================================
function [srcPts, dstPts] = buildCorrespondences(torso, garmentSize, cfg)
% Pair the garment anchor points (normalised -> garment pixels) with the
% body keypoints, in the point order warpGarment expects.
gH = garmentSize(1);
gW = garmentSize(2);
anc = cfg.garment.anchors;
toPx = @(p) [1 + p(1) * (gW - 1), 1 + p(2) * (gH - 1)];

gLS = toPx(anc.leftShoulder);   gRS = toPx(anc.rightShoulder);
gLH = toPx(anc.leftHip);        gRH = toPx(anc.rightHip);

switch lower(cfg.warp.mode)
    case 'piecewise'
        % Around the quad: LS, RS, RH, LH (see warpGarment).
        srcPts = [gLS; gRS; gRH; gLH];
        dstPts = [torso.leftShoulder; torso.rightShoulder; torso.rightHip; torso.leftHip];
    case 'affine'
        % One triangle: both shoulders + the midpoint of the hips.
        srcPts = [gLS; gRS; (gLH + gRH) / 2];
        dstPts = [torso.leftShoulder; torso.rightShoulder; (torso.leftHip + torso.rightHip) / 2];
    otherwise
        error('tryOnPipeline:mode', 'Unknown cfg.warp.mode "%s".', cfg.warp.mode);
end
end

function roi = torsoBoundingBox(dstPts, frameSize, shrink)
% Bounding box of the torso keypoints, shrunk toward its centre so that it
% contains mostly the person's body (little background), clamped to the
% frame. Returned as [rowStart rowEnd colStart colEnd].
x = dstPts(:, 1);
y = dstPts(:, 2);
dx = (max(x) - min(x)) * shrink;
dy = (max(y) - min(y)) * shrink;
c1 = round(min(x) + dx);   c2 = round(max(x) - dx);
r1 = round(min(y) + dy);   r2 = round(max(y) - dy);

c1 = min(max(c1, 1), frameSize(2));   c2 = min(max(c2, 1), frameSize(2));
r1 = min(max(r1, 1), frameSize(1));   r2 = min(max(r2, 1), frameSize(1));
if c2 < c1, [c1, c2] = deal(c2, c1); end
if r2 < r1, [r1, r2] = deal(r2, r1); end
roi = [r1 r2 c1 c2];
end
