function [outFrame, dbg] = tryOnPipeline(webcamFrame, poseKeypoints, garmentImg, garmentAlpha, cfg, bodyMask)
%TRYONPIPELINE  One frame of the virtual try-on.
%
%   outFrame        = tryOnPipeline(webcamFrame, poseKeypoints, garmentImg, garmentAlpha)
%   [outFrame, dbg] = tryOnPipeline(..., cfg)
%   [outFrame, dbg] = tryOnPipeline(..., cfg, bodyMask)
%
%   STAGES
%     1. torso warp        warpGarment          (piecewise affine, 4 keypoints)
%     2. body-shape fit    fitGarmentToBody     (only if bodyMask is given)
%     3. sleeves           warpSleeve           (rotate with the upper arms,
%                                                only if elbows are detected)
%     4. lighting match    matchGarmentLighting (global brightness/contrast)
%     5. shading transfer  transferBodyShading  (folds + body roundness)
%     6. feathered blend   blendGarmentOntoFrame (from scratch)
%   Every stage can be switched off in cfg (see defaultTryOnConfig).
%
%   INPUTS
%     webcamFrame    H x W x 3 frame from the camera (uint8 typically).
%     poseKeypoints  Output of the (external, pretrained) pose estimator; see
%                    getTorsoKeypoints for the accepted formats.
%     garmentImg     garment RGB image (e.g. from [img,~,alpha] = imread(png)).
%     garmentAlpha   garment alpha channel, same height/width as garmentImg.
%     cfg            (optional) parameter struct, see defaultTryOnConfig.
%     bodyMask       (optional) logical person mask, e.g. from
%                    segmentPersonBackground. Without it stage 2 is skipped.
%
%   OUTPUTS
%     outFrame   composited frame, same size/class as webcamFrame. If the
%                torso is not detected (or the pose is degenerate) the
%                original frame is returned unchanged.
%     dbg        struct with all intermediate results, for the demo / debug.
%
%   See also defaultTryOnConfig, liveTryOn.

if nargin < 5 || isempty(cfg)
    cfg = defaultTryOnConfig();
end
if nargin < 6
    bodyMask = [];
end
cfg = fillMissingSections(cfg);

outFrame = webcamFrame;               % fallback result: frame untouched
dbg = struct('ok', false, 'reason', '');
frameSize = size(webcamFrame);

%% ---- 0. Keypoints ---------------------------------------------------------------
torso = getTorsoKeypoints(poseKeypoints, frameSize, cfg.pose.minScore);
dbg.torso = torso;
if ~torso.valid
    dbg.reason = torso.reason;
    return
end

%% ---- 1. Point correspondences + torso warp ------------------------------------
anchorsPx = anchorsToPixels(cfg.garment.anchors, size(garmentImg));
[srcPts, dstPts] = buildCorrespondences(torso, anchorsPx, cfg.warp.mode);
dbg.srcPts = srcPts;
dbg.dstPts = dstPts;

% Which sleeves get their own (arm-following) transform?
doLeft  = cfg.sleeves.enable && torso.hasLeftElbow  && isfield(anchorsPx, 'leftSleeveEnd');
doRight = cfg.sleeves.enable && torso.hasRightElbow && isfield(anchorsPx, 'rightSleeveEnd');
torsoAlphaG = im2double(garmentAlpha);
if size(torsoAlphaG, 3) > 1, torsoAlphaG = torsoAlphaG(:, :, 1); end
if doLeft || doRight
    parts = splitGarmentSleeves(torsoAlphaG, anchorsPx);
    % A sleeve that is NOT articulated stays part of the torso layer.
    torsoAlphaG = parts.torso;
    if ~doLeft,  torsoAlphaG = max(torsoAlphaG, parts.leftSleeve);  end
    if ~doRight, torsoAlphaG = max(torsoAlphaG, parts.rightSleeve); end
end

try
    [layerRGB, layerAlpha, warpInfo] = warpGarment(garmentImg, torsoAlphaG, srcPts, dstPts, frameSize);
catch err
    % e.g. a degenerate triangle when the person stands exactly side-on:
    % skip the overlay for this frame instead of crashing the live app.
    dbg.reason = err.message;
    return
end
dbg.warpedRGB   = layerRGB;
dbg.warpedAlpha = layerAlpha;
dbg.warpInfo    = warpInfo;

%% ---- 2. Fit the torso part to the body silhouette -----------------------------
dbg.fitInfo = struct('ok', false);
if cfg.fit.enable && ~isempty(bodyMask)
    [layerRGB, layerAlpha, dbg.fitInfo] = fitGarmentToBody(layerRGB, layerAlpha, bodyMask, torso, cfg.fit);
end

%% ---- 3. Sleeves follow the upper arms -------------------------------------------
% Garment -> frame scale from the shoulder width, for the sleeve length.
scale = norm(torso.leftShoulder - torso.rightShoulder) / ...
        max(norm(anchorsPx.leftShoulder - anchorsPx.rightShoulder), eps);
sides = {'left', 'right'};
doSide = [doLeft, doRight];
for i = 1:2
    if ~doSide(i), continue; end
    s = sides{i};
    try
        [sRGB, sAlpha] = warpSleeve(garmentImg, parts.([s 'Sleeve']), ...
            anchorsPx.([s 'Shoulder']), anchorsPx.([s 'SleeveEnd']), ...
            torso.([s 'Shoulder']), torso.([s 'Elbow']), scale, frameSize);
    catch
        continue                             % elbow on the shoulder: skip sleeve
    end
    % Sleeve drawn OVER the torso layer ("over" operator on the two layers).
    layerRGB   = sRGB .* sAlpha + layerRGB .* (1 - sAlpha);
    layerAlpha = sAlpha + layerAlpha .* (1 - sAlpha);
end

%% ---- 4. Global lighting match ---------------------------------------------------
torsoRoi = torsoBoundingBox(dstPts, frameSize, cfg.lighting.roiShrink);
lightStats = [];
if cfg.lighting.enable
    torsoCrop = webcamFrame(torsoRoi(1):torsoRoi(2), torsoRoi(3):torsoRoi(4), :);
    [layerRGB, lightStats] = matchGarmentLighting(layerRGB, torsoCrop, layerAlpha, cfg.lighting);
end
dbg.litRGB = layerRGB;

%% ---- 5. Shading transfer (folds, roundness) ------------------------------------
shade = [];
if cfg.shading.enable
    [layerRGB, shade] = transferBodyShading(layerRGB, layerAlpha, webcamFrame, cfg.shading);
end

%% ---- 6. Feathered alpha blend onto the frame --------------------------------
if nargout > 1
    [outFrame, finalAlpha, blendDbg] = blendGarmentOntoFrame(webcamFrame, layerRGB, layerAlpha, cfg.blend);
else
    outFrame = blendGarmentOntoFrame(webcamFrame, layerRGB, layerAlpha, cfg.blend);
end

%% ---- Debug output -------------------------------------------------------------
if nargout > 1
    dbg.ok          = true;
    dbg.layerRGB    = layerRGB;          % final garment layer before blending
    dbg.layerAlpha  = layerAlpha;
    dbg.torsoRoi    = torsoRoi;          % [rowStart rowEnd colStart colEnd]
    dbg.lightStats  = lightStats;
    dbg.shade       = shade;
    dbg.finalAlpha  = finalAlpha;
    dbg.blend       = blendDbg;
    dbg.sleeves     = doSide;
end
end

% ==========================================================================
function anchorsPx = anchorsToPixels(anchors, garmentSize)
% Normalised [0,1] anchor positions -> garment pixel coordinates.
gH = garmentSize(1);
gW = garmentSize(2);
names = fieldnames(anchors);
for i = 1:numel(names)
    p = anchors.(names{i});
    anchorsPx.(names{i}) = [1 + p(1) * (gW - 1), 1 + p(2) * (gH - 1)];
end
end

function [srcPts, dstPts] = buildCorrespondences(torso, a, mode)
% Pair garment anchors with body keypoints in the order warpGarment expects.
switch lower(mode)
    case 'piecewise'
        % Around the quad: LS, RS, RH, LH (see warpGarment).
        srcPts = [a.leftShoulder; a.rightShoulder; a.rightHip; a.leftHip];
        dstPts = [torso.leftShoulder; torso.rightShoulder; torso.rightHip; torso.leftHip];
    case 'affine'
        % One triangle: both shoulders + the midpoint of the hips.
        srcPts = [a.leftShoulder; a.rightShoulder; (a.leftHip + a.rightHip) / 2];
        dstPts = [torso.leftShoulder; torso.rightShoulder; (torso.leftHip + torso.rightHip) / 2];
    otherwise
        error('tryOnPipeline:mode', 'Unknown cfg.warp.mode "%s".', mode);
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

function cfg = fillMissingSections(cfg)
% Configs saved from an older version may lack the newer sections: take
% those from the defaults so old App Designer code keeps working.
defaults = defaultTryOnConfig();
sections = fieldnames(defaults);
for i = 1:numel(sections)
    if ~isfield(cfg, sections{i})
        cfg.(sections{i}) = defaults.(sections{i});
    end
end
end
