function test_tryOn()
%TEST_TRYON  Automated sanity checks for every stage of the try-on pipeline.
%
%   Run:  >> test_tryOn
%
%   Non-visual checks (numbers, not pictures) that each function does what
%   it claims. Prints PASS/FAIL per check and errors at the end if anything
%   failed. Use demo_tryOn for the visual, stage-by-stage inspection.

addpath(fileparts(mfilename('fullpath')));
nFail = 0;

[frame, kp, garmentImg, garmentAlpha] = makeSyntheticTestData();
cfg = defaultTryOnConfig();
frameSize = size(frame);

%% ===== warpGarment ===========================================================
% 1. Identity: same source and destination points -> garment unchanged.
g  = im2double(garmentImg);
ga = im2double(garmentAlpha);
src = [300 60; 100 60; 130 390; 270 390];
[wRGB, wA] = warpGarment(g, ga, src, src, size(g));
% Compare without the outermost pixel ring: there a 1e-14 round-off in the
% fitted transform can put the sample just outside the image -> fill value.
in = @(I) I(2:end-1, 2:end-1, :);
dRGB = abs(in(wRGB) - in(g));
dA   = abs(in(wA) - in(ga));
nFail = nFail + check(max(dRGB(:)) < 1e-6 && max(dA(:)) < 1e-6, ...
    'warp: identity mapping leaves the garment unchanged');

% 2. The anchor points really land on the keypoints (both modes). A small
%    white dot is painted at every source anchor; its centroid after warping
%    must be at the matching destination point.
dst = [395 165; 245 165; 272 365; 368 365];      % LS, RS, RH, LH in the frame
for mode = {'piecewise', 'affine'}
    if strcmp(mode{1}, 'piecewise')
        s = src;  d = dst;
    else
        s = [src(1:2, :); mean(src(3:4, :))];
        d = [dst(1:2, :); mean(dst(3:4, :))];
    end
    maxErr = 0;
    for i = 1:size(s, 1)
        dot = zeros(size(ga));
        [gx, gy] = meshgrid(1:size(ga, 2), 1:size(ga, 1));
        dot((gx - s(i, 1)).^2 + (gy - s(i, 2)).^2 <= 3^2) = 1;
        [~, wDot] = warpGarment(g, dot, s, d, frameSize);
        [fx, fy] = meshgrid(1:frameSize(2), 1:frameSize(1));
        c = [sum(fx(:) .* wDot(:)), sum(fy(:) .* wDot(:))] / sum(wDot(:));
        maxErr = max(maxErr, norm(c - d(i, :)));
    end
    nFail = nFail + check(maxErr < 1.0, ...
        sprintf('warp (%s): anchors land on keypoints (max error %.2f px)', mode{1}, maxErr));
end

% 3. If the body motion is itself one affine transform, the piecewise warp
%    must equal the single affine warp (both triangles get the same
%    transform) -> proves the two halves are stitched without a seam.
t = deg2rad(10);
Aff = [1.1 * cos(t), -sin(t); sin(t), 0.9 * cos(t)];
dstAffine = src * Aff' + [150 -20];
[pRGB, pA] = warpGarment(g, ga, src, dstAffine, frameSize);
[aRGB, aA] = warpGarment(g, ga, [src(1:2, :); mean(src(3:4, :))], ...
                         [dstAffine(1:2, :); mean(dstAffine(3:4, :))], frameSize);
nFail = nFail + check(max(abs(pRGB(:) - aRGB(:))) < 1e-6 && max(abs(pA(:) - aA(:))) < 1e-6, ...
    'warp: piecewise == affine when the motion is affine (seamless stitching)');

% 4. For a turning torso the piecewise warp must NOT equal the affine warp.
kpTurn = makeTestPose(kp, 'turn');
dstTurn = [kpTurn.leftShoulder(1:2); kpTurn.rightShoulder(1:2); kpTurn.rightHip(1:2); kpTurn.leftHip(1:2)];
[~, pA] = warpGarment(g, ga, src, dstTurn, frameSize);
[~, aA] = warpGarment(g, ga, [src(1:2, :); mean(src(3:4, :))], ...
                      [dstTurn(1:2, :); mean(dstTurn(3:4, :))], frameSize);
nFail = nFail + check(mean(abs(pA(:) - aA(:))) > 1e-3, ...
    'warp: piecewise differs from affine for a turning torso');

% 5. Degenerate (collinear) points raise a clear error.
try
    warpGarment(g, ga, src(1:3, :), [0 0; 10 10; 20 20], frameSize);
    ok = false;
catch err
    ok = strcmp(err.identifier, 'warpGarment:degenerate');
end
nFail = nFail + check(ok, 'warp: collinear triangle is rejected');

%% ===== blendGarmentOntoFrame ==================================================
F  = im2double(frame);
Gf = repmat(reshape([0.1 0.9 0.2], 1, 1, 3), frameSize(1), frameSize(2));  % green garment
M  = zeros(frameSize(1), frameSize(2));
M(100:300, 200:400) = 1;                                                   % square alpha

% 6. Zero alpha -> frame returned untouched.
out = blendGarmentOntoFrame(frame, Gf, zeros(frameSize(1:2)), cfg.blend);
nFail = nFail + check(isequal(out, frame), 'blend: alpha = 0 leaves the frame untouched');

% 7. Output class and size match the input frame.
[out, alpha] = blendGarmentOntoFrame(frame, Gf, M, cfg.blend);
nFail = nFail + check(isa(out, 'uint8') && isequal(size(out), frameSize), ...
    'blend: output has the same size and class as the frame');

% 8. Deep inside the garment the result is exactly the garment; far outside
%    it is exactly the frame.
outD = im2double(out);
inside  = squeeze(outD(200, 300, :))';
outside = squeeze(outD(50, 50, :))';
nFail = nFail + check(norm(inside - [0.1 0.9 0.2]) < 2 / 255 && ...
                      norm(outside - squeeze(F(50, 50, :))') < 1e-9, ...
    'blend: opaque inside = garment, far outside = frame');

% 9. Feathering: alpha rises smoothly across the edge, stays in [0,1] and
%    never exceeds the original alpha (no bleeding outside the garment).
edgeProfile = alpha(200, 190:215);
nFail = nFail + check(all(diff(edgeProfile) >= -1e-12) && ...
                      numel(unique(round(edgeProfile * 100))) > 4 && ...
                      all(alpha(:) >= 0) && all(alpha(:) <= M(:) + 1e-12), ...
    'blend: soft, monotonic, non-bleeding edge');

% 10. The explicit per-pixel loop and the vectorised version are identical.
pLoop = cfg.blend;  pLoop.useLoops = true;
outLoop = blendGarmentOntoFrame(F, Gf, M, pLoop);
outVec  = blendGarmentOntoFrame(F, Gf, M, cfg.blend);
nFail = nFail + check(max(abs(outLoop(:) - outVec(:))) < 1e-12, ...
    'blend: loop implementation == vectorised implementation');

%% ===== matchGarmentLighting ===================================================
darkScene = im2double(frame(200:340, 280:360, :)) * 0.4;
[adj, st] = matchGarmentLighting(garmentImg, darkScene, garmentAlpha, cfg.lighting);

% 11. The garment's mean brightness moves toward the scene by ~strength.
expected = st.garmentMeanV + cfg.lighting.strength * (st.sceneMeanV - st.garmentMeanV);
nFail = nFail + check(st.adjustedMeanV < st.garmentMeanV && abs(st.adjustedMeanV - expected) < 0.03, ...
    sprintf('lighting: mean V %.2f -> %.2f (scene %.2f, expected ~%.2f)', ...
            st.garmentMeanV, st.adjustedMeanV, st.sceneMeanV, expected));

% 12. strength = 0 leaves the garment unchanged.
p0 = cfg.lighting;  p0.strength = 0;
adj0 = matchGarmentLighting(garmentImg, darkScene, garmentAlpha, p0);
nFail = nFail + check(max(abs(adj0(:) - im2double(garmentImg(:)))) < 1e-6, ...
    'lighting: strength 0 is the identity');

% 13. Output stays a valid image.
nFail = nFail + check(isequal(size(adj), size(garmentImg)) && all(adj(:) >= 0 & adj(:) <= 1), ...
    'lighting: output is a valid [0,1] RGB image of the same size');

%% ===== getTorsoKeypoints ======================================================
% 14. COCO-17 array and MediaPipe normalised array are parsed correctly.
coco = zeros(17, 3);
coco([6 7 12 13], :) = [395 165 0.9; 245 165 0.9; 368 365 0.9; 272 365 0.9];
tc = getTorsoKeypoints(coco, frameSize, 0.3);
mp = zeros(33, 3);
mp([12 13 24 25], :) = [395/640 165/480 0.9; 245/640 165/480 0.9; 368/640 365/480 0.9; 272/640 365/480 0.9];
tm = getTorsoKeypoints(mp, frameSize, 0.3);
nFail = nFail + check(tc.valid && tm.valid && ...
                      norm(tc.leftShoulder - [395 165]) < 1e-9 && ...
                      norm(tm.rightHip - [272 365]) < 1e-9, ...
    'keypoints: COCO-17 and normalised MediaPipe arrays are parsed');

% 15. Low-confidence keypoint -> invalid.
coco(13, 3) = 0.1;
nFail = nFail + check(~getTorsoKeypoints(coco, frameSize, 0.3).valid, ...
    'keypoints: low-confidence hip makes the torso invalid');

%% ===== tryOnPipeline ==========================================================
% 16. Full pipeline runs, changes the torso, leaves the far background alone.
[out, dbg] = tryOnPipeline(frame, kp, garmentImg, garmentAlpha, cfg);
chest = mean(mean(abs(double(out(220:300, 300:340, :)) - double(frame(220:300, 300:340, :)))));
nFail = nFail + check(dbg.ok && isequal(size(out), frameSize) && isa(out, 'uint8') && ...
                      all(chest(:) > 10) && isequal(out(1:20, 1:20, :), frame(1:20, 1:20, :)), ...
    'pipeline: garment drawn on the torso, background untouched');

% 17. Missing keypoints -> frame returned unchanged, no error.
badKp = kp;  badKp.leftHip = [NaN NaN 0];
[out, dbg] = tryOnPipeline(frame, badKp, garmentImg, garmentAlpha, cfg);
nFail = nFail + check(isequal(out, frame) && ~dbg.ok, ...
    'pipeline: missing keypoint -> frame unchanged');

% 18. Every pose variant and both warp modes run without error.
ok = true;
for v = {'neutral', 'tilt', 'turn', 'lean', 'closer'}
    for mode = {'piecewise', 'affine'}
        c = cfg;  c.warp.mode = mode{1};
        [~, dbg] = tryOnPipeline(frame, makeTestPose(kp, v{1}), garmentImg, garmentAlpha, c);
        ok = ok && dbg.ok;
    end
end
nFail = nFail + check(ok, 'pipeline: all pose variants x both warp modes succeed');

%% ===== Phase 2b: segmentation, body fit, sleeves, shading ====================
[frame, kp, garmentImg, garmentAlpha, background] = makeSyntheticTestData();

% 19. Background subtraction finds the person's torso but not the wall.
%     (The synthetic skin is almost wall-coloured, so the head is not
%     required - only the torso is used by the body fit.)
bodyMask = segmentPersonBackground(frame, background, cfg.segment);
nFail = nFail + check(bodyMask(300, 320) && bodyMask(200, 260) && ~bodyMask(20, 20) && ~bodyMask(300, 600), ...
    'segment: torso is foreground, wall is background');

% 20. Body fit: at waist height the garment edges end up at the body edges
%     (+ ease), with strength 1.
c = cfg;  c.fit.strength = 1;  c.sleeves.enable = false;
c.shading.enable = false;  c.lighting.enable = false;
[~, dbg] = tryOnPipeline(frame, kp, garmentImg, garmentAlpha, c, bodyMask);
y = 330;
row = dbg.layerAlpha(y, :) > 0.5;
bodyRow = bodyMask(y, :);
gEdges = [find(row, 1, 'first'), find(row, 1, 'last')];
bEdges = [find(bodyRow(200:440), 1, 'first') + 199, find(bodyRow(200:440), 1, 'last') + 199];
err = abs(gEdges - (bEdges + [-c.fit.ease c.fit.ease]));
nFail = nFail + check(dbg.fitInfo.ok && all(err <= 3), ...
    sprintf('fit: garment edges follow the body at the waist (error %d / %d px)', err(1), err(2)));

% 21. Sleeves: with the arms raised sideways, the garment covers a point
%     along the upper arm; with sleeves disabled it does not.
kpUp = kp;
kpUp.leftElbow  = [kp.leftShoulder(1) + 80,  kp.leftShoulder(2)  + 5, 0.9];
kpUp.rightElbow = [kp.rightShoulder(1) - 80, kp.rightShoulder(2) + 5, 0.9];
c = cfg;  c.fit.enable = false;
[~, dOn] = tryOnPipeline(frame, kpUp, garmentImg, garmentAlpha, c);
c.sleeves.enable = false;
[~, dOff] = tryOnPipeline(frame, kpUp, garmentImg, garmentAlpha, c);
probeL = round(kp.leftShoulder(1:2)  + [45 3]);
probeR = round(kp.rightShoulder(1:2) + [-45 3]);
nFail = nFail + check(all(dOn.sleeves) && ...
    dOn.layerAlpha(probeL(2), probeL(1)) > 0.5 && dOn.layerAlpha(probeR(2), probeR(1)) > 0.5 && ...
    dOff.layerAlpha(probeL(2), probeL(1)) < 0.5, ...
    'sleeves: sleeves follow raised arms');

% 22. Shading: a flat frame leaves the garment unchanged; a dark crease in
%     the frame darkens the garment there.
gRGB = repmat(reshape([0.8 0.2 0.2], 1, 1, 3), 100, 100);
gA = ones(100, 100);
flat = 0.6 * ones(100, 100, 3);
[sFlat, shadeFlat] = transferBodyShading(gRGB, gA, flat, cfg.shading);
crease = flat;  crease(:, 48:52, :) = 0.3;
[sCrease, shadeCrease] = transferBodyShading(gRGB, gA, crease, cfg.shading);
nFail = nFail + check(max(abs(sFlat(:) - gRGB(:))) < 1e-6 && ...
                      shadeCrease(50, 50) < 0.8 && abs(shadeCrease(50, 10) - 1) < 0.1 && ...
                      sCrease(50, 50, 1) < gRGB(50, 50, 1), ...
    'shading: flat frame = no change, crease in frame = darker garment');

% 23. Elbows are parsed from arrays and are optional.
coco = zeros(17, 3);
coco([6 7 12 13 8 9], :) = [395 165 .9; 245 165 .9; 368 365 .9; 272 365 .9; 433 255 .9; 207 255 .1];
t = getTorsoKeypoints(coco, frameSize, 0.3);
nFail = nFail + check(t.valid && t.hasLeftElbow && ~t.hasRightElbow && ...
                      norm(t.leftElbow - [433 255]) < 1e-9, ...
    'keypoints: elbows parsed, low-confidence elbow ignored, torso still valid');

%% ===== Summary ===============================================================
if nFail == 0
    fprintf('\nAll checks passed.\n');
else
    error('test_tryOn:failed', '%d check(s) failed.', nFail);
end
end

% ==========================================================================
function failed = check(condition, name)
% Print one PASS/FAIL line; return 1 on failure so failures can be counted.
if condition
    fprintf('PASS  %s\n', name);
    failed = 0;
else
    fprintf('FAIL  %s\n', name);
    failed = 1;
end
end
