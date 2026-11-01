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

% 18. Every pose variant and every warp mode run without error.
ok = true;
for v = {'neutral', 'tilt', 'turn', 'lean', 'closer'}
    for mode = {'piecewise', 'affine', 'homography'}
        c = cfg;  c.warp.mode = mode{1};
        [~, dbg] = tryOnPipeline(frame, makeTestPose(kp, v{1}), garmentImg, garmentAlpha, c);
        ok = ok && dbg.ok;
    end
end
nFail = nFail + check(ok, 'pipeline: all pose variants x all three warp modes succeed');

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
c = cfg;  c.fit.strength = 1;  c.warp.mode = 'piecewise';  c.sleeves.enable = false;
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

%% ===== Phase 3-4: homography (from scratch), preprocessing, tracking =========
[frame, kp, garmentImg, garmentAlpha, background] = makeSyntheticTestData();
g  = im2double(garmentImg);
ga = im2double(garmentAlpha);
src = [300 60; 100 60; 130 390; 270 390];
dst = [395 165; 245 165; 272 365; 368 365];

% 24. DLT with 4 exact pairs = MATLAB's projective fit.
H = estimateHomographyDLT(src, dst);
ref = fitgeotform2d(src, dst, 'projective');
probe = [50 50; 200 200; 390 400; 10 300];
dRef = max(abs(applyHomography(H, probe) - applyHomography(ref.A, probe)), [], 'all');
nFail = nFail + check(max(abs(applyHomography(H, src) - dst), [], 'all') < 1e-9 && dRef < 1e-6, ...
    sprintf('homography: DLT maps the 4 points exactly and equals fitgeotform2d (%.1e px)', dRef));

% 25. Normalisation: huge coordinates and 8 noisy pairs still give an
%     accurate least-squares fit.
rng(3);
Htrue = [1.1 0.05 300; -0.04 0.95 200; 1e-5 -2e-5 1];
big = [1000 1000; 5000 1100; 5200 4800; 900 5000; 3000 1000; 5100 3000; 3000 4900; 950 3000];
dstBig = applyHomography(Htrue, big) + 0.5 * randn(8, 2);
[Hb, infoB] = estimateHomographyDLT(big, dstBig);
errB = max(sqrt(sum((applyHomography(Hb, big) - applyHomography(Htrue, big)).^2, 2)));
nFail = nFail + check(errB < 2 && infoB.rmse < 2, ...
    sprintf('homography: normalised DLT is accurate on 8 noisy pairs at 5000 px (max error %.2f px)', errB));

% 26. Three collinear points are rejected.
try
    estimateHomographyDLT(src, [0 0; 10 10; 20 20; 0 30]);
    ok = false;
catch err
    ok = strcmp(err.identifier, 'estimateHomographyDLT:degenerate');
end
nFail = nFail + check(ok, 'homography: collinear points are rejected');

% 27. Our inverse warp + bilinear interpolation = imwarp (away from the
%     1-px border, which imwarp blends with the fill value).
own = inverseWarpBilinear(cat(3, g, ga), H, frameSize);
refW = imwarp(cat(3, g, ga), projtform2d(H), 'linear', 'OutputView', imref2d(frameSize(1:2)), 'FillValues', 0);
inner = imerode(own(:, :, 4) > 0 & refW(:, :, 4) > 0, strel('square', 3));
d = abs(own - refW);
d = d(repmat(inner, [1 1 4]));
nFail = nFail + check(max(d) < 1e-6, ...
    sprintf('homography: inverseWarpBilinear equals imwarp (max difference %.1e)', max(d)));

% 28. Homography mode: the anchors land on the keypoints.
maxErr = 0;
[gx, gy] = meshgrid(1:size(ga, 2), 1:size(ga, 1));
[fx, fy] = meshgrid(1:frameSize(2), 1:frameSize(1));
for i = 1:4
    dotImg = double((gx - src(i, 1)).^2 + (gy - src(i, 2)).^2 <= 3^2);
    [~, wDot] = warpGarment(g, dotImg, src, dst, frameSize, 'homography');
    c = [sum(fx(:) .* wDot(:)), sum(fy(:) .* wDot(:))] / sum(wDot(:));
    maxErr = max(maxErr, norm(c - dst(i, :)));
end
nFail = nFail + check(maxErr < 1.0, ...
    sprintf('warp (homography): anchors land on keypoints (max error %.2f px)', maxErr));

% 29. An affine body motion gives the same picture with the homography as
%     with the affine warp (a homography generalises the affine map).
t = deg2rad(10);
Aff = [1.1 * cos(t), -sin(t); sin(t), 0.9 * cos(t)];
dstAffine = src * Aff' + [150 -20];
[hRGB, hA] = warpGarment(g, ga, src, dstAffine, frameSize, 'homography');
[aRGB, aA] = warpGarment(g, ga, [src(1:2, :); mean(src(3:4, :))], ...
                         [dstAffine(1:2, :); mean(dstAffine(3:4, :))], frameSize);
both = imerode(hA > 0 & aA > 0, strel('square', 3));
dd = abs(hRGB - aRGB);
nFail = nFail + check(max(dd(repmat(both, [1 1 3]))) < 1e-6, ...
    'warp: homography == affine when the motion is affine');

% 30. A crossed (non-convex) body quad is rejected, and the pipeline then
%     returns the frame unchanged instead of crashing.
crossed = kp;
[crossed.leftHip, crossed.rightHip] = deal(kp.rightHip, kp.leftHip);
[outX, dbgX] = tryOnPipeline(frame, crossed, garmentImg, garmentAlpha, cfg);
nFail = nFail + check(~dbgX.ok && isequal(outX, frame) && contains(dbgX.reason, 'convex'), ...
    'warp (homography): crossed keypoints -> overlay skipped, no crash');

% 31. Preprocessing: resizes to 640 px wide; a dark frame is enhanced
%     (brighter), a normal frame is left alone.
bigFrame = imresize(frame, 2);
[p1, i1] = preprocessFrame(bigFrame, cfg.preprocess);
dark = makeSyntheticTestData(0.3);
[p2, i2] = preprocessFrame(dark, cfg.preprocess);
[~, i3] = preprocessFrame(frame, cfg.preprocess);
nFail = nFail + check(size(p1, 2) == 640 && size(p1, 1) == 480 && i2.enhanced && ~i3.enhanced && ...
                      mean(im2double(p2(:))) > mean(im2double(dark(:))) + 0.05, ...
    sprintf('preprocess: resize to 640 px, dark frame enhanced (mean %.2f -> %.2f), normal frame untouched', ...
            mean(im2double(dark(:))), mean(im2double(p2(:)))));

% 32. Automatic garment anchors are close to the hand-placed ones.
auto = autoGarmentAnchors(garmentAlpha);
names = {'leftShoulder', 'rightShoulder', 'leftHip', 'rightHip', 'leftArmpit', 'rightArmpit'};
errA = 0;
for i = 1:numel(names)
    errA = max(errA, norm(auto.(names{i}) - cfg.garment.anchors.(names{i})));
end
nFail = nFail + check(errA < 0.03, ...
    sprintf('anchors: automatic anchors within %.3f of the hand-placed ones', errA));

% 33. KLT tracking follows a moving person between detections.
fakeKp = toCoco(kp);
tracker = PoseTracker('function', 'DetectFcn', @(~) fakeKp, 'DetectEvery', 10, 'Smoothing', 0);
tracker.step(frame);                                   % detection
shift = [6 3];                                         % person moves 6 px right, 3 px down
moved = imtranslate(frame, shift);
[kpT, infoT] = tracker.step(moved);                    % tracking
errT = max(sqrt(sum((kpT([6 7 12 13], 1:2) - (fakeKp([6 7 12 13], 1:2) + shift)).^2, 2)));
nFail = nFail + check(strcmp(infoT.source, 'tracker') && errT < 1.5, ...
    sprintf('tracking: KLT follows a 6/3 px move (max error %.2f px)', errT));

% 34. Full session (preprocess -> pose -> mask -> pipeline) and the torso
%     alignment measure on the synthetic person.
lib = struct('name', 'synthetic', 'file', '', 'rgb', garmentImg, 'alpha', garmentAlpha, ...
             'anchors', cfg.garment.anchors, 'anchorSource', 'file', 'thumb', []);
c = cfg;  c.eval.torsoMask = true;
session = TryOnSession(lib, PoseTracker('function', 'DetectFcn', @(~) fakeKp), c);
session.setBackground(background);
[outS, infoS] = session.processFrame(frame);
gtPoly = [235 160; 300 150; 340 150; 405 160; 400 225; 385 375; 255 375; 240 225];
gt = poly2mask(gtPoly(:, 1), gtPoly(:, 2), frameSize(1), frameSize(2));
pred = infoS.dbg.torsoOnlyAlpha > 0.5;
iou = nnz(pred & gt) / nnz(pred | gt);
nFail = nFail + check(infoS.ok && isequal(size(outS), frameSize) && ~isempty(infoS.bodyMask) && ...
                      all(isfield(infoS.timing, {'preprocess', 'pose', 'segment', 'pipeline'})) && iou > 0.75, ...
    sprintf('session: full frame processed, torso IoU %.2f on the synthetic person', iou));

% 36. The person mask is still right when the empty-scene picture gets
%     contrast-enhanced and the photo with the person does not (the session
%     compares the un-enhanced frames).
[~, bgInfo] = preprocessFrame(background, cfg.preprocess);
[~, frInfo] = preprocessFrame(frame, cfg.preprocess);
m = infoS.bodyMask;
nFail = nFail + check(bgInfo.enhanced ~= frInfo.enhanced && m(300, 320) && m(200, 260) && ...
                      ~m(20, 20) && ~m(300, 600) && nnz(m) < 0.3 * numel(m), ...
    'session: person mask correct even when only the background is contrast-enhanced');

% 35. prepareGarment cuts a garment out of a plain background.
tmp = tempname;  mkdir(tmp);
cleanupTmp = onCleanup(@() rmdir(tmp, 's'));
a = im2double(garmentAlpha);
photo = im2double(garmentImg) .* a + 0.97 * (1 - a);     % garment on a white background
photo = padarray(photo, [30 30], 0.97, 'both');
imwrite(photo, fullfile(tmp, 'shirt.jpg'), 'Quality', 95);
out = prepareGarment(fullfile(tmp, 'shirt.jpg'), fullfile(tmp, 'garments'));
[~, ~, alphaOut] = imread(out{1});
truth = padarray(a > 0.5, [30 30], false, 'both');
rows = find(any(truth, 2));  cols = find(any(truth, 1));
truth = truth(max(1, rows(1) - 8):min(end, rows(end) + 8), max(1, cols(1) - 8):min(end, cols(end) + 8));
cut = imresize(alphaOut, size(truth)) > 127;
iouCut = nnz(cut & truth) / nnz(cut | truth);
nFail = nFail + check(iouCut > 0.95 && exist(fullfile(tmp, 'garments', 'shirt_anchors.mat'), 'file') == 2, ...
    sprintf('garments: background removed (IoU %.3f with the true silhouette), anchors saved', iouCut));

%% ===== Summary ===============================================================
if nFail == 0
    fprintf('\nAll checks passed.\n');
else
    error('test_tryOn:failed', '%d check(s) failed.', nFail);
end
end

% ==========================================================================
function c = toCoco(s)
% Keypoint struct -> 17x3 COCO-17 array.
c = zeros(17, 3);
c(6, :) = s.leftShoulder;   c(7, :) = s.rightShoulder;
c(8, :) = s.leftElbow;      c(9, :) = s.rightElbow;
c(12, :) = s.leftHip;       c(13, :) = s.rightHip;
end

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
