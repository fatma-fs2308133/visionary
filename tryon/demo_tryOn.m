%% DEMO_TRYON  Stage-by-stage visual check of the virtual try-on pipeline.
%
%   Runs the pipeline on a static test image with fake keypoints (no webcam
%   or pose model needed) and opens one figure per stage:
%
%     Figure 1 - inputs: garment + its anchor points, frame + torso keypoints
%     Figure 2 - WARPING: the same garment on 4 different body poses
%     Figure 3 - WARPING: single affine vs piecewise affine on a turning torso
%     Figure 4 - BLENDING: the masks (hard -> eroded -> feathered) and a
%                zoomed comparison of a hard vs a feathered edge
%     Figure 5 - LIGHTING: dim scene without / with lighting matching, and
%                the V-channel histograms
%     Figure 6 - final result + timing (frames per second estimate)
%
%   To try your own images, set useOwnImages = true below.
%   Set saveFigures = true to write every figure as a PNG (for the report).

clear; close all; clc;
demoDir = fileparts(mfilename('fullpath'));
addpath(demoDir);

useOwnImages = false;       % true -> use your own photo + garment PNG
saveFigures  = false;       % true -> save each figure to demo_output/*.png
outDir = fullfile(demoDir, 'demo_output');

cfg = defaultTryOnConfig();

%% ---- 0. Load the test data ---------------------------------------------------
if ~useOwnImages
    [frame, kp, garmentImg, garmentAlpha] = makeSyntheticTestData();
else
    % A photo of a person facing the camera, and a garment PNG WITH
    % transparency. Adjust cfg.garment.anchors to your garment (Figure 1
    % shows where they are).
    frame = imread('person.jpg');                       % <-- your file
    [garmentImg, ~, garmentAlpha] = imread('shirt.png'); % <-- your file
    if isempty(garmentAlpha)
        error('The garment PNG has no alpha channel (needs a transparent background).');
    end
    % "Fake pose model": click the 4 torso keypoints yourself.
    figure('Name', 'Click the keypoints');
    imshow(frame);
    title({'Click, in this order, the PERSON''S:', ...
           'right shoulder, left shoulder, left hip, right hip'});
    [cx, cy] = ginput(4);
    close(gcf);
    kp.rightShoulder = [cx(1) cy(1)];
    kp.leftShoulder  = [cx(2) cy(2)];
    kp.leftHip       = [cx(3) cy(3)];
    kp.rightHip      = [cx(4) cy(4)];
end
if saveFigures && ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Colours used to mark keypoints in every figure.
ptColour = [0 1 1];       % cyan
diagColour = [1 0.5 0];   % orange = the diagonal that splits the quad

%% ---- Figure 1: inputs --------------------------------------------------------
f1 = figure('Name', '1 - Inputs', 'Position', [50 400 1100 450]);

subplot(1, 2, 1);
imshow(garmentImg, 'InitialMagnification', 'fit');
hold on;
% Show the alpha: draw the garment over a checkerboard would be nicer, but
% an outline of alpha = 0.5 is enough to see the silhouette.
contour(double(garmentAlpha) / 255, [0.5 0.5], 'k', 'LineWidth', 1);
gH = size(garmentImg, 1);  gW = size(garmentImg, 2);
anc = cfg.garment.anchors;
ancNames = {'leftShoulder', 'rightShoulder', 'rightHip', 'leftHip'};
ancPx = zeros(4, 2);
for i = 1:4
    a = anc.(ancNames{i});
    ancPx(i, :) = [1 + a(1) * (gW - 1), 1 + a(2) * (gH - 1)];
    text(ancPx(i, 1) + 6, ancPx(i, 2), ancNames{i}, 'Color', 'k', 'FontSize', 8);
end
plot(ancPx([1:4 1], 1), ancPx([1:4 1], 2), '-o', 'Color', ptColour, ...
     'LineWidth', 1.5, 'MarkerFaceColor', ptColour);
plot(ancPx([1 3], 1), ancPx([1 3], 2), '--', 'Color', diagColour, 'LineWidth', 1.5);
title('Garment + anchor points (source points)');

subplot(1, 2, 2);
imshow(frame);
hold on;
torso = getTorsoKeypoints(kp, size(frame), cfg.pose.minScore);
bodyPx = [torso.leftShoulder; torso.rightShoulder; torso.rightHip; torso.leftHip];
plot(bodyPx([1:4 1], 1), bodyPx([1:4 1], 2), '-o', 'Color', ptColour, ...
     'LineWidth', 1.5, 'MarkerFaceColor', ptColour);
plot(bodyPx([1 3], 1), bodyPx([1 3], 2), '--', 'Color', diagColour, 'LineWidth', 1.5);
for i = 1:4
    text(bodyPx(i, 1) + 6, bodyPx(i, 2), ancNames{i}, 'Color', 'w', 'FontSize', 8);
end
title('Webcam frame + torso keypoints (destination points)');
if saveFigures, print(f1, fullfile(outDir, 'fig1_inputs.png'), '-dpng', '-r100'); end

%% ---- Figure 2: warping on different poses ------------------------------------
% Lighting is switched off here so only the geometry is compared.
cfgWarp = cfg;
cfgWarp.lighting.enable = false;
poses = {'neutral', 'tilt', 'lean', 'turn'};

f2 = figure('Name', '2 - Warping on different poses', 'Position', [50 50 1400 420]);
for i = 1:numel(poses)
    kpPose = makeTestPose(kp, poses{i});
    [outPose, dbgPose] = tryOnPipeline(frame, kpPose, garmentImg, garmentAlpha, cfgWarp);
    subplot(1, numel(poses), i);
    imshow(outPose);
    hold on;
    d = dbgPose.dstPts;
    plot(d([1:4 1], 1), d([1:4 1], 2), '-o', 'Color', ptColour, 'LineWidth', 1, ...
         'MarkerFaceColor', ptColour, 'MarkerSize', 4);
    plot(d([1 3], 1), d([1 3], 2), '--', 'Color', diagColour, 'LineWidth', 1);
    title(sprintf('pose: %s', poses{i}));
end
if saveFigures, print(f2, fullfile(outDir, 'fig2_poses.png'), '-dpng', '-r100'); end

%% ---- Figure 3: affine vs piecewise affine on a turning torso -----------------
kpTurn = makeTestPose(kp, 'turn');
cfgAff = cfgWarp;  cfgAff.warp.mode = 'affine';
cfgPw  = cfgWarp;  cfgPw.warp.mode  = 'piecewise';
[outAff, dbgAff] = tryOnPipeline(frame, kpTurn, garmentImg, garmentAlpha, cfgAff);
[outPw,  dbgPw ] = tryOnPipeline(frame, kpTurn, garmentImg, garmentAlpha, cfgPw);
bodyTurn = [kpTurn.leftShoulder(1:2); kpTurn.rightShoulder(1:2); kpTurn.rightHip(1:2); kpTurn.leftHip(1:2)];

f3 = figure('Name', '3 - Affine vs piecewise affine (torso turning)', 'Position', [100 100 1300 450]);
subplot(1, 3, 1);
imshow(outAff);  hold on;
plot(bodyTurn([1:4 1], 1), bodyTurn([1:4 1], 2), '-o', 'Color', ptColour, 'MarkerFaceColor', ptColour);
title({'Single affine (3 points)', 'hips cannot follow the turn'});

subplot(1, 3, 2);
imshow(outPw);  hold on;
plot(bodyTurn([1:4 1], 1), bodyTurn([1:4 1], 2), '-o', 'Color', ptColour, 'MarkerFaceColor', ptColour);
title({'Piecewise affine (2 triangles)', 'all 4 corners hit the keypoints'});

subplot(1, 3, 3);
% Which output pixels use which triangle's transform (half-planes).
triMap = double(dbgPw.warpInfo.useTriangle1) + 1;          % 1 or 2
triMap(dbgPw.warpedAlpha < 0.5) = 0;                        % no garment
imshow(label2rgb(triMap, [0.95 0.55 0.2; 0.25 0.6 0.95], 'k'));
hold on;
plot(bodyTurn([1 3], 1), bodyTurn([1 3], 2), '--w', 'LineWidth', 1.5);
title({'Triangle used per pixel', 'orange = triangle 1, blue = triangle 2'});
if saveFigures, print(f3, fullfile(outDir, 'fig3_affine_vs_piecewise.png'), '-dpng', '-r100'); end

%% ---- Figure 4: blending (the from-scratch algorithm) -------------------------
[outSoft, dbgSoft] = tryOnPipeline(frame, kp, garmentImg, garmentAlpha, cfgWarp);
% Same frame with feathering switched off = what Phase 1 looked like.
cfgHard = cfgWarp;
cfgHard.blend.erodeRadius = 0;
cfgHard.blend.gaussSigma  = 0;
outHard = tryOnPipeline(frame, kp, garmentImg, garmentAlpha, cfgHard);

% Zoom window around the image-right shoulder/sleeve edge.
zc = round(torso.leftShoulder);
zr = max(1, zc(2) - 45):min(size(frame, 1), zc(2) + 45);
zcCols = max(1, zc(1) - 10):min(size(frame, 2), zc(1) + 80);

f4 = figure('Name', '4 - Alpha blending + boundary smoothing', 'Position', [50 50 1400 700]);
subplot(2, 4, 1); imshow(dbgSoft.warpedAlpha);          title('a) warped alpha');
subplot(2, 4, 2); imshow(dbgSoft.blend.hardMask);       title('1) binary mask');
subplot(2, 4, 3); imshow(dbgSoft.blend.erodedMask);     title(sprintf('2) eroded (disk r = %d)', cfg.blend.erodeRadius));
subplot(2, 4, 4); imshow(dbgSoft.finalAlpha);           title(sprintf('3-4) feathered, sigma = %.1f', cfg.blend.gaussSigma));
subplot(2, 4, 5); imshow(outHard(zr, zcCols, :), 'InitialMagnification', 'fit'); title('hard edge (Phase 1 style)');
subplot(2, 4, 6); imshow(outSoft(zr, zcCols, :), 'InitialMagnification', 'fit'); title('feathered edge (this work)');
subplot(2, 4, 7);
row = zc(2) + 30;                                        % a row through the sleeve
plot(dbgSoft.warpedAlpha(row, zcCols), 'k--', 'LineWidth', 1.2); hold on;
plot(double(dbgSoft.blend.hardMask(row, zcCols)), 'b:', 'LineWidth', 1.2);
plot(dbgSoft.finalAlpha(row, zcCols), 'r', 'LineWidth', 1.5);
legend({'warped alpha', 'binary mask', 'final alpha'}, 'Location', 'best');
xlabel('pixel along the row');  ylabel('alpha');  ylim([-0.05 1.05]);
title('alpha profile across the edge');
subplot(2, 4, 8);
surf(dbgSoft.blend.kernel);  shading interp;  axis tight;
title(sprintf('Gaussian kernel %dx%d', size(dbgSoft.blend.kernel, 1), size(dbgSoft.blend.kernel, 2)));
if saveFigures, print(f4, fullfile(outDir, 'fig4_blending.png'), '-dpng', '-r100'); end

%% ---- Figure 5: lighting matching in a dim scene ------------------------------
if useOwnImages
    dimFrame = im2uint8(im2double(frame) * 0.45);   % darken your photo
else
    dimFrame = makeSyntheticTestData(0.45);          % same scene, 45% light
end
cfgNoLight = cfg;  cfgNoLight.lighting.enable = false;
outNoLight        = tryOnPipeline(dimFrame, kp, garmentImg, garmentAlpha, cfgNoLight);
[outLight, dbgL]  = tryOnPipeline(dimFrame, kp, garmentImg, garmentAlpha, cfg);
st = dbgL.lightStats;
roi = dbgL.torsoRoi;

f5 = figure('Name', '5 - Lighting matching', 'Position', [100 100 1400 450]);
subplot(1, 3, 1);
imshow(outNoLight);  hold on;
rectangle('Position', [roi(3) roi(1) roi(4) - roi(3) roi(2) - roi(1)], 'EdgeColor', 'y', 'LineStyle', '--');
title({'dim scene, NO lighting match', '(yellow = torso region measured)'});
subplot(1, 3, 2);
imshow(outLight);
title({'dim scene, WITH lighting match', ...
       sprintf('gain %.2f, offset %+.2f, strength %.1f', st.gain, st.offset, cfg.lighting.strength)});
subplot(1, 3, 3);
centres = (st.histEdges(1:end-1) + st.histEdges(2:end)) / 2;
plot(centres, st.sceneHist,    'k',  'LineWidth', 1.5); hold on;
plot(centres, st.garmentHist,  'r--', 'LineWidth', 1.2);
plot(centres, st.adjustedHist, 'r',  'LineWidth', 1.5);
legend({sprintf('torso region (mean V %.2f)', st.sceneMeanV), ...
        sprintf('garment before (mean V %.2f)', st.garmentMeanV), ...
        sprintf('garment after (mean V %.2f)', st.adjustedMeanV)}, 'Location', 'north');
xlabel('V (brightness)');  ylabel('fraction of pixels');
title('V-channel histograms');
if saveFigures, print(f5, fullfile(outDir, 'fig5_lighting.png'), '-dpng', '-r100'); end

%% ---- Figure 6: final result + timing ------------------------------------------
nRuns = 20;
tic;
for i = 1:nRuns
    outFinal = tryOnPipeline(frame, kp, garmentImg, garmentAlpha, cfg);
end
msPerFrame = 1000 * toc / nRuns;

f6 = figure('Name', '6 - Final result', 'Position', [150 150 1000 420]);
subplot(1, 2, 1); imshow(frame);    title('input frame');
subplot(1, 2, 2); imshow(outFinal);
title(sprintf('warp -> light-match -> blend: %.1f ms/frame (~%.0f fps, pose model not included)', ...
              msPerFrame, 1000 / msPerFrame));
if saveFigures, print(f6, fullfile(outDir, 'fig6_final.png'), '-dpng', '-r100'); end

fprintf('Pipeline: %.1f ms per frame (%.0f fps) at %dx%d, excluding pose estimation.\n', ...
        msPerFrame, 1000 / msPerFrame, size(frame, 2), size(frame, 1));
