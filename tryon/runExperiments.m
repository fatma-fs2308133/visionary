function results = runExperiments(opts)
%RUNEXPERIMENTS  All quantitative results for the report, in one run.
%
%   results = runExperiments()          % defaults below
%   results = runExperiments(opts)
%
%   EXPERIMENTS
%     E1  From-scratch validation: our DLT homography vs MATLAB's
%         fitgeotform2d, and our inverse warp vs imwarp (accuracy + speed).
%     E2  Detection rate: share of frames WITH a person where all 4 torso
%         keypoints are found (target >= 90%), per condition (bright, dim,
%         ...), with and without preprocessing. Frames annotated as "no
%         person" count false detections.
%     E3  Torso alignment: IoU between the warped garment torso and the
%         hand-marked torso (annotateTorso), per garment and per warp mode
%         (affine / piecewise / homography). Target: mean IoU >= 0.75.
%     E4  Speed: time of every stage, and the resulting live frame rate
%         with HRNet every N frames + KLT tracking in between. Target >= 10 fps.
%     E5  Failure cases: the frames with the lowest IoU, drawn with the
%         ground truth (green) and the garment torso (red) outlines.
%
%   DATA
%     Test sessions recorded with captureTestFrames and annotated with
%     annotateTorso: data/test_frames/<person>_<condition>/frame_*.png,
%     background.png, gt/. Garments from data/garments (loadGarmentLibrary).
%     Without annotated sessions a SYNTHETIC smoke test runs instead (only
%     to check that everything works - not a result for the report).
%
%   opts (all optional):
%     .dataRoot      'data/test_frames'
%     .garmentDir    'data/garments'
%     .outDir        'results'
%     .poseMode      'auto' (HRNet) | 'hrnet' | 'synthetic'
%     .maxGarments   garments evaluated (5)
%     .warpModes     {'affine', 'piecewise', 'homography'}
%     .preprocessAblation  also measure detection without preprocessing (true)
%     .maxFramesPerSession  Inf
%
%   OUTPUT (in outDir): summary.txt, e1_from_scratch.csv, frames.csv (one
%   row per frame/garment/mode), timing.csv, fig_*.png. Also returned as a
%   struct.
%
%   See also captureTestFrames, annotateTorso, TryOnSession.

if nargin < 1 || isempty(opts), opts = struct(); end
opts = fillDefaults(opts);
if ~exist(opts.outDir, 'dir'), mkdir(opts.outDir); end
cfg = defaultTryOnConfig();
cfg.eval.torsoMask = true;

%% ---- Garments ------------------------------------------------------------------
lib = loadGarmentLibrary(opts.garmentDir);
if isempty(lib)
    fprintf('No garments in %s - creating the sample garments.\n', opts.garmentDir);
    makeSampleGarments(opts.garmentDir);
    lib = loadGarmentLibrary(opts.garmentDir);
end
lib = lib(1:min(opts.maxGarments, numel(lib)));
fprintf('Garments: %s\n', strjoin({lib.name}, ', '));

%% ---- E1: from-scratch validation -----------------------------------------------
fprintf('\n== E1 from-scratch validation ==\n');
results.e1 = validateFromScratch(lib(1));
writetable(results.e1, fullfile(opts.outDir, 'e1_from_scratch.csv'));
disp(results.e1);

%% ---- Sessions --------------------------------------------------------------------
sessions = findSessions(opts.dataRoot);
synthetic = isempty(sessions) || strcmp(opts.poseMode, 'synthetic');
if synthetic
    fprintf(['\nNo annotated sessions in %s -> SYNTHETIC smoke test.\n' ...
             'Record real data with captureTestFrames + annotateTorso for the report.\n'], opts.dataRoot);
    sessions = syntheticSessions();
    detectFcnMaker = @(s) s.detectFcn;
    trackerMode = 'function';
else
    trackerMode = opts.poseMode;
    detectFcnMaker = @(s) [];
end

rows = {};          % per frame x garment x mode
detRows = {};       % per frame (detection, with/without preprocessing)
timeRows = {};      % per frame timings
failures = struct('iou', {}, 'frame', {}, 'gt', {}, 'pred', {}, 'label', {});

for si = 1:numel(sessions)
    S = sessions(si);
    fprintf('\n== session %s (%s), %d frames ==\n', S.name, S.condition, numel(S.frames));
    detector = makeTracker(trackerMode, detectFcnMaker(S));
    detector.Smoothing = 0;                 % raw detections, no smoothing
    session = TryOnSession(lib, detector, cfg);
    if ~isempty(S.background)
        session.setBackground(S.background);
    end
    % A second tracker that only TRACKS between frames, to time KLT.
    klt = makeTracker(trackerMode, detectFcnMaker(S));
    klt.DetectEvery = Inf;                  % never re-detect: pure KLT steps

    nFrames = min(numel(S.frames), opts.maxFramesPerSession);
    for fi = 1:nFrames
        raw = S.frames{fi};
        hasPerson = S.hasPerson(fi);

        % -- E2: detection on this frame (fresh detection, no tracking) --
        detector.reset();
        tPre = tic;
        [frame, plain] = session.preprocess(raw);
        timePre = toc(tPre);
        tPose = tic;
        [kp, ~] = detector.step(frame);
        timeDetect = toc(tPose);
        torso = getTorsoKeypoints(kpOrEmpty(kp), size(frame), cfg.pose.minScore);
        detected = ~isempty(kp) && torso.valid;
        conf = 0;
        if ~isempty(kp), conf = mean(kp(PoseTracker.TorsoRows, 3)); end

        detectedRaw = NaN;
        if opts.preprocessAblation
            detector.reset();
            rawResized = imresize(im2uint8(raw), [size(frame, 1) size(frame, 2)]);
            kpRaw = detector.step(rawResized);
            tr = getTorsoKeypoints(kpOrEmpty(kpRaw), size(rawResized), cfg.pose.minScore);
            detectedRaw = ~isempty(kpRaw) && tr.valid;
        end
        detRows(end + 1, :) = {S.name, S.condition, fi, hasPerson, detected, detectedRaw, conf}; %#ok<AGROW>

        % -- KLT tracking time (track from the previous frame) --
        timeTrack = NaN;
        if ~isempty(kp)
            if fi == 1 || klt.needsManualKeypoints() || isempty(klt.Kp)
                klt.setManualKeypoints(kp, frame);      % (re)start tracking here
            else
                tT = tic;
                klt.step(frame);
                timeTrack = toc(tT);
            end
        end

        % -- E3: alignment IoU for every garment and warp mode --
        gt = [];
        if hasPerson && ~isempty(S.gt{fi})
            gt = imresize(S.gt{fi}, [size(frame, 1) size(frame, 2)], 'nearest');
        end
        bodyMask = [];
        if ~isempty(session.Background)
            bodyMask = segmentPersonBackground(plain, session.Background, cfg.segment);
        end
        for gi = 1:numel(lib)
            for mi = 1:numel(opts.warpModes)
                c = cfg;
                c.warp.mode = opts.warpModes{mi};
                c.garment.anchors = lib(gi).anchors;
                iou = NaN;
                pipeTiming = struct('warp', NaN, 'fit', NaN, 'sleeves', NaN, ...
                                    'lighting', NaN, 'shading', NaN, 'blend', NaN, 'total', NaN);
                if detected
                    [~, dbg] = tryOnPipeline(frame, kp, lib(gi).rgb, lib(gi).alpha, c, bodyMask);
                    if dbg.ok
                        pipeTiming = dbg.timing;
                        if ~isempty(gt)
                            pred = dbg.torsoOnlyAlpha > 0.5;
                            iou = nnz(pred & gt) / max(nnz(pred | gt), 1);
                            failures(end + 1) = struct('iou', iou, 'frame', frame, 'gt', gt, ... %#ok<AGROW>
                                'pred', pred, 'label', sprintf('%s #%d, %s, %s', S.name, fi, ...
                                lib(gi).name, opts.warpModes{mi}));
                            failures = keepWorst(failures, 6);
                        end
                    else
                        iou = 0;                       % detected but the warp failed
                    end
                elseif ~isempty(gt)
                    iou = 0;                           % missed detection
                end
                rows(end + 1, :) = {S.name, S.condition, fi, lib(gi).name, opts.warpModes{mi}, ... %#ok<AGROW>
                                    detected, iou, pipeTiming.warp, pipeTiming.total};
                if gi == 1 && strcmp(opts.warpModes{mi}, cfg.warp.mode)
                    timeRows(end + 1, :) = {S.name, fi, timePre, timeDetect, timeTrack, ... %#ok<AGROW>
                        pipeTiming.warp, pipeTiming.fit, pipeTiming.sleeves, pipeTiming.lighting, ...
                        pipeTiming.shading, pipeTiming.blend, pipeTiming.total};
                end
            end
        end
        fprintf('  frame %3d: person=%d detected=%d conf=%.2f\n', fi, hasPerson, detected, conf);
    end
end

%% ---- Tables -------------------------------------------------------------------------
frames = cell2table(rows, 'VariableNames', {'session', 'condition', 'frame', 'garment', ...
    'warpMode', 'detected', 'iou', 'warpTime', 'pipelineTime'});
det = cell2table(detRows, 'VariableNames', {'session', 'condition', 'frame', 'hasPerson', ...
    'detected', 'detectedNoPreprocess', 'confidence'});
timing = cell2table(timeRows, 'VariableNames', {'session', 'frame', 'preprocess', 'detect', ...
    'track', 'warp', 'fit', 'sleeves', 'lighting', 'shading', 'blend', 'pipeline'});
writetable(frames, fullfile(opts.outDir, 'frames.csv'));
writetable(det, fullfile(opts.outDir, 'detection.csv'));
writetable(timing, fullfile(opts.outDir, 'timing.csv'));
results.frames = frames;
results.detection = det;
results.timing = timing;

%% ---- E2: detection rate ----------------------------------------------------------
withPerson = det(det.hasPerson, :);
results.detectionRate = mean(withPerson.detected);
results.detectionRateNoPre = mean(withPerson.detectedNoPreprocess);
noPerson = det(~det.hasPerson, :);
results.falseDetections = sum(noPerson.detected);
conds = unique(det.condition, 'stable');
condRate = zeros(numel(conds), 2);
for k = 1:numel(conds)
    sel = withPerson(strcmp(withPerson.condition, conds{k}), :);
    condRate(k, :) = [mean(sel.detected), mean(sel.detectedNoPreprocess)];
end

%% ---- E3: IoU --------------------------------------------------------------------------
modes = opts.warpModes;
iouGiven = zeros(1, numel(modes));        % mean IoU over frames where detection worked
iouE2E   = zeros(1, numel(modes));        % misses count as 0
for mi = 1:numel(modes)
    sel = frames(strcmp(frames.warpMode, modes{mi}) & ~isnan(frames.iou), :);
    iouE2E(mi) = mean(sel.iou);
    iouGiven(mi) = mean(sel.iou(sel.detected));
end
garmentIou = zeros(numel(lib), 1);
for gi = 1:numel(lib)
    sel = frames(strcmp(frames.garment, lib(gi).name) & strcmp(frames.warpMode, cfg.warp.mode) & ...
                 frames.detected & ~isnan(frames.iou), :);
    garmentIou(gi) = mean(sel.iou);
end
results.iouByMode = table(modes(:), iouGiven(:), iouE2E(:), ...
    'VariableNames', {'warpMode', 'meanIoU_detected', 'meanIoU_endToEnd'});
results.iouByGarment = table({lib.name}', garmentIou, 'VariableNames', {'garment', 'meanIoU'});

%% ---- E4: speed --------------------------------------------------------------------------
% Skip the first frame of every session (MATLAB compiles functions on the
% first call, which is not representative).
steady = timing(timing.frame > 1, :);
if isempty(steady), steady = timing; end
m = @(v) mean(v(~isnan(v)));
N = cfg.tracking.detectEvery;
poseAvg = (m(steady.detect) + (N - 1) * m(steady.track)) / N;
if isnan(poseAvg), poseAvg = m(steady.detect); end
stageMs = 1000 * [m(steady.preprocess), poseAvg, m(steady.warp), m(steady.fit), m(steady.sleeves), ...
                  m(steady.lighting), m(steady.shading), m(steady.blend)];
stageMs(isnan(stageMs)) = 0;
stageNames = {'preprocess', sprintf('pose (HRNet 1/%d + KLT)', N), 'warp', 'body fit', 'sleeves', ...
              'lighting', 'shading', 'blend'};
results.msPerFrame = sum(stageMs);
results.fps = 1000 / results.msPerFrame;
results.fpsDetectEveryFrame = 1000 / (sum(stageMs) - stageMs(2) + 1000 * m(steady.detect));
results.stageMs = table(stageNames(:), stageMs(:), 'VariableNames', {'stage', 'ms'});

%% ---- Figures ------------------------------------------------------------------------------
makeFigures(opts.outDir, results, conds, condRate, modes, iouGiven, iouE2E, lib, garmentIou, ...
            stageNames, stageMs, failures, synthetic);

%% ---- Summary ---------------------------------------------------------------------------
summary = '';
if synthetic
    summary = [summary sprintf('*** SYNTHETIC SMOKE TEST - record real sessions for the report ***\n\n')];
end
summary = [summary sprintf('Sessions: %d, frames with a person: %d, without: %d\n', ...
    numel(sessions), height(withPerson), height(noPerson))];
summary = [summary sprintf('Garments: %d, warp modes: %s\n\n', numel(lib), strjoin(modes, ', '))];
summary = [summary sprintf('%-46s %-12s %-10s %s\n', 'Objective (Phase 1)', 'Target', 'Achieved', 'Result')];
summary = [summary summaryLine('Live frame rate (fps)', '>= 10', results.fps, results.fps >= 10, '%.1f')];
summary = [summary summaryLine('Torso detection rate', '>= 90%', 100 * results.detectionRate, ...
                        results.detectionRate >= 0.9, '%.1f%%')];
[~, mainMode] = ismember(cfg.warp.mode, modes);
if mainMode > 0
    summary = [summary summaryLine(sprintf('Mean torso IoU (%s)', cfg.warp.mode), '>= 0.75', ...
                            iouGiven(mainMode), iouGiven(mainMode) >= 0.75, '%.3f')];
end
summary = [summary summaryLine('Selectable garments', '>= 5', numel(lib), numel(lib) >= 5, '%d')];
summary = [summary sprintf('\nDetection without preprocessing: %.1f%%\n', 100 * results.detectionRateNoPre)];
summary = [summary sprintf('False detections on empty frames: %d\n', results.falseDetections)];
summary = [summary sprintf('fps if HRNet ran on every frame: %.1f\n\n', results.fpsDetectEveryFrame)];
summary = [summary sprintf('IoU by warp mode:\n%s\n', plainTable(results.iouByMode))];
summary = [summary sprintf('IoU by garment (%s):\n%s\n', cfg.warp.mode, plainTable(results.iouByGarment))];
summary = [summary sprintf('Time per stage (ms):\n%s\n', plainTable(results.stageMs))];
summary = [summary sprintf('From-scratch validation (E1):\n%s\n', plainTable(results.e1))];
fid = fopen(fullfile(opts.outDir, 'summary.txt'), 'w');
fprintf(fid, '%s', summary);
fclose(fid);
fprintf('\n%s\nResults written to %s\n', summary, opts.outDir);
end

% ==========================================================================
%  E1
% ==========================================================================
function T = validateFromScratch(garment)
rng(7);
nTrials = 200;
errExact = zeros(nTrials, 1);
rmseOwn = zeros(nTrials, 1);  rmseRef = zeros(nTrials, 1);
tOwn = 0;  tRef = 0;
for k = 1:nTrials
    src = [100 60; 300 60; 270 390; 130 390] + 10 * randn(4, 2);
    % A random perspective view of the source quad.
    Htrue = [1 + 0.2 * randn, 0.2 * randn, 200 * rand;
             0.2 * randn, 1 + 0.2 * randn, 100 * rand;
             2e-4 * randn, 2e-4 * randn, 1];
    dst = applyHomography(Htrue, src);
    t = tic;  H = estimateHomographyDLT(src, dst);  tOwn = tOwn + toc(t);
    t = tic;  T2 = fitgeotform2d(src, dst, 'projective');  tRef = tRef + toc(t);
    Href = T2.A / T2.A(3, 3);
    errExact(k) = max(abs(applyHomography(H, src) - applyHomography(Href, src)), [], 'all');

    % 8 noisy correspondences (1 px noise): least-squares accuracy.
    srcN = [src; (src + circshift(src, 1)) / 2];
    dstN = applyHomography(Htrue, srcN) + randn(8, 2);
    Hn = estimateHomographyDLT(srcN, dstN);
    Tn = fitgeotform2d(srcN, dstN, 'projective');
    test = [linspace(110, 290, 20)', linspace(70, 380, 20)'];
    rmseOwn(k) = sqrt(mean(sum((applyHomography(Hn, test) - applyHomography(Htrue, test)).^2, 2)));
    rmseRef(k) = sqrt(mean(sum((applyHomography(Tn.A, test) - applyHomography(Htrue, test)).^2, 2)));
end

% Inverse warp: ours vs imwarp with the same homography.
img = im2double(garment.rgb);
alpha = im2double(garment.alpha);
src = [1 1; size(img, 2) 1; size(img, 2) size(img, 1); 1 size(img, 1)];
dst = [180 120; 430 140; 410 420; 200 400];
H = estimateHomographyDLT(src, dst);
outSize = [480 640];
t = tic;
for r = 1:10, own = inverseWarpBilinear(cat(3, img, alpha), H, outSize); end
tWarpOwn = toc(t) / 10;
tf = projtform2d(H);
t = tic;
for r = 1:10
    ref = imwarp(cat(3, img, alpha), tf, 'linear', 'OutputView', imref2d(outSize), 'FillValues', 0);
end
tWarpRef = toc(t) / 10;
% Compare where both are well inside the garment image (the 1-px border is
% handled slightly differently: we drop it, imwarp blends with the fill).
inner = imerode(own(:, :, 4) > 0 & ref(:, :, 4) > 0, strel('square', 3));
d = abs(own - ref);
d = d(repmat(inner, [1 1 4]));

T = table( ...
    {'DLT: max point difference vs fitgeotform2d (4 exact pairs), px'; ...
     'DLT: RMSE vs truth, 8 noisy pairs (ours), px'; ...
     'DLT: RMSE vs truth, 8 noisy pairs (fitgeotform2d), px'; ...
     'DLT: time per call (ours), ms'; ...
     'DLT: time per call (fitgeotform2d), ms'; ...
     'Inverse warp: mean abs difference vs imwarp (0..1 intensity)'; ...
     'Inverse warp: max abs difference vs imwarp'; ...
     'Inverse warp: time 640x480 (ours), ms'; ...
     'Inverse warp: time 640x480 (imwarp), ms'}, ...
    [max(errExact); mean(rmseOwn); mean(rmseRef); 1000 * tOwn / nTrials; 1000 * tRef / nTrials; ...
     mean(d); max(d); 1000 * tWarpOwn; 1000 * tWarpRef], ...
    'VariableNames', {'check', 'value'});
end

% ==========================================================================
%  Data
% ==========================================================================
function sessions = findSessions(root)
sessions = struct('name', {}, 'condition', {}, 'frames', {}, 'hasPerson', {}, ...
                  'gt', {}, 'background', {}, 'detectFcn', {});
if ~isfolder(root), return; end
dirs = dir(root);
dirs = dirs([dirs.isdir] & ~startsWith({dirs.name}, '.'));
for i = 1:numel(dirs)
    d = fullfile(dirs(i).folder, dirs(i).name);
    annFile = fullfile(d, 'gt', 'annotations.mat');
    if ~exist(annFile, 'file')
        fprintf('Skipping %s (not annotated - run annotateTorso).\n', dirs(i).name);
        continue
    end
    a = load(annFile);
    ann = a.ann;
    S.name = dirs(i).name;
    parts = split(S.name, '_');
    S.condition = parts{end};
    S.frames = cell(1, height(ann));
    S.gt = cell(1, height(ann));
    S.hasPerson = ann.hasPerson(:)';
    for k = 1:height(ann)
        [~, base] = fileparts(char(ann.frame(k)));
        S.frames{k} = imread(fullfile(d, char(ann.frame(k))));
        if ann.hasPerson(k)
            S.gt{k} = imread(fullfile(d, 'gt', [base '_torso.png'])) > 0;
        end
    end
    bgFile = fullfile(d, 'background.png');
    S.background = [];
    if exist(bgFile, 'file'), S.background = imread(bgFile); end
    S.detectFcn = [];
    sessions(end + 1) = S; %#ok<AGROW>
end
end

function sessions = syntheticSessions()
% Synthetic person in normal and dim light, with the torso hand-marked by
% construction (the shirt outline without its sleeves).
[f, kp, ~, ~, bg] = makeSyntheticTestData();
fDim = makeSyntheticTestData(0.45);
[~, ~, ~, ~, bgDim] = makeSyntheticTestData(0.45);
gtPoly = [235 160; 300 150; 340 150; 405 160; 400 225; 385 375; 255 375; 240 225];
gt = poly2mask(gtPoly(:, 1), gtPoly(:, 2), size(f, 1), size(f, 2));
S1 = struct('name', 'synthetic_bright', 'condition', 'bright', 'frames', {{f, f}}, ...
            'hasPerson', [true true], 'gt', {{gt, gt}}, 'background', bg, ...
            'detectFcn', @(~) kp);
S2 = S1;
S2.name = 'synthetic_dim';  S2.condition = 'dim';
S2.frames = {fDim, fDim};  S2.background = bgDim;
sessions = [S1, S2];
end

function tr = makeTracker(mode, detectFcn)
if strcmp(mode, 'function')
    tr = PoseTracker('function', 'DetectFcn', detectFcn);
else
    tr = PoseTracker(mode);
    if ~tr.hasDetector()
        error('runExperiments:noPose', ['The experiments need the HRNet pose model ' ...
            '("Computer Vision Toolbox Model for Object Keypoint Detection" add-on).']);
    end
end
end

function s = plainTable(T)
% Table as plain text (no HTML bold markup, which disp adds in some modes).
s = char(formattedDisplayText(T, 'SuppressMarkup', true));
end

function kp = kpOrEmpty(kp)
if isempty(kp), kp = NaN(17, 3); end
end

function f = keepWorst(f, n)
[~, order] = sort([f.iou]);
f = f(order(1:min(n, numel(order))));
end

function s = summaryLine(name, target, value, pass, fmt)
if pass, verdict = 'PASS'; else, verdict = 'NOT MET'; end
s = sprintf(['%-46s %-12s ' fmt '%s %s\n'], name, target, value, ...
            repmat(' ', 1, max(1, 10 - numel(sprintf(fmt, value)))), verdict);
end

% ==========================================================================
%  Figures
% ==========================================================================
function makeFigures(outDir, R, conds, condRate, modes, iouGiven, iouE2E, lib, garmentIou, ...
                     stageNames, stageMs, failures, synthetic)
blue = [0.20 0.45 0.75];  grey = [0.65 0.65 0.65];  red = [0.80 0.25 0.20];
tag = '';
if synthetic, tag = ' (SYNTHETIC smoke test)'; end

% Detection rate per condition, with / without preprocessing
f = figure('Visible', 'off', 'Position', [100 100 700 420]);
b = groupedBar(conds, 100 * condRate);
b(1).FaceColor = blue;  b(2).FaceColor = grey;
yline(90, '--', 'target 90%', 'Color', red, 'LabelHorizontalAlignment', 'left');
ylim([0 105]);  ylabel('frames with all 4 torso keypoints (%)');
legend({'with preprocessing', 'without'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
title(['Torso detection rate by condition' tag]);
exportgraphics(f, fullfile(outDir, 'fig_detection.png'), 'Resolution', 150);
close(f);

% IoU per warp mode
f = figure('Visible', 'off', 'Position', [100 100 700 420]);
b = groupedBar(modes, [iouGiven(:) iouE2E(:)]);
b(1).FaceColor = blue;  b(2).FaceColor = grey;
yline(0.75, '--', 'target 0.75', 'Color', red, 'LabelHorizontalAlignment', 'left');
ylim([0 1]);  ylabel('mean torso IoU');
legend({'frames with a detection', 'end-to-end (misses = 0)'}, 'Location', 'southoutside', ...
       'Orientation', 'horizontal');
title(['Torso alignment by warp model' tag]);
exportgraphics(f, fullfile(outDir, 'fig_iou_by_mode.png'), 'Resolution', 150);
close(f);

% IoU per garment
f = figure('Visible', 'off', 'Position', [100 100 760 420]);
names = strrep({lib.name}, '_', ' ');
barh(categorical(names, names), garmentIou, 'FaceColor', blue);
xline(0.75, '--', 'target', 'Color', red);
xlim([0 1]);  xlabel('mean torso IoU');
title(['Torso alignment by garment' tag]);
exportgraphics(f, fullfile(outDir, 'fig_iou_by_garment.png'), 'Resolution', 150);
close(f);

% Time per stage
f = figure('Visible', 'off', 'Position', [100 100 760 420]);
barh(categorical(stageNames, stageNames), stageMs, 'FaceColor', blue);
xlabel('ms per frame');
title(sprintf('Time per stage: %.0f ms total = %.1f fps%s', sum(stageMs), 1000 / sum(stageMs), tag));
exportgraphics(f, fullfile(outDir, 'fig_timing.png'), 'Resolution', 150);
close(f);

% Failure cases
if ~isempty(failures)
    f = figure('Visible', 'off', 'Position', [100 100 1200 640]);
    for k = 1:numel(failures)
        subplot(2, 3, k);
        imshow(failures(k).frame);  hold on;
        visboundaries(failures(k).gt, 'Color', 'g', 'LineWidth', 1);
        visboundaries(failures(k).pred, 'Color', 'r', 'LineWidth', 1);
        title({strrep(failures(k).label, '_', ' '), sprintf('IoU %.2f', failures(k).iou)}, 'FontSize', 8);
    end
    sgtitle(['Lowest-IoU cases: green = hand-marked torso, red = garment torso' tag]);
    exportgraphics(f, fullfile(outDir, 'fig_failure_cases.png'), 'Resolution', 120);
    close(f);
end
end

function b = groupedBar(labels, Y)
% Grouped bar chart that also works with a single group (bar() would
% otherwise treat a 1-by-2 Y as two separate groups).
labels = strrep(labels, '_', ' ');
n = numel(labels);
if n == 1
    b = bar([1 2], [Y; NaN(1, size(Y, 2))], 'grouped');
    xlim([0.5 1.5]);
else
    b = bar(1:n, Y, 'grouped');
end
set(gca, 'XTick', 1:n, 'XTickLabel', labels);
end

function opts = fillDefaults(opts)
defaults = struct('dataRoot', fullfile('data', 'test_frames'), 'garmentDir', fullfile('data', 'garments'), ...
                  'outDir', 'results', 'poseMode', 'auto', 'maxGarments', 5, ...
                  'warpModes', {{'affine', 'piecewise', 'homography'}}, ...
                  'preprocessAblation', true, 'maxFramesPerSession', Inf);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(opts, names{i}) || isempty(opts.(names{i}))
        opts.(names{i}) = defaults.(names{i});
    end
end
end
