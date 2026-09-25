function liveTryOn(garmentFile, opts)
%LIVETRYON  Live webcam try-on: camera -> pose -> tryOnPipeline -> screen.
%
%   liveTryOn                    % synthetic striped T-shirt
%   liveTryOn('shirt.png')       % your own garment PNG (with transparency)
%   liveTryOn('shirt.png', opts) % options, see below
%
%   KEYS (click the video window first):
%     B  capture the empty background (3-second countdown: step OUT of view).
%        Needed for the body-shape fit. Press again if the light changes.
%     F  body-shape fit on/off       S  shading (folds) on/off
%     A  sleeves follow arms on/off  L  lighting match on/off
%     M  show the person mask        Q  quit (or close the window)
%
%   REQUIREMENTS
%     - MATLAB Support Package for USB Webcams  (for webcam / snapshot)
%       Home > Add-Ons > search "USB Webcams".
%     - For automatic pose: Computer Vision Toolbox + Deep Learning Toolbox
%       and the "Computer Vision Toolbox Model for Object Keypoint Detection"
%       add-on (hrnetObjectKeypointDetector, R2023b+).
%       If that is not installed, the script falls back to MANUAL mode:
%       you click your shoulders and hips once, then stand still. That is
%       enough to see the warp / blend / lighting working on your camera.
%
%   STAND BACK: the pose model needs to see your HIPS. At a desk the webcam
%   usually only sees your chest, the hips are missing and nothing is drawn.
%
%   YOUR OWN GARMENT: run pickGarmentAnchors('shirt.png') once. It saves
%   shirt_anchors.mat next to the PNG, which liveTryOn loads automatically.
%
%   opts fields (all optional):
%     .poseMode    'auto' (HRNet if available, else manual) | 'hrnet' | 'manual'
%     .mirror      true = selfie view (default true)
%     .smoothing   0..1 keypoint smoothing between frames (default 0.5;
%                  0 = none, higher = steadier but laggier)
%     .cfg         pipeline config (default defaultTryOnConfig())
%     .maxFrames   stop after this many frames (default Inf)

addpath(fileparts(mfilename('fullpath')));
if nargin < 2 || isempty(opts), opts = struct(); end
opts = fillDefaults(opts);
cfg = opts.cfg;

%% ---- Garment ---------------------------------------------------------------
if nargin >= 1 && ~isempty(garmentFile)
    [garmentImg, ~, garmentAlpha] = imread(garmentFile);
    if isempty(garmentAlpha)
        error('"%s" has no alpha channel. Use a PNG with a transparent background.', garmentFile);
    end
    fprintf('Garment: %s\n', garmentFile);
    [folder, name] = fileparts(garmentFile);
    anchorsFile = fullfile(folder, [name '_anchors.mat']);
    if exist(anchorsFile, 'file')
        loaded = load(anchorsFile, 'anchors');
        fn = fieldnames(loaded.anchors);
        for i = 1:numel(fn)
            cfg.garment.anchors.(fn{i}) = loaded.anchors.(fn{i});
        end
        fprintf('Anchors: loaded %s\n', anchorsFile);
    else
        fprintf(['Anchors: using defaults - run pickGarmentAnchors(''%s'') ' ...
                 'for a better fit.\n'], garmentFile);
    end
else
    [~, ~, garmentImg, garmentAlpha] = makeSyntheticTestData();
    fprintf('Garment: synthetic test T-shirt\n');
end

%% ---- Camera ------------------------------------------------------------------
try
    cam = webcam();
catch err
    error(['Could not open the webcam: %s\n' ...
           'Install "MATLAB Support Package for USB Webcams" (Add-Ons) and ' ...
           'close other apps using the camera (Teams, Zoom, browser).'], err.message);
end
frame = grabFrame(cam, opts.mirror);

%% ---- Pose model ----------------------------------------------------------------
poseMode = lower(opts.poseMode);
detector = [];
if any(strcmp(poseMode, {'auto', 'hrnet'}))
    try
        detector = hrnetObjectKeypointDetector();
        poseMode = 'hrnet';
        fprintf('Pose: HRNet keypoint detector (automatic)\n');
    catch err
        if strcmp(poseMode, 'hrnet')
            rethrow(err);
        end
        fprintf(['Pose: HRNet not available (%s)\n' ...
                 '      -> MANUAL mode: click your keypoints once, then stay still.\n'], err.message);
        poseMode = 'manual';
    end
end
if strcmp(poseMode, 'manual')
    manualKp = clickKeypoints(frame);
end

%% ---- Display window ------------------------------------------------------------
fig = figure('Name', 'Live try-on  (B = background, F/S/A/L = toggles, Q = quit)', ...
             'NumberTitle', 'off');
set(fig, 'KeyPressFcn', @(src, evt) setappdata(src, 'key', lower(evt.Key)));
setappdata(fig, 'key', '');
hImg = imshow(frame);
hTitle = title('starting...');
fprintf('Press B (with the video window focused) to capture the background for the body fit.\n');

%% ---- Main loop -----------------------------------------------------------------
smoothKp = [];
background = [];                 % empty-scene frame for background subtraction
bgCountdown = [];                % tic of a running background countdown
showMask = false;
nFrames = 0;
while ishandle(fig) && nFrames < opts.maxFrames
    tLoop = tic;
    frame = grabFrame(cam, opts.mirror);

    % ---- keyboard ----
    key = getappdata(fig, 'key');
    setappdata(fig, 'key', '');
    switch key
        case 'q', break
        case 'b', bgCountdown = tic;
        case 'f', cfg.fit.enable      = ~cfg.fit.enable;
        case 's', cfg.shading.enable  = ~cfg.shading.enable;
        case 'a', cfg.sleeves.enable  = ~cfg.sleeves.enable;
        case 'l', cfg.lighting.enable = ~cfg.lighting.enable;
        case 'm', showMask = ~showMask;
    end

    % ---- background capture: 3 s countdown so the user can step away ----
    if ~isempty(bgCountdown)
        remaining = 3 - toc(bgCountdown);
        if remaining > 0
            set(hImg, 'CData', frame);
            set(hTitle, 'String', sprintf('STEP OUT OF VIEW - capturing background in %.0f s', ceil(remaining)));
            drawnow;
            continue
        end
        background = captureBackground(cam, opts.mirror);
        bgCountdown = [];
        fprintf('Background captured. Step back in.\n');
    end

    % Person mask (only if a background was captured)
    bodyMask = [];
    if ~isempty(background)
        bodyMask = segmentPersonBackground(frame, background, cfg.segment);
    end

    % 1. Pose keypoints (COCO-17 layout: rows = keypoints, cols = [x y score])
    if strcmp(poseMode, 'hrnet')
        kp = detectPoseHRNet(detector, frame);
    else
        kp = manualKp;
    end

    % 2. Temporal smoothing: blend with the previous frame's keypoints so the
    %    garment does not jitter (exponential moving average).
    if ~isempty(kp) && ~isempty(smoothKp) && opts.smoothing > 0
        kp(:, 1:2) = opts.smoothing * smoothKp(:, 1:2) + (1 - opts.smoothing) * kp(:, 1:2);
    end
    smoothKp = kp;

    % 3. Warp -> light-match -> blend
    if isempty(kp)
        out = frame;
        status = 'no person detected';
    else
        [out, dbg] = tryOnPipeline(frame, kp, garmentImg, garmentAlpha, cfg, bodyMask);
        if dbg.ok
            status = 'tracking';
        else
            status = ['no overlay: ' dbg.reason];
        end
    end
    if showMask && ~isempty(bodyMask)
        % Tint the detected person green to check the segmentation.
        out(:, :, 2) = max(out(:, :, 2), uint8(160 * bodyMask));
    end

    % 4. Show (updating CData is much faster than calling imshow again)
    if ~ishandle(fig), break; end
    set(hImg, 'CData', out);
    set(hTitle, 'String', sprintf('%s | %.1f fps | fit:%s%s shade:%s arms:%s light:%s', ...
        status, 1 / toc(tLoop), onOff(cfg.fit.enable), noBgNote(background, cfg.fit.enable), ...
        onOff(cfg.shading.enable), onOff(cfg.sleeves.enable), onOff(cfg.lighting.enable)));
    drawnow limitrate;
    nFrames = nFrames + 1;
end
if ishandle(fig), close(fig); end
fprintf('Stopped after %d frames.\n', nFrames);
end

% ==========================================================================
function frame = grabFrame(cam, mirror)
frame = snapshot(cam);
if mirror
    % Selfie view. Mirroring BEFORE pose detection keeps keypoints, garment
    % and display consistent, and garment text still reads correctly.
    frame = fliplr(frame);
end
end

function background = captureBackground(cam, mirror)
% Median of a few frames = a noise-free picture of the empty scene.
nShots = 5;
shots = cell(1, nShots);
for i = 1:nShots
    shots{i} = grabFrame(cam, mirror);
end
background = median(cat(4, shots{:}), 4);
end

function s = onOff(flag)
if flag, s = 'on'; else, s = 'off'; end
end

function s = noBgNote(background, fitEnabled)
% Remind the user that the fit needs a background frame.
if fitEnabled && isempty(background), s = '(press B)'; else, s = ''; end
end

function kp = detectPoseHRNet(detector, frame)
% Run HRNet on the whole frame (assumes one person in view, so no separate
% person detector is needed). Returns a 17x3 [x y score] COCO array, or []
% if nothing was found.
bbox = [1 1 size(frame, 2) size(frame, 1)];
try
    [points, scores] = detect(detector, frame, bbox);
catch
    kp = [];
    return
end
if iscell(points), points = points{1}; end
if iscell(scores), scores = scores{1}; end
if isempty(points)
    kp = [];
    return
end
points = points(:, :, 1);          % first (only) person: 17x2
scores = scores(:, 1);             % 17x1
kp = [double(points) double(scores(:))];
end

function kp = clickKeypoints(frame)
% Manual stand-in for a pose model: the user clicks 4 points once.
f = figure('Name', 'Click your keypoints', 'NumberTitle', 'off');
imshow(frame);
title({'Click on the IMAGE, in this order:', ...
       '1) shoulder on the image LEFT   2) shoulder on the image RIGHT', ...
       '3) hip on the image RIGHT   4) hip on the image LEFT', ...
       '5) elbow on the image LEFT   6) elbow on the image RIGHT'});
[x, y] = ginput(6);
close(f);
% Build a COCO-17 array. Image-left shoulder = person's RIGHT shoulder for a
% front-facing person (also true in the mirrored view, see getTorsoKeypoints).
kp = zeros(17, 3);
kp(7,  :) = [x(1) y(1) 1];   % right shoulder
kp(6,  :) = [x(2) y(2) 1];   % left shoulder
kp(12, :) = [x(3) y(3) 1];   % left hip
kp(13, :) = [x(4) y(4) 1];   % right hip
kp(9,  :) = [x(5) y(5) 1];   % right elbow
kp(8,  :) = [x(6) y(6) 1];   % left elbow
end

function opts = fillDefaults(opts)
defaults = struct('poseMode', 'auto', 'mirror', true, 'smoothing', 0.5, ...
                  'cfg', [], 'maxFrames', Inf);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(opts, names{i}) || isempty(opts.(names{i}))
        opts.(names{i}) = defaults.(names{i});
    end
end
if isempty(opts.cfg)
    opts.cfg = defaultTryOnConfig();
end
end
