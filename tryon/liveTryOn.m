function liveTryOn(garmentFile, opts)
%LIVETRYON  Live webcam try-on: camera -> pose -> tryOnPipeline -> screen.
%
%   liveTryOn                    % synthetic striped T-shirt
%   liveTryOn('shirt.png')       % your own garment PNG (with transparency)
%   liveTryOn('shirt.png', opts) % options, see below
%
%   Close the window or press Q to stop.
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
    fprintf('Garment: %s  (check cfg.garment.anchors match this image!)\n', garmentFile);
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
fig = figure('Name', 'Live try-on  (press Q or close to stop)', 'NumberTitle', 'off');
set(fig, 'KeyPressFcn', @(src, evt) setappdata(src, 'stop', strcmpi(evt.Key, 'q')));
setappdata(fig, 'stop', false);
hImg = imshow(frame);
hTitle = title('starting...');

%% ---- Main loop -----------------------------------------------------------------
smoothKp = [];
nFrames = 0;
while ishandle(fig) && ~getappdata(fig, 'stop') && nFrames < opts.maxFrames
    tLoop = tic;
    frame = grabFrame(cam, opts.mirror);

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
        [out, dbg] = tryOnPipeline(frame, kp, garmentImg, garmentAlpha, cfg);
        if dbg.ok
            status = 'tracking';
        else
            status = ['no overlay: ' dbg.reason];
        end
    end

    % 4. Show (updating CData is much faster than calling imshow again)
    if ~ishandle(fig), break; end
    set(hImg, 'CData', out);
    set(hTitle, 'String', sprintf('%s  |  %.1f fps  |  Q = quit', status, 1 / toc(tLoop)));
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
       '3) hip on the image RIGHT   4) hip on the image LEFT'});
[x, y] = ginput(4);
close(f);
% Build a COCO-17 array. Image-left shoulder = person's RIGHT shoulder for a
% front-facing person (also true in the mirrored view, see getTorsoKeypoints).
kp = zeros(17, 3);
kp(7,  :) = [x(1) y(1) 1];   % right shoulder
kp(6,  :) = [x(2) y(2) 1];   % left shoulder
kp(12, :) = [x(3) y(3) 1];   % left hip
kp(13, :) = [x(4) y(4) 1];   % right hip
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
