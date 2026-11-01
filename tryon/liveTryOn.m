function liveTryOn(garments, opts)
%LIVETRYON  Live webcam try-on in a plain figure window (keyboard controls).
%
%   liveTryOn                      % all garments in data/garments
%   liveTryOn('shirt.png')         % one garment PNG (with transparency)
%   liveTryOn('my_garments')       % every PNG in a folder
%   liveTryOn(garments, opts)      % options, see below
%
%   For the full graphical interface use TryOnApp; this script is the
%   lightweight version (and handy for measuring the frame rate).
%
%   KEYS (click the video window first):
%     B  capture the empty background (3-second countdown: step OUT of view).
%        Needed for the body-shape fit. Press again if the light changes.
%     G  next garment                W  next warp model (homography/piecewise/affine)
%     F  body-shape fit on/off       S  shading (folds) on/off
%     A  sleeves follow arms on/off  L  lighting match on/off
%     P  preprocessing on/off        M  show the person mask
%     K  click the keypoints again (manual mode)
%     Q  quit (or close the window)
%
%   REQUIREMENTS
%     - MATLAB Support Package for USB Webcams  (for webcam / snapshot)
%       Home > Add-Ons > search "USB Webcams".
%     - For automatic pose: Computer Vision Toolbox + Deep Learning Toolbox
%       and the "Computer Vision Toolbox Model for Object Keypoint Detection"
%       add-on (hrnetObjectKeypointDetector, R2023b+).
%       Without it the script falls back to MANUAL mode: you click your
%       shoulders, hips and elbows once and KLT tracking follows you.
%
%   STAND BACK: the pose model needs to see your HIPS. At a desk the webcam
%   usually only sees your chest, the hips are missing and nothing is drawn.
%
%   opts fields (all optional):
%     .poseMode    'auto' (HRNet if available, else manual) | 'hrnet' | 'manual'
%     .mirror      true = selfie view (default true)
%     .cfg         pipeline config (default defaultTryOnConfig())
%     .maxFrames   stop after this many frames (default Inf)
%     .logFile     CSV file for per-frame timings (default '' = no log)
%
%   See also TryOnApp, TryOnSession, PoseTracker.

root = fileparts(mfilename('fullpath'));
addpath(root);
if nargin < 2 || isempty(opts), opts = struct(); end
opts = fillDefaults(opts);

%% ---- Garments ---------------------------------------------------------------
if nargin < 1 || isempty(garments)
    garments = fullfile(root, 'data', 'garments');
    if isempty(loadGarmentLibrary(garments))
        makeSampleGarments(garments);
    end
end
lib = loadLibrary(garments);
fprintf('Garments: %s\n', strjoin({lib.name}, ', '));

%% ---- Camera, pose, session ----------------------------------------------------
try
    cam = webcam();
catch err
    error(['Could not open the webcam: %s\n' ...
           'Install "MATLAB Support Package for USB Webcams" (Add-Ons) and ' ...
           'close other apps using the camera (Teams, Zoom, browser).'], err.message);
end
tracker = PoseTracker(opts.poseMode);
if tracker.hasDetector()
    fprintf('Pose: HRNet every %d frames + KLT tracking\n', opts.cfg.tracking.detectEvery);
else
    fprintf('Pose: MANUAL - click your keypoints once, KLT tracking follows you.\n');
end
session = TryOnSession(lib, tracker, opts.cfg);
frame = session.preprocess(grabFrame(cam, opts.mirror));
if ~tracker.hasDetector()
    tracker.setManualKeypoints(clickTorsoKeypoints(frame), frame);
end

%% ---- Display window ------------------------------------------------------------
fig = figure('Name', 'Live try-on  (B background, G garment, W warp, Q quit)', ...
             'NumberTitle', 'off');
set(fig, 'KeyPressFcn', @(src, evt) setappdata(src, 'key', lower(evt.Key)));
setappdata(fig, 'key', '');
hImg = imshow(frame);
hTitle = title('starting...');

%% ---- Main loop -----------------------------------------------------------------
warpModes = {'homography', 'piecewise', 'affine'};
log = zeros(0, 7);
bgCountdown = [];
showMask = false;
nFrames = 0;
while ishandle(fig) && nFrames < opts.maxFrames
    tLoop = tic;
    raw = grabFrame(cam, opts.mirror);
    c = session.Cfg;

    % ---- keyboard ----
    key = getappdata(fig, 'key');
    setappdata(fig, 'key', '');
    switch key
        case 'q', break
        case 'b', bgCountdown = tic;
        case 'g', session.GarmentIndex = mod(session.GarmentIndex, numel(lib)) + 1;
        case 'w'
            k = find(strcmp(warpModes, c.warp.mode));
            session.Cfg.warp.mode = warpModes{mod(k, numel(warpModes)) + 1};
        case 'f', session.Cfg.fit.enable        = ~c.fit.enable;
        case 's', session.Cfg.shading.enable    = ~c.shading.enable;
        case 'a', session.Cfg.sleeves.enable    = ~c.sleeves.enable;
        case 'l', session.Cfg.lighting.enable   = ~c.lighting.enable;
        case 'p', session.Cfg.preprocess.enable = ~c.preprocess.enable;
        case 'm', showMask = ~showMask;
        case 'k'
            f = session.preprocess(raw);
            kp = clickTorsoKeypoints(f);
            if ~isempty(kp), tracker.setManualKeypoints(kp, f); end
            continue
    end

    % ---- background capture: 3 s countdown so the user can step away ----
    if ~isempty(bgCountdown)
        remaining = 3 - toc(bgCountdown);
        if remaining > 0
            set(hImg, 'CData', session.preprocess(raw));
            set(hTitle, 'String', sprintf('STEP OUT OF VIEW - capturing background in %.0f s', ceil(remaining)));
            drawnow;
            continue
        end
        shots = cell(1, 5);
        for i = 1:5, shots{i} = grabFrame(cam, opts.mirror); end
        session.setBackground(shots);
        tracker.reset();
        if ~tracker.hasDetector()
            f = session.preprocess(grabFrame(cam, opts.mirror));
            tracker.setManualKeypoints(clickTorsoKeypoints(f), f);
        end
        bgCountdown = [];
        fprintf('Background captured.\n');
        continue
    end

    % ---- one frame: preprocess -> pose -> mask -> try-on ----
    [out, info] = session.processFrame(raw);
    if showMask && ~isempty(info.bodyMask)
        out(:, :, 2) = max(out(:, :, 2), uint8(160 * info.bodyMask));
    end

    if ~ishandle(fig), break; end
    if ~isequal(size(get(hImg, 'CData')), size(out))
        hImg = imshow(out);
        hTitle = title('');
    end
    set(hImg, 'CData', out);
    loopTime = toc(tLoop);
    c = session.Cfg;
    set(hTitle, 'String', sprintf('%s | %.1f fps | %s | %s | fit:%s%s shade:%s arms:%s light:%s pre:%s', ...
        info.status, 1 / loopTime, strrep(session.currentGarment().name, '_', ' '), c.warp.mode, ...
        onOff(c.fit.enable), noBgNote(session.Background, c.fit.enable), onOff(c.shading.enable), ...
        onOff(c.sleeves.enable), onOff(c.lighting.enable), onOff(c.preprocess.enable)), ...
        'Interpreter', 'none');
    drawnow limitrate;
    nFrames = nFrames + 1;
    t = info.timing;
    log(end + 1, :) = [nFrames, t.preprocess, t.pose, t.segment, t.pipeline, loopTime, info.ok]; %#ok<AGROW>
end
if ishandle(fig), close(fig); end
clear cam

fprintf('Stopped after %d frames.\n', nFrames);
if nFrames > 1
    fprintf('Mean frame rate: %.1f fps (excluding the first frame)\n', 1 / mean(log(2:end, 6)));
end
if ~isempty(opts.logFile) && nFrames > 0
    T = array2table(log, 'VariableNames', {'frame', 'preprocess', 'pose', 'segment', ...
                                           'pipeline', 'loop', 'overlayOk'});
    writetable(T, opts.logFile);
    fprintf('Timings written to %s\n', opts.logFile);
end
end

% ==========================================================================
function lib = loadLibrary(garments)
% A folder of PNGs, or a single PNG file.
if isfolder(garments)
    lib = loadGarmentLibrary(garments);
else
    [folder, name] = fileparts(garments);
    if isempty(folder), folder = '.'; end
    lib = loadGarmentLibrary(folder);
    lib = lib(strcmp({lib.name}, name));
end
if isempty(lib)
    error('liveTryOn:noGarment', ['No usable garment in "%s" (needs PNGs with ' ...
          'transparency - see prepareGarment).'], garments);
end
end

function frame = grabFrame(cam, mirror)
frame = snapshot(cam);
if mirror
    % Selfie view. Mirroring BEFORE pose detection keeps keypoints, garment
    % and display consistent, and garment text still reads correctly.
    frame = fliplr(frame);
end
end

function s = onOff(flag)
if flag, s = 'on'; else, s = 'off'; end
end

function s = noBgNote(background, fitEnabled)
% Remind the user that the fit needs a background frame.
if fitEnabled && isempty(background), s = '(press B)'; else, s = ''; end
end

function opts = fillDefaults(opts)
defaults = struct('poseMode', 'auto', 'mirror', true, 'cfg', [], 'maxFrames', Inf, 'logFile', '');
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
