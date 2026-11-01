function sessionDir = captureTestFrames(sessionName, opts)
%CAPTURETESTFRAMES  Record a labelled test session from the webcam.
%
%   captureTestFrames('habiba_bright')
%   captureTestFrames('ashley_dim', struct('numFrames', 40, 'interval', 0.5))
%
%   Name each session <person>_<condition>, e.g. habiba_bright, fatma_dim,
%   ashley_backlit, mahajuba_far. runExperiments groups the results by the
%   part after the last underscore (the lighting/condition label).
%
%   What it does:
%     1. 5-second countdown: STEP OUT of view -> background.png (median of
%        5 frames of the empty scene, needed for the body-shape fit)
%     2. 3-second countdown: step back in
%     3. numFrames frames, one every interval seconds -> frame_0001.png, ...
%        Move a little between frames: turn, lean, raise the arms, step
%        closer/further, so the test set covers realistic poses.
%     4. session.mat with the date, camera name and resolution, and the
%        condition label.
%
%   PRIVACY: only record people who have agreed to it (team members), and
%   keep the frames out of any public repository.
%
%   opts (all optional):
%     .numFrames  frames to record           (60)
%     .interval   seconds between frames     (0.5)
%     .mirror     selfie view                (true)
%     .root       parent folder              ('data/test_frames')
%     .camera     webcam name or index       (1)
%
%   See also annotateTorso, runExperiments.

if nargin < 1 || isempty(sessionName)
    error('captureTestFrames:name', 'Give the session a name, e.g. ''habiba_bright''.');
end
if nargin < 2 || isempty(opts), opts = struct(); end
defaults = struct('numFrames', 60, 'interval', 0.5, 'mirror', true, ...
                  'root', fullfile('data', 'test_frames'), 'camera', 1);
fn = fieldnames(defaults);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}), opts.(fn{i}) = defaults.(fn{i}); end
end

sessionDir = fullfile(opts.root, sessionName);
if exist(sessionDir, 'dir') && ~isempty(dir(fullfile(sessionDir, 'frame_*.png')))
    error('captureTestFrames:exists', 'Session "%s" already exists. Pick another name.', sessionName);
end
mkdir(sessionDir);

try
    cam = webcam(opts.camera);
catch err
    error(['Could not open the webcam: %s\nInstall "MATLAB Support Package for USB ' ...
           'Webcams" (Add-Ons) and close other apps using the camera.'], err.message);
end
cleanup = onCleanup(@() clear('cam'));

fig = figure('Name', ['Recording ' sessionName], 'NumberTitle', 'off');
hImg = imshow(grab(cam, opts.mirror));
hTitle = title('');

% 1. Background
countdown(cam, opts.mirror, hImg, hTitle, 5, 'STEP OUT OF VIEW - background in %d s');
shots = cell(1, 5);
for i = 1:5, shots{i} = grab(cam, opts.mirror); end
imwrite(median(cat(4, shots{:}), 4), fullfile(sessionDir, 'background.png'));

% 2. Frames
countdown(cam, opts.mirror, hImg, hTitle, 3, 'STEP IN - recording starts in %d s');
for k = 1:opts.numFrames
    if ~ishandle(fig), break; end
    t = tic;
    frame = grab(cam, opts.mirror);
    imwrite(frame, fullfile(sessionDir, sprintf('frame_%04d.png', k)));
    set(hImg, 'CData', frame);
    set(hTitle, 'String', sprintf('frame %d / %d - move a little (turn, lean, arms)', k, opts.numFrames));
    drawnow;
    pause(max(0, opts.interval - toc(t)));
end

% 3. Session info
parts = split(sessionName, '_');
info.name = sessionName;
info.condition = parts{end};
info.date = datetime('now');
info.camera = cam.Name;
info.resolution = cam.Resolution;
info.mirror = opts.mirror;
save(fullfile(sessionDir, 'session.mat'), 'info');
if ishandle(fig), close(fig); end
fprintf('Saved %d frames to %s\nNext: annotateTorso(''%s'')\n', k, sessionDir, sessionDir);
end

% ==========================================================================
function frame = grab(cam, mirror)
frame = snapshot(cam);
if mirror, frame = fliplr(frame); end
end

function countdown(cam, mirror, hImg, hTitle, seconds, msg)
t = tic;
while toc(t) < seconds
    set(hImg, 'CData', grab(cam, mirror));
    set(hTitle, 'String', sprintf(msg, ceil(seconds - toc(t))));
    drawnow;
end
end
