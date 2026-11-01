classdef PoseTracker < handle
%POSETRACKER  Body keypoints for every frame: detect (HRNet) + track (KLT).
%
%   tracker = PoseTracker('auto')      % HRNet if installed, else manual
%   tracker = PoseTracker('hrnet')     % HRNet, error if not installed
%   tracker = PoseTracker('manual')    % user clicks the points once
%   tracker = PoseTracker('function', 'DetectFcn', @myDetector)  % tests
%   tracker = PoseTracker(..., 'DetectEvery', 5, 'UseKLT', true, 'Smoothing', 0.5)
%
%   [kp, info] = tracker.step(frame)   % call once per frame
%
%   kp is a 17x3 COCO-17 array [x y score] (the layout getTorsoKeypoints
%   reads), or [] when nobody is found.
%
%   WHY DETECT + TRACK?
%   The pretrained HRNet keypoint detector is accurate but slow on a CPU
%   (100-400 ms per frame), so on its own it limits the system to a few
%   frames per second. Between two detections the body moves only a few
%   pixels, which the KLT (Kanade-Lucas-Tomasi) optical-flow tracker
%   (vision.PointTracker) follows in a few milliseconds. So:
%     - every DetectEvery-th frame: run HRNet (corrects any tracking drift)
%     - the frames in between:      track the keypoints with KLT
%     - a torso point lost by KLT:  detect again on the very next frame
%   In manual mode there is no detector: the clicked points are tracked by
%   KLT so the garment still follows the person.
%
%   HRNet needs a person bounding box. We use the box around the previous
%   keypoints when we have them, otherwise MATLAB's ACF people detector,
%   otherwise the whole frame.
%
%   See also hrnetObjectKeypointDetector, vision.PointTracker,
%   peopleDetectorACF, getTorsoKeypoints, clickTorsoKeypoints.

    properties
        DetectEvery = 5      % run the detector every N frames (1 = always)
        UseKLT      = true   % track between detections
        Smoothing   = 0.5    % exponential smoothing of the output (0..1)
        MinScore    = 0.3    % keypoints below this score are not tracked
    end

    properties (SetAccess = private)
        Mode = 'manual'      % 'hrnet' | 'manual' | 'function'
        Detector = []        % hrnetObjectKeypointDetector
        PersonDetector = []  % peopleDetectorACF (created on first use)
        DetectFcn = []       % function handle, 'function' mode
        PointTracker = []    % vision.PointTracker
        TrackedRows = []     % rows of Kp currently tracked by KLT
        Kp = []              % latest keypoints (17x3, not smoothed)
        SmoothKp = []        % latest smoothed output
        FramesSinceDetect = Inf
        Lost = false         % KLT lost a torso point -> detect next frame
    end

    properties (Constant)
        TorsoRows = [6 7 12 13]   % COCO-17: L/R shoulder, L/R hip
    end

    methods
        function obj = PoseTracker(mode, varargin)
            if nargin < 1 || isempty(mode), mode = 'auto'; end
            p = inputParser;
            p.addParameter('DetectFcn', []);
            p.addParameter('DetectEvery', 5);
            p.addParameter('UseKLT', true);
            p.addParameter('Smoothing', 0.5);
            p.addParameter('MinScore', 0.3);
            p.parse(varargin{:});
            obj.DetectEvery = p.Results.DetectEvery;
            obj.UseKLT      = p.Results.UseKLT;
            obj.Smoothing   = p.Results.Smoothing;
            obj.MinScore    = p.Results.MinScore;

            switch lower(mode)
                case {'auto', 'hrnet'}
                    try
                        obj.Detector = hrnetObjectKeypointDetector();
                        obj.Mode = 'hrnet';
                    catch err
                        if strcmpi(mode, 'hrnet')
                            rethrow(err);
                        end
                        obj.Mode = 'manual';
                    end
                case 'manual'
                    obj.Mode = 'manual';
                case 'function'
                    if isempty(p.Results.DetectFcn)
                        error('PoseTracker:noFcn', 'Mode "function" needs a ''DetectFcn''.');
                    end
                    obj.DetectFcn = p.Results.DetectFcn;
                    obj.Mode = 'function';
                otherwise
                    error('PoseTracker:mode', 'Unknown mode "%s".', mode);
            end
        end

        function tf = hasDetector(obj)
            %HASDETECTOR  true if keypoints can be found automatically.
            tf = ~strcmp(obj.Mode, 'manual');
        end

        function tf = needsManualKeypoints(obj)
            %NEEDSMANUALKEYPOINTS  manual mode and nothing clicked yet.
            tf = strcmp(obj.Mode, 'manual') && isempty(obj.Kp);
        end

        function setManualKeypoints(obj, kp, frame)
            %SETMANUALKEYPOINTS  Start from user-clicked keypoints (17x3).
            obj.Kp = double(kp);
            obj.SmoothKp = obj.Kp;
            obj.Lost = false;
            obj.FramesSinceDetect = 1;
            obj.initTracker(toGray(frame));
        end

        function reset(obj)
            %RESET  Forget everything (e.g. after a new garment or camera).
            obj.Kp = [];
            obj.SmoothKp = [];
            obj.TrackedRows = [];
            obj.FramesSinceDetect = Inf;
            obj.Lost = false;
            if ~isempty(obj.PointTracker)
                release(obj.PointTracker);
            end
        end

        function [kp, info] = step(obj, frame)
            %STEP  Keypoints for this frame. info: source, detectTime, trackTime.
            info = struct('source', 'none', 'detectTime', 0, 'trackTime', 0);
            gray = toGray(frame);
            isManual = strcmp(obj.Mode, 'manual');

            needDetect = ~isManual && (isempty(obj.Kp) || obj.Lost || ~obj.UseKLT || ...
                                       obj.FramesSinceDetect >= obj.DetectEvery);
            if needDetect
                t = tic;
                kpNew = obj.detect(frame);
                info.detectTime = toc(t);
                if isempty(kpNew)
                    obj.Kp = [];
                    obj.SmoothKp = [];
                else
                    obj.Kp = kpNew;
                    obj.FramesSinceDetect = 1;
                    obj.Lost = false;
                    if obj.UseKLT
                        obj.initTracker(gray);
                    end
                    info.source = 'detector';
                end
            elseif ~isempty(obj.Kp)
                if obj.UseKLT && ~isempty(obj.TrackedRows)
                    t = tic;
                    obj.track(gray);
                    info.trackTime = toc(t);
                    info.source = 'tracker';
                else
                    info.source = 'manual';
                end
                obj.FramesSinceDetect = obj.FramesSinceDetect + 1;
            end

            % Temporal smoothing (exponential moving average) against jitter.
            if isempty(obj.Kp)
                kp = [];
                return
            end
            if isempty(obj.SmoothKp) || obj.Smoothing <= 0 || ~isequal(size(obj.SmoothKp), size(obj.Kp))
                obj.SmoothKp = obj.Kp;
            else
                s = obj.Smoothing;
                obj.SmoothKp(:, 1:2) = s * obj.SmoothKp(:, 1:2) + (1 - s) * obj.Kp(:, 1:2);
                obj.SmoothKp(:, 3) = obj.Kp(:, 3);
            end
            kp = obj.SmoothKp;
        end
    end

    methods (Access = private)
        function kp = detect(obj, frame)
            switch obj.Mode
                case 'function'
                    kp = obj.DetectFcn(frame);
                    if isstruct(kp)
                        kp = structToCoco(kp);
                    end
                case 'hrnet'
                    kp = obj.detectHRNet(frame);
                otherwise
                    kp = [];
            end
            if ~isempty(kp)
                kp = double(kp);
                if size(kp, 2) == 2
                    kp(:, 3) = 1;                 % no scores given: confident
                end
            end
        end

        function kp = detectHRNet(obj, frame)
            bbox = obj.personBox(frame);
            try
                [points, scores] = detect(obj.Detector, frame, bbox);
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
            points = double(points(:, :, 1));     % first person: 17x2
            if size(points, 2) ~= 2 && size(points, 1) == 2
                points = points';
            end
            scores = double(scores);
            scores = scores(:, 1);
            kp = [points, scores(1:size(points, 1))];
        end

        function bbox = personBox(obj, frame)
            % [x y w h] box around the person for HRNet.
            H = size(frame, 1);  W = size(frame, 2);
            bbox = [1 1 W H];
            if ~isempty(obj.Kp)
                ok = obj.Kp(:, 3) >= obj.MinScore & all(isfinite(obj.Kp(:, 1:2)), 2);
                if nnz(ok) >= 4
                    p = obj.Kp(ok, 1:2);
                    x1 = min(p(:, 1));  x2 = max(p(:, 1));
                    y1 = min(p(:, 2));  y2 = max(p(:, 2));
                    w = x2 - x1;  h = y2 - y1;
                    % Generous margin: the person may have moved, and the
                    % head / arms may not all have been detected.
                    x1 = x1 - 0.4 * w;  x2 = x2 + 0.4 * w;
                    y1 = y1 - 0.3 * h;  y2 = y2 + 0.3 * h;
                    x1 = max(1, x1);  y1 = max(1, y1);
                    x2 = min(W, x2);  y2 = min(H, y2);
                    if x2 - x1 > 20 && y2 - y1 > 20
                        bbox = [x1 y1 x2 - x1 y2 - y1];
                        return
                    end
                end
            end
            % No previous keypoints: look for a person with the ACF detector.
            try
                if isempty(obj.PersonDetector)
                    obj.PersonDetector = peopleDetectorACF();
                end
                [boxes, scores] = detect(obj.PersonDetector, frame);
                if ~isempty(boxes)
                    [~, best] = max(scores);
                    b = boxes(best, :);
                    % ACF boxes are tight: widen them a little.
                    b = [b(1) - 0.15 * b(3), b(2) - 0.1 * b(4), 1.3 * b(3), 1.2 * b(4)];
                    x1 = max(1, b(1));  y1 = max(1, b(2));
                    x2 = min(W, b(1) + b(3));  y2 = min(H, b(2) + b(4));
                    bbox = [x1 y1 x2 - x1 y2 - y1];
                end
            catch
                % keep the whole-frame box
            end
        end

        function initTracker(obj, gray)
            ok = obj.Kp(:, 3) >= obj.MinScore & all(isfinite(obj.Kp(:, 1:2)), 2);
            obj.TrackedRows = find(ok);
            if isempty(obj.TrackedRows)
                return
            end
            pts = obj.Kp(obj.TrackedRows, 1:2);
            pts(:, 1) = min(max(pts(:, 1), 1), size(gray, 2));
            pts(:, 2) = min(max(pts(:, 2), 1), size(gray, 1));
            if isempty(obj.PointTracker)
                % Large block: shoulders/hips sit on clothing with little
                % texture, so the tracker needs a big window to lock on.
                obj.PointTracker = vision.PointTracker('MaxBidirectionalError', 3, ...
                    'BlockSize', [31 31], 'NumPyramidLevels', 3);
            else
                release(obj.PointTracker);
            end
            initialize(obj.PointTracker, pts, gray);
        end

        function track(obj, gray)
            [pts, valid] = obj.PointTracker(gray);
            rows = obj.TrackedRows;
            obj.Kp(rows(valid), 1:2) = pts(valid, :);
            if strcmp(obj.Mode, 'manual')
                return                            % no detector: keep lost points where they were
            end
            obj.Kp(rows(~valid), 3) = 0;          % lost -> not confident
            if any(ismember(obj.TorsoRows, rows(~valid)))
                obj.Lost = true;                  % re-detect next frame
            end
        end
    end
end

% ==========================================================================
function g = toGray(frame)
if size(frame, 3) == 3
    g = rgb2gray(frame);
else
    g = frame;
end
end

function kp = structToCoco(s)
% Struct with named keypoints -> 17x3 COCO array (missing rows score 0).
names = {'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftHip', 'rightHip'};
rows  = [6 7 8 9 12 13];
kp = zeros(17, 3);
for i = 1:numel(names)
    if isfield(s, names{i}) && ~isempty(s.(names{i}))
        v = double(s.(names{i}));
        if numel(v) == 2, v(3) = 1; end
        kp(rows(i), :) = v(1:3);
    end
end
end
