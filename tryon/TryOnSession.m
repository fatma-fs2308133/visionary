classdef TryOnSession < handle
%TRYONSESSION  Everything that happens to one camera frame, in one place.
%
%   session = TryOnSession(garments, tracker)
%   session = TryOnSession(garments, tracker, cfg)
%   [out, info] = session.processFrame(rawFrame)
%
%   Used by TryOnApp (GUI), liveTryOn (script) and runExperiments, so all
%   three run exactly the same processing:
%
%     raw frame -> preprocessFrame -> PoseTracker.step (HRNet + KLT)
%               -> segmentPersonBackground (if a background was captured)
%               -> tryOnPipeline (warp, fit, sleeves, light, shading, blend)
%
%   INPUTS
%     garments  struct array from loadGarmentLibrary (fields rgb, alpha,
%               anchors, name).
%     tracker   a PoseTracker.
%     cfg       (optional) defaultTryOnConfig() or a modified copy.
%
%   info (per frame) has: frame (the preprocessed input), kp, torso,
%   confidence (mean score of the 4 torso keypoints, 0..1), source
%   ('detector' | 'tracker' | 'manual' | 'none'), status text, ok, bodyMask,
%   dbg (from tryOnPipeline) and timing (seconds: preprocess, pose,
%   segment, pipeline, total).
%
%   See also TryOnApp, liveTryOn, runExperiments.

    properties
        Cfg                  % pipeline configuration (change any field live)
        Garments             % struct array (loadGarmentLibrary)
        GarmentIndex = 1     % garment currently worn
        Tracker              % PoseTracker
    end

    properties (SetAccess = private)
        Background = []      % empty-scene frame (resized + denoised, NOT enhanced)
    end

    methods
        function obj = TryOnSession(garments, tracker, cfg)
            if nargin < 3 || isempty(cfg), cfg = defaultTryOnConfig(); end
            if isempty(garments)
                error('TryOnSession:noGarments', 'The garment library is empty.');
            end
            obj.Garments = garments;
            obj.Tracker = tracker;
            obj.Cfg = cfg;
        end

        function g = currentGarment(obj)
            obj.GarmentIndex = min(max(obj.GarmentIndex, 1), numel(obj.Garments));
            g = obj.Garments(obj.GarmentIndex);
        end

        function [frame, plain] = preprocess(obj, rawFrame)
            %PREPROCESS  The preprocessing step alone.
            %   frame  fully preprocessed (what the pose model and the
            %          try-on see)
            %   plain  resized + denoised, not contrast-enhanced (what the
            %          background subtraction compares)
            %   With preprocessing switched off the frame is still resized
            %   to the working width: a 4000 px phone photo would otherwise
            %   be slow and every pixel-based parameter would be off.
            p = obj.Cfg.preprocess;
            if ~p.enable
                p.denoise = 'none';
                p.enhance = 'off';
            end
            [frame, ~, plain] = preprocessFrame(rawFrame, p);
        end

        function setBackground(obj, rawFrames)
            %SETBACKGROUND  Empty-scene picture = median of a few raw frames.
            %   rawFrames: one frame, or a cell array of frames.
            if ~iscell(rawFrames), rawFrames = {rawFrames}; end
            plains = cell(size(rawFrames));
            for i = 1:numel(rawFrames)
                [~, plains{i}] = obj.preprocess(rawFrames{i});
            end
            obj.Background = median(cat(4, plains{:}), 4);
        end

        function clearBackground(obj)
            obj.Background = [];
        end

        function [out, info] = processFrame(obj, rawFrame)
            tTotal = tic;
            cfg = obj.Cfg;
            info = struct('frame', [], 'kp', [], 'torso', [], 'confidence', 0, ...
                          'source', 'none', 'status', '', 'ok', false, ...
                          'bodyMask', [], 'dbg', [], 'timing', []);
            timing = struct('preprocess', 0, 'pose', 0, 'segment', 0, 'pipeline', 0, 'total', 0);

            % 1. Preprocessing
            t = tic;
            [frame, plain] = obj.preprocess(rawFrame);
            timing.preprocess = toc(t);
            info.frame = frame;

            % 2. Pose keypoints (detect or track)
            t = tic;
            obj.Tracker.MinScore  = cfg.pose.minScore;
            obj.Tracker.Smoothing = cfg.tracking.smoothing;
            obj.Tracker.DetectEvery = cfg.tracking.detectEvery;
            obj.Tracker.UseKLT = cfg.tracking.useKLT;
            [kp, poseInfo] = obj.Tracker.step(frame);
            timing.pose = toc(t);
            info.kp = kp;
            info.source = poseInfo.source;
            if ~isempty(kp) && size(kp, 1) == 17
                info.confidence = mean(kp(PoseTracker.TorsoRows, 3));
            end

            % 3. Person mask (only with a background frame)
            t = tic;
            bodyMask = [];
            if ~isempty(obj.Background) && (cfg.fit.enable)
                bg = obj.Background;
                if ~isequal(size(bg), size(plain))
                    bg = imresize(bg, [size(plain, 1) size(plain, 2)]);
                end
                bodyMask = segmentPersonBackground(plain, bg, cfg.segment);
            end
            timing.segment = toc(t);
            info.bodyMask = bodyMask;

            % 4. Try-on pipeline
            t = tic;
            if isempty(kp)
                out = frame;
                if obj.Tracker.needsManualKeypoints()
                    info.status = 'click the keypoints first';
                else
                    info.status = 'no person detected';
                end
            else
                g = obj.currentGarment();
                cfg.garment.anchors = g.anchors;
                [out, dbg] = tryOnPipeline(frame, kp, g.rgb, g.alpha, cfg, bodyMask);
                info.dbg = dbg;
                info.torso = dbg.torso;
                info.ok = dbg.ok;
                if dbg.ok
                    info.status = 'tracking';
                else
                    info.status = ['no overlay: ' dbg.reason];
                end
            end
            timing.pipeline = toc(t);
            timing.total = toc(tTotal);
            info.timing = timing;
        end
    end
end
