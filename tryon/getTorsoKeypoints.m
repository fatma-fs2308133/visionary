function torso = getTorsoKeypoints(poseKeypoints, frameSize, minScore)
%GETTORSOKEYPOINTS  Extract the 4 torso landmarks from a pose-estimator output.
%
%   torso = getTorsoKeypoints(poseKeypoints, frameSize, minScore)
%
%   This is the adapter between whatever pose model you use and the rest of
%   the pipeline, which only ever needs 4 points: both shoulders and both
%   hips, in MATLAB pixel coordinates [x y] (x = column, y = row, 1-based).
%
%   INPUTS
%     poseKeypoints  One of:
%       (a) struct with fields leftShoulder, rightShoulder, leftHip, rightHip,
%           each [x y] or [x y score].          <- easiest for stubbing/tests
%       (b) N-by-2 or N-by-3 numeric array [x y (score)], one row per keypoint,
%           where the layout is recognised from N:
%             N = 17 : COCO-17 (MATLAB hrnetObjectKeypointDetector, MMPose, ...)
%             N = 18 : OpenPose COCO-18
%             N = 25 : OpenPose BODY_25
%             N = 33 : MediaPipe BlazePose
%     frameSize      size(frame) - used to convert normalised coordinates.
%     minScore       keypoints with score < minScore count as missing
%                    (ignored if no score column is given). Default 0.3.
%
%   OUTPUT
%     torso  struct with fields leftShoulder, rightShoulder, leftHip,
%            rightHip (1x2 [x y] pixels), valid (logical) and reason (char,
%            why it is invalid). Also leftElbow / rightElbow with flags
%            hasLeftElbow / hasRightElbow: elbows are OPTIONAL (only used to
%            rotate the sleeves); a missing elbow never makes torso invalid.
%
%   ASSUMPTIONS
%   - If every x and y is <= 1.5, the coordinates are taken to be normalised
%     to [0,1] (MediaPipe's default) and are scaled by the frame size.
%   - 0-based pixel coordinates from Python tools are used as-is; the 1 px
%     offset is irrelevant at this scale.
%   - "left"/"right" are the person's anatomical sides, as every pose model
%     reports them. This also works for a mirrored (selfie) preview: the
%     model then simply sees a mirrored person and labels it consistently.

if nargin < 3 || isempty(minScore)
    minScore = 0.3;
end

% The first 4 are required, the elbows (5-6) are optional.
names = {'leftShoulder', 'rightShoulder', 'leftHip', 'rightHip', 'leftElbow', 'rightElbow'};
nRequired = 4;
torso = struct('leftShoulder', [NaN NaN], 'rightShoulder', [NaN NaN], ...
               'leftHip', [NaN NaN], 'rightHip', [NaN NaN], ...
               'leftElbow', [NaN NaN], 'rightElbow', [NaN NaN], ...
               'hasLeftElbow', false, 'hasRightElbow', false, ...
               'valid', false, 'reason', '');

% ---- 1. Collect the raw rows [x y score] -----------------------------------
raw = NaN(numel(names), 3);         % one row per name, score NaN = "no score"
if isstruct(poseKeypoints)
    for i = 1:numel(names)
        if ~isfield(poseKeypoints, names{i}) || isempty(poseKeypoints.(names{i}))
            if i > nRequired
                continue            % optional elbow not given -> stays NaN
            end
            torso.reason = sprintf('missing field "%s"', names{i});
            return
        end
        p = double(poseKeypoints.(names{i}));
        raw(i, 1:numel(p)) = p(:)';
    end

elseif isnumeric(poseKeypoints)
    kp = double(poseKeypoints);
    % Row indices (1-based) of
    % [leftShoulder rightShoulder leftHip rightHip leftElbow rightElbow]
    switch size(kp, 1)
        case 17, idx = [6 7 12 13 8 9];      % COCO-17
        case 18, idx = [6 3 12 9 7 4];       % OpenPose COCO-18
        case 25, idx = [6 3 13 10 7 4];      % OpenPose BODY_25
        case 33, idx = [12 13 24 25 14 15];  % MediaPipe BlazePose
        otherwise
            torso.reason = sprintf('unknown keypoint layout with %d rows', size(kp, 1));
            return
    end
    nCols = min(size(kp, 2), 3);
    raw(:, 1:nCols) = kp(idx, 1:nCols);

else
    torso.reason = 'poseKeypoints must be a struct or a numeric array';
    return
end

% ---- 2. Validity: finite coordinates and a high enough score ---------------
xy    = raw(:, 1:2);
score = raw(:, 3);
% NaN scores compare false -> "no score given" counts as confident.
ok = all(isfinite(xy), 2) & ~(score < minScore);
if any(~all(isfinite(xy(1:nRequired, :)), 2))
    torso.reason = 'a torso keypoint is NaN/Inf (not detected)';
    return
end
if ~all(ok(1:nRequired))
    torso.reason = 'a torso keypoint is below the confidence threshold';
    return
end

% ---- 3. Normalised -> pixel coordinates ------------------------------------
req = xy(1:nRequired, :);
if all(req(:) <= 1.5)
    xy(:, 1) = xy(:, 1) * frameSize(2);     % x scales with width
    xy(:, 2) = xy(:, 2) * frameSize(1);     % y scales with height
end

for i = 1:numel(names)
    torso.(names{i}) = xy(i, :);
end
torso.hasLeftElbow  = ok(5);
torso.hasRightElbow = ok(6);
torso.valid = true;
end
