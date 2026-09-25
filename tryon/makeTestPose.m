function kp = makeTestPose(baseKp, variant)
%MAKETESTPOSE  Fake body movements by moving the 4 torso keypoints.
%
%   kp = makeTestPose(baseKp, variant)
%
%   baseKp   keypoint struct (leftShoulder, rightShoulder, leftHip, rightHip).
%   variant  one of
%     'neutral' - unchanged
%     'tilt'    - whole torso rotated 15 degrees (leaning sideways)
%     'turn'    - torso turned ~35 degrees: the person's LEFT side (image
%                 right) moves away from the camera and gets narrower, the
%                 near side gets a little wider. A single affine transform
%                 cannot reproduce this; piecewise affine can.
%     'lean'    - shoulders shifted sideways relative to the hips + slight
%                 shoulder tilt (shear)
%     'closer'  - person steps toward the camera (uniform 1.25x scale)
%
%   Only used by the demo and the tests.

names = {'leftShoulder', 'rightShoulder', 'leftHip', 'rightHip'};
P = zeros(4, 2);
for i = 1:4
    P(i, :) = baseKp.(names{i})(1:2);
end
centre = mean(P, 1);

switch lower(variant)
    case 'neutral'
        % nothing
    case 'tilt'
        t = deg2rad(15);
        R = [cos(t) -sin(t); sin(t) cos(t)];
        P = (P - centre) * R' + centre;
    case 'turn'
        halfWidth = (P(1, 1) - P(2, 1)) / 2;          % shoulder half-width
        P([1 3], 1) = P([1 3], 1) - 0.45 * halfWidth; % far side (image right) moves in
        P([2 4], 1) = P([2 4], 1) - 0.10 * halfWidth; % near side shifts slightly
        P(1, 2) = P(1, 2) + 6;                        % far shoulder a bit lower (perspective)
    case 'lean'
        P(1:2, 1) = P(1:2, 1) + 40;                   % shoulders move right
        P(1, 2) = P(1, 2) + 12;                       % left shoulder drops
        P(2, 2) = P(2, 2) - 6;
    case 'closer'
        P = (P - centre) * 1.25 + centre;
    otherwise
        error('makeTestPose:variant', 'Unknown pose variant "%s".', variant);
end

kp = baseKp;
for i = 1:4
    kp.(names{i})(1:2) = P(i, :);
end
end
