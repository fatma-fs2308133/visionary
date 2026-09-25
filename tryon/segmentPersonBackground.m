function bodyMask = segmentPersonBackground(frame, background, params)
%SEGMENTPERSONBACKGROUND  Person silhouette by background subtraction.
%
%   bodyMask = segmentPersonBackground(frame, background, params)
%
%   Classical foreground segmentation: the camera does not move, so a frame
%   captured while nobody is in view ("background") tells us what every
%   pixel looks like without the person. Pixels whose colour now differs
%   enough from it belong to the person.
%
%   INPUTS
%     frame       current webcam frame (RGB).
%     background  frame of the empty scene, same size (capture it with the
%                 person out of view; liveTryOn does this when you press B).
%     params      (optional) struct with
%       .threshold    colour distance (0..1) counted as foreground (0.12)
%       .blurSigma    Gaussian pre-blur against camera noise, px   (1.5)
%       .openRadius   imopen disk radius - removes speckles        (2)
%       .closeRadius  imclose disk radius - closes gaps in the body (6)
%
%   OUTPUT
%     bodyMask    logical H x W, true = person. Only the largest blob is kept.
%
%   LIMITATIONS (say these in the demo): the camera must not move, auto-
%   exposure / lights switching change the background colour, shadows the
%   person casts on the wall count as "person", and clothes that have the
%   same colour as the wall behind them are missed. Re-capture the
%   background (B) when the lighting changes.

if nargin < 3 || isempty(params), params = struct(); end
params = fillDefaults(params);

F = im2double(frame);
B = im2double(background);
if params.blurSigma > 0
    F = imgaussfilt(F, params.blurSigma);
    B = imgaussfilt(B, params.blurSigma);
end

% 1. Per-pixel colour distance between frame and background (Euclidean in
%    RGB, divided by sqrt(3) so it lies in 0..1).
difference = sqrt(sum((F - B).^2, 3)) / sqrt(3);

% 2. Threshold -> raw foreground.
bodyMask = difference > params.threshold;

% 3. Morphological clean-up.
if params.openRadius > 0     % opening = erode then dilate: removes small specks
    bodyMask = imopen(bodyMask, strel('disk', params.openRadius, 0));
end
if params.closeRadius > 0    % closing = dilate then erode: fills thin gaps
    bodyMask = imclose(bodyMask, strel('disk', params.closeRadius, 0));
end
bodyMask = imfill(bodyMask, 'holes');   % fill enclosed holes (e.g. a white logo)

% 4. Keep only the largest connected region = the person.
if any(bodyMask(:))
    bodyMask = bwareafilt(bodyMask, 1);
end
end

% ==========================================================================
function params = fillDefaults(params)
defaults = struct('threshold', 0.12, 'blurSigma', 1.5, 'openRadius', 2, 'closeRadius', 6);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(params, names{i}) || isempty(params.(names{i}))
        params.(names{i}) = defaults.(names{i});
    end
end
end
