function [warpedRGB, warpedAlpha, info] = warpGarment(garmentImg, garmentAlpha, srcPts, dstPts, outputSize, mode)
%WARPGARMENT  Warp a garment image and its alpha mask onto body landmarks.
%
%   [warpedRGB, warpedAlpha, info] = warpGarment(garmentImg, garmentAlpha, ...
%                                                srcPts, dstPts, outputSize)
%   [...] = warpGarment(..., mode)     mode = 'affine' | 'piecewise' | 'homography'
%
%   The garment is mapped from its own image (source points = anchor points
%   on the garment PNG) into the webcam frame (destination points = pose
%   keypoints). Because the result is rendered straight into frame
%   coordinates, it is already positioned and only needs to be blended.
%
%   INPUTS
%     garmentImg    H_g x W_g x 3 garment RGB (any class; converted to double).
%     garmentAlpha  H_g x W_g x K alpha mask(s) (0 = transparent). Usually
%                   K = 1; extra channels (e.g. a torso-only mask used for
%                   evaluation) are warped with exactly the same transform.
%     srcPts        Points on the garment image, [x y] per row (pixels).
%     dstPts        Matching points in the webcam frame, [x y] per row.
%     mode          (optional) the transform model. Default: chosen from the
%                   number of rows (3 -> 'affine', 4 -> 'piecewise').
%       'affine'     3 rows -> ONE affine transform ("triangle" mode)
%                    order: [leftShoulder; rightShoulder; hipCentre]
%       'piecewise'  4 rows -> PIECEWISE affine, quad split into two
%                    triangles. Order (around the quad): [leftShoulder;
%                    rightShoulder; rightHip; leftHip]
%       'homography' 4+ rows (same order) -> ONE projective transform,
%                    estimated with our own DLT (estimateHomographyDLT) and
%                    applied with our own inverse warping + bilinear
%                    interpolation (inverseWarpBilinear). FROM SCRATCH.
%     outputSize    size(frame): the warped images get this height/width so
%                   they line up pixel-for-pixel with the frame.
%
%   OUTPUTS
%     warpedRGB     outputSize(1) x outputSize(2) x 3, double in [0,1].
%     warpedAlpha   outputSize(1) x outputSize(2) x K, double in [0,1]; 0
%                   wherever the garment does not land.
%     info          struct: mode, tforms (cell of affine2d, affine modes) or
%                   H (3x3, homography mode), triangles (vertex indices), and
%                   for piecewise mode useTriangle1 (logical map of which
%                   output pixels use triangle 1). For the demo.
%
%   WHY A HOMOGRAPHY?
%   An affine map keeps parallel lines parallel. A homography (8 degrees of
%   freedom) maps ANY convex quad onto any other convex quad, so it models
%   the perspective of a torso plane that turns or tilts toward the camera
%   with a single smooth transform - no seam, unlike the piecewise affine.
%
%   WHY PIECEWISE?
%   A single affine transform keeps parallel lines parallel, so it can
%   rotate, scale and shear the garment but can NOT make the left half of
%   the torso narrower than the right half, which is exactly what happens
%   when the person turns. With two triangles, each half of the torso gets
%   its own affine transform, so each side deforms independently while the
%   garment stays continuous along the shared edge.
%
%   HOW THE TWO TRIANGLES COVER THE WHOLE GARMENT (JUDGEMENT CALL)
%   The quad is split along its diagonal leftShoulder -> rightHip:
%       triangle 1 = [leftShoulder, rightShoulder, rightHip]
%       triangle 2 = [leftShoulder, rightHip,      leftHip ]
%   Sleeves and the collar lie OUTSIDE the shoulder/hip quad, so instead of
%   only warping the inside of each triangle (fitgeotrans 'pwl' would leave
%   those parts undefined and cut them off), the diagonal is extended to an
%   infinite line and each triangle's affine transform is used for its whole
%   HALF-PLANE. The two transforms agree exactly on that line (they share
%   both of its end points, and an affine map is fixed on a line by two
%   points), so the garment stays seamless across it. Using the other
%   diagonal (rightShoulder -> leftHip) is equally valid; this choice is
%   arbitrary and only matters for strongly non-planar poses.
%
%   Uses fitgeotrans + imwarp (Image Processing Toolbox). In R2022b+
%   fitgeotform2d is the newer equivalent; fitgeotrans still works.
%
%   See also fitgeotrans, imwarp, imref2d, tryOnPipeline.

if nargin < 5 || isempty(outputSize)
    error('warpGarment:noOutputSize', ...
        'Pass outputSize = size(frame) so the garment is rendered in frame coordinates.');
end

% ---- Input conditioning ----------------------------------------------------
garmentImg   = im2double(garmentImg);
garmentAlpha = im2double(garmentAlpha);
srcPts = double(srcPts);
dstPts = double(dstPts);
if ~isequal(size(srcPts), size(dstPts)) || size(srcPts, 2) ~= 2
    error('warpGarment:badPoints', 'srcPts and dstPts must both be N-by-2 with equal N.');
end
if nargin < 6 || isempty(mode)
    if size(srcPts, 1) == 3, mode = 'affine'; else, mode = 'piecewise'; end
end
mode = lower(mode);
expected = struct('affine', 3, 'piecewise', 4, 'homography', 4);
if ~isfield(expected, mode)
    error('warpGarment:mode', 'Unknown mode "%s" (use affine, piecewise or homography).', mode);
end
if size(srcPts, 1) < expected.(mode) || (~strcmp(mode, 'homography') && size(srcPts, 1) ~= expected.(mode))
    error('warpGarment:badPoints', 'Mode "%s" needs %d points, got %d.', ...
          mode, expected.(mode), size(srcPts, 1));
end

% The output "canvas" is the webcam frame: world coordinates = pixel
% coordinates of the frame (x = column, y = row).
outputView = imref2d(outputSize(1:2));

switch mode
    % ======================================================================
    case 'homography'   % ---- one projective transform, all from scratch --
        if size(dstPts, 1) == 4
            checkConvexQuad(dstPts);
        end
        [H, fitInfo] = estimateHomographyDLT(srcPts, dstPts);

        % Only compute output pixels inside the bounding box of the warped
        % garment (much faster than the whole frame).
        roi = homographyRoi(H, garmentAlpha, outputSize);

        % Warp colour and alpha in ONE call so they are sampled at exactly
        % the same positions.
        nA = size(garmentAlpha, 3);
        stacked = inverseWarpBilinear(cat(3, garmentImg, garmentAlpha), H, outputSize, roi);
        warpedRGB   = stacked(:, :, 1:3);
        warpedAlpha = stacked(:, :, 4:3 + nA);

        info.mode        = 'homography';
        info.H           = H;
        info.tforms      = {};
        info.triangles   = {1:size(srcPts, 1)};
        info.reprojError = fitInfo.reprojError;
        info.roi         = roi;

    % ======================================================================
    case 'affine'       % ---- single affine transform from one triangle ---
        checkTriangle(srcPts, 'source');
        checkTriangle(dstPts, 'destination');

        % fitgeotrans(moving, fixed, ...) returns the transform that maps the
        % garment ("moving") points onto the body ("fixed") points.
        tform = fitgeotrans(srcPts, dstPts, 'affine');

        [warpedRGB, warpedAlpha] = applyTransform(garmentImg, garmentAlpha, tform, outputView);

        info.mode      = 'affine';
        info.tforms    = {tform};
        info.triangles = {[1 2 3]};

    % ======================================================================
    case 'piecewise'    % ---- piecewise affine: quad -> two triangles ------
        tri1 = [1 2 3];     % leftShoulder, rightShoulder, rightHip
        tri2 = [1 3 4];     % leftShoulder, rightHip,      leftHip
        checkTriangle(srcPts(tri1, :), 'source triangle 1');
        checkTriangle(srcPts(tri2, :), 'source triangle 2');
        checkTriangle(dstPts(tri1, :), 'destination triangle 1');
        checkTriangle(dstPts(tri2, :), 'destination triangle 2');

        % One affine transform per triangle.
        tform1 = fitgeotrans(srcPts(tri1, :), dstPts(tri1, :), 'affine');
        tform2 = fitgeotrans(srcPts(tri2, :), dstPts(tri2, :), 'affine');

        % Warp the WHOLE garment with each transform.
        [rgb1, alpha1] = applyTransform(garmentImg, garmentAlpha, tform1, outputView);
        [rgb2, alpha2] = applyTransform(garmentImg, garmentAlpha, tform2, outputView);

        % For every output pixel, decide which triangle's transform applies:
        % the side of the (destination) diagonal leftShoulder -> rightHip the
        % pixel is on. The sign of the 2-D cross product
        %     (B - A) x (P - A)
        % tells on which side of the line A->B a point P lies.
        A = dstPts(1, :);   % leftShoulder
        B = dstPts(3, :);   % rightHip
        [X, Y] = meshgrid(1:outputSize(2), 1:outputSize(1));
        sidePixels = (B(1) - A(1)) .* (Y - A(2)) - (B(2) - A(2)) .* (X - A(1));
        % Triangle 1 is the side that contains its third vertex, rightShoulder.
        sideTri1   = (B(1) - A(1)) .* (dstPts(2, 2) - A(2)) - (B(2) - A(2)) .* (dstPts(2, 1) - A(1));
        useTriangle1 = (sidePixels .* sideTri1) > 0;

        % Stitch the two warps together (pixels exactly on the diagonal go to
        % triangle 2; both transforms give the same result there anyway).
        m = double(useTriangle1);
        warpedRGB   = rgb1   .* m + rgb2   .* (1 - m);
        warpedAlpha = alpha1 .* m + alpha2 .* (1 - m);

        info.mode         = 'piecewise';
        info.tforms       = {tform1, tform2};
        info.triangles    = {tri1, tri2};
        info.useTriangle1 = useTriangle1;
end

% Interpolation can overshoot very slightly; keep everything in [0,1].
warpedRGB   = min(max(warpedRGB, 0), 1);
warpedAlpha = min(max(warpedAlpha, 0), 1);
end

% ==========================================================================
function [rgbOut, alphaOut] = applyTransform(rgb, alpha, tform, outputView)
% Apply the same geometric transform to the colour image and to its alpha
% mask, so colour and transparency stay perfectly aligned. Bilinear
% interpolation; everything outside the garment becomes 0 (= transparent).
rgbOut   = imwarp(rgb,   tform, 'linear', 'OutputView', outputView, 'FillValues', 0);
alphaOut = imwarp(alpha, tform, 'linear', 'OutputView', outputView, 'FillValues', 0);
end

% ==========================================================================
function checkTriangle(P, label)
% A triangle whose points are (almost) on one line has no well-defined
% affine transform (this happens e.g. when the person is seen exactly from
% the side and both shoulders coincide). Fail loudly; tryOnPipeline catches
% this and simply skips the overlay for that frame.
v1 = P(2, :) - P(1, :);
v2 = P(3, :) - P(1, :);
twiceArea = abs(v1(1) * v2(2) - v1(2) * v2(1));
if twiceArea < 1        % less than half a square pixel
    error('warpGarment:degenerate', 'The %s is degenerate (points are collinear).', label);
end
end

% ==========================================================================
function checkConvexQuad(P)
% The body quad [LS RS RH LH] must be convex: all 4 corners turn the same
% way. A crossed or dented quad (e.g. a wrong keypoint, or the person seen
% from the back with left/right swapped) has no sensible homography, so we
% fail and tryOnPipeline skips the overlay for this frame.
turns = zeros(1, 4);
for k = 1:4
    a = P(k, :);
    b = P(mod(k, 4) + 1, :);
    c = P(mod(k + 1, 4) + 1, :);
    turns(k) = (b(1) - a(1)) * (c(2) - b(2)) - (b(2) - a(2)) * (c(1) - b(1));
end
if ~(all(turns > 1) || all(turns < -1))
    error('warpGarment:degenerate', 'The body keypoints do not form a convex quad.');
end
end

function roi = homographyRoi(H, alpha, outputSize)
% Output rows/cols that the visible part of the garment can reach: map the
% corners of the garment's bounding box with H and take their bounding box
% (a homography maps a convex quad to a convex quad, so the corners bound
% everything). Falls back to the whole frame if a corner maps behind the
% horizon (w <= 0).
visible = any(alpha > 0, 3);
rows = find(any(visible, 2));
cols = find(any(visible, 1));
fullFrame = [1 outputSize(1) 1 outputSize(2)];
if isempty(rows)
    roi = [1 0 1 0];                      % empty: nothing to draw
    return
end
corners = [cols(1) rows(1); cols(end) rows(1); cols(end) rows(end); cols(1) rows(end)];
ph = [corners, ones(4, 1)] * H';
centre = [mean(cols), mean(rows), 1] * H';
if any(sign(ph(:, 3)) ~= sign(centre(3))) || any(abs(ph(:, 3)) < eps)
    roi = fullFrame;
    return
end
q = ph(:, 1:2) ./ ph(:, 3);
roi = [floor(min(q(:, 2))) - 1, ceil(max(q(:, 2))) + 1, ...
       floor(min(q(:, 1))) - 1, ceil(max(q(:, 1))) + 1];
roi = [max(roi(1), 1), min(roi(2), outputSize(1)), max(roi(3), 1), min(roi(4), outputSize(2))];
end
