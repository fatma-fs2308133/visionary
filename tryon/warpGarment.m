function [warpedRGB, warpedAlpha, info] = warpGarment(garmentImg, garmentAlpha, srcPts, dstPts, outputSize)
%WARPGARMENT  Warp a garment image and its alpha mask onto body landmarks.
%
%   [warpedRGB, warpedAlpha, info] = warpGarment(garmentImg, garmentAlpha, ...
%                                                srcPts, dstPts, outputSize)
%
%   The garment is mapped from its own image (source points = anchor points
%   on the garment PNG) into the webcam frame (destination points = pose
%   keypoints). Because the result is rendered straight into frame
%   coordinates, it is already positioned and only needs to be blended.
%
%   INPUTS
%     garmentImg    H_g x W_g x 3 garment RGB (any class; converted to double).
%     garmentAlpha  H_g x W_g alpha / transparency mask (0 = transparent).
%     srcPts        Points on the garment image, [x y] per row (pixels).
%     dstPts        Matching points in the webcam frame, [x y] per row.
%                   The number of rows selects the model:
%       3 rows -> ONE affine transform ("triangle" mode)
%                 order: [leftShoulder; rightShoulder; hipCentre]
%       4 rows -> PIECEWISE affine, quad split into two triangles
%                 order (around the quad): [leftShoulder; rightShoulder;
%                                           rightHip; leftHip]
%     outputSize    size(frame): the warped images get this height/width so
%                   they line up pixel-for-pixel with the frame.
%
%   OUTPUTS
%     warpedRGB     outputSize(1) x outputSize(2) x 3, double in [0,1].
%     warpedAlpha   outputSize(1) x outputSize(2), double in [0,1]; 0 wherever
%                   the garment does not land.
%     info          struct: mode, tforms (cell of affine2d), triangles (vertex
%                   indices), and for piecewise mode useTriangle1 (logical
%                   map of which output pixels use triangle 1). For the demo.
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
if size(garmentAlpha, 3) > 1
    garmentAlpha = garmentAlpha(:, :, 1);
end
srcPts = double(srcPts);
dstPts = double(dstPts);
if ~isequal(size(srcPts), size(dstPts)) || size(srcPts, 2) ~= 2
    error('warpGarment:badPoints', 'srcPts and dstPts must both be N-by-2 with equal N.');
end

% The output "canvas" is the webcam frame: world coordinates = pixel
% coordinates of the frame (x = column, y = row).
outputView = imref2d(outputSize(1:2));

switch size(srcPts, 1)
    % ======================================================================
    case 3      % ---- single affine transform from one triangle -----------
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
    case 4      % ---- piecewise affine: quad -> two triangles --------------
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

    otherwise
        error('warpGarment:badPoints', ...
            'Use 3 points (single affine) or 4 points (piecewise affine), got %d.', size(srcPts, 1));
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
