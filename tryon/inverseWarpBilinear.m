function warped = inverseWarpBilinear(img, H, outputSize, roi)
%INVERSEWARPBILINEAR  Warp an image with a homography (inverse mapping + bilinear).
%
%   warped = inverseWarpBilinear(img, H, outputSize)
%   warped = inverseWarpBilinear(img, H, outputSize, roi)
%
%   *** From-scratch algorithm. No imwarp, interp2, griddedInterpolant or
%   *** projective2d: the inverse mapping and the bilinear interpolation are
%   *** written out below.
%
%   INPUTS
%     img         Hs x Ws x C source image (double). Any number of channels,
%                 e.g. [R G B alpha] stacked, so colour and transparency are
%                 sampled at exactly the same positions.
%     H           3x3 homography mapping SOURCE pixel coordinates [x y] to
%                 OUTPUT pixel coordinates (as from estimateHomographyDLT).
%     outputSize  [Ho Wo] (or size(frame)) of the output image.
%     roi         (optional) [rowStart rowEnd colStart colEnd]: only these
%                 output pixels are computed, everything else is 0. Used to
%                 skip the part of the frame the garment cannot reach.
%
%   OUTPUT
%     warped      Ho x Wo x C double; 0 where the output pixel maps outside
%                 the source image.
%
%   WHY INVERSE MAPPING?
%   Forward mapping (push every source pixel to H*p) leaves holes when the
%   garment is enlarged and overlaps when it is shrunk. Inverse mapping goes
%   the other way: for EVERY output pixel q we ask "where in the source does
%   it come from?", p = inv(H) * q, and read the colour there. Every output
%   pixel gets exactly one value, so there are no holes.
%
%   BILINEAR INTERPOLATION
%   p is generally not on the pixel grid. With x0 = floor(x), fx = x - x0
%   (same for y), the value is the weighted mean of the 4 neighbours:
%       I(p) = (1-fx)(1-fy) I(x0,y0)   + fx(1-fy) I(x0+1,y0)
%            + (1-fx) fy    I(x0,y0+1) + fx fy    I(x0+1,y0+1)
%   The weights are the areas of the opposite sub-rectangles and sum to 1.
%
%   See also estimateHomographyDLT, applyHomography, warpGarment.

img = double(img);
[Hs, Ws, C] = size(img);
Ho = outputSize(1);
Wo = outputSize(2);
if nargin < 4 || isempty(roi)
    roi = [1 Ho 1 Wo];
end
warped = zeros(Ho, Wo, C);
r1 = max(1, roi(1));  r2 = min(Ho, roi(2));
c1 = max(1, roi(3));  c2 = min(Wo, roi(4));
if r2 < r1 || c2 < c1 || Hs < 2 || Ws < 2
    return
end

%% ---- 1. Inverse mapping: output pixel -> source position ------------------
% Make sure the source image lies on the w > 0 side of H (H and -H are the
% same homography); then a valid source point always has w > 0 after the
% inverse map as well, and points "behind the horizon" can be rejected.
centre = [(Ws + 1) / 2; (Hs + 1) / 2; 1];
if H(3, :) * centre < 0
    H = -H;
end
Hinv = inv(H);

[X, Y] = meshgrid(c1:c2, r1:r2);
X = X(:);
Y = Y(:);
w  = Hinv(3, 1) * X + Hinv(3, 2) * Y + Hinv(3, 3);
xs = (Hinv(1, 1) * X + Hinv(1, 2) * Y + Hinv(1, 3)) ./ w;
ys = (Hinv(2, 1) * X + Hinv(2, 2) * Y + Hinv(2, 3)) ./ w;

% Keep only output pixels whose source position is inside the image.
inside = w > 0 & xs >= 1 & xs <= Ws & ys >= 1 & ys <= Hs;
xs = xs(inside);
ys = ys(inside);

%% ---- 2. Bilinear interpolation --------------------------------------------
% Top-left neighbour; at the right/bottom border step back one pixel so the
% 2x2 neighbourhood stays inside (fx/fy then become exactly 1 there).
x0 = min(floor(xs), Ws - 1);
y0 = min(floor(ys), Hs - 1);
fx = xs - x0;
fy = ys - y0;
w00 = (1 - fx) .* (1 - fy);
w10 = fx .* (1 - fy);
w01 = (1 - fx) .* fy;
w11 = fx .* fy;

% Linear indices of the 4 neighbours (MATLAB is column-major:
% index = row + (col - 1) * nRows).
i00 = y0 + (x0 - 1) * Hs;
i10 = i00 + Hs;          % one column to the right
i01 = i00 + 1;           % one row down
i11 = i10 + 1;

% Linear index of each kept output pixel inside the ROI block.
roiH = r2 - r1 + 1;
roiW = c2 - c1 + 1;
outIdx = find(inside);

for c = 1:C
    I = img(:, :, c);
    values = w00 .* I(i00) + w10 .* I(i10) + w01 .* I(i01) + w11 .* I(i11);
    block = zeros(roiH, roiW);
    block(outIdx) = values;
    warped(r1:r2, c1:c2, c) = block;
end
end
