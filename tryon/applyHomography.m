function q = applyHomography(H, p)
%APPLYHOMOGRAPHY  Map points through a 3x3 homography.
%
%   q = applyHomography(H, p)
%
%   p  N-by-2 [x y] points. Each point is lifted to homogeneous coordinates
%      [x y 1], multiplied by H, and divided by the resulting w:
%          [x' y' w'] = H * [x; y; 1],   q = [x'/w', y'/w']
%
%   See also estimateHomographyDLT, inverseWarpBilinear.

ph = [double(p), ones(size(p, 1), 1)] * H';
q = ph(:, 1:2) ./ ph(:, 3);
end
