function [sleeveRGB, sleeveAlpha] = warpSleeve(garmentImg, sleeveAlphaG, gShoulder, gSleeveEnd, fShoulder, fElbow, scale, outputSize)
%WARPSLEEVE  Place one sleeve so that it follows the upper arm.
%
%   [sleeveRGB, sleeveAlpha] = warpSleeve(garmentImg, sleeveAlphaG, ...
%       gShoulder, gSleeveEnd, fShoulder, fElbow, scale, outputSize)
%
%   The sleeve is treated as a rigid piece hinged at the shoulder (like a
%   2-D puppet limb). A SIMILARITY transform (rotation + uniform scale +
%   translation) maps
%       garment shoulder anchor  -> body shoulder keypoint
%       garment sleeve axis      -> direction shoulder -> elbow
%   and the sleeve keeps its own length times the torso scale factor.
%
%   INPUTS
%     garmentImg     garment RGB (original image).
%     sleeveAlphaG   alpha of this sleeve only (splitGarmentSleeves).
%     gShoulder, gSleeveEnd   anchors on the garment image [x y] (px).
%     fShoulder, fElbow       keypoints in the frame [x y] (px).
%     scale          garment-to-frame size ratio (frame shoulder width /
%                    garment shoulder width).
%     outputSize     size(frame).
%
%   OUTPUTS
%     sleeveRGB, sleeveAlpha   the sleeve rendered in frame coordinates.

gAxis = gSleeveEnd - gShoulder;
armDir = fElbow - fShoulder;
if norm(gAxis) < 1 || norm(armDir) < 1
    error('warpSleeve:degenerate', 'Sleeve axis or upper arm has zero length.');
end
sleeveLength = norm(gAxis) * scale;
fSleeveEnd = fShoulder + sleeveLength * armDir / norm(armDir);

% A similarity transform is fixed by 2 point pairs. fitgeotrans 'affine'
% needs 3, so we add a third point: the axis rotated by 90 degrees, on both
% sides. An affine fit through these 3 pairs is exactly the similarity
% (same rotation, same scale in both directions).
perp = @(v) [-v(2), v(1)];
src = [gShoulder; gSleeveEnd; gShoulder + perp(gAxis)];
dst = [fShoulder; fSleeveEnd; fShoulder + perp(fSleeveEnd - fShoulder)];
tform = fitgeotrans(src, dst, 'affine');

outputView = imref2d(outputSize(1:2));
sleeveRGB   = imwarp(im2double(garmentImg), tform, 'linear', 'OutputView', outputView, 'FillValues', 0);
sleeveAlpha = imwarp(im2double(sleeveAlphaG), tform, 'linear', 'OutputView', outputView, 'FillValues', 0);
sleeveRGB   = min(max(sleeveRGB, 0), 1);
sleeveAlpha = min(max(sleeveAlpha, 0), 1);
end
