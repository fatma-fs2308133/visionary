function parts = splitGarmentSleeves(garmentAlpha, anchorsPx)
%SPLITGARMENTSLEEVES  Cut the garment's alpha into torso / left / right sleeve.
%
%   parts = splitGarmentSleeves(garmentAlpha, anchorsPx)
%
%   A sleeve is everything on the OUTER side of the straight line from the
%   shoulder anchor to the armpit anchor (outer = the side where the sleeve
%   end lies). This works for normal T-shirts; for unusual cuts adjust the
%   armpit anchor (pickGarmentAnchors).
%
%   INPUTS
%     garmentAlpha  alpha of the ORIGINAL garment image (double 0..1).
%     anchorsPx     struct of anchor points in garment PIXELS, with fields
%                   leftShoulder, rightShoulder, leftArmpit, rightArmpit,
%                   leftSleeveEnd, rightSleeveEnd.
%
%   OUTPUT
%     parts.torso, parts.leftSleeve, parts.rightSleeve  (alpha maps, same
%     size as garmentAlpha). The sleeve alphas fade in over ~4 px across the
%     cut line and the torso keeps a 3 px overlap under them, so no gap
%     opens at the shoulder seam when a sleeve is rotated.

garmentAlpha = im2double(garmentAlpha);
[H, W] = size(garmentAlpha);
[X, Y] = meshgrid(1:W, 1:H);

distL = outerDistance(X, Y, anchorsPx.leftShoulder,  anchorsPx.leftArmpit,  anchorsPx.leftSleeveEnd);
distR = outerDistance(X, Y, anchorsPx.rightShoulder, anchorsPx.rightArmpit, anchorsPx.rightSleeveEnd);

% Sleeves: soft ramp from -2 px (0) to +2 px (1) across the cut line.
parts.leftSleeve  = garmentAlpha .* min(max((distL + 2) / 4, 0), 1);
parts.rightSleeve = garmentAlpha .* min(max((distR + 2) / 4, 0), 1);
% Torso: everything up to 3 px beyond the cut lines (hidden under sleeves).
parts.torso = garmentAlpha .* (distL <= 3 & distR <= 3);
end

% ==========================================================================
function d = outerDistance(X, Y, A, B, outerPoint)
% Signed distance (px) of every pixel from the line A->B, positive on the
% side where outerPoint lies.
dirAB = (B - A) / max(norm(B - A), eps);
normal = [-dirAB(2), dirAB(1)];                 % unit normal of the line
d = (X - A(1)) * normal(1) + (Y - A(2)) * normal(2);
if dot(outerPoint - A, normal) < 0
    d = -d;
end
end
