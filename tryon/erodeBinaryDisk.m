function out = erodeBinaryDisk(mask, radius)
%ERODEBINARYDISK  Binary erosion with a disk, implemented from scratch.
%
%   out = erodeBinaryDisk(mask, radius)
%
%   *** From-scratch replacement for imerode(mask, strel('disk', radius, 0)).
%   *** No imerode, strel, bwmorph or conv2 is used.
%
%   INPUTS
%     mask    H x W logical (or 0/1) mask.
%     radius  disk radius in pixels (integer >= 0).
%
%   OUTPUT
%     out     H x W logical: true where the WHOLE disk centred on the pixel
%             fits inside the mask.
%
%   THE ALGORITHM
%     Structuring element  B = { (dx,dy) : dx^2 + dy^2 <= r^2 }
%     Erosion              out(y,x) = AND over (dx,dy) in B of mask(y+dy, x+dx)
%     Instead of a window loop per pixel, we loop over the offsets of B
%     (29 offsets for r = 3) and AND the whole shifted mask each time.
%     Pixels outside the image count as true (same as imerode), so a
%     garment touching the frame border is not eaten away from that side.
%
%   See also convolve2D, blendGarmentOntoFrame.

mask = logical(mask);
radius = max(0, round(radius));
if radius == 0
    out = mask;
    return
end
[H, W] = size(mask);

% Disk offsets (the structuring element)
[dx, dy] = meshgrid(-radius:radius, -radius:radius);
inDisk = dx.^2 + dy.^2 <= radius^2;
dx = dx(inDisk);
dy = dy(inDisk);

% Pad with true by the radius so every shifted window stays in range.
padded = true(H + 2 * radius, W + 2 * radius);
padded(radius + 1:radius + H, radius + 1:radius + W) = mask;

out = true(H, W);
for k = 1:numel(dx)
    rows = (1:H) + radius + dy(k);
    cols = (1:W) + radius + dx(k);
    out = out & padded(rows, cols);
end
end
