function [shadedRGB, shade] = transferBodyShading(garmentRGB, garmentAlpha, frame, params)
%TRANSFERBODYSHADING  Copy folds and body curvature from the frame to the garment.
%
%   [shadedRGB, shade] = transferBodyShading(garmentRGB, garmentAlpha, frame, params)
%
%   A pasted garment looks flat because its brightness is the same everywhere,
%   while real fabric on a body is darker toward the sides (the torso is
%   round), has folds and wrinkles, and shadows under the chest and arms.
%   The clothes the person is REALLY wearing in the webcam frame already show
%   all of this. We extract that shading and multiply it onto the garment:
%
%     1. L      = grey-level brightness of the frame (slightly blurred to
%                 remove camera noise)
%     2. base   = local average of L, computed ONLY over the garment area
%                 ("normalised convolution": blur(L.*M) ./ blur(M)), so the
%                 background around the body does not leak in
%     3. ratio  = L ./ base   -> >1 where locally brighter (a fold facing the
%                 light), <1 where darker (sides of the torso, creases)
%     4. shade  = 1 + strength*(ratio - 1), clamped
%     5. garment_out = garment .* shade
%
%   Because the ratio is recomputed every frame from the live image, the
%   garment's folds move when the person moves - this is the "flow".
%
%   INPUTS
%     garmentRGB    warped garment (frame coordinates), double RGB.
%     garmentAlpha  its alpha (defines where to measure/apply shading).
%     frame         current webcam frame.
%     params        cfg.shading (strength, baseSigma, fineSigma, minShade,
%                   maxShade).
%
%   OUTPUTS
%     shadedRGB  garment with shading applied.
%     shade      H x W shading factor (1 outside the garment) - for the demo.
%
%   LIMITATION: prints or strong patterns on the real shirt partly show
%   through as "shading" (lower strength or raise baseSigma), and a very dark
%   real shirt carries little shading information. Works best with a plain,
%   light-coloured shirt underneath.

L = im2double(frame);
if size(L, 3) == 3
    L = rgb2gray(L);
end
if params.fineSigma > 0
    L = imgaussfilt(L, params.fineSigma);
end

M = double(garmentAlpha > 0.05);            % where the garment is
if ~any(M(:))
    shadedRGB = garmentRGB;
    shade = ones(size(M));
    return
end

% Local average brightness inside the garment region only.
num  = imgaussfilt(L .* M, params.baseSigma);
den  = imgaussfilt(M, params.baseSigma);
base = num ./ max(den, 1e-3);

ratio = L ./ max(base, 0.02);
shade = 1 + params.strength * (ratio - 1);
shade = min(max(shade, params.minShade), params.maxShade);
shade(M == 0) = 1;

shadedRGB = min(max(garmentRGB .* shade, 0), 1);
end
