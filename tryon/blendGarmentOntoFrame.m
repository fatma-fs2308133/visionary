function [outFrame, alpha, dbg] = blendGarmentOntoFrame(frame, warpedGarment, warpedAlphaMask, params)
%BLENDGARMENTONTOFRAME  Feathered alpha compositing, implemented from scratch.
%
%   [outFrame, alpha, dbg] = blendGarmentOntoFrame(frame, warpedGarment, ...
%                                                  warpedAlphaMask, params)
%
%   *** This is the project's "implement one algorithm from scratch"
%   *** deliverable. No built-in blending/compositing function is used
%   *** (no imfuse, imoverlay, alphablend, vision.AlphaBlender, ...). Toolbox
%   *** calls are limited to imerode (morphology) and imfilter (convolution);
%   *** the Gaussian kernel and the compositing maths are written by hand.
%
%   WHAT PROBLEM DOES IT SOLVE?
%   Pasting the garment with its raw (hard, 0-or-1) mask gives a jagged,
%   "cut-out sticker" edge, and the PNG's semi-transparent border often
%   carries a dark/white fringe. Phase 1 looked flat partly because of that.
%   Here the mask is (1) pulled slightly inward and (2) blurred, so the
%   garment fades into the person over a few pixels, like real fabric edges
%   seen through a camera.
%
%   INPUTS
%     frame            H x W x 3 webcam frame (uint8 / double / ...).
%     warpedGarment    H x W x 3 garment already warped into frame coordinates.
%     warpedAlphaMask  H x W alpha of the warped garment, values in [0,1].
%     params           (optional) struct with fields
%       .alphaThreshold  binarisation threshold for the mask   (default 0.5)
%       .erodeRadius     disk radius for imerode, in px         (default 3)
%       .gaussSigma      std-dev of the Gaussian feather, in px (default 2)
%       .useLoops        true = explicit per-pixel loops        (default false)
%
%   OUTPUTS
%     outFrame   composited frame, same size and class as the input frame.
%     alpha      H x W final (feathered) alpha that was used, in [0,1].
%     dbg        struct with every intermediate mask (for the demo figures):
%                hardMask, erodedMask, softMask, kernel, roi. Only filled when
%                requested (nargout == 3) to keep the live loop fast.
%
%   THE ALGORITHM
%     1. Binarise:   M_hard  = alpha >= threshold
%     2. Erode:      M_erode = M_hard eroded by a disk of radius r
%                    (shrinks the garment silhouette r px inward, removing
%                    the fringe pixels of the PNG border)
%     3. Feather:    M_soft  = M_erode convolved with a normalised Gaussian
%                    G(x,y) = exp(-(x^2+y^2) / (2 sigma^2)) / sum(G)
%                    (turns the 0->1 step at the edge into a smooth ramp)
%     4. Final alpha a = M_soft .* alpha_original
%                    (keeps any intentional semi-transparency of the PNG,
%                    and guarantees the garment never bleeds outside its
%                    original silhouette)
%     5. Composite, for every pixel (x,y) and colour channel c:
%            out(x,y,c) = a(x,y) * garment(x,y,c) + (1 - a(x,y)) * frame(x,y,c)
%        This is the standard "over" operator of Porter & Duff (1984) for an
%        opaque background.
%
%   See also imerode, imfilter, strel, warpGarment, tryOnPipeline.

%% ---- Parameters (fill in any that were not given) ------------------------
if nargin < 4 || isempty(params)
    params = struct();
end
params = fillDefaults(params);

%% ---- Step 0: bring everything to double precision in [0,1] ---------------
% All maths below assumes intensities in [0,1]; uint8 frames (0..255) would
% overflow/round in "alpha * garment". We remember the class to convert back.
outClass = class(frame);
F = im2double(frame);                       % background (the webcam frame)
if size(F, 3) == 1
    F = repmat(F, [1 1 3]);                 % grey camera -> 3 identical channels
end
G = im2double(warpedGarment);               % foreground (the garment)
A = im2double(warpedAlphaMask);             % original alpha of the garment
if size(A, 3) > 1
    A = A(:, :, 1);
end

[H, W, nChannels] = size(F);
if size(G, 1) ~= H || size(G, 2) ~= W || size(A, 1) ~= H || size(A, 2) ~= W
    error('blendGarmentOntoFrame:size', ...
        'frame, warpedGarment and warpedAlphaMask must have the same height/width.');
end

%% ---- Step 1: binary mask ---------------------------------------------------
% Every pixel that is "mostly garment" becomes true. This removes the fuzzy
% (and often discoloured) border of the PNG so we can build our OWN clean
% soft edge in steps 2-3.
hardMask = A >= params.alphaThreshold;

% Nothing to draw (garment completely outside the frame)? Return unchanged.
if ~any(hardMask(:))
    outFrame = frame;
    alpha    = zeros(H, W);
    dbg      = struct('hardMask', hardMask, 'erodedMask', hardMask, ...
                      'softMask', zeros(H, W), 'kernel', [], 'roi', []);
    return
end

%% ---- Speed-up: only process the region around the garment ----------------
% The garment covers maybe 20-30% of the frame; erosion, convolution and
% compositing only matter near it. We crop a Region Of Interest (ROI) = the
% garment's bounding box plus a margin as wide as the Gaussian kernel, so
% the blur is not cut off at the ROI border. This does not change the
% result, it only makes it faster for real-time use.
kernelRadius = ceil(3 * params.gaussSigma);
margin = kernelRadius + params.erodeRadius + 1;
rowsWithGarment = find(any(hardMask, 2));
colsWithGarment = find(any(hardMask, 1));
r1 = max(1, rowsWithGarment(1)   - margin);
r2 = min(H, rowsWithGarment(end) + margin);
c1 = max(1, colsWithGarment(1)   - margin);
c2 = min(W, colsWithGarment(end) + margin);

hardROI = hardMask(r1:r2, c1:c2);
A_ROI   = A(r1:r2, c1:c2);
F_ROI   = F(r1:r2, c1:c2, :);
G_ROI   = G(r1:r2, c1:c2, :);

%% ---- Step 2: erosion - pull the garment edge inward ------------------------
% Morphological erosion with a disk-shaped structuring element: a pixel stays
% "on" only if the WHOLE disk centred on it fits inside the mask. The result
% is the same silhouette, shrunk by erodeRadius pixels all around. Doing
% this before the blur means the blur's soft ramp ends up INSIDE the
% original garment outline instead of straddling it.
if params.erodeRadius > 0
    se = strel('disk', params.erodeRadius, 0);   % 0 = exact disk (no approximation)
    erodedROI = imerode(hardROI, se);
else
    erodedROI = hardROI;
end

%% ---- Step 3: Gaussian feathering of the MASK (not of the image) ------------
% We blur the 0/1 mask so the step from 0 to 1 at the garment edge becomes a
% smooth ramp: pixels deep inside stay 1, pixels far outside stay 0, and
% pixels near the edge get fractional values = partial transparency.
% The garment image itself is NOT blurred, so its texture stays sharp.
if params.gaussSigma > 0
    kernel = makeGaussianKernel(params.gaussSigma);
    % imfilter = 2-D correlation (same as convolution for this symmetric
    % kernel). 'replicate' padding: where the garment touches the frame
    % border it stays opaque instead of fading toward a fake black border.
    softROI = imfilter(double(erodedROI), kernel, 'replicate');
else
    kernel  = 1;
    softROI = double(erodedROI);
end

%% ---- Step 4: final per-pixel alpha -----------------------------------------
% Multiply by the original alpha: inside the garment the original alpha is
% ~1, so the feathered mask decides; if the PNG has intentionally see-through
% parts (lace, mesh), they stay see-through; and nothing can become visible
% where the original garment was fully transparent.
alphaROI = softROI .* A_ROI;
alphaROI = min(max(alphaROI, 0), 1);        % guard against rounding

%% ---- Step 5: alpha compositing  out = a*garment + (1-a)*background --------
if params.useLoops
    outROI = compositeWithLoops(F_ROI, G_ROI, alphaROI);
else
    outROI = compositeVectorised(F_ROI, G_ROI, alphaROI);
end

%% ---- Write the ROI back and convert to the input class ---------------------
out = F;
out(r1:r2, c1:c2, :) = outROI;
outFrame = convertToClass(out, outClass);

alpha = zeros(H, W);
alpha(r1:r2, c1:c2) = alphaROI;

% Full-size intermediate masks for visualisation (only if asked for).
if nargout > 2
    erodedMask = false(H, W);  erodedMask(r1:r2, c1:c2) = erodedROI;
    softMask   = zeros(H, W);  softMask(r1:r2, c1:c2)   = softROI;
    dbg = struct('hardMask', hardMask, 'erodedMask', erodedMask, ...
                 'softMask', softMask, 'kernel', kernel, ...
                 'roi', [r1 r2 c1 c2], 'nChannels', nChannels);
end
end

% ==========================================================================
%  Local helper functions
% ==========================================================================

function kernel = makeGaussianKernel(sigma)
% Build a 2-D Gaussian kernel by hand (equivalent to fspecial('gaussian')).
%
%   G(x,y) = exp( -(x^2 + y^2) / (2*sigma^2) )
%
% - Kernel radius = ceil(3*sigma): +-3 sigma holds 99.7% of the Gaussian, so
%   cutting it off there is invisible.
% - The 1/(2*pi*sigma^2) constant is skipped because we normalise anyway:
%   dividing by the sum makes the weights add up to exactly 1, so a region
%   that is fully opaque (all ones) stays exactly 1 after filtering.
radius = ceil(3 * sigma);
[x, y] = meshgrid(-radius:radius, -radius:radius);
kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
kernel = kernel / sum(kernel(:));
end

function out = compositeWithLoops(background, foreground, alpha)
% Literal, pixel-by-pixel implementation of the compositing formula.
% Slow in MATLAB (~3 nested loops), but it is the clearest way to SHOW the
% algorithm during the demo. compositeVectorised gives identical results.
[h, w, nc] = size(background);
out = background;                           % a = 0 -> pixel stays background
for y = 1:h                                 % every row
    for x = 1:w                             % every column
        a = alpha(y, x);                    % how much garment at this pixel
        if a > 0                            % skip fully transparent pixels
            for c = 1:nc                    % R, G, B
                out(y, x, c) = a * foreground(y, x, c) + (1 - a) * background(y, x, c);
            end
        end
    end
end
end

function out = compositeVectorised(background, foreground, alpha)
% Exactly the same formula, but applied to all pixels of one colour channel
% at once with element-wise operators (.*). Much faster in MATLAB.
out = zeros(size(background));
for c = 1:size(background, 3)
    out(:, :, c) = alpha .* foreground(:, :, c) + (1 - alpha) .* background(:, :, c);
end
end

function params = fillDefaults(params)
defaults = struct('alphaThreshold', 0.5, 'erodeRadius', 3, ...
                  'gaussSigma', 2, 'useLoops', false);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(params, names{i}) || isempty(params.(names{i}))
        params.(names{i}) = defaults.(names{i});
    end
end
params.erodeRadius = max(0, round(params.erodeRadius));
params.gaussSigma  = max(0, params.gaussSigma);
end

function img = convertToClass(img, className)
% Convert a double [0,1] image back to the class the frame came in with.
switch className
    case 'uint8',  img = im2uint8(img);
    case 'uint16', img = im2uint16(img);
    case 'single', img = single(img);
    otherwise      % double (or anything else): keep double
end
end
