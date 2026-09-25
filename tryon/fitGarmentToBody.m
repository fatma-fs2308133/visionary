function [fitRGB, fitAlpha, info] = fitGarmentToBody(warpedRGB, warpedAlpha, bodyMask, torso, params)
%FITGARMENTTOBODY  Bend the garment's side edges onto the body silhouette.
%
%   [fitRGB, fitAlpha, info] = fitGarmentToBody(warpedRGB, warpedAlpha, ...
%                                               bodyMask, torso, params)
%
%   The piecewise-affine warp only uses 4 points (shoulders + hips), so the
%   sides of the garment are always straight lines between them. Real bodies
%   are not: there is a chest, a waist, hips. This step fixes the SHAPE:
%
%     for every image row between the armpits and the hem:
%       1. find the garment's left/right edge in that row      (gL, gR)
%       2. find the body's left/right edge in that row          (bL, bR)
%       3. target edges = body edges + a little "ease"          (tL, tR)
%       4. re-sample the row so that gL lands on tL and gR on tR,
%          stretching/squeezing linearly in between
%
%   That is a 1-D piecewise-linear remap per row (a "row-wise mesh warp").
%   The edge offsets are smoothed from row to row so the result is a smooth
%   curve, not a staircase, and faded in below the armpits so the shoulders
%   are not touched.
%
%   INPUTS
%     warpedRGB, warpedAlpha  garment already warped into frame coordinates
%                             (output of warpGarment; torso part only works
%                             best, so arms beside the body are not mixed in).
%     bodyMask   logical person mask (segmentPersonBackground).
%     torso      struct from getTorsoKeypoints.
%     params     cfg.fit (see defaultTryOnConfig).
%
%   OUTPUTS
%     fitRGB, fitAlpha  re-shaped garment layer (same size as input).
%     info              struct: rows, gL, gR, tL, tR (for plotting), ok.

fitRGB = warpedRGB;
fitAlpha = warpedAlpha;
info = struct('ok', false, 'rows', [], 'gL', [], 'gR', [], 'tL', [], 'tR', []);
if isempty(bodyMask) || ~any(bodyMask(:))
    return
end
[H, W] = size(warpedAlpha);

%% ---- Rows to process ----------------------------------------------------------
shoulderMid = (torso.leftShoulder + torso.rightShoulder) / 2;
hipMid      = (torso.leftHip + torso.rightHip) / 2;
torsoLength = hipMid(2) - shoulderMid(2);
if torsoLength < 10
    return                                  % person upside down / too small
end
garmentRows = find(any(warpedAlpha > 0.5, 2));
if isempty(garmentRows), return; end

yStart = max(1, round(shoulderMid(2) + params.startFrac * torsoLength));
yFull  = yStart + max(1, round(params.fadeFrac * torsoLength));
yEnd   = min(H, garmentRows(end));
rows = (yStart:yEnd)';
if numel(rows) < 3, return; end

%% ---- 1-2. Garment and body edges in every row ---------------------------------
n = numel(rows);
gL = NaN(n, 1);  gR = NaN(n, 1);
dL = NaN(n, 1);  dR = NaN(n, 1);        % desired edge displacement
for k = 1:n
    y = rows(k);
    % Torso centre line at this height (between shoulder and hip midpoints):
    % we take the run of pixels that CONTAINS the centre, so a separate arm
    % next to the body is not mistaken for the body edge.
    t  = (y - shoulderMid(2)) / torsoLength;
    xc = round(shoulderMid(1) + t * (hipMid(1) - shoulderMid(1)));
    xc = min(max(xc, 1), W);

    [gl, gr] = runAround(warpedAlpha(y, :) > 0.5, xc);
    if isnan(gl), continue; end
    gL(k) = gl;  gR(k) = gr;

    [bl, br] = runAround(bodyMask(y, :), xc);
    if isnan(bl), continue; end
    ratio = (br - bl) / max(gr - gl, 1);
    if ratio < params.minRatio || ratio > params.maxRatio
        continue                             % unreliable row -> interpolate
    end
    % 3. target = body edge moved outward by "ease"
    dL(k) = (bl - params.ease) - gl;
    dR(k) = (br + params.ease) - gr;
end

valid = ~isnan(dL);
if nnz(valid) < 3
    return
end

%% ---- Fill unreliable rows and smooth along the vertical ---------------------
dL = fillAndSmooth(dL, valid, params.smoothSigma);
dR = fillAndSmooth(dR, valid, params.smoothSigma);

% Fade in below the armpits, then apply the overall strength.
w = min(max((rows - yStart) / max(yFull - yStart, 1), 0), 1) * params.strength;
dL = w .* dL;
dR = w .* dR;

%% ---- 4. Build the source-x map and resample ---------------------------------
[X, Y] = meshgrid(1:W, 1:H);
srcX = X;                                    % default: pixel maps to itself
tL = gL + dL;
tR = gR + dR;
for k = 1:n
    if isnan(gL(k)), continue; end
    % keep the target at least 30% of the original width (never fold over)
    if tR(k) - tL(k) < 0.3 * (gR(k) - gL(k))
        continue
    end
    y = rows(k);
    x = 1:W;
    sx = x - dL(k);                                       % left of the garment: shift
    inside = x >= tL(k) & x <= tR(k);                     % garment: stretch
    sx(inside) = gL(k) + (x(inside) - tL(k)) * (gR(k) - gL(k)) / (tR(k) - tL(k));
    right = x > tR(k);                                    % right of it: shift
    sx(right) = x(right) - dR(k);
    srcX(y, :) = sx;
end

% Inverse mapping: output pixel (x,y) takes its colour from (srcX, y).
% Only the processed rows change, so only those are resampled (faster).
fitAlpha(rows, :) = interp2(warpedAlpha, srcX(rows, :), Y(rows, :), 'linear', 0);
for c = 1:size(warpedRGB, 3)
    fitRGB(rows, :, c) = interp2(warpedRGB(:, :, c), srcX(rows, :), Y(rows, :), 'linear', 0);
end

info = struct('ok', true, 'rows', rows, 'gL', gL, 'gR', gR, 'tL', tL, 'tR', tR);
end

% ==========================================================================
function [left, right] = runAround(rowMask, xc)
% Left and right end of the run of true pixels that contains column xc.
left = NaN;  right = NaN;
if ~rowMask(xc), return; end
before = find(~rowMask(1:xc), 1, 'last');
after  = find(~rowMask(xc:end), 1, 'first');
if isempty(before), left = 1; else, left = before + 1; end
if isempty(after), right = numel(rowMask); else, right = xc + after - 2; end
end

function v = fillAndSmooth(v, valid, sigma)
% Linear interpolation over missing rows (nearest value at the ends), then a
% 1-D Gaussian along the rows (normalised at the ends).
idx = (1:numel(v))';
v = interp1(idx(valid), v(valid), idx, 'linear');
first = find(valid, 1, 'first');  last = find(valid, 1, 'last');
v(1:first) = v(first);
v(last:end) = v(last);
if sigma > 0
    r = ceil(3 * sigma);
    g = exp(-(-r:r)'.^2 / (2 * sigma^2));
    v = conv(v, g, 'same') ./ conv(ones(size(v)), g, 'same');
end
end
