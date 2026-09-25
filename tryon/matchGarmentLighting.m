function [adjustedGarment, stats] = matchGarmentLighting(garmentImg, torsoRegion, garmentAlpha, params)
%MATCHGARMENTLIGHTING  Shift the garment's brightness/contrast toward the scene.
%
%   [adjustedGarment, stats] = matchGarmentLighting(garmentImg, torsoRegion)
%   [adjustedGarment, stats] = matchGarmentLighting(garmentImg, torsoRegion, garmentAlpha, params)
%
%   A garment PNG is usually a bright studio photo, while the webcam scene
%   may be dim, over-exposed or low-contrast. Pasting it unchanged makes it
%   "glow" on top of the person. This function measures the brightness of
%   the person's torso in the live frame and moves the garment's brightness
%   distribution toward it.
%
%   INPUTS
%     garmentImg    garment RGB (any class). Can be the raw PNG or the garment
%                   already warped into frame coordinates.
%     torsoRegion   RGB crop of the webcam frame covering the torso.
%     garmentAlpha  (optional) alpha of garmentImg. Only opaque garment pixels
%                   are measured; without it the transparent background
%                   would be counted as "black garment".
%     params        (optional) struct:
%       .strength        0..1, how far to move toward the scene (default 0.6)
%       .minGain/.maxGain clamp for the contrast ratio      (default 0.7 / 1.4)
%       .alphaThreshold  alpha above which a pixel is garment (default 0.5)
%
%   OUTPUTS
%     adjustedGarment  double RGB in [0,1], same size as garmentImg.
%     stats            struct with the measured means/std-devs, the gain and
%                      offset used, the imadjust limits and V histograms of
%                      garment and scene (for the demo / a live debug plot).
%
%   METHOD
%     1. RGB -> HSV for both images; keep V (= brightness = max(R,G,B)).
%     2. Measure mean (overall brightness) and standard deviation (contrast)
%        of V: garment -> (muG, sdG), torso region -> (muS, sdS).
%     3. Brightness delta = muS - muG, contrast ratio = sdS / sdG.
%        Scaled by params.strength this gives a linear brightness map
%            v' = gain * v + offset
%        that sends the garment mean to muG + strength*(muS - muG) and
%        scales its spread by  1 + strength*(sdS/sdG - 1).
%     4. That linear map is expressed as imadjust's [low_in high_in] ->
%        [low_out high_out] limits and applied to the garment RGB.
%
%   Applying the same monotonic map to R, G and B changes V = max(R,G,B) by
%   exactly that map, so the measured V statistics move as intended while
%   the hue is essentially kept (a large positive offset slightly lowers
%   saturation, like a washed-out bright scene does).
%
%   ASSUMPTION / LIMITATION: the torso crop mixes lighting with the colour of
%   the clothes the user is really wearing (hence strength < 1 by default).
%   For a stronger model, compare against a reference frame of the same
%   person, or use imhistmatch on V for a full histogram match.
%
%   See also rgb2hsv, imadjust, tryOnPipeline.

%% ---- Parameters ------------------------------------------------------------
if nargin < 3
    garmentAlpha = [];
end
if nargin < 4 || isempty(params)
    params = struct();
end
params = fillDefaults(params);

G = im2double(garmentImg);
S = im2double(torsoRegion);
if size(S, 3) == 1
    S = repmat(S, [1 1 3]);
end

%% ---- Step 1: HSV, keep the V (brightness) channel ---------------------------
hsvGarment = rgb2hsv(G);
hsvScene   = rgb2hsv(S);
vGarmentImg = hsvGarment(:, :, 3);
vScene      = hsvScene(:, :, 3);
vScene      = vScene(:);

% Only measure pixels that actually belong to the garment.
if isempty(garmentAlpha)
    garmentMask = true(size(vGarmentImg));
else
    a = im2double(garmentAlpha);
    garmentMask = a(:, :, 1) >= params.alphaThreshold;
end
vGarment = vGarmentImg(garmentMask);

% Histograms (32 bins over [0,1]) - only for display / debugging.
edges = linspace(0, 1, 33);
stats.histEdges   = edges;
stats.garmentHist = normalisedHistogram(vGarment, edges);
stats.sceneHist   = normalisedHistogram(vScene, edges);

if isempty(vGarment) || isempty(vScene)
    % Nothing to measure (garment off-screen): return unchanged.
    adjustedGarment = G;
    stats = fillIdentityStats(stats);
    return
end

%% ---- Step 2: brightness (mean) and contrast (std) ---------------------------
muG = mean(vGarment);   sdG = std(vGarment);
muS = mean(vScene);     sdS = std(vScene);

%% ---- Step 3: delta -> linear brightness map  v' = gain*v + offset ------------
brightnessDelta = muS - muG;
contrastRatio   = sdS / max(sdG, eps);
contrastRatio   = min(max(contrastRatio, params.minGain), params.maxGain);

s = min(max(params.strength, 0), 1);
gain       = 1 + s * (contrastRatio - 1);    % partial contrast change
targetMean = muG + s * brightnessDelta;      % partial brightness change
offset     = targetMean - gain * muG;        % so that muG maps to targetMean

%% ---- Step 4: express the map as imadjust limits and apply it ----------------
% imadjust(I, [lowIn highIn], [lowOut highOut]) maps lowIn -> lowOut and
% highIn -> highOut linearly and clips outside [lowIn, highIn]. We pick
% lowIn/highIn as the part of [0,1] that the line v' = gain*v + offset keeps
% inside [0,1]; below/above it the output simply clips to 0/1.
lowIn  = max(0, (0 - offset) / gain);
highIn = min(1, (1 - offset) / gain);

if highIn - lowIn < 1e-6
    % Degenerate (the whole range would clip): flat image at the target.
    adjustedGarment = repmat(min(max(targetMean, 0), 1), size(G));
    lowOut = targetMean;  highOut = targetMean;
else
    lowOut  = min(max(gain * lowIn  + offset, 0), 1);
    highOut = min(max(gain * highIn + offset, 0), 1);
    % 2-by-3 limits = same mapping for the R, G and B channels.
    adjustedGarment = imadjust(G, [lowIn highIn]' * ones(1, 3), ...
                                  [lowOut highOut]' * ones(1, 3));
end

%% ---- Report ----------------------------------------------------------------
vAdjusted = max(adjustedGarment, [], 3);     % V channel = max(R,G,B)
stats.garmentMeanV    = muG;
stats.garmentStdV     = sdG;
stats.sceneMeanV      = muS;
stats.sceneStdV       = sdS;
stats.brightnessDelta = brightnessDelta;
stats.contrastRatio   = contrastRatio;
stats.gain            = gain;
stats.offset          = offset;
stats.imadjustIn      = [lowIn highIn];
stats.imadjustOut     = [lowOut highOut];
stats.adjustedMeanV   = mean(vAdjusted(garmentMask));
stats.adjustedHist    = normalisedHistogram(vAdjusted(garmentMask), edges);
end

% ==========================================================================
function h = normalisedHistogram(v, edges)
% Fraction of pixels per bin (sums to 1), so garment and scene histograms
% can be plotted on the same axis despite different pixel counts.
h = histcounts(v, edges);
h = h / max(sum(h), 1);
end

function stats = fillIdentityStats(stats)
stats.garmentMeanV = NaN;  stats.garmentStdV = NaN;
stats.sceneMeanV   = NaN;  stats.sceneStdV   = NaN;
stats.brightnessDelta = 0; stats.contrastRatio = 1;
stats.gain = 1;            stats.offset = 0;
stats.imadjustIn = [0 1];  stats.imadjustOut = [0 1];
stats.adjustedMeanV = NaN; stats.adjustedHist = stats.garmentHist;
end

function params = fillDefaults(params)
defaults = struct('strength', 0.6, 'minGain', 0.7, 'maxGain', 1.4, ...
                  'alphaThreshold', 0.5);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(params, names{i}) || isempty(params.(names{i}))
        params.(names{i}) = defaults.(names{i});
    end
end
end
