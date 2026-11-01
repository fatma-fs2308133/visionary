function anchors = autoGarmentAnchors(alpha)
%AUTOGARMENTANCHORS  Estimate the 8 garment anchor points from the silhouette.
%
%   anchors = autoGarmentAnchors(alpha)
%
%   Gives a usable first guess for a new garment, so every garment works
%   without clicking. For the best fit, refine with pickGarmentAnchors.
%
%   INPUT
%     alpha    H x W alpha of a FRONT-facing upper-body garment, collar up.
%
%   OUTPUT
%     anchors  struct with normalised [x y] (0..1) fields leftShoulder,
%              rightShoulder, leftHip, rightHip, leftArmpit, rightArmpit,
%              leftSleeveEnd, rightSleeveEnd - same format as
%              cfg.garment.anchors. Wearer's LEFT = image RIGHT.
%
%   METHOD (shape analysis of the binary mask, row by row)
%     - Torso edges from the BOTTOM OUTLINE (lowest garment pixel of every
%       column). Walking outward from the centre, the outline follows the
%       hem upward; where it drops down again (a sleeve or cuff hanging
%       lower) or the garment ends, we have passed the side of the torso.
%       This works even when long sleeves hang against the body and merge
%       with it in the silhouette. Fallback: the narrowest run of pixels
%       through the centre column in the lower part (55-88% of the height).
%     - Armpit: the lowest row (in the upper 70%) where the run still
%       reaches clearly beyond the torso edge - for sleeves that stick out.
%       If that is not at a plausible height (18-45% from the top: long
%       sleeves hanging down, sleeveless), it is placed at 30% of the
%       height, the typical armpit level of a shirt.
%     - Shoulder seam: slightly inside the armpit, 4% of the height below
%       the top outline of the garment at that column.
%     - Hip: 6% above the bottom, 16% of the torso width inside its edges
%       (pose models put the hip keypoint at the hip joint, inside the
%       body outline).
%     - Sleeve opening: centroid of the sleeve pixels (outside the torso
%       edge) farthest from the shoulder (top 15% by distance).
%     Sleeveless garments get sleeve ends just outside the armpits, so no
%     sleeve is cut off.
%
%   See also pickGarmentAnchors, prepareGarment, defaultTryOnConfig.

M = im2double(alpha);
if size(M, 3) > 1, M = M(:, :, 1); end
M = M > 0.5;
[H, W] = size(M);
if ~any(M(:))
    error('autoGarmentAnchors:empty', 'The alpha mask is empty.');
end
M = bwareafilt(M, 1);                         % ignore stray specks

rows = find(any(M, 2));
top = rows(1);  bottom = rows(end);
h = bottom - top + 1;
[~, cols] = find(M);
xc = round(mean(cols));

%% ---- Left/right edge of the run through the centre, for every row -------
L = NaN(H, 1);  R = NaN(H, 1);
for y = top:bottom
    [L(y), R(y)] = runAround(M(y, :), xc);
end

%% ---- Torso edges ------------------------------------------------------------
bottomY = NaN(1, W);
for x = 1:W
    r = find(M(:, x), 1, 'last');
    if ~isempty(r), bottomY(x) = r; end
end
bottomY = movmean(bottomY, 3, 'omitnan');           % smooth the jagged outline
tL = edgeFromBottom(bottomY, xc, -1, h);
tR = edgeFromBottom(bottomY, xc, +1, h);
if isnan(tL) || isnan(tR) || tR - tL < 0.25 * (max(cols) - min(cols))
    % Fallback: narrowest centre run in the lower part.
    lower = (round(top + 0.55 * h):round(top + 0.88 * h))';
    widths = R(lower) - L(lower);
    widths(isnan(widths)) = Inf;
    [~, k] = min(widths);
    if isinf(widths(k))
        tL = min(cols);  tR = max(cols);
    else
        tL = L(lower(k));  tR = R(lower(k));
    end
end
w = tR - tL;

%% ---- Armpits -------------------------------------------------------------------
upper = (top:round(top + 0.7 * h))';
margin = 0.08 * w;
armL = armpit(upper(L(upper) < tL - margin), L, tL, top, h, w);
armR = armpit(upper(R(upper) > tR + margin), R, tR, top, h, w);

%% ---- Shoulder seams, hips ----------------------------------------------------
shL = shoulder(M, armL(1) + 0.06 * w, h);       % image-left = wearer's right
shR = shoulder(M, armR(1) - 0.06 * w, h);
yHip = top + round(0.94 * h);
hipL = [tL + 0.16 * w, yHip];
hipR = [tR - 0.16 * w, yHip];

%% ---- Sleeve openings -------------------------------------------------------
[X, Y] = meshgrid(1:W, 1:H);
endL = sleeveEnd(M & X < armL(1) - 0.02 * w, X, Y, shL, armL, -1, w);
endR = sleeveEnd(M & X > armR(1) + 0.02 * w, X, Y, shR, armR, +1, w);

%% ---- Normalise (image-left = wearer's right) -----------------------------
n = @(p) [(p(1) - 1) / (W - 1), (p(2) - 1) / (H - 1)];
anchors.rightShoulder  = n(shL);
anchors.leftShoulder   = n(shR);
anchors.rightHip       = n(hipL);
anchors.leftHip        = n(hipR);
anchors.rightArmpit    = n(armL);
anchors.leftArmpit     = n(armR);
anchors.rightSleeveEnd = n(endL);
anchors.leftSleeveEnd  = n(endR);
end

% ==========================================================================
function edge = edgeFromBottom(bottomY, xc, step, h)
% Walk from the centre column outward along the bottom outline. The torso
% side is where the outline turns downward again by more than 2% of the
% height (a sleeve hangs lower) or where the garment ends. NaN if neither
% happens.
edge = NaN;
best = bottomY(xc);  xBest = xc;
x = xc;
while true
    x = x + step;
    if x < 1 || x > numel(bottomY)
        return
    end
    b = bottomY(x);
    if isnan(b)
        edge = x - step;                 % garment ends (sleeveless)
        return
    end
    if b < best
        best = b;  xBest = x;            % outline still rising (hem curve)
    elseif b > best + 0.02 * h
        edge = xBest;                    % outline drops again: sleeve / cuff
        return
    end
end
end

function p = armpit(sleeveRows, edge, torsoEdge, top, h, w)
% Armpit = torso edge just below the lowest row where a sleeve sticks out;
% at 30% of the height if that row is implausible (hanging long sleeves,
% sleeveless).
y = NaN;
if ~isempty(sleeveRows)
    y = min(sleeveRows(end) + max(1, round(0.02 * h)), top + h - 1);
end
if isnan(y) || y < top + 0.18 * h || y > top + 0.45 * h
    y = top + round(0.30 * h);
end
x = edge(y);
if isnan(x) || abs(x - torsoEdge) > 0.08 * w
    x = torsoEdge;                  % edge merged with a sleeve: use the torso line
end
p = [x, y];
end

function p = shoulder(M, x, h)
% 4% of the garment height below the top outline at column x.
x = min(max(round(x), 1), size(M, 2));
y = find(M(:, x), 1, 'first');
if isempty(y), y = find(any(M, 2), 1, 'first'); end
p = [x, y + 0.04 * h];
end

function p = sleeveEnd(S, X, Y, shoulder, arm, side, w)
% Centroid of the sleeve pixels farthest from the shoulder.
if nnz(S) < 0.002 * numel(S)
    p = [arm(1) + side * 0.05 * w, arm(2)];   % no sleeve: just outside the armpit
    return
end
xs = X(S);  ys = Y(S);
dist = hypot(xs - shoulder(1), ys - shoulder(2));
sorted = sort(dist);
far = dist >= sorted(max(1, ceil(0.85 * numel(sorted))));
p = [mean(xs(far)), mean(ys(far))];
end

function [left, right] = runAround(rowMask, xc)
% Left and right end of the run of true pixels that contains column xc.
left = NaN;  right = NaN;
if ~rowMask(xc), return; end
before = find(~rowMask(1:xc), 1, 'last');
after  = find(~rowMask(xc:end), 1, 'first');
if isempty(before), left = 1; else, left = before + 1; end
if isempty(after), right = numel(rowMask); else, right = xc + after - 2; end
end
