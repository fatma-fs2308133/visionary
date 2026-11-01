function [H, info] = estimateHomographyDLT(srcPts, dstPts)
%ESTIMATEHOMOGRAPHYDLT  Planar homography from point pairs (normalised DLT).
%
%   H         = estimateHomographyDLT(srcPts, dstPts)
%   [H, info] = estimateHomographyDLT(srcPts, dstPts)
%
%   *** From-scratch algorithm (core vision step of the project). Only basic
%   *** linear algebra (svd) is used - no fitgeotform2d, fitgeotrans,
%   *** estimateGeometricTransform2D or projective2d.
%
%   Finds the 3x3 matrix H that maps every source point onto its destination
%   point in homogeneous coordinates:
%
%       [x'; y'; w'] = H * [x; y; 1]      destination = [x'/w', y'/w']
%
%   INPUTS
%     srcPts  N-by-2 [x y] points on the garment image (pixels), N >= 4.
%     dstPts  N-by-2 [x y] matching points in the webcam frame (pixels).
%             With N = 4 the fit is exact; with N > 4 it is the algebraic
%             least-squares solution.
%
%   OUTPUTS
%     H      3x3 homography, scaled so that H(3,3) = 1.
%     info   struct: reprojError (N-by-1, px), rmse (px), singularValues
%            (of the normalised DLT matrix A).
%
%   THE ALGORITHM (Hartley & Zisserman, "Multiple View Geometry", Alg. 4.2)
%     1. Normalise both point sets: move the centroid to the origin and
%        scale so the mean distance from it is sqrt(2) (matrices T1, T2).
%        Without this the entries of A range from 1 to ~10^5 (pixel
%        coordinates squared) and the SVD becomes badly conditioned.
%     2. Each pair (x,y) -> (u,v) gives two linear equations in the nine
%        entries h = [h1 .. h9] of H. From
%            u = (h1 x + h2 y + h3) / (h7 x + h8 y + h9)
%            v = (h4 x + h5 y + h6) / (h7 x + h8 y + h9)
%        multiply out the denominator:
%            [x y 1 0 0 0 -u*x -u*y -u] * h = 0
%            [0 0 0 x y 1 -v*x -v*y -v] * h = 0
%        Stacking all pairs gives A * h = 0 with A of size 2N-by-9.
%     3. h is defined only up to scale, so we minimise |A h| subject to
%        |h| = 1. The solution is the right singular vector of A that
%        belongs to its smallest singular value (last column of V).
%     4. Undo the normalisation: H = inv(T2) * Hn * T1, then scale H(3,3) = 1.
%
%   See also applyHomography, inverseWarpBilinear, warpGarment.

srcPts = double(srcPts);
dstPts = double(dstPts);
N = size(srcPts, 1);
if ~isequal(size(srcPts), size(dstPts)) || size(srcPts, 2) ~= 2
    error('estimateHomographyDLT:badPoints', 'srcPts and dstPts must both be N-by-2 with equal N.');
end
if N < 4
    error('estimateHomographyDLT:badPoints', 'A homography needs at least 4 point pairs, got %d.', N);
end
if any(~isfinite(srcPts(:))) || any(~isfinite(dstPts(:)))
    error('estimateHomographyDLT:badPoints', 'Points must be finite (no NaN/Inf).');
end
if N == 4
    % 4 points define H only if no 3 of them lie on one line.
    checkNoThreeCollinear(srcPts, 'source');
    checkNoThreeCollinear(dstPts, 'destination');
end

%% ---- Step 1: normalisation ---------------------------------------------------
[srcN, T1] = normalisePoints(srcPts);
[dstN, T2] = normalisePoints(dstPts);

%% ---- Step 2: build A (2N x 9) ----------------------------------------------
A = zeros(2 * N, 9);
for i = 1:N
    x = srcN(i, 1);  y = srcN(i, 2);
    u = dstN(i, 1);  v = dstN(i, 2);
    A(2 * i - 1, :) = [x y 1 0 0 0 -u * x -u * y -u];
    A(2 * i,     :) = [0 0 0 x y 1 -v * x -v * y -v];
end

%% ---- Step 3: h = right singular vector of the smallest singular value ------
[~, S, V] = svd(A);
s = diag(S);
h = V(:, end);
Hn = reshape(h, 3, 3)';                 % rows of H = [h1 h2 h3; h4 h5 h6; h7 h8 h9]

% A unique solution needs a one-dimensional null space: the second-smallest
% singular value must be clearly above zero (for N = 4, A is 8x9, so s has
% 8 entries and the 9th singular value is implicitly 0).
if numel(s) >= 8 && s(8) < 1e-10 * s(1)
    error('estimateHomographyDLT:degenerate', ...
        'The point configuration does not determine a unique homography.');
end

%% ---- Step 4: denormalise ------------------------------------------------------
H = T2 \ Hn * T1;
if abs(H(3, 3)) < eps
    error('estimateHomographyDLT:degenerate', 'H(3,3) is zero (point at infinity).');
end
H = H / H(3, 3);

if nargout > 1
    proj = applyHomography(H, srcPts);
    info.reprojError    = sqrt(sum((proj - dstPts).^2, 2));
    info.rmse           = sqrt(mean(info.reprojError.^2));
    info.singularValues = s;
end
end

% ==========================================================================
function [pN, T] = normalisePoints(p)
% Similarity transform T (3x3) such that the points T*[p 1] have their
% centroid at the origin and an average distance of sqrt(2) from it.
c = mean(p, 1);
d = mean(sqrt(sum((p - c).^2, 2)));
if d < eps
    error('estimateHomographyDLT:degenerate', 'All points coincide.');
end
s = sqrt(2) / d;
T = [s 0 -s * c(1);
     0 s -s * c(2);
     0 0  1];
pN = (p - c) * s;
end

function checkNoThreeCollinear(P, label)
% Any 3 of the 4 points (almost) on one line -> no unique homography.
% Tolerance: a triangle area below 1 square pixel.
triples = nchoosek(1:4, 3);
for k = 1:size(triples, 1)
    a = P(triples(k, 1), :);  b = P(triples(k, 2), :);  c = P(triples(k, 3), :);
    twiceArea = abs((b(1) - a(1)) * (c(2) - a(2)) - (b(2) - a(2)) * (c(1) - a(1)));
    if twiceArea < 2
        error('estimateHomographyDLT:degenerate', ...
            'Three of the 4 %s points are collinear.', label);
    end
end
end
