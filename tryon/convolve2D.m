function out = convolve2D(img, kernel)
%CONVOLVE2D  2-D convolution with replicate padding, implemented from scratch.
%
%   out = convolve2D(img, kernel)
%
%   *** From-scratch replacement for imfilter(img, kernel, 'replicate', 'conv').
%   *** No imfilter, conv2, filter2 or fft is used.
%
%   INPUTS
%     img     H x W double image (one channel).
%     kernel  kh x kw kernel with odd height and width.
%
%   OUTPUT
%     out     H x W double, same size as img.
%
%   THE ALGORITHM
%     out(y,x) = SUM over (i,j) of kernel(i,j) * img(y - i, x - j)
%     with (i,j) measured from the kernel centre. Pixels outside the image
%     take the value of the nearest border pixel ("replicate"), so a region
%     that touches the frame border does not fade toward a fake black edge.
%     Instead of a 4-deep loop we loop over the kernel weights and add the
%     whole shifted image times that weight ("shift and add").
%     If the kernel is separable (rank 1, e.g. a Gaussian) it is split into
%     a column and a row kernel and applied in two 1-D passes:
%     2k operations per pixel instead of k^2, with the same result.
%
%   See also erodeBinaryDisk, blendGarmentOntoFrame.

[kh, kw] = size(kernel);
if mod(kh, 2) == 0 || mod(kw, 2) == 0
    error('convolve2D:kernelSize', 'The kernel must have an odd height and width.');
end
img = double(img);

% Separable kernel? K = u * v' (singular value decomposition, rank 1)
[U, S, V] = svd(kernel);
s = diag(S);
if numel(s) > 1 && s(2) <= 1e-10 * s(1)
    colKernel = U(:, 1) * sqrt(s(1));
    rowKernel = V(:, 1)' * sqrt(s(1));
    out = shiftAndAdd(shiftAndAdd(img, colKernel), rowKernel);
else
    out = shiftAndAdd(img, kernel);
end
end

function out = shiftAndAdd(img, kernel)
[H, W] = size(img);
[kh, kw] = size(kernel);
ry = (kh - 1) / 2;
rx = (kw - 1) / 2;

% Replicate padding: index of the nearest valid row / column
rowIdx = min(max((1 - ry):(H + ry), 1), H);
colIdx = min(max((1 - rx):(W + rx), 1), W);
padded = img(rowIdx, colIdx);

% Convolution flips the kernel: weight kernel(i,j) multiplies the pixel at
% offset (-(i - centre), -(j - centre)).
out = zeros(H, W);
for i = 1:kh
    for j = 1:kw
        w = kernel(kh + 1 - i, kw + 1 - j);
        if w == 0, continue; end
        out = out + w * padded((1:H) + i - 1, (1:W) + j - 1);
    end
end
end
