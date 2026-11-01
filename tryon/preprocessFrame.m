function [out, info, plain] = preprocessFrame(frame, params)
%PREPROCESSFRAME  Resize, denoise and (if needed) contrast-enhance a camera frame.
%
%   [out, info] = preprocessFrame(frame, params)
%   [out, info, plain] = preprocessFrame(frame, params)
%
%   Runs on every camera frame BEFORE pose estimation, so the pose model,
%   the segmentation and the try-on all see the same cleaned-up image.
%
%     1. Resize      to params.targetWidth (aspect ratio kept). All pixel-
%                    based parameters (blur sigmas, erosion radius, ...) are
%                    tuned for 640 px, and a fixed size keeps the speed
%                    predictable whatever camera is plugged in.
%     2. Denoise     light Gaussian or 3x3 median filter against sensor noise.
%     3. Measure     brightness (mean of L) and contrast (std of L), with L
%                    the grey-level luma (rgb2gray), scaled to 0..1.
%     4. Enhance     if the frame is dark or flat (or always, see params),
%                    on the luma channel Y of YCbCr only, so colours (Cb, Cr)
%                    are not shifted:
%                      a) global stretch: the 1st..99th percentile of Y is
%                         mapped to the full range (stretchlim + imadjust)
%                         - this is what brightens a dark room
%                      b) CLAHE (contrast-limited adaptive histogram
%                         equalisation, adapthisteq) for local contrast
%                    YCbCr instead of Lab because the conversion is linear
%                    and ~10x faster.
%
%   INPUTS
%     frame   H x W x 3 camera frame (uint8 typically).
%     params  cfg.preprocess (see defaultTryOnConfig): targetWidth, denoise,
%             denoiseSigma, enhance ('auto' | 'always' | 'off'), lowLight,
%             lowContrast, claheClip.
%
%   OUTPUTS
%     out     preprocessed frame, uint8, H' x W' x 3.
%     info    struct: scale (resize factor), meanL, stdL (before
%             enhancement), enhanced (logical), time (seconds).
%     plain   the frame after resizing and denoising but BEFORE contrast
%             enhancement. Background subtraction uses this: the
%             enhancement adapts to the image content, so an empty-scene
%             picture and a picture with the person would otherwise be
%             enhanced differently and no longer match.
%
%   NOTE: the enhancement adapts to the image content, which is why
%   background subtraction compares the un-enhanced "plain" frames.
%
%   See also adapthisteq, imgaussfilt, medfilt2, defaultTryOnConfig.

tStart = tic;
if nargin < 2 || isempty(params)
    params = struct();
end
params = fillDefaults(params);

out = im2uint8(frame);
if size(out, 3) == 1
    out = repmat(out, [1 1 3]);          % grey camera -> 3 channels
end

%% ---- 1. Resize ---------------------------------------------------------------
info.scale = 1;
if ~isempty(params.targetWidth) && size(out, 2) ~= params.targetWidth
    info.scale = params.targetWidth / size(out, 2);
    out = imresize(out, info.scale);
end

%% ---- 2. Denoise --------------------------------------------------------------
switch lower(params.denoise)
    case 'gaussian'
        if params.denoiseSigma > 0
            out = imgaussfilt(out, params.denoiseSigma);
        end
    case 'median'
        for c = 1:3
            out(:, :, c) = medfilt2(out(:, :, c), [3 3], 'symmetric');
        end
    case 'none'
        % nothing
    otherwise
        error('preprocessFrame:denoise', 'Unknown denoise mode "%s".', params.denoise);
end

%% ---- 3. Measure brightness and contrast ---------------------------------------
% A quarter-resolution copy is plenty for two statistics and 16x cheaper.
L = im2double(rgb2gray(out(1:4:end, 1:4:end, :)));   % 0..1
info.meanL = mean(L(:));
info.stdL  = std(L(:));

%% ---- 4. Contrast enhancement (CLAHE on the luma) -------------------------------
switch lower(params.enhance)
    case 'always', doEnhance = true;
    case 'auto',   doEnhance = info.meanL < params.lowLight || info.stdL < params.lowContrast;
    case 'off',    doEnhance = false;
    otherwise
        error('preprocessFrame:enhance', 'Unknown enhance mode "%s".', params.enhance);
end
info.enhanced = doEnhance;
plain = out;
if doEnhance
    ycc = rgb2ycbcr(out);
    Y = ycc(:, :, 1);
    Y = imadjust(Y, stretchlim(Y, [0.01 0.99]));
    ycc(:, :, 1) = adapthisteq(Y, 'ClipLimit', params.claheClip);
    out = ycbcr2rgb(ycc);
end
info.time = toc(tStart);
end

% ==========================================================================
function params = fillDefaults(params)
defaults = struct('targetWidth', 640, 'denoise', 'gaussian', 'denoiseSigma', 0.7, ...
                  'enhance', 'auto', 'lowLight', 0.35, 'lowContrast', 0.07, ...
                  'claheClip', 0.01);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(params, names{i})
        params.(names{i}) = defaults.(names{i});
    end
end
end
