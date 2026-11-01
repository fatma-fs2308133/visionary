function outFiles = prepareGarment(input, outDir, opts)
%PREPAREGARMENT  Turn dataset garment photos into transparent PNGs + anchors.
%
%   outFiles = prepareGarment('cloth/00006_00.jpg', 'data/garments')
%   outFiles = prepareGarment('cloth', 'data/garments')        % whole folder
%   outFiles = prepareGarment(..., opts)
%
%   Garment images in try-on datasets (VITON-HD, DressCode, MV-Fashion
%   product shots) are flat, front-facing photos on a plain background.
%   This function cuts the garment out, saves it as an RGBA PNG and
%   estimates its anchor points (autoGarmentAnchors), so the garment can be
%   used by liveTryOn / TryOnApp straight away.
%
%   BACKGROUND REMOVAL
%     If the dataset provides a mask (VITON-HD: the folder "cloth-mask" next
%     to "cloth", same file name) it is used. Otherwise:
%       1. background colour = median of the image border pixels
%       2. colour distance of every pixel to it (after a light blur)
%       3. threshold -> opening/closing -> fill holes -> largest region
%     A garment with the same colour as the background (white shirt on
%     white) needs the dataset mask or a manual fix.
%
%   INPUTS
%     input   an image file, or a folder (all .jpg/.jpeg/.png inside).
%     outDir  output folder (created if missing). Default 'data/garments'.
%     opts    (optional) struct:
%       .maskFile   explicit mask image for a single input file
%       .threshold  colour distance (0..1) counted as garment   (0.10)
%       .maxSide    longest side of the saved PNG, px            (512)
%       .margin     transparent border around the garment, px     (8)
%       .overwrite  re-create existing PNGs                      (false)
%
%   OUTPUT
%     outFiles  cell array of the PNG files written.
%
%   See also autoGarmentAnchors, pickGarmentAnchors, loadGarmentLibrary.

if nargin < 2 || isempty(outDir), outDir = fullfile('data', 'garments'); end
if nargin < 3 || isempty(opts), opts = struct(); end
opts = fillDefaults(opts);
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% ---- A folder: process every image in it ---------------------------------
if isfolder(input)
    files = [dir(fullfile(input, '*.jpg')); dir(fullfile(input, '*.jpeg')); dir(fullfile(input, '*.png'))];
    outFiles = {};
    for i = 1:numel(files)
        f = fullfile(files(i).folder, files(i).name);
        try
            outFiles = [outFiles, prepareGarment(f, outDir, rmfield(opts, 'maskFile'))]; %#ok<AGROW>
        catch err
            warning('prepareGarment:skip', 'Skipped %s: %s', files(i).name, err.message);
        end
    end
    fprintf('Prepared %d of %d garments in %s\n', numel(outFiles), numel(files), outDir);
    return
end

%% ---- A single file ------------------------------------------------------------
[~, name] = fileparts(input);
outFile = fullfile(outDir, [name '.png']);
outFiles = {outFile};
if exist(outFile, 'file') && ~opts.overwrite
    return
end

[img, ~, existingAlpha] = imread(input);
img = im2double(img);
if size(img, 3) == 1, img = repmat(img, [1 1 3]); end

maskFile = opts.maskFile;
if isempty(maskFile)
    maskFile = findDatasetMask(input);
end
if ~isempty(existingAlpha)
    mask = im2double(existingAlpha) > 0.5;           % already transparent
elseif ~isempty(maskFile)
    m = im2double(imread(maskFile));
    mask = m(:, :, 1) > 0.5;                           % JPEG masks are not exactly 0/1
    if ~isequal(size(mask), [size(img, 1) size(img, 2)])
        mask = imresize(mask, [size(img, 1) size(img, 2)], 'nearest');
    end
else
    mask = segmentFromBorder(img, opts.threshold);
end
if nnz(mask) < 0.02 * numel(mask)
    error('prepareGarment:noGarment', 'No garment found in %s (try a dataset mask).', input);
end
mask = bwareafilt(imfill(mask, 'holes'), 1);

% Soft 1-px edge so the cut-out is not jagged.
alpha = min(max(imgaussfilt(double(mask), 0.8), 0), 1);
alpha(~imdilate(mask, strel('disk', 2))) = 0;

%% ---- Crop to the garment + margin, limit the size -----------------------------
rows = find(any(mask, 2));  cols = find(any(mask, 1));
r1 = max(1, rows(1) - opts.margin);  r2 = min(size(mask, 1), rows(end) + opts.margin);
c1 = max(1, cols(1) - opts.margin);  c2 = min(size(mask, 2), cols(end) + opts.margin);
img = img(r1:r2, c1:c2, :);
alpha = alpha(r1:r2, c1:c2);
scale = opts.maxSide / max(size(alpha));
if scale < 1
    img = imresize(img, scale);
    alpha = imresize(alpha, scale);
end
img = min(max(img, 0), 1);
alpha = min(max(alpha, 0), 1);

imwrite(im2uint8(img), outFile, 'Alpha', im2uint8(alpha));

% Anchors: only if the user has not already made (better) manual ones.
anchorsFile = fullfile(outDir, [name '_anchors.mat']);
if ~exist(anchorsFile, 'file') || (opts.overwrite && isAutoAnchors(anchorsFile))
    anchors = autoGarmentAnchors(alpha); %#ok<NASGU>
    source = 'auto'; %#ok<NASGU>
    save(anchorsFile, 'anchors', 'source');
end
fprintf('Garment: %s\n', outFile);
end

% ==========================================================================
function tf = isAutoAnchors(file)
% true if the anchors file was written automatically (not clicked by hand).
s = load(file);
tf = isfield(s, 'source') && strcmp(s.source, 'auto');
end

function maskFile = findDatasetMask(imgFile)
% VITON-HD layout: .../cloth/NAME.jpg  ->  .../cloth-mask/NAME.jpg
maskFile = '';
[folder, name, ext] = fileparts(imgFile);
parent = fileparts(folder);
candidates = {fullfile(parent, 'cloth-mask', [name ext]), ...
              fullfile(parent, 'cloth-mask', [name '.jpg']), ...
              fullfile(parent, 'cloth-mask', [name '.png']), ...
              fullfile(parent, 'cloth_mask', [name ext])};
for i = 1:numel(candidates)
    if exist(candidates{i}, 'file')
        maskFile = candidates{i};
        return
    end
end
end

function mask = segmentFromBorder(img, threshold)
% Garment = pixels whose colour differs from the background colour, which
% is estimated as the median of the image border.
b = imgaussfilt(img, 1);
border = [reshape(b(1, :, :), [], 3); reshape(b(end, :, :), [], 3); ...
          reshape(b(:, 1, :), [], 3); reshape(b(:, end, :), [], 3)];
bg = median(border, 1);
dist = sqrt(sum((b - reshape(bg, 1, 1, 3)).^2, 3)) / sqrt(3);
mask = dist > threshold;
% Clean-up radii scale with the picture so small web images keep the
% gaps between sleeves and body (1% / 0.5% of the longest side).
n = max(size(mask));
mask = imopen(mask, strel('disk', max(1, round(0.005 * n))));
mask = imclose(mask, strel('disk', max(1, round(0.01 * n))));
end

function opts = fillDefaults(opts)
defaults = struct('maskFile', '', 'threshold', 0.10, 'maxSide', 512, ...
                  'margin', 8, 'overwrite', false);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(opts, names{i}) || (isempty(opts.(names{i})) && ~strcmp(names{i}, 'maskFile'))
        opts.(names{i}) = defaults.(names{i});
    end
end
end
