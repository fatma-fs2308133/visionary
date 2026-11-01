function files = makeSampleGarments(outDir)
%MAKESAMPLEGARMENTS  Write 5 synthetic garment PNGs so the app works out of the box.
%
%   files = makeSampleGarments()             % into data/garments
%   files = makeSampleGarments(outDir)
%
%   Five different garments (3 cuts x different colours/patterns), drawn
%   with poly2mask, each saved as an RGBA PNG plus NAME_anchors.mat from
%   autoGarmentAnchors. They are placeholders for testing the GUI before
%   the real dataset garments are prepared with prepareGarment.
%   Existing files are not overwritten.
%
%   See also prepareGarment, loadGarmentLibrary.

if nargin < 1 || isempty(outDir), outDir = fullfile('data', 'garments'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end

gH = 420;  gW = 400;
% Outlines in normalised coordinates (x right, y down), collar at the top.
tshirt = [0.38 0.05; 0.44 0.11; 0.50 0.13; 0.56 0.11; 0.62 0.05; ...
          0.78 0.10; 0.98 0.33; 0.86 0.43; 0.76 0.33; ...
          0.75 0.99; 0.25 0.99; ...
          0.24 0.33; 0.14 0.43; 0.02 0.33; 0.22 0.10];
longSleeve = [0.38 0.05; 0.44 0.11; 0.50 0.13; 0.56 0.11; 0.62 0.05; ...
              0.78 0.10; 0.99 0.80; 0.88 0.84; 0.76 0.36; ...
              0.75 0.99; 0.25 0.99; ...
              0.24 0.36; 0.12 0.84; 0.01 0.80; 0.22 0.10];
tank = [0.36 0.04; 0.42 0.20; 0.50 0.24; 0.58 0.20; 0.64 0.04; ...
        0.72 0.06; 0.76 0.33; 0.76 0.99; 0.24 0.99; 0.24 0.33; 0.28 0.06];

specs = {
    'sample_1_navy_tshirt',    tshirt,     [0.12 0.18 0.40], 'plain'
    'sample_2_red_stripes',    tshirt,     [0.80 0.16 0.18], 'stripes'
    'sample_3_green_check',    longSleeve, [0.15 0.50 0.30], 'check'
    'sample_4_yellow_dots',    tshirt,     [0.95 0.80 0.25], 'dots'
    'sample_5_white_tank',     tank,       [0.93 0.93 0.90], 'logo'
};

[gx, gy] = meshgrid(1:gW, 1:gH);
files = cell(1, size(specs, 1));
for k = 1:size(specs, 1)
    [name, outline, colour, pattern] = specs{k, :};
    files{k} = fullfile(outDir, [name '.png']);
    if exist(files{k}, 'file')
        continue
    end
    % Anti-aliased alpha: rasterise at 4x and average down.
    s = 4;
    big = poly2mask(outline(:, 1) * gW * s, outline(:, 2) * gH * s, gH * s, gW * s);
    alpha = imresize(double(big), [gH gW], 'box');

    img = repmat(reshape(colour, 1, 1, 3), gH, gW);
    switch pattern
        case 'stripes'
            m = mod(gy, 40) < 10;
            img = paint(img, m, [0.95 0.95 0.95]);
        case 'check'
            m = xor(mod(floor(gx / 30), 2), mod(floor(gy / 30), 2));
            img = paint(img, m, colour * 0.6);
            img = paint(img, mod(gx, 30) < 3 | mod(gy, 30) < 3, [0.9 0.9 0.8]);
        case 'dots'
            m = (mod(gx, 36) - 18).^2 + (mod(gy, 36) - 18).^2 < 7^2;
            img = paint(img, m, [0.20 0.25 0.60]);
        case 'logo'
            m = (gx - 0.5 * gW).^2 + (gy - 0.45 * gH).^2 < 45^2 & ...
                (gx - 0.5 * gW).^2 + (gy - 0.45 * gH).^2 > 32^2;
            img = paint(img, m, [0.85 0.20 0.20]);
    end
    % Soft studio shading: a little darker toward the hem and the sides.
    shade = (0.92 + 0.08 * (1 - gy / gH)) .* (1 - 0.10 * ((gx - gW / 2) / (gW / 2)).^2);
    img = min(max(img .* shade, 0), 1);

    imwrite(im2uint8(img), files{k}, 'Alpha', im2uint8(alpha));
    anchors = autoGarmentAnchors(alpha); %#ok<NASGU>
    source = 'auto'; %#ok<NASGU>
    save(fullfile(outDir, [name '_anchors.mat']), 'anchors', 'source');
end
fprintf('Sample garments in %s\n', outDir);
end

% ==========================================================================
function img = paint(img, mask, colour)
for c = 1:3
    ch = img(:, :, c);
    ch(mask) = colour(c);
    img(:, :, c) = ch;
end
end
