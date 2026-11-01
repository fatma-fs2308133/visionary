function lib = loadGarmentLibrary(folder, maxSide)
%LOADGARMENTLIBRARY  Load every garment PNG (with transparency) in a folder.
%
%   lib = loadGarmentLibrary('data/garments')
%   lib = loadGarmentLibrary(folder, maxSide)
%
%   OUTPUT: struct array, one element per garment, with fields
%     name          file name without extension (shown in the GUI)
%     file          full path of the PNG
%     rgb, alpha    uint8 image and alpha (longest side <= maxSide, 512)
%     anchors       normalised anchor struct for cfg.garment.anchors:
%                   clicked anchors from NAME_anchors.mat (pickGarmentAnchors)
%                   if they exist, otherwise autoGarmentAnchors (recomputed
%                   on every load, so improvements to it always apply)
%     anchorSource  'file' or 'auto'
%     thumb         small RGB preview on a light background (for the GUI)
%
%   PNGs without an alpha channel are skipped with a warning. Garments are
%   sorted by name.
%
%   See also prepareGarment, makeSampleGarments, TryOnApp.

if nargin < 2 || isempty(maxSide), maxSide = 512; end
lib = struct('name', {}, 'file', {}, 'rgb', {}, 'alpha', {}, ...
             'anchors', {}, 'anchorSource', {}, 'thumb', {});
if ~isfolder(folder)
    return
end
files = dir(fullfile(folder, '*.png'));
[~, order] = sort(lower({files.name}));
files = files(order);

for i = 1:numel(files)
    f = fullfile(files(i).folder, files(i).name);
    [~, name] = fileparts(f);
    try
        [rgb, ~, alpha] = imread(f);
    catch err
        warning('loadGarmentLibrary:read', 'Cannot read %s: %s', files(i).name, err.message);
        continue
    end
    if isempty(alpha)
        warning('loadGarmentLibrary:noAlpha', ...
            '%s has no transparency - run prepareGarment on it first.', files(i).name);
        continue
    end
    if size(rgb, 3) == 1, rgb = repmat(rgb, [1 1 3]); end
    rgb = im2uint8(rgb);
    alpha = im2uint8(alpha);
    scale = maxSide / max(size(alpha));
    if scale < 1
        rgb = imresize(rgb, scale);
        alpha = imresize(alpha, scale);
    end

    g.name = name;
    g.file = f;
    g.rgb = rgb;
    g.alpha = alpha;
    anchorsFile = fullfile(files(i).folder, [name '_anchors.mat']);
    g.anchors = [];
    if exist(anchorsFile, 'file')
        s = load(anchorsFile);
        % Files written by prepareGarment / makeSampleGarments are marked
        % source = 'auto': recompute those; keep hand-clicked ones.
        isAuto = isfield(s, 'source') && strcmp(s.source, 'auto');
        if isfield(s, 'anchors') && ~isAuto
            g.anchors = s.anchors;
            g.anchorSource = 'file';
        end
    end
    if isempty(g.anchors)
        try
            g.anchors = autoGarmentAnchors(alpha);
            g.anchorSource = 'auto';
        catch err
            warning('loadGarmentLibrary:anchors', 'No anchors for %s: %s', name, err.message);
            continue
        end
    end
    % Thumbnail: garment over a light grey background.
    a = im2double(imresize(alpha, [96 NaN]));
    t = im2double(imresize(rgb, [96 NaN]));
    g.thumb = im2uint8(t .* a + 0.93 * (1 - a));
    lib(end + 1) = g; %#ok<AGROW>
end
end
