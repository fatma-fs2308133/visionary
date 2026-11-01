function counts = splitDataset(srcDir, outRoot, ratios, seed)
%SPLITDATASET  Copy a folder of images into train / val / test sub-folders.
%
%   splitDataset('downloads/viton/test/cloth', 'data/garment_split')
%   splitDataset(srcDir, outRoot, [0.7 0.15 0.15], 1)
%
%   The pose model is pretrained, so nothing is trained on "train". The
%   split keeps tuning and testing honest:
%     train  garments used while developing (e.g. picking anchors)
%     val    used to tune parameters (blend sigma, thresholds, ...)
%     test   only touched for the final numbers in the report
%   A fixed random seed makes the split reproducible. If a VITON-HD style
%   "cloth-mask" folder sits next to srcDir, the masks are split the same
%   way, so prepareGarment still finds them.
%
%   OUTPUT: counts = [nTrain nVal nTest].
%
%   See also prepareGarment.

if nargin < 3 || isempty(ratios), ratios = [0.7 0.15 0.15]; end
if nargin < 4 || isempty(seed), seed = 1; end
ratios = ratios / sum(ratios);

files = [dir(fullfile(srcDir, '*.jpg')); dir(fullfile(srcDir, '*.jpeg')); dir(fullfile(srcDir, '*.png'))];
n = numel(files);
if n == 0
    error('splitDataset:empty', 'No images found in %s.', srcDir);
end
rng(seed);
order = randperm(n);
nTrain = round(ratios(1) * n);
nVal = round(ratios(2) * n);
groups = {order(1:nTrain), order(nTrain + 1:nTrain + nVal), order(nTrain + nVal + 1:end)};
names = {'train', 'val', 'test'};

[srcParent, srcName] = fileparts(srcDir);
maskDir = fullfile(srcParent, 'cloth-mask');
hasMasks = isfolder(maskDir);
counts = zeros(1, 3);
for g = 1:3
    outDir = fullfile(outRoot, names{g}, srcName);
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    if hasMasks
        outMask = fullfile(outRoot, names{g}, 'cloth-mask');
        if ~exist(outMask, 'dir'), mkdir(outMask); end
    end
    for k = groups{g}
        copyfile(fullfile(files(k).folder, files(k).name), outDir);
        m = fullfile(maskDir, files(k).name);
        if hasMasks && exist(m, 'file')
            copyfile(m, outMask);
        end
    end
    counts(g) = numel(groups{g});
end
fprintf('Split %d images: %d train / %d val / %d test -> %s\n', n, counts, outRoot);
end
