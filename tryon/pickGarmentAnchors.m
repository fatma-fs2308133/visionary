function anchors = pickGarmentAnchors(garmentFile)
%PICKGARMENTANCHORS  Click the 8 anchor points on your own garment PNG.
%
%   anchors = pickGarmentAnchors('shirt.png')
%
%   Shows the garment and asks you to click 8 points, then saves them
%   (normalised 0..1) as shirt_anchors.mat next to the PNG. liveTryOn loads
%   that file automatically; in your own code use:
%       s = load('shirt_anchors.mat');  cfg.garment.anchors = s.anchors;
%
%   The garment is assumed to be FRONT-facing, so the wearer's RIGHT side is
%   on the IMAGE LEFT. Click, in order:
%     1 shoulder seam, image-left      5 armpit, image-left
%     2 shoulder seam, image-right     6 armpit, image-right
%     3 hem/hip, image-right           7 sleeve opening centre, image-left
%     4 hem/hip, image-left            8 sleeve opening centre, image-right
%   Hip points: slightly inside the hem corners, where the hip joints would be.

[img, ~, alpha] = imread(garmentFile);
if isempty(alpha)
    error('"%s" has no alpha channel. Use a PNG with a transparent background.', garmentFile);
end
[H, W, ~] = size(img);

prompts = {'1/8  shoulder seam on the image LEFT', ...
           '2/8  shoulder seam on the image RIGHT', ...
           '3/8  hip / hem on the image RIGHT', ...
           '4/8  hip / hem on the image LEFT', ...
           '5/8  armpit on the image LEFT', ...
           '6/8  armpit on the image RIGHT', ...
           '7/8  centre of the sleeve opening on the image LEFT', ...
           '8/8  centre of the sleeve opening on the image RIGHT'};
% Image-left = wearer's right (front-facing garment).
names = {'rightShoulder', 'leftShoulder', 'leftHip', 'rightHip', ...
         'rightArmpit', 'leftArmpit', 'rightSleeveEnd', 'leftSleeveEnd'};

f = figure('Name', 'Click the garment anchors', 'NumberTitle', 'off');
% Show the garment over a checkerboard so the transparent parts are visible.
[cx, cy] = meshgrid(1:W, 1:H);
checker = 0.75 + 0.2 * xor(mod(floor(cx / 16), 2), mod(floor(cy / 16), 2));
a = im2double(alpha);
shown = im2double(img) .* a + checker .* (1 - a);
imshow(shown);
hold on;

anchors = struct();
for i = 1:numel(names)
    title(prompts{i});
    [x, y] = ginput(1);
    plot(x, y, 'c+', 'MarkerSize', 14, 'LineWidth', 2);
    text(x + 6, y, names{i}, 'Color', 'c', 'FontSize', 8);
    anchors.(names{i}) = [(x - 1) / (W - 1), (y - 1) / (H - 1)];
end
title('Done - saved. You can close this window.');

[folder, name] = fileparts(garmentFile);
outFile = fullfile(folder, [name '_anchors.mat']);
save(outFile, 'anchors');
fprintf('Saved %s\n', outFile);
end
