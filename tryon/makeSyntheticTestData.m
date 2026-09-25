function [frame, keypoints, garmentImg, garmentAlpha, background] = makeSyntheticTestData(brightness)
%MAKESYNTHETICTESTDATA  Generate a fake webcam frame, keypoints and garment.
%
%   [frame, keypoints, garmentImg, garmentAlpha] = makeSyntheticTestData()
%   [...] = makeSyntheticTestData(brightness)
%
%   Lets the whole pipeline be tested without a webcam, a pose model or any
%   image files. Everything is drawn with poly2mask so it is deterministic.
%
%     frame        480x640x3 uint8: simple background + a cartoon person
%                  wearing a dark shirt.
%     keypoints    struct with leftShoulder, rightShoulder, leftHip, rightHip
%                  as [x y score] - the same format a stubbed pose model
%                  would return. Person faces the camera, so the person's
%                  LEFT shoulder is on the IMAGE RIGHT.
%     garmentImg   420x400x3 uint8 T-shirt with stripes (so warping is easy to
%                  see) and a chest logo.
%     garmentAlpha 420x400 uint8 alpha (anti-aliased edge, like a real PNG).
%     background   the same scene WITHOUT the person (for background
%                  subtraction / segmentPersonBackground).
%
%   brightness     scene brightness multiplier (default 1). Use e.g. 0.45 to
%                  simulate a dim room when testing matchGarmentLighting.

if nargin < 1 || isempty(brightness)
    brightness = 1;
end

%% ---- Keypoints (pixel coordinates in a 640x480 frame) ------------------------
keypoints.rightShoulder = [245 165 0.95];   % image left
keypoints.leftShoulder  = [395 165 0.95];   % image right
keypoints.rightHip      = [272 365 0.90];
keypoints.leftHip       = [368 365 0.90];
keypoints.rightElbow    = [207 255 0.90];   % arms hang down beside the body
keypoints.leftElbow     = [433 255 0.90];

%% ---- Webcam frame ------------------------------------------------------------
H = 480;  W = 640;
[X, Y] = meshgrid(1:W, 1:H);

% Background: soft wall gradient with a little sensor noise.
frame = zeros(H, W, 3);
frame(:, :, 1) = 0.62 + 0.15 * X / W;
frame(:, :, 2) = 0.64 + 0.05 * X / W;
frame(:, :, 3) = 0.66 - 0.12 * Y / H;

emptyScene = frame;               % remember the scene before the person is drawn

skin  = [0.86 0.70 0.58];
shirt = [0.55 0.72 0.95];          % the (real) light-blue shirt the user wears
pants = [0.20 0.22 0.25];

% Legs / pants.
frame = paint(frame, poly2mask([262 378 392 342 320 298 248], ...
                               [360 360 480 480 400 480 480], H, W), pants);
% Arms (skin), drawn before the torso so the shirt covers the shoulders.
frame = paint(frame, poly2mask([232 205 180 204 222 250], [160 250 345 352 262 190], H, W), skin);
frame = paint(frame, poly2mask([408 435 460 436 418 390], [160 250 345 352 262 190], H, W), skin);
% Neck and head.
frame = paint(frame, poly2mask([302 338 338 302], [110 110 160 160], H, W), skin);
frame = paint(frame, (X - 320).^2 + ((Y - 95) / 1.15).^2 <= 42^2, skin);
% Torso with short sleeves, plus a bit of vertical shading (for contrast).
torsoMask = poly2mask([300 340 405 440 425 400 385 255 240 215 200 235], ...
                      [150 150 160 215 240 225 375 375 225 240 215 160], H, W);
% Round torso (brighter in the middle) + a few diagonal folds, so the
% shading-transfer stage has something to copy.
shade = 0.75 + 0.3 * exp(-((X - 320) / 70).^2);
folds = 1 - 0.18 * max(0, sin((X + 0.6 * Y) / 9)).^6 .* (Y > 250);
shade = shade .* folds;
for c = 1:3
    ch = frame(:, :, c);
    val = shirt(c) * shade;
    ch(torsoMask) = val(torsoMask);
    frame(:, :, c) = ch;
end

rng(0);                                          % reproducible noise
noise = 0.015 * randn(H, W, 3);
frame = im2uint8(min(max(frame * brightness + noise, 0), 1));
background = im2uint8(min(max(emptyScene * brightness + noise, 0), 1));

%% ---- Garment (front-facing T-shirt, collar at the top) ----------------------
gH = 420;  gW = 400;
% Outline in normalised coordinates (x right, y down), clockwise from the
% image-left side of the collar. Matches the default anchors in
% defaultTryOnConfig (shoulder seams at x = 0.27 / 0.73, y = 0.14).
outline = [0.38 0.05; 0.44 0.11; 0.50 0.13; 0.56 0.11; 0.62 0.05; ...   % collar
           0.78 0.10; 0.98 0.33; 0.86 0.43; 0.76 0.33;               ...   % image-right sleeve
           0.75 0.99; 0.25 0.99;                                     ...   % hem
           0.24 0.33; 0.14 0.43; 0.02 0.33; 0.22 0.10];                    % image-left sleeve

% Anti-aliased alpha: rasterise at 4x resolution and average down.
s = 4;
bigMask = poly2mask(outline(:, 1) * gW * s, outline(:, 2) * gH * s, gH * s, gW * s);
garmentAlpha = imresize(double(bigMask), [gH gW], 'box');

% Texture: red shirt, white horizontal stripes, a darker vertical centre
% band and a round chest "logo" - all make rotation/shear/turn visible.
[gx, gy] = meshgrid(1:gW, 1:gH);
base = cat(3, 0.80 * ones(gH, gW), 0.16 * ones(gH, gW), 0.18 * ones(gH, gW));
stripes = mod(gy, 40) < 10;
centre  = abs(gx - gW / 2) < 12;
logo    = (gx - 0.62 * gW).^2 + (gy - 0.30 * gH).^2 < 22^2;
garment = base;
for c = 1:3
    ch = garment(:, :, c);
    ch(stripes) = 0.95;
    ch(centre)  = 0.55 * ch(centre);
    ch(logo)    = [0.95 0.85 0.20] * [c == 1; c == 2; c == 3];
    garment(:, :, c) = ch;
end
garment = garment .* (0.9 + 0.1 * gy / gH);      % slight studio shading
garmentImg   = im2uint8(garment);
garmentAlpha = im2uint8(garmentAlpha);
end

% ==========================================================================
function img = paint(img, mask, color)
% Fill the pixels of MASK in IMG with a solid RGB COLOR.
for c = 1:3
    ch = img(:, :, c);
    ch(mask) = color(c);
    img(:, :, c) = ch;
end
end
