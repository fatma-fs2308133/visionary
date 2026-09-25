function cfg = defaultTryOnConfig()
%DEFAULTTRYONCONFIG  Default (tunable) parameters for the virtual try-on pipeline.
%
%   cfg = defaultTryOnConfig()
%
%   Every tunable number of the pipeline lives in this one struct, so that in
%   App Designer each field can be bound to a slider / spinner / dropdown and
%   changed live (just overwrite the field and the next frame uses it).
%
%   The defaults assume a 640x480 webcam frame with the person's torso roughly
%   150-250 px wide. If you work at 1280x720, scale the pixel-based values
%   (erodeRadius, gaussSigma) up by about 2x.
%
%   See also tryOnPipeline, warpGarment, blendGarmentOntoFrame,
%   matchGarmentLighting, getTorsoKeypoints.

%% ------------------------------------------------------------------------
%  Pose / keypoint input
%  ------------------------------------------------------------------------
% Keypoints whose confidence is below this value are treated as missing. If
% any of the 4 torso keypoints is missing, the frame is returned unchanged
% (no garment is drawn), which is better than drawing a garment in a random
% place. ASSUMPTION: 0.3 is a common default for MediaPipe / OpenPose scores.
cfg.pose.minScore = 0.3;

%% ------------------------------------------------------------------------
%  Garment anchor points (where the body landmarks sit on the garment PNG)
%  ------------------------------------------------------------------------
% Normalised [x y] coordinates inside the garment image (0 = left/top edge,
% 1 = right/bottom edge). These are the garment-side "source" points of the
% warp; the body keypoints are the "destination" points.
%
% ASSUMPTION: the garment PNG is a front-facing T-shirt photographed flat,
% centred, with the collar at the top. "Left"/"right" are the WEARER's left
% and right, so for a front-facing garment the wearer's LEFT shoulder is on
% the IMAGE RIGHT - exactly like the person in a (non-mirrored) webcam
% frame. Hip anchors are a bit inside the hem corners because pose models
% put the hip keypoint at the hip JOINT, which is narrower than the body.
%
% >>> Adjust these per garment image. The demo script draws them on the
% >>> garment so you can check them visually.
cfg.garment.anchors.rightShoulder = [0.27 0.14];   % image-left shoulder seam
cfg.garment.anchors.leftShoulder  = [0.73 0.14];   % image-right shoulder seam
cfg.garment.anchors.rightHip      = [0.33 0.93];   % image-left, near the hem
cfg.garment.anchors.leftHip       = [0.67 0.93];   % image-right, near the hem
% Sleeve anchors (used to cut the sleeves off and rotate them with the arms):
% the armpit point and the centre of each sleeve opening. The sleeve is
% everything on the outer side of the line shoulder -> armpit.
cfg.garment.anchors.rightArmpit    = [0.24 0.33];
cfg.garment.anchors.leftArmpit     = [0.76 0.33];
cfg.garment.anchors.rightSleeveEnd = [0.08 0.38];
cfg.garment.anchors.leftSleeveEnd  = [0.92 0.38];
% >>> Easiest way to set all 8 anchors for YOUR garment:
% >>>     pickGarmentAnchors('shirt.png')   (click them; saved next to the PNG)

%% ------------------------------------------------------------------------
%  Warping
%  ------------------------------------------------------------------------
% 'piecewise' : 4 points (both shoulders + both hips), quad split into two
%               triangles, one affine transform per triangle. Handles torso
%               turning / leaning, where the two sides of the body deform
%               differently. (DEFAULT)
% 'affine'    : 3 points (left shoulder, right shoulder, hip centre), one
%               affine transform for the whole garment. Simpler, but a
%               single affine cannot squash one side more than the other.
cfg.warp.mode = 'piecewise';

%% ------------------------------------------------------------------------
%  Sleeves follow the arms (needs elbow keypoints)
%  ------------------------------------------------------------------------
% Each sleeve is cut off the garment and rotated about the shoulder so its
% axis points along the upper arm (shoulder -> elbow). Without this the
% sleeves keep the angle they have in the garment photo (sticking out) no
% matter where the arms are. If an elbow is not detected, that sleeve just
% moves with the torso.
cfg.sleeves.enable = true;

%% ------------------------------------------------------------------------
%  Fit the garment to the body silhouette (needs a background frame)
%  ------------------------------------------------------------------------
% The 4-point warp only knows where shoulders and hips are, so the garment
% sides stay straight. With a person mask (background subtraction) each
% garment row below the armpits is stretched/squeezed horizontally so its
% left/right edges follow the body's edges (waist, chest, hips).
cfg.fit.enable = true;
% 0 = ignore the body shape, 1 = hug it completely. ASSUMPTION 0.8: a
% T-shirt is a bit looser than the body.
cfg.fit.strength = 0.8;
% Extra width (px) on each side so the garment sits slightly outside the
% skin, like real fabric. ASSUMPTION: 4 px at 640x480.
cfg.fit.ease = 4;
% Fitting starts this far down from the shoulders (fraction of shoulder->hip
% distance, ~armpit level) and fades in over fadeFrac, so the shoulders and
% sleeves are not distorted.
cfg.fit.startFrac = 0.25;
cfg.fit.fadeFrac  = 0.20;
% Smoothing of the edge offsets along the vertical direction (rows), so a
% noisy mask does not make the garment edge wobble.
cfg.fit.smoothSigma = 6;
% Rows where body width / garment width is outside this range are treated as
% segmentation errors (arm touching the body, shadow) and interpolated.
cfg.fit.minRatio = 0.6;
cfg.fit.maxRatio = 1.5;

% Background subtraction (segmentPersonBackground)
cfg.segment.threshold  = 0.12;  % colour difference (0..1) counted as "person"
cfg.segment.blurSigma  = 1.5;   % denoise both images before differencing
cfg.segment.openRadius = 2;     % remove speckles
cfg.segment.closeRadius = 6;    % close small gaps inside the body

%% ------------------------------------------------------------------------
%  Shading transfer: folds / body curvature onto the garment ("flow")
%  ------------------------------------------------------------------------
% The clothes the person is really wearing show how light falls on the body:
% darker at the sides (the torso is round), wrinkles, folds, shadows under
% the chest. We measure that shading in the webcam frame (brightness divided
% by its local average) and multiply it onto the garment, so the virtual
% garment gets the same folds and 3-D roundness and moves with the person.
cfg.shading.enable = true;
% 0 = flat garment, 1 = full shading of the real clothes. ASSUMPTION 0.8.
cfg.shading.strength = 0.8;
% Size (px) of the "local average": structures smaller than ~2x this count
% as shading/folds. ASSUMPTION: 15 px at 640x480 (torso ~150-250 px wide).
cfg.shading.baseSigma = 15;
% Light pre-blur to remove camera noise and very fine fabric texture.
cfg.shading.fineSigma = 1.2;
% Clamp for the shading factor (avoid black holes / blown-out spots).
cfg.shading.minShade = 0.5;
cfg.shading.maxShade = 1.4;

%% ------------------------------------------------------------------------
%  Lighting matching
%  ------------------------------------------------------------------------
cfg.lighting.enable = true;

% How far to move the garment's brightness/contrast toward the scene
% (0 = untouched, 1 = match the torso region's statistics exactly).
% ASSUMPTION: 0.6. The torso region we measure also contains the colour of
% the shirt the user is really wearing (a black T-shirt looks "dark" even
% in good light), so matching 100% would wrongly darken a white garment on
% someone wearing black. A partial match follows the lighting trend without
% destroying the garment's own colour.
cfg.lighting.strength = 0.6;

% Clamp for the contrast (std-dev) ratio so a nearly flat torso region or a
% nearly flat garment cannot produce an extreme contrast stretch.
cfg.lighting.minGain = 0.7;
cfg.lighting.maxGain = 1.4;

% Garment pixels with alpha below this are ignored when measuring the
% garment's own brightness (otherwise the transparent background counts).
cfg.lighting.alphaThreshold = 0.5;

% The torso region is the bounding box of the 4 torso keypoints, shrunk
% toward its centre by this fraction on every side so that it contains
% mostly body and little background.
cfg.lighting.roiShrink = 0.15;

%% ------------------------------------------------------------------------
%  Alpha blending + boundary smoothing (the "from scratch" part)
%  ------------------------------------------------------------------------
% Alpha values >= this are considered "inside" the garment when building
% the binary mask that is then eroded and feathered.
cfg.blend.alphaThreshold = 0.5;

% Radius (px) of the disk used by imerode: how far the hard garment edge is
% pulled inward before feathering. ASSUMPTION: 3 px at 640x480. Should be
% about 1-2x gaussSigma so the feathered edge stays inside the original
% garment silhouette (no dark halo from the PNG's transparent border).
cfg.blend.erodeRadius = 3;

% Standard deviation (px) of the Gaussian used to feather the mask.
% ASSUMPTION: 2 px at 640x480 -> soft transition ~2*3*sigma = 12 px wide,
% visible as "soft" but still crisp. 0 disables feathering.
cfg.blend.gaussSigma = 2;

% true  = composite with explicit per-pixel for-loops (slow, but reads
%         exactly like the formula; nice for the live explanation)
% false = same maths, vectorised per colour channel (fast, use for live video)
cfg.blend.useLoops = false;
end
