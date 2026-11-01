function annotateTorso(sessionDir, redo)
%ANNOTATETORSO  Hand-mark the torso area in every test frame (ground truth).
%
%   annotateTorso('data/test_frames/habiba_bright')
%   annotateTorso(sessionDir, true)      % redo frames already annotated
%
%   For each frame_XXXX.png you draw a polygon around the TORSO AREA THE
%   SHIRT SHOULD COVER:
%       start at one shoulder seam (where the shoulder meets the arm), go
%       along the armpit and the side of the body down to the hip/waistband,
%       across, up the other side and back over the other shoulder and the
%       neckline. Do NOT include the arms, sleeves or the neck/head.
%   Click the corners, double-click to finish.
%
%   Press Esc while drawing (empty polygon) if NO person/torso is visible:
%   the frame is then a "no person" frame, used to check for false
%   detections. Close the window to stop; your progress is kept.
%
%   OUTPUT (in sessionDir/gt/):
%     frame_XXXX_torso.png   logical mask of the torso area
%     annotations.mat        table: frame, hasPerson, polygon
%
%   Measured later by runExperiments as the Intersection over Union (IoU)
%   between this mask and the warped garment torso.
%
%   See also captureTestFrames, runExperiments.

if nargin < 2, redo = false; end
gtDir = fullfile(sessionDir, 'gt');
if ~exist(gtDir, 'dir'), mkdir(gtDir); end
annFile = fullfile(gtDir, 'annotations.mat');
if exist(annFile, 'file')
    s = load(annFile);
    ann = s.ann;
else
    ann = table('Size', [0 3], 'VariableTypes', {'string', 'logical', 'cell'}, ...
                'VariableNames', {'frame', 'hasPerson', 'polygon'});
end

files = dir(fullfile(sessionDir, 'frame_*.png'));
fig = figure('Name', ['Annotate ' sessionDir], 'NumberTitle', 'off');
for i = 1:numel(files)
    name = string(files(i).name);
    if ~redo && any(ann.frame == name)
        continue
    end
    if ~ishandle(fig), break; end
    frame = imread(fullfile(sessionDir, files(i).name));
    clf(fig);
    ax = axes(fig);
    imshow(frame, 'Parent', ax);
    title(ax, {sprintf('%s  (%d / %d)', name, i, numel(files)), ...
               'Outline the torso area (no arms/sleeves). Double-click = done, Esc = no person.'});
    try
        roi = drawpolygon(ax, 'Color', 'g');
    catch
        break                                   % window closed
    end
    if ~ishandle(fig), break; end
    pos = roi.Position;
    hasPerson = size(pos, 1) >= 3;
    if hasPerson
        mask = createMask(roi);
    else
        mask = false(size(frame, 1), size(frame, 2));
        pos = zeros(0, 2);
    end
    [~, base] = fileparts(files(i).name);
    imwrite(mask, fullfile(gtDir, [base '_torso.png']));

    ann(ann.frame == name, :) = [];
    ann = [ann; {name, hasPerson, {pos}}]; %#ok<AGROW>
    save(annFile, 'ann');
end
if ishandle(fig), close(fig); end
fprintf('%d frames annotated in %s\n', height(ann), gtDir);
end
