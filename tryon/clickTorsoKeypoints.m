function kp = clickTorsoKeypoints(frame)
%CLICKTORSOKEYPOINTS  Manual stand-in for the pose model: click 6 points.
%
%   kp = clickTorsoKeypoints(frame)
%
%   Asks for both shoulders, both hips and both elbows. Within each pair
%   the order does not matter: the points are sorted afterwards.
%     - shoulders and hips: the one further LEFT in the image is the
%       person's RIGHT side (person facing the camera; also true in a
%       mirrored selfie, see getTorsoKeypoints)
%     - elbows: each elbow goes with the nearer shoulder, so an arm bent
%       across the body is still assigned correctly
%   Returns a 17x3 COCO-17 array [x y score] (unclicked rows have score 0),
%   or [] if the window is closed before all points are clicked.
%
%   See also PoseTracker, getTorsoKeypoints.

prompts = {'1/6  click ONE SHOULDER (where the shoulder meets the arm)', ...
           '2/6  click the OTHER SHOULDER', ...
           '3/6  click ONE HIP (at the waistband, side of the body)', ...
           '4/6  click the OTHER HIP', ...
           '5/6  click ONE ELBOW', ...
           '6/6  click the OTHER ELBOW'};
colours = {'c', 'c', 'm', 'm', 'y', 'y'};

f = figure('Name', 'Click the body points', 'NumberTitle', 'off');
imshow(frame);
hold on;
P = zeros(6, 2);
for i = 1:6
    title(prompts{i}, 'FontSize', 13);
    try
        [x, y] = ginput(1);
    catch
        kp = [];                   % window closed
        return
    end
    if isempty(x)
        kp = [];
        if ishandle(f), close(f); end
        return
    end
    P(i, :) = [x y];
    plot(x, y, '+', 'Color', colours{i}, 'MarkerSize', 16, 'LineWidth', 2.5);
end
close(f);

% Shoulders / hips: image-left = person's right.
sh = sortrows(P(1:2, :), 1);           % sh(1,:) = image left
hp = sortrows(P(3:4, :), 1);
% Elbows: pair each with the nearer shoulder (try both assignments).
el = P(5:6, :);
if norm(el(1, :) - sh(1, :)) + norm(el(2, :) - sh(2, :)) > ...
   norm(el(2, :) - sh(1, :)) + norm(el(1, :) - sh(2, :))
    el = el([2 1], :);
end

% COCO-17 rows: 6/7 = left/right shoulder, 8/9 = left/right elbow,
% 12/13 = left/right hip ("left" = the person's left = image right).
kp = zeros(17, 3);
kp(7, :)  = [sh(1, :) 1];   kp(6, :)  = [sh(2, :) 1];
kp(13, :) = [hp(1, :) 1];   kp(12, :) = [hp(2, :) 1];
kp(9, :)  = [el(1, :) 1];   kp(8, :)  = [el(2, :) 1];
end
