classdef TryOnApp < handle
%TRYONAPP  Graphical user interface for the virtual try-on (photo based).
%
%   TryOnApp                % open the app
%   app = TryOnApp;         % keep a handle (e.g. app.Session.Cfg)
%
%   HOW TO USE
%     1. YOUR PHOTO   "Upload photo..." (a full-body or waist-up photo,
%                     facing the camera), or "Take photo" to capture one
%                     snapshot with the webcam after a countdown. With "Also
%                     capture the empty background" ticked, the webcam first
%                     takes a picture without you (step out of view), which
%                     enables the body-shape fit.
%     2. GARMENT      pick one, or "Upload garment picture..." (a flat,
%                     front-facing product photo on a plain background, or
%                     a transparent PNG). The background is removed and the
%                     anchor points are found automatically.
%     3. BODY POINTS  found automatically by HRNet if the add-on is
%                     installed; otherwise a window opens and you click
%                     shoulders, hips and elbows (6 clicks).
%     Then adjust the warp model, the stages and the sliders, and press
%     "Save result". "Live video" (optional) runs the same processing on
%     the webcam stream.
%
%   LAYOUT
%     your photo with the detected body points | the try-on result
%     person mask / garment layer | status (confidence, body-point source,
%     timings) | controls on the right
%
%   The GUI is written programmatically (uifigure) rather than as a binary
%   .mlapp file so the code can be read, diffed and modified like any other
%   .m file. All image processing is in TryOnSession.
%
%   See also TryOnSession, PoseTracker, prepareGarment, liveTryOn.

    properties
        Session              % TryOnSession (Cfg, garments, tracker)
        AutoClick = true     % open the click window when no body points are found
    end

    properties (SetAccess = private)
        Fig
        Running = false      % live video loop active
        Busy = false         % a webcam countdown is running
        Camera = []          % webcam object
        Photo = []           % the person photo (raw)
        LastFrame = []       % last preprocessed frame
        LastOut = []         % last result
        LastInfo = []
        BgCountdown = []     % tic of a running background countdown (live)
        Fps = 0
        GarmentDir
        % --- components ---
        ImPhoto, ImResult, ImMask, MaskTitle
        UploadBtn, TakeBtn, CamDrop, MirrorBox, BgBox, BgBtn
        GarmentDrop, GarmentImg, AddGarmentBtn, AnchorsBtn
        PoseLabel, ClickBtn, DetectBtn
        WarpDrop
        PreBox, FitBox, SleeveBox, LightBox, ShadeBox, LandmarkBox, MaskBox
        SigmaSlider, ErodeSlider, LightSlider, ShadeSlider
        SaveBtn, LiveBtn, StudioBtn
        FpsLabel, ConfLabel, SourceLabel, StatusLabel, Lamp
    end

    properties (Constant, Access = private)
        WarpItems = {'Homography (DLT, from scratch)', 'Piecewise affine', 'Affine'}
        WarpModes = {'homography', 'piecewise', 'affine'}
        Green = [0.30 0.62 0.38]
        Red   = [0.85 0.33 0.31]
    end

    methods
        function app = TryOnApp(visible)
            if nargin < 1, visible = true; end
            root = fileparts(mfilename('fullpath'));
            addpath(root);
            app.GarmentDir = fullfile(root, 'data', 'garments');
            lib = loadGarmentLibrary(app.GarmentDir);
            if isempty(lib)
                makeSampleGarments(app.GarmentDir);
                lib = loadGarmentLibrary(app.GarmentDir);
            end
            app.AutoClick = visible;          % no click windows in headless tests
            app.Session = TryOnSession(lib, PoseTracker('manual'), defaultTryOnConfig());
            app.buildUi(visible);
            app.setPoseMode('auto');
            app.refreshGarments(1);
            app.syncControlsFromCfg();
            app.StatusLabel.Text = 'Step 1: upload a photo of yourself or take one with the webcam.';
        end

        function delete(app)
            app.Running = false;
            app.releaseCamera();
            if ~isempty(app.Fig) && isvalid(app.Fig)
                delete(app.Fig);
            end
        end

        function setPhoto(app, img, background)
            %SETPHOTO  Use this person photo (array or file name).
            %   app.setPhoto(img)               without a background
            %   app.setPhoto(img, background)   with an empty-scene picture
            if ischar(img) || isstring(img), img = readPhotoUpright(img); end
            if size(img, 3) == 1, img = repmat(img, [1 1 3]); end
            app.stopLive();
            app.Photo = im2uint8(img);
            app.Session.Tracker.reset();
            if ~app.Session.Tracker.hasDetector()
                app.setPoseMode('auto');      % try HRNet again for a new photo
            end
            app.Session.clearBackground();
            if nargin > 2 && ~isempty(background)
                app.Session.setBackground(background);
            end
            app.findBodyPoints();
        end

        function setManualKeypoints(app, kp)
            %SETMANUALKEYPOINTS  Programmatic version of "Click body points".
            if ~strcmp(app.Session.Tracker.Mode, 'manual')
                app.setPoseMode('manual');
            end
            app.Session.Tracker.setManualKeypoints(kp, app.currentFrame());
            app.safeStep();
        end

        function [out, info] = stepOnce(app)
            %STEPONCE  Process and show the current photo / live frame.
            raw = app.grabFrame();
            [out, info] = app.Session.processFrame(raw);
            app.LastFrame = info.frame;
            app.LastOut = out;
            app.LastInfo = info;
            app.show(out, info);
        end

        function startLive(app)
            %STARTLIVE  Optional: run the same processing on the webcam stream.
            if app.Running || app.Busy, return; end
            if ~app.openCamera(), return; end
            app.Running = true;
            app.Session.Tracker.reset();
            app.LiveBtn.Text = 'Stop live video';
            app.LiveBtn.BackgroundColor = app.Red;
            app.StatusLabel.Text = 'Live video. Stand back so your hips are visible.';
            if app.Session.Tracker.needsManualKeypoints() && app.AutoClick
                app.clickBodyPoints();
            end
            while app.Running && isvalid(app.Fig)
                tFrame = tic;
                try
                    if ~isempty(app.BgCountdown)
                        app.runBackgroundCountdown();
                    else
                        app.stepOnce();
                    end
                catch err
                    if ~isvalid(app.Fig), break; end
                    app.StatusLabel.Text = ['Error: ' err.message];
                end
                app.Fps = 0.8 * app.Fps + 0.2 / max(toc(tFrame), eps);
                if isvalid(app.Fig)
                    app.FpsLabel.Text = sprintf('%.1f fps', app.Fps);
                end
                drawnow limitrate;
            end
            app.Running = false;
            app.releaseCamera();
            if isvalid(app.Fig)
                app.LiveBtn.Text = 'Live video (optional)';
                app.LiveBtn.BackgroundColor = [0.94 0.94 0.94];
                app.FpsLabel.Text = '-';
                if ~isempty(app.Photo)
                    app.Session.Tracker.reset();
                    app.findBodyPoints();     % back to the photo
                end
            end
        end

        function stopLive(app)
            app.Running = false;
        end
    end

    % ======================================================================
    %  UI construction
    % ======================================================================
    methods (Access = private)
        function buildUi(app, visible)
            % Window size: 1320 x 860, or smaller to fit the screen.
            screen = get(0, 'ScreenSize');
            w = min(1320, screen(3) - 40);
            h = min(860, screen(4) - 200);     % room for menu bar, title bar and Dock
            app.Fig = uifigure('Name', 'Virtual Try-On', 'Position', [20 40 w h], ...
                'Visible', matlab.lang.OnOffSwitchState(visible), ...
                'CloseRequestFcn', @(~, ~) app.onClose());
            main = uigridlayout(app.Fig, [1 2], 'ColumnWidth', {'1x', 340}, 'Padding', 8);

            % ---- left: views + status -------------------------------------
            leftPanel = uipanel(main, 'BorderType', 'none');
            left = uigridlayout(leftPanel, [2 1], 'RowHeight', {'1x', 220}, 'Padding', 0);
            top = uigridlayout(left, [1 2], 'Padding', 0);
            app.ImPhoto  = makeView(top, 'Your photo + body points');
            app.ImResult = makeView(top, 'Virtual try-on');
            bottom = uigridlayout(left, [1 2], 'ColumnWidth', {'1x', 300}, 'Padding', 0);
            [app.ImMask, app.MaskTitle] = makeView(bottom, 'Garment layer');

            status = uipanel(bottom, 'Title', 'Status');
            sg = uigridlayout(status, [5 2], 'ColumnWidth', {90, '1x'}, ...
                'RowHeight', {22, 22, 22, 22, '1x'}, 'Padding', 6, 'RowSpacing', 4);
            place(uilabel(sg, 'Text', 'State'), 1, 1);
            app.Lamp = place(uilamp(sg, 'Color', [0.6 0.6 0.6]), 1, 2);
            place(uilabel(sg, 'Text', 'Confidence'), 2, 1);
            app.ConfLabel = place(uilabel(sg, 'Text', '-', 'FontWeight', 'bold'), 2, 2);
            place(uilabel(sg, 'Text', 'Body points'), 3, 1);
            app.SourceLabel = place(uilabel(sg, 'Text', '-'), 3, 2);
            place(uilabel(sg, 'Text', 'Live video'), 4, 1);
            app.FpsLabel = place(uilabel(sg, 'Text', '-'), 4, 2);
            app.StatusLabel = place(uilabel(sg, 'Text', '', 'WordWrap', 'on', ...
                'VerticalAlignment', 'top'), 5, [1 2]);

            % ---- right: controls ------------------------------------------
            % Every control is placed explicitly at (row, column); h collects
            % the row heights.
            panel = uipanel(main, 'Scrollable', 'on', 'BorderType', 'none');
            g = uigridlayout(panel, [1 2], 'ColumnWidth', {130, '1x'}, ...
                'RowSpacing', 4, 'Padding', [6 4 6 4]);
            h = {};

            [r, h] = newRow(h, 20);  header(g, r, '1  Your photo');
            [r, h] = newRow(h, 30);
            app.UploadBtn = place(uibutton(g, 'Text', 'Upload photo...', 'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~, ~) app.onUploadPhoto()), r, 1);
            app.TakeBtn = place(uibutton(g, 'Text', 'Take photo (webcam)', 'FontWeight', 'bold', ...
                'ButtonPushedFcn', @(~, ~) app.onTakePhoto()), r, 2);
            [r, h] = newRow(h);
            place(uilabel(g, 'Text', 'Camera'), r, 1);
            cams = listCameras();
            if isempty(cams), cams = {'(no webcam found)'}; end
            app.CamDrop = place(uidropdown(g, 'Items', cams), r, 2);
            [r, h] = newRow(h);
            app.MirrorBox = place(uicheckbox(g, 'Text', 'Mirror (selfie)', 'Value', true), r, 1);
            app.BgBox = place(uicheckbox(g, 'Text', 'Also capture empty background', ...
                'Value', true), r, 2);
            [r, h] = newRow(h);
            app.BgBtn = place(uibutton(g, 'Text', 'Background photo (optional)...', ...
                'Tooltip', ['A photo of the same scene WITHOUT you, taken from the same spot. ' ...
                            'Enables the body-shape fit.'], ...
                'ButtonPushedFcn', @(~, ~) app.onBackground()), r, [1 2]);

            [r, h] = newRow(h, 20);  header(g, r, '2  Garment');
            [r, h] = newRow(h);
            place(uilabel(g, 'Text', 'Garment'), r, 1);
            app.GarmentDrop = place(uidropdown(g, 'Items', {''}, ...
                'ValueChangedFcn', @(src, ~) app.onGarmentChanged(src.Value)), r, 2);
            [r, h] = newRow(h, 34);
            r2 = r;
            app.AddGarmentBtn = place(uibutton(g, 'Text', 'Upload garment picture...', ...
                'FontWeight', 'bold', 'ButtonPushedFcn', @(~, ~) app.onAddGarment()), r, 2);
            [r, h] = newRow(h, 34);
            app.AnchorsBtn = place(uibutton(g, 'Text', 'Fix garment points...', ...
                'Tooltip', 'Click the 8 anchor points on the garment if it sits badly', ...
                'ButtonPushedFcn', @(~, ~) app.onEditAnchors()), r, 2);
            app.GarmentImg = place(uiimage(g, 'ScaleMethod', 'fit'), [r2 r], 1);

            [r, h] = newRow(h, 20);  header(g, r, '3  Body points');
            [r, h] = newRow(h);
            app.PoseLabel = place(uilabel(g, 'Text', '', 'FontAngle', 'italic', 'WordWrap', 'on'), r, [1 2]);
            [r, h] = newRow(h);
            app.DetectBtn = place(uibutton(g, 'Text', 'Detect automatically', ...
                'ButtonPushedFcn', @(~, ~) app.onDetect()), r, 1);
            app.ClickBtn = place(uibutton(g, 'Text', 'Click body points...', ...
                'ButtonPushedFcn', @(~, ~) app.clickBodyPoints()), r, 2);

            [r, h] = newRow(h, 20);  header(g, r, 'Adjust');
            [r, h] = newRow(h);
            place(uilabel(g, 'Text', 'Warp model'), r, 1);
            app.WarpDrop = place(uidropdown(g, 'Items', app.WarpItems, ...
                'ValueChangedFcn', @(src, ~) app.setCfg('warp', 'mode', ...
                    app.WarpModes{strcmp(app.WarpItems, src.Value)})), r, 2);
            [r, h] = newRow(h);
            app.PreBox = place(uicheckbox(g, 'Text', 'Preprocessing', ...
                'ValueChangedFcn', @(src, ~) app.setCfg('preprocess', 'enable', src.Value)), r, 1);
            app.FitBox = place(uicheckbox(g, 'Text', 'Body-shape fit', ...
                'ValueChangedFcn', @(src, ~) app.setCfg('fit', 'enable', src.Value)), r, 2);
            [r, h] = newRow(h);
            app.LightBox = place(uicheckbox(g, 'Text', 'Lighting match', ...
                'ValueChangedFcn', @(src, ~) app.setCfg('lighting', 'enable', src.Value)), r, 1);
            app.SleeveBox = place(uicheckbox(g, 'Text', 'Sleeves follow arms', ...
                'ValueChangedFcn', @(src, ~) app.setCfg('sleeves', 'enable', src.Value)), r, 2);
            [r, h] = newRow(h);
            app.ShadeBox = place(uicheckbox(g, 'Text', 'Shading (folds)', ...
                'ValueChangedFcn', @(src, ~) app.setCfg('shading', 'enable', src.Value)), r, 1);
            app.LandmarkBox = place(uicheckbox(g, 'Text', 'Points on result', 'Value', false, ...
                'ValueChangedFcn', @(~, ~) app.refresh()), r, 2);
            [r, h] = newRow(h);
            app.MaskBox = place(uicheckbox(g, 'Text', 'Tint person mask', 'Value', false, ...
                'ValueChangedFcn', @(~, ~) app.refresh()), r, [1 2]);
            [app.SigmaSlider, h] = addSlider(g, h, 'Edge softness', [0 6], ...
                @(v) app.setCfg('blend', 'gaussSigma', v));
            [app.ErodeSlider, h] = addSlider(g, h, 'Edge shrink (px)', [0 8], ...
                @(v) app.setCfg('blend', 'erodeRadius', round(v)));
            [app.LightSlider, h] = addSlider(g, h, 'Lighting strength', [0 1], ...
                @(v) app.setCfg('lighting', 'strength', v));
            [app.ShadeSlider, h] = addSlider(g, h, 'Shading strength', [0 1], ...
                @(v) app.setCfg('shading', 'strength', v));

            [r, h] = newRow(h, 8);
            [r, h] = newRow(h, 32);
            app.SaveBtn = place(uibutton(g, 'Text', 'Save result', 'FontWeight', 'bold', ...
                'BackgroundColor', app.Green, 'FontColor', 'w', ...
                'ButtonPushedFcn', @(~, ~) app.onSave()), r, 1);
            app.LiveBtn = place(uibutton(g, 'Text', 'Live video (optional)', ...
                'ButtonPushedFcn', @(~, ~) app.onLive()), r, 2);
            [r, h] = newRow(h, 32);
            app.StudioBtn = place(uibutton(g, 'Text', 'Studio render (photorealistic, local AI)', ...
                'Tooltip', 'Needs the local studio server: cd webapp; .venv/bin/uvicorn server:app', ...
                'ButtonPushedFcn', @(~, ~) app.onStudio()), r, [1 2]);
            g.RowHeight = h;
        end

        function syncControlsFromCfg(app)
            c = app.Session.Cfg;
            app.WarpDrop.Value = app.WarpItems{strcmp(app.WarpModes, c.warp.mode)};
            app.PreBox.Value = c.preprocess.enable;
            app.FitBox.Value = c.fit.enable;
            app.SleeveBox.Value = c.sleeves.enable;
            app.LightBox.Value = c.lighting.enable;
            app.ShadeBox.Value = c.shading.enable;
            app.SigmaSlider.Value = c.blend.gaussSigma;
            app.ErodeSlider.Value = c.blend.erodeRadius;
            app.LightSlider.Value = c.lighting.strength;
            app.ShadeSlider.Value = c.shading.strength;
        end

        function setCfg(app, section, field, value)
            app.Session.Cfg.(section).(field) = value;
            app.refresh();
        end

        function refresh(app)
            % Re-process the photo so a changed setting shows right away
            % (in live mode the next frame picks it up anyway).
            if ~app.Running && ~isempty(app.Photo)
                app.safeStep();
            end
        end

        % ==================================================================
        %  Display
        % ==================================================================
        function show(app, out, info)
            if ~isvalid(app.Fig), return; end
            frame = info.frame;
            if app.MaskBox.Value && ~isempty(info.bodyMask)
                out(:, :, 2) = max(out(:, :, 2), uint8(140 * info.bodyMask));
            end
            % Body points: torso quad (cyan), upper arms (yellow), keypoints
            % (red), drawn into the pictures with insertShape.
            t = info.torso;
            pts = zeros(0, 2);
            if ~isempty(info.kp) && size(info.kp, 1) == 17
                ok = info.kp(:, 3) >= app.Session.Cfg.pose.minScore & all(isfinite(info.kp(:, 1:2)), 2);
                pts = info.kp(ok, 1:2);
            end
            quad = [];  arms = zeros(0, 4);
            if ~isempty(t) && t.valid
                quad = reshape([t.leftShoulder; t.rightShoulder; t.rightHip; t.leftHip]', 1, []);
                if t.hasLeftElbow,  arms(end + 1, :) = [t.leftShoulder t.leftElbow];   end
                if t.hasRightElbow, arms(end + 1, :) = [t.rightShoulder t.rightElbow]; end
            end
            lw = max(2, round(size(frame, 2) / 250));        % visible on large photos too
            annotated = frame;
            if ~isempty(arms)
                annotated = insertShape(annotated, 'Line', arms, 'Color', [255 200 50], 'LineWidth', lw);
            end
            if ~isempty(quad)
                annotated = insertShape(annotated, 'Polygon', quad, 'Color', [0 255 255], 'LineWidth', lw);
            end
            if ~isempty(pts)
                annotated = insertShape(annotated, 'FilledCircle', [pts, 2 * lw * ones(size(pts, 1), 1)], ...
                    'Color', [255 70 70], 'Opacity', 1);
            end
            if app.LandmarkBox.Value && ~isempty(quad)
                out = insertShape(out, 'Polygon', quad, 'Color', [0 255 255], 'LineWidth', lw);
            end
            app.ImResult.ImageSource = out;
            app.ImPhoto.ImageSource = annotated;

            % Mask / garment layer view
            if ~isempty(info.bodyMask)
                app.ImMask.ImageSource = im2uint8(repmat(info.bodyMask, [1 1 3]));
                app.MaskTitle.Text = 'Person mask (background subtraction)';
            elseif ~isempty(info.dbg) && isfield(info.dbg, 'layerAlpha')
                app.ImMask.ImageSource = im2uint8(repmat(info.dbg.layerAlpha, [1 1 3]));
                app.MaskTitle.Text = 'Garment layer (add a background photo for the person mask)';
            end

            % Status
            app.ConfLabel.Text = sprintf('%.0f %%', 100 * info.confidence);
            switch info.source
                case 'detector', src = 'HRNet (automatic)';
                case 'tracker',  src = ternary(app.Running, 'KLT tracking', 'HRNet (automatic)');
                case 'manual',   src = 'clicked';
                otherwise,       src = '-';
            end
            if strcmp(app.Session.Tracker.Mode, 'manual') && ~isempty(info.kp)
                src = 'clicked';
            end
            app.SourceLabel.Text = src;
            if info.ok
                app.Lamp.Color = [0.2 0.75 0.3];
            elseif isempty(info.kp)
                app.Lamp.Color = [0.85 0.3 0.25];
            else
                app.Lamp.Color = [0.95 0.75 0.2];
            end
            tm = info.timing;
            if info.ok
                msg = sprintf('Done. Garment: %s', strrep(app.Session.currentGarment().name, '_', ' '));
            else
                msg = friendlyStatus(info.status);
            end
            app.StatusLabel.Text = sprintf('%s\nprocessing: %.0f ms (pre %.0f, pose %.0f, mask %.0f, try-on %.0f)', ...
                msg, 1000 * tm.total, 1000 * tm.preprocess, 1000 * tm.pose, ...
                1000 * tm.segment, 1000 * tm.pipeline);
        end

        % ==================================================================
        %  Callbacks
        % ==================================================================
        function onUploadPhoto(app)
            if app.Busy, return; end
            [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif', 'Photos (JPG, PNG, BMP, TIF)'}, ...
                'Photo of yourself, facing the camera, shoulders to hips visible');
            figure(app.Fig);
            if isequal(f, 0), return; end
            try
                img = readPhotoUpright(fullfile(p, f));
            catch err
                uialert(app.Fig, err.message, 'Could not read the photo');
                return
            end
            app.setPhoto(img);
        end

        function onTakePhoto(app)
            %ONTAKEPHOTO  Countdown, then ONE webcam snapshot (+ optional background).
            if app.Busy, return; end
            app.stopLive();
            if ~app.openCamera(), return; end
            app.Busy = true;
            cleanup = onCleanup(@() app.endBusy());
            background = [];
            if app.BgBox.Value
                app.countdown(5, 'STEP OUT of the picture - empty background in %d s');
                shots = cell(1, 5);
                for i = 1:5, shots{i} = app.grabCamera(); end
                background = median(cat(4, shots{:}), 4);
            end
            app.countdown(5, ['STEP IN, face the camera, arms slightly away from the body. ' ...
                              'Photo in %d s']);
            photo = app.grabCamera();
            app.releaseCamera();
            app.setPhoto(photo, background);
        end

        function onBackground(app)
            if isempty(app.Photo)
                uialert(app.Fig, 'Upload or take your photo first.', 'No photo');
                return
            end
            if app.Running
                app.BgCountdown = tic;            % live: the loop shows a countdown
                return
            end
            [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif', 'Photos'}, ...
                'The SAME scene WITHOUT you, taken from the same spot');
            figure(app.Fig);
            if isequal(f, 0), return; end
            try
                bg = readPhotoUpright(fullfile(p, f));
            catch err
                uialert(app.Fig, err.message, 'Could not read the photo');
                return
            end
            if ~isequal(size(bg, 1, 2), size(app.Photo, 1, 2))
                uialert(app.Fig, sprintf(['The background photo must have the same size as your ' ...
                    'photo (%d x %d), so take both from the same spot with the same camera.'], ...
                    size(app.Photo, 2), size(app.Photo, 1)), 'Size mismatch');
                return
            end
            app.Session.setBackground(bg);
            app.safeStep();
        end

        function onDetect(app)
            if isempty(app.Photo) && ~app.Running
                uialert(app.Fig, 'Upload or take your photo first.', 'No photo');
                return
            end
            app.setPoseMode('auto');
            if ~app.Session.Tracker.hasDetector()
                uialert(app.Fig, ['Automatic detection needs the "Computer Vision Toolbox Model ' ...
                    'for Object Keypoint Detection" add-on (Home > Add-Ons). Until then, use ' ...
                    '"Click body points".'], 'HRNet not installed');
                return
            end
            app.findBodyPoints();
        end

        function clickBodyPoints(app)
            frame = app.currentFrame();
            if isempty(frame)
                uialert(app.Fig, 'Upload or take your photo first.', 'No photo');
                return
            end
            kp = clickTorsoKeypoints(frame);
            if isvalid(app.Fig), figure(app.Fig); end
            if isempty(kp), return; end
            if ~strcmp(app.Session.Tracker.Mode, 'manual')
                app.setPoseMode('manual');
            end
            app.Session.Tracker.setManualKeypoints(kp, frame);
            app.safeStep();
        end

        function onGarmentChanged(app, value)
            idx = find(strcmp(app.GarmentDrop.Items, value), 1);
            if isempty(idx), return; end
            app.Session.GarmentIndex = idx;
            app.GarmentImg.ImageSource = app.Session.Garments(idx).thumb;
            app.refresh();
        end

        function onAddGarment(app)
            [f, p] = uigetfile({'*.png;*.jpg;*.jpeg', 'Garment pictures (PNG, JPG)'}, ...
                'Garment picture: flat, front-facing, plain background (or a transparent PNG)');
            figure(app.Fig);
            if isequal(f, 0), return; end
            d = uiprogressdlg(app.Fig, 'Title', 'Preparing the garment', ...
                'Message', 'Removing the background and finding the garment points...', ...
                'Indeterminate', 'on');
            try
                out = prepareGarment(fullfile(p, f), app.GarmentDir, struct('overwrite', true));
                [~, name] = fileparts(out{1});
                close(d);
                app.refreshGarments(name);
            catch err
                close(d);
                uialert(app.Fig, err.message, 'Could not prepare the garment');
            end
        end

        function onEditAnchors(app)
            g = app.Session.currentGarment();
            app.stopLive();
            try
                pickGarmentAnchors(g.file);
            catch err
                uialert(app.Fig, err.message, 'Garment points not saved');
            end
            figure(app.Fig);
            app.refreshGarments(g.name);
        end

        function onStudio(app)
            % Photorealistic render of the current photo + garment by the
            % local hybrid pipeline (see studioRender).
            if app.Running
                uialert(app.Fig, 'Stop the live video first, or take a photo.', 'Live video running');
                return
            end
            if isempty(app.Photo)
                uialert(app.Fig, 'Upload or take your photo first.', 'No photo');
                return
            end
            dlg = uiprogressdlg(app.Fig, 'Title', 'Studio render', 'Indeterminate', 'on', ...
                'Message', 'Rendering locally on the Mac''s GPU (about 30 s; the first render also loads the model)...');
            cleanup = onCleanup(@() close(dlg));
            try
                [out, info] = studioRender(app.Photo, app.Session.currentGarment());
            catch err
                uialert(app.Fig, err.message, 'Studio render failed');
                return
            end
            app.ImResult.ImageSource = out;
            app.LastOut = out;
            app.LastFrame = imresize(im2uint8(app.Photo), [size(out, 1) size(out, 2)]);
            if isempty(app.LastInfo), app.LastInfo = struct('ok', true); else, app.LastInfo.ok = true; end
            app.StatusLabel.Text = sprintf('Studio render done in %.0f s (local CatVTON + our geometry).', info.seconds);
        end

        function onSave(app)
            if isempty(app.LastOut) || isempty(app.LastInfo) || ~app.LastInfo.ok
                uialert(app.Fig, ['There is no try-on result to save yet: the garment is not ' ...
                    'drawn. See the message in the Status box.'], 'No result');
                return
            end
            root = fileparts(mfilename('fullpath'));
            folder = fullfile(root, 'results', 'tryon');
            if ~exist(folder, 'dir'), mkdir(folder); end
            stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
            file = fullfile(folder, ['tryon_' stamp '.png']);
            imwrite(app.LastOut, file);
            imwrite([app.LastFrame, app.LastOut], fullfile(folder, ['tryon_' stamp '_before_after.png']));
            app.StatusLabel.Text = ['Saved ' file];
        end

        function onLive(app)
            if app.Running
                app.stopLive();
            else
                app.startLive();
            end
        end

        function onClose(app)
            app.Running = false;
            app.releaseCamera();
            delete(app.Fig);
        end

        % ==================================================================
        %  Helpers
        % ==================================================================
        function findBodyPoints(app)
            % Process the photo; if no torso is found, ask for clicks.
            app.safeStep();
            if isempty(app.LastInfo) || app.LastInfo.ok
                return
            end
            if app.Session.Tracker.hasDetector()
                app.StatusLabel.Text = ['HRNet could not find your shoulders and hips. ' ...
                    'Click them yourself ("Click body points").'];
            else
                app.StatusLabel.Text = 'Click your shoulders, hips and elbows in the window that opened.';
            end
            if app.AutoClick
                drawnow;
                app.clickBodyPoints();
            end
        end

        function setPoseMode(app, mode)
            if strcmp(mode, 'auto') && strcmp(app.Session.Tracker.Mode, 'hrnet')
                return                                % already loaded
            end
            d = [];
            if strcmp(mode, 'auto')
                try, d = uiprogressdlg(app.Fig, 'Title', 'Loading pose model', 'Indeterminate', 'on'); catch, end
            end
            tracker = PoseTracker(mode);
            if ~isempty(d), close(d); end
            app.Session.Tracker = tracker;
            if tracker.hasDetector()
                app.PoseLabel.Text = 'Found automatically with HRNet. Wrong? Click them yourself.';
            elseif strcmp(mode, 'auto')
                app.PoseLabel.Text = 'HRNet add-on not installed: you click 6 body points.';
            else
                app.PoseLabel.Text = 'Body points clicked by hand.';
            end
        end

        function refreshGarments(app, select)
            lib = loadGarmentLibrary(app.GarmentDir);
            if isempty(lib), return; end
            app.Session.Garments = lib;
            names = {lib.name};
            app.GarmentDrop.Items = names;
            if ischar(select) || isstring(select)
                idx = find(strcmp(names, select), 1);
            else
                idx = select;
            end
            if isempty(idx), idx = 1; end
            app.GarmentDrop.Value = names{idx};
            app.onGarmentChanged(names{idx});
        end

        function ok = openCamera(app)
            ok = ~isempty(app.Camera);
            if ok, return; end
            try
                if strcmp(app.CamDrop.Value, '(no webcam found)')
                    error('No webcam was found.');
                end
                app.Camera = webcam(app.CamDrop.Value);
                ok = true;
            catch err
                uialert(app.Fig, sprintf(['Could not open the webcam:\n%s\n\nInstall the ' ...
                    '"MATLAB Support Package for USB Webcams" (Home > Add-Ons), restart the app ' ...
                    'and close other apps using the camera. You can also upload a photo instead.'], ...
                    err.message), 'Webcam');
            end
        end

        function releaseCamera(app)
            app.Camera = [];                      % deleting the object frees the camera
        end

        function frame = grabCamera(app)
            frame = snapshot(app.Camera);
            if app.MirrorBox.Value
                frame = fliplr(frame);
            end
        end

        function raw = grabFrame(app)
            if app.Running
                raw = app.grabCamera();
            elseif ~isempty(app.Photo)
                raw = app.Photo;
            else
                error('TryOnApp:noPhoto', 'Upload or take a photo first.');
            end
        end

        function frame = currentFrame(app)
            % The preprocessed frame the body points refer to.
            if app.Running && ~isempty(app.LastFrame)
                frame = app.LastFrame;
            elseif ~isempty(app.Photo)
                frame = app.Session.preprocess(app.Photo);
            else
                frame = [];
            end
        end

        function countdown(app, seconds, message)
            % Show the webcam preview with a countdown message.
            t = tic;
            while toc(t) < seconds && isvalid(app.Fig)
                frame = app.grabCamera();
                app.ImPhoto.ImageSource = frame;
                app.StatusLabel.Text = sprintf(message, ceil(seconds - toc(t)));
                drawnow;
            end
        end

        function endBusy(app)
            app.Busy = false;
            app.releaseCamera();
        end

        function runBackgroundCountdown(app)
            remaining = 3 - toc(app.BgCountdown);
            app.ImResult.ImageSource = app.Session.preprocess(app.grabCamera());
            if remaining > 0
                app.StatusLabel.Text = sprintf('STEP OUT OF VIEW - capturing background in %d s', ...
                                               ceil(remaining));
                return
            end
            shots = cell(1, 5);
            for i = 1:5, shots{i} = app.grabCamera(); end
            app.Session.setBackground(shots);
            app.BgCountdown = [];
            app.Session.Tracker.reset();          % re-detect when the person is back
            app.StatusLabel.Text = 'Background captured. Step back in.';
        end

        function safeStep(app)
            try
                app.stepOnce();
            catch err
                app.StatusLabel.Text = ['Error: ' err.message];
            end
        end
    end
end

% ==========================================================================
%  Local functions
% ==========================================================================
function img = readPhotoUpright(file)
% Read a photo and undo the camera rotation stored in its EXIF data
% (phone photos are saved sideways with an "Orientation" tag that imread
% ignores).
[~, ~, ext] = fileparts(file);
if any(strcmpi(ext, {'.heic', '.heif'}))
    error(['HEIC photos cannot be read by MATLAB. Export the photo as JPG ' ...
           '(Photos app: File > Export) and upload that.']);
end
img = imread(file);
try
    info = imfinfo(file);
    if isfield(info, 'Orientation')
        switch info(1).Orientation
            case 3, img = rot90(img, 2);
            case 6, img = rot90(img, -1);
            case 8, img = rot90(img, 1);
        end
    end
catch
    % no EXIF information: keep as is
end
if size(img, 3) == 4, img = img(:, :, 1:3); end    % CMYK / RGBA -> RGB
end

function msg = friendlyStatus(status)
% Turn the pipeline's technical reason into advice for the user.
if contains(status, 'convex') || contains(status, 'collinear') || contains(status, 'degenerate')
    msg = ['The garment could not be placed: the body points look crossed or squashed. ' ...
           'Press "Click body points..." and click them again.'];
elseif contains(status, 'click the keypoints')
    msg = 'Press "Click body points..." and click shoulders, hips and elbows.';
elseif contains(status, 'no person')
    msg = 'No person found. Use a photo facing the camera with shoulders and hips visible.';
elseif contains(status, 'confidence') || contains(status, 'NaN')
    msg = 'Shoulders or hips not found. Press "Click body points..." to place them by hand.';
else
    msg = status;
end
end

function [im, label] = makeView(parent, titleText)
% A titled picture: label on top, uiimage below (scaled to fit, aspect kept).
g = uigridlayout(parent, [2 1], 'RowHeight', {18, '1x'}, 'Padding', 0, 'RowSpacing', 2);
label = uilabel(g, 'Text', titleText, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
im = uiimage(g, 'ScaleMethod', 'fit', 'ImageSource', 0.94 * ones(480, 640, 3));
end

function [r, h] = newRow(h, height)
% Append a grid row (default 24 px high) and return its index.
if nargin < 2, height = 24; end
h{end + 1} = height;
r = numel(h);
end

function c = place(c, row, col)
c.Layout.Row = row;
c.Layout.Column = col;
end

function header(g, row, text)
place(uilabel(g, 'Text', upper(text), 'FontWeight', 'bold', 'FontSize', 11, ...
    'FontColor', [0.25 0.35 0.55], 'VerticalAlignment', 'bottom'), row, [1 2]);
end

function [s, h] = addSlider(g, h, label, limits, onChange)
% Label in column 1, slider (with 3 tick labels) in column 2.
[r, h] = newRow(h, 34);
place(uilabel(g, 'Text', label, 'VerticalAlignment', 'top'), r, 1);
s = place(uislider(g, 'Limits', limits, 'MajorTicks', linspace(limits(1), limits(2), 3), ...
    'MinorTicks', [], 'ValueChangedFcn', @(src, ~) onChange(src.Value)), r, 2);
end

function cams = listCameras()
try
    cams = webcamlist();
    if isempty(cams), cams = {}; end
    cams = reshape(cams, 1, []);
catch
    cams = {};
end
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
