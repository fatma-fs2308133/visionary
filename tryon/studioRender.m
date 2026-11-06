function [out, info] = studioRender(personImg, garment, opts)
%STUDIORENDER  Photorealistic try-on through the local studio server.
%
%   out = studioRender(personImg, garment)
%   [out, info] = studioRender(personImg, garment, opts)
%
%   The studio render is the hybrid pipeline of the Python web app
%   (webapp/hybrid): our pose + DLT homography + TPS build the try-on mask
%   and a pretrained diffusion model (CatVTON) renders the garment inside it,
%   locally on the Mac's GPU. Our feathered blend pastes it back, so pixels
%   outside the garment area are the original photo.
%
%   WHY A LOCAL SERVER (and not py.* calls)
%     MATLAB R2023b on Apple silicon runs as an Intel (x86_64) app under
%     Rosetta. PyTorch's Apple-GPU back end (MPS) needs a native arm64
%     process, so the model cannot run inside MATLAB. The server runs on the
%     same Mac (no internet, no hosted service):
%         cd webapp
%         .venv/bin/uvicorn server:app --port 8000
%
%   INPUTS
%     personImg  H x W x 3 photo of the person (any class).
%     garment    struct from loadGarmentLibrary (fields name, rgb, alpha).
%                If the server's library has a garment with the same name,
%                its original product photo is used (best quality);
%                otherwise this cut-out is sent.
%     opts       (optional) struct: .server ('http://localhost:8000'),
%                .timeout (600 s).
%
%   OUTPUTS
%     out   the try-on result, uint8, same size as the server's working
%           copy of the photo (longest side at most 1024 px).
%     info  struct: .seconds (server time).
%
%   See also TryOnApp, loadGarmentLibrary.

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'server'),  opts.server = 'http://localhost:8000'; end
if ~isfield(opts, 'timeout'), opts.timeout = 600; end

body = struct( ...
    'person',    toBase64(im2uint8(personImg), '.jpg'), ...
    'garmentId', garment.name, ...
    'garment',   toBase64(cat(3, im2uint8(garment.rgb), im2uint8(garment.alpha(:, :, 1))), '.png'), ...
    'quality',   'standard');

options = weboptions('MediaType', 'application/json', 'ContentType', 'json', ...
                     'Timeout', opts.timeout);
try
    resp = webwrite([opts.server '/api/studio/render'], body, options);
catch err
    if contains(err.message, {'Could not establish', 'connection', 'refused', 'resolve'})
        error('studioRender:server', ['The studio server is not running. In a terminal:\n' ...
              '  cd webapp\n  .venv/bin/uvicorn server:app --port 8000\n(%s)'], err.message);
    end
    rethrow(err);
end

out = fromBase64(resp.image);
info = struct('seconds', resp.seconds);
end

% ==========================================================================
function s = toBase64(img, ext)
% Encode an image (RGB, or RGBA as H x W x 4) as a base64 JPEG/PNG string.
f = [tempname ext];
cleanup = onCleanup(@() delete(f));
if size(img, 3) == 4
    imwrite(img(:, :, 1:3), f, 'Alpha', img(:, :, 4));
else
    imwrite(img, f, 'Quality', 95);
end
fid = fopen(f, 'r');
bytes = fread(fid, Inf, '*uint8');
fclose(fid);
s = matlab.net.base64encode(bytes);
end

function img = fromBase64(s)
f = [tempname '.jpg'];
cleanup = onCleanup(@() delete(f));
fid = fopen(f, 'w');
fwrite(fid, matlab.net.base64decode(s), 'uint8');
fclose(fid);
img = imread(f);
end
