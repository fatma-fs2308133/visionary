"""Inverse warping with bilinear interpolation, implemented from scratch
(port of inverseWarpBilinear.m). No cv2.warpPerspective / remap is used.

For every output pixel p we compute the source position q = H^-1 p and
interpolate the source image there from its 4 neighbours:

    out = (1-a)(1-b) I[y0,x0] + a(1-b) I[y0,x1] + (1-a)b I[y1,x0] + ab I[y1,x1]

Only the bounding box of the warped source corners is evaluated, which keeps
it fast enough for live video.
"""

import numpy as np

from .homography import apply_homography


def inverse_warp_bilinear(src: np.ndarray, h: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """src: Hs x Ws x C float image. Returns out_h x out_w x C, 0 outside."""
    out_h, out_w = out_hw
    hs, ws = src.shape[:2]
    channels = src.shape[2]
    out = np.zeros((out_h, out_w, channels), dtype=np.float32)

    corners = apply_homography(h, np.array([[0, 0], [ws - 1, 0], [ws - 1, hs - 1], [0, hs - 1]]))
    if not np.isfinite(corners).all():
        return out
    x0 = int(max(0, np.floor(corners[:, 0].min())))
    x1 = int(min(out_w - 1, np.ceil(corners[:, 0].max())))
    y0 = int(max(0, np.floor(corners[:, 1].min())))
    y1 = int(min(out_h - 1, np.ceil(corners[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return out

    # Output pixel grid of the bounding box -> source coordinates through H^-1
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float32), np.arange(y0, y1 + 1, dtype=np.float32))
    hinv = np.linalg.inv(h).astype(np.float32)
    w = hinv[2, 0] * xs + hinv[2, 1] * ys + hinv[2, 2]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    sx = (hinv[0, 0] * xs + hinv[0, 1] * ys + hinv[0, 2]) / w
    sy = (hinv[1, 0] * xs + hinv[1, 1] * ys + hinv[1, 2]) / w

    # Keep only positions with all 4 neighbours inside the source image
    valid = (sx >= 0) & (sx <= ws - 1) & (sy >= 0) & (sy <= hs - 1)
    sx = sx[valid]
    sy = sy[valid]
    ix = np.minimum(np.floor(sx).astype(np.int64), ws - 2)
    iy = np.minimum(np.floor(sy).astype(np.int64), hs - 2)
    a = (sx - ix).astype(np.float32)[:, None]
    b = (sy - iy).astype(np.float32)[:, None]

    # Gather the 4 neighbours through one flat index (faster than 2-D indexing)
    flat = src.reshape(-1, channels)
    i00 = iy * ws + ix
    top = (1 - a) * flat.take(i00, axis=0) + a * flat.take(i00 + 1, axis=0)
    bottom = (1 - a) * flat.take(i00 + ws, axis=0) + a * flat.take(i00 + ws + 1, axis=0)
    box = out[y0:y1 + 1, x0:x1 + 1]
    box[valid] = (1 - b) * top + b * bottom
    return out
