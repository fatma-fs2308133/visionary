"""Feathered alpha compositing, implemented from scratch
(port of blendGarmentOntoFrame.m + erodeBinaryDisk.m + convolve2D.m).

1. binarise the garment alpha            M = alpha >= 0.5
2. erode with a disk of radius r         (pull the edge inward, drop PNG fringe)
3. convolve with a normalised Gaussian   (0->1 step becomes a soft ramp)
4. final alpha = soft mask * original alpha
5. out = a * garment + (1 - a) * frame   (Porter-Duff "over")

Only the bounding box of the garment (plus the kernel margin) is processed.
"""

import numpy as np


def erode_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion with an exact disk; outside the image counts as True."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.ones((h + 2 * radius, w + 2 * radius), dtype=bool)
    padded[radius:radius + h, radius:radius + w] = mask
    out = np.ones((h, w), dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                out &= padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
    return out


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def convolve_separable(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convolve with k along columns then rows (Gaussian is separable),
    replicate padding, shift-and-add."""
    r = len(k) // 2
    out = img
    for axis in (0, 1):
        n = out.shape[axis]
        idx = np.clip(np.arange(-r, n + r), 0, n - 1)
        padded = np.take(out, idx, axis=axis)
        acc = np.zeros_like(out, dtype=np.float32)
        for i, wgt in enumerate(k[::-1]):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(i, i + n)
            acc += wgt * padded[tuple(sl)]
        out = acc
    return out


def feather_blend(frame: np.ndarray, garment: np.ndarray, alpha: np.ndarray,
                  erode_radius: int = 3, sigma: float = 2.0, threshold: float = 0.5):
    """frame, garment: H x W x 3 float32 in [0,1]; alpha: H x W in [0,1].
    Returns (composited frame, final alpha)."""
    h, w = alpha.shape
    hard = alpha >= threshold
    final = np.zeros((h, w), dtype=np.float32)
    if not hard.any():
        return frame, final

    margin = int(np.ceil(3 * sigma)) + erode_radius + 1
    rows = np.flatnonzero(hard.any(axis=1))
    cols = np.flatnonzero(hard.any(axis=0))
    r1, r2 = max(0, rows[0] - margin), min(h, rows[-1] + margin + 1)
    c1, c2 = max(0, cols[0] - margin), min(w, cols[-1] + margin + 1)

    soft = erode_disk(hard[r1:r2, c1:c2], erode_radius).astype(np.float32)
    if sigma > 0:
        soft = convolve_separable(soft, gaussian_kernel_1d(sigma))
    a = np.clip(soft * alpha[r1:r2, c1:c2], 0, 1)

    out = frame.copy()
    out[r1:r2, c1:c2] = a[..., None] * garment[r1:r2, c1:c2] + (1 - a[..., None]) * frame[r1:r2, c1:c2]
    final[r1:r2, c1:c2] = a
    return out, final
