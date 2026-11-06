"""Planar homography, implemented from scratch (port of estimateHomographyDLT.m).

Normalised Direct Linear Transform: build the 2N x 9 matrix A from the point
pairs and solve A h = 0 with the SVD (h = right singular vector of the
smallest singular value). Hartley normalisation (centroid at the origin,
mean distance sqrt(2)) keeps A well conditioned.
"""

from itertools import combinations

import numpy as np


class DegenerateError(ValueError):
    """The points do not determine a unique homography."""


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src and dst must both be N x 2 with equal N")
    n = src.shape[0]
    if n < 4:
        raise ValueError(f"a homography needs at least 4 point pairs, got {n}")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("points must be finite")
    if n == 4:
        _check_no_three_collinear(src, "source")
        _check_no_three_collinear(dst, "destination")

    src_n, t1 = _normalise(src)
    dst_n, t2 = _normalise(dst)

    a = np.zeros((2 * n, 9))
    for i, ((x, y), (u, v)) in enumerate(zip(src_n, dst_n)):
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]

    _, s, vt = np.linalg.svd(a)
    if len(s) >= 8 and s[7] < 1e-10 * s[0]:
        raise DegenerateError("the point configuration does not determine a unique homography")
    hn = vt[-1].reshape(3, 3)

    h = np.linalg.solve(t2, hn @ t1)          # denormalise: H = T2^-1 Hn T1
    if abs(h[2, 2]) < np.finfo(float).eps:
        raise DegenerateError("H(3,3) is zero (point at infinity)")
    return h / h[2, 2]


def apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


def _normalise(p):
    c = p.mean(axis=0)
    d = np.mean(np.linalg.norm(p - c, axis=1))
    if d < np.finfo(float).eps:
        raise DegenerateError("all points coincide")
    s = np.sqrt(2) / d
    t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (p - c) * s, t


def _check_no_three_collinear(p, label):
    for i, j, k in combinations(range(4), 3):
        a, b, c = p[i], p[j], p[k]
        twice_area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if twice_area < 2:
            raise DegenerateError(f"three of the 4 {label} points are collinear")
