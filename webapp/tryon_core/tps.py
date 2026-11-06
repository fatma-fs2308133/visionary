"""Thin-plate spline warping, implemented from scratch.

A TPS maps R^2 -> R^2 through N control points p_i -> q_i:

    f(x) = a0 + a1 x + a2 y + sum_i w_i U(|x - p_i|),   U(r) = r^2 log r^2

with the side conditions sum w_i = 0, sum w_i p_i = 0. The coefficients
solve the (N+3) x (N+3) linear system

    [ K + lambda I   P ] [ w ]   [ q ]
    [ P^T            0 ] [ a ] = [ 0 ]

(K_ij = U(|p_i - p_j|), P = [1 x y]). lambda > 0 relaxes exact
interpolation into a smoothing spline. This is the classical shape-context +
TPS warp of Belongie et al. (PAMI 2002) used by the original VITON
(Han et al., CVPR 2018) before learned warping modules.
"""

import numpy as np


def _u(r2):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = r2 * np.log(r2)
    return np.nan_to_num(v)


def fit_tps(src: np.ndarray, dst: np.ndarray, lam: float = 1e-3) -> dict:
    """TPS that maps src points (N x 2) onto dst points (N x 2)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    # normalise for conditioning (and so lambda is scale-free)
    c, s = src.mean(0), max(src.std(), 1e-6)
    p = (src - c) / s
    k = _u(((p[:, None] - p[None]) ** 2).sum(-1))
    pmat = np.c_[np.ones(n), p]
    a = np.zeros((n + 3, n + 3))
    a[:n, :n] = k + lam * np.eye(n)
    a[:n, n:] = pmat
    a[n:, :n] = pmat.T
    b = np.zeros((n + 3, 2))
    b[:n] = dst
    coef = np.linalg.solve(a, b)
    return {"ctrl": p, "w": coef[:n], "a": coef[n:], "c": c, "s": s}


def apply_tps(tps: dict, pts: np.ndarray) -> np.ndarray:
    """Evaluate the TPS at M points (M x 2), in chunks to bound memory."""
    out = np.empty((len(pts), 2))
    for i in range(0, len(pts), 20000):
        q = (pts[i:i + 20000] - tps["c"]) / tps["s"]
        u = _u(((q[:, None] - tps["ctrl"][None]) ** 2).sum(-1))
        out[i:i + 20000] = u @ tps["w"] + np.c_[np.ones(len(q)), q] @ tps["a"]
    return out


def remap_bilinear(src: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """out[y, x] = src sampled at (mx[y, x], my[y, x]) with bilinear
    interpolation (same scheme as warp.inverse_warp_bilinear); 0 outside."""
    hs, ws = src.shape[:2]
    ch = src.shape[2]
    out = np.zeros(mx.shape + (ch,), np.float32)
    valid = (mx >= 0) & (mx <= ws - 1) & (my >= 0) & (my <= hs - 1)
    sx, sy = mx[valid], my[valid]
    ix = np.minimum(np.floor(sx).astype(np.int64), ws - 2)
    iy = np.minimum(np.floor(sy).astype(np.int64), hs - 2)
    a = (sx - ix).astype(np.float32)[:, None]
    b = (sy - iy).astype(np.float32)[:, None]
    flat = src.reshape(-1, ch)
    i00 = iy * ws + ix
    top = (1 - a) * flat.take(i00, 0) + a * flat.take(i00 + 1, 0)
    bot = (1 - a) * flat.take(i00 + ws, 0) + a * flat.take(i00 + ws + 1, 0)
    out[valid] = (1 - b) * top + b * bot
    return out
