"""Multi-view torso reconstruction from silhouettes (front, side, back).

Our own geometry, no learned model:
  1. each view's silhouette is scaled to centimetres with the person's known
     height (head top -> feet), so rows of different photos line up;
  2. per height, the front (and back) silhouette gives the torso WIDTH, the
     side silhouette gives the DEPTH and how far the centre sits forward;
  3. each height becomes an elliptical (or superellipse) cross-section. That
     is the visual hull of two orthogonal views, rounded: the true shape
     inside the width x depth box is not observed, which is the main error
     source (see the roundness test in run_poc.py).

The single-view baseline (what a single 2D photo allows) has the width only
and must ASSUME the depth (depth/width = DEPTH_PRIOR).
"""

from dataclasses import dataclass

import numpy as np

from .body import superellipse

DEPTH_PRIOR = 0.70          # assumed depth / width when only a front view exists


@dataclass
class Torso:
    ys: np.ndarray     # heights (cm)
    a: np.ndarray      # half-width
    b: np.ndarray      # half-depth
    c: np.ndarray      # forward offset of the centre (cm, + = toward the front)
    n: float = 2.0

    def section(self, y):
        return np.interp(y, self.ys, self.a), np.interp(y, self.ys, self.b), np.interp(y, self.ys, self.c), self.n

    def girth(self, y):
        a, b, _, n = self.section(y)
        return perimeter(a, b, n)


def perimeter(a, b, n=2.0, m=2048):
    x, z = superellipse(a, b, n, m)
    return float(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[z, z[0]])).sum())


def _scale(mask, height_cm):
    rows = np.flatnonzero(mask.any(axis=1))
    top, bottom = rows[0], rows[-1] + 1
    return top, (bottom - top) / height_cm          # (row of head top, px per cm)


def _runs(mask, rows, xc):
    """For each row: (start, end) of the silhouette run containing column xc (arms excluded)."""
    out = []
    for r in rows:
        xs = np.flatnonzero(mask[r])
        if not len(xs):
            out.append((np.nan, np.nan))
            continue
        cuts = np.flatnonzero(np.diff(xs) > 1)
        starts, ends = np.r_[xs[0], xs[cuts + 1]], np.r_[xs[cuts], xs[-1]]
        k = np.flatnonzero((starts <= xc) & (ends >= xc))
        k = k[0] if len(k) else int(np.argmin(np.minimum(abs(starts - xc), abs(ends - xc))))
        out.append((starts[k], ends[k] + 1))
    return np.array(out, float)


def reconstruct(masks: dict, height_cm: float, y_range=(84, 144), step=0.5, n=2.0) -> Torso:
    """masks: {"front": HxW bool, "side": ..., "back": ... (optional)}. The side
    view must show the body's front toward image right."""
    ys = np.arange(y_range[0], y_range[1] + step, step)
    prof = {}
    for view, m in masks.items():
        top, ppc = _scale(m, height_cm)
        rows = np.clip((top + (height_cm - ys) * ppc).astype(int), 0, m.shape[0] - 1)
        band = m[rows[len(rows) // 3]: rows[0]]
        xc = np.nonzero(band)[1].mean()                 # torso centre column
        prof[view] = _runs(m, rows, xc) / ppc            # in cm
    widths = [prof[v][:, 1] - prof[v][:, 0] for v in ("front", "back") if v in prof]
    a = np.nanmean(widths, axis=0) / 2
    side = prof["side"]
    b = (side[:, 1] - side[:, 0]) / 2
    centre = (side[:, 1] + side[:, 0]) / 2
    c = centre - np.interp(88, ys, centre)              # forward offset relative to the hips
    # silhouettes are quantised to whole pixels (~0.25 cm here): smooth the
    # profiles over ~1 cm of height so the mesh has no row-to-row steps
    k = np.exp(-0.5 * (np.arange(-4, 5) * step / 1.0) ** 2)
    k /= k.sum()
    a, b, c = (np.convolve(np.pad(v, 4, mode="edge"), k, mode="valid") for v in (a, b, c))
    return Torso(ys, a, b, c, n)


def single_view(front_mask, height_cm, **kw) -> Torso:
    """What one front photo allows: width measured, depth assumed."""
    t = reconstruct({"front": front_mask, "side": front_mask}, height_cm, **kw)
    t.b = DEPTH_PRIOR * t.a
    t.c = np.zeros_like(t.a)
    return t


def landmarks(t: Torso):
    """Waist = narrowest point between 96 and 116 cm, hip = widest between
    84 and 96, chest = deepest point between 116 and 132 (bust/chest)."""
    def pick(lo, hi, v, fn):
        k = (t.ys >= lo) & (t.ys <= hi)
        return float(t.ys[k][fn(v[k])])
    return {"chest": pick(116, 132, t.b, np.argmax), "waist": pick(96, 116, t.a, np.argmin),
            "hip": pick(84, 96, t.a, np.argmax)}


def projected_width(a, b, n, yaw):
    """Silhouette width of a cross-section turned by yaw (radians)."""
    x, z = superellipse(a, b, n, 720)
    xr = x * np.cos(yaw) + z * np.sin(yaw)
    return float(xr.max() - xr.min())


def estimate_yaw(t: Torso, front_width_cm: dict):
    """Torso rotation in a NEW photo from its silhouette widths at known
    heights, given the reconstructed torso. |yaw| only: a turn to the left
    and to the right give the same widths (MediaPipe's z gives the sign)."""
    grid = np.radians(np.arange(0, 70.5, 0.5))
    err = np.zeros_like(grid)
    for y, w in front_width_cm.items():
        a, b, _, n = t.section(y)
        err += np.array([(projected_width(a, b, n, g) - w) ** 2 for g in grid])
    return float(grid[np.argmin(err)])


def torso_parts(t: Torso, m=72):
    """Mesh of the reconstructed torso (for rendering / OBJ export)."""
    from .body import _tube
    rings = []
    for y in t.ys[::2]:
        a, b, c, n = t.section(y)
        x, z = superellipse(a, b, n, m)
        rings.append(np.c_[x, np.full_like(x, y), z + c])
    return [_tube(rings)]
