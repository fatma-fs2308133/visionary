"""Synthetic test bodies with KNOWN 3D geometry, and a small z-buffer renderer.

Ground truth for the 3D proof of concept: we have no multi-view photos of a
real person with tape-measured girths, so the reconstruction is first checked
on bodies whose every cross-section is known exactly.

A body is a stack of superellipse torso cross-sections (half-width a,
half-depth b, forward offset c, roundness n per height, in cm) plus a head,
A-pose arms and legs. World axes: x to the body's left (image right in a
front view), y up from the floor, z toward the camera (front of the body).
"""

from dataclasses import dataclass

import numpy as np

# Key heights (cm) and torso cross-sections: (y, a, b, c, n)
BODIES = {
    # average proportions, slim
    "A": dict(height=177, rows=[(84, 17.0, 11.5, -0.5, 2.4), (96, 15.5, 10.0, 0.0, 2.3), (104, 13.5, 9.0, 0.3, 2.2),
                                (116, 14.0, 9.5, 0.6, 2.2), (124, 15.5, 11.5, 1.5, 2.1), (132, 16.5, 10.5, 0.8, 2.3),
                                (140, 18.5, 8.0, -0.5, 2.6), (145, 7.0, 6.0, -1.0, 2.0), (155, 5.5, 5.5, -1.0, 2.0)]),
    # same frontal widths as A, much deeper torso (rounder belly / chest):
    # a single front photo cannot tell A and B apart
    "B": dict(height=177, rows=[(84, 17.0, 13.5, 0.0, 2.4), (96, 15.5, 13.0, 1.5, 2.2), (104, 13.5, 12.5, 2.0, 2.1),
                                (116, 14.0, 12.0, 1.5, 2.1), (124, 15.5, 13.0, 1.5, 2.1), (132, 16.5, 11.5, 0.8, 2.3),
                                (140, 18.5, 8.5, -0.5, 2.6), (145, 7.0, 6.0, -1.0, 2.0), (155, 5.5, 5.5, -1.0, 2.0)]),
}
SHOULDER_JOINT = (15.5, 140.0)      # (|x|, y) of the shoulder joints
HIP_JOINT = (9.0, 88.0)


@dataclass
class Body:
    name: str
    height: float
    ys: np.ndarray       # torso sample heights (cm)
    a: np.ndarray        # half-width
    b: np.ndarray        # half-depth
    c: np.ndarray        # forward offset of the centre
    n: np.ndarray        # superellipse exponent (2 = ellipse, larger = boxier)

    def section(self, y):
        """Cross-section parameters (a, b, c, n) at height y (linear interpolation)."""
        return tuple(np.interp(y, self.ys, v) for v in (self.a, self.b, self.c, self.n))

    def girth(self, y):
        """True circumference (cm) at height y."""
        a, b, _, n = self.section(y)
        x, z = superellipse(a, b, n, 4096)
        return float(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[z, z[0]])).sum())

    def joints(self, yaw=0.0):
        """World positions of the shoulder and hip joints (for the 2D pipeline's keypoints)."""
        (sx, sy), (hx, hy) = SHOULDER_JOINT, HIP_JOINT
        pts = {"leftShoulder": (sx, sy, 0.0), "rightShoulder": (-sx, sy, 0.0),
               "leftHip": (hx, hy, 0.0), "rightHip": (-hx, hy, 0.0),
               "leftElbow": (24.7, 113.5, 0.2), "rightElbow": (-24.7, 113.5, 0.2)}   # middle of the A-pose arm
        return {k: rotate_y(np.array([v]), yaw)[0] for k, v in pts.items()}


def make_body(name: str) -> Body:
    spec = BODIES[name]
    rows = np.array(spec["rows"], float)
    ys = np.linspace(rows[0, 0], rows[-1, 0], 160)
    cols = [np.interp(ys, rows[:, 0], rows[:, k]) for k in range(1, 5)]
    # smooth the piecewise-linear profiles (bodies have no kinks)
    kern = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    kern /= kern.sum()
    cols = [np.convolve(np.pad(v, 12, mode="edge"), kern, mode="valid") for v in cols]
    return Body(name, spec["height"], ys, *cols)


def superellipse(a, b, n, m=96):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    # t = 0 is the front centre (+z), increasing toward +x
    s, c = np.sin(t), np.cos(t)
    return a * np.sign(s) * np.abs(s) ** (2 / n), b * np.sign(c) * np.abs(c) ** (2 / n)


def rotate_y(p, yaw):
    """Turn the body by `yaw` radians about the vertical axis (positive = left side toward camera)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    out = p.copy()
    out[:, 0] = p[:, 0] * cy + p[:, 2] * sy
    out[:, 2] = -p[:, 0] * sy + p[:, 2] * cy
    return out


# ---------------------------------------------------------------------------
#  Mesh
# ---------------------------------------------------------------------------
def _tube(rings):
    """rings: list of (m, 3) closed loops -> (verts, faces)."""
    m = len(rings[0])
    v = np.concatenate(rings)
    f = []
    for i in range(len(rings) - 1):
        for j in range(m):
            p, q = i * m + j, i * m + (j + 1) % m
            f += [(p, q, q + m), (p, q + m, p + m)]
    return v, np.array(f)


def _limb(p0, p1, r0, r1, k=14, m=16):
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    u = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(d, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    return _tube([p0 + (p1 - p0) * s + (r0 + (r1 - r0) * s) * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)
                  for s in np.linspace(0, 1, k)])


def body_mesh(body: Body):
    """Closed-ish triangle mesh of the whole body: list of (verts, faces) parts."""
    parts = []
    rings = []
    for y in body.ys:
        a, b, c, n = body.section(y)
        x, z = superellipse(a, b, n, 72)
        rings.append(np.c_[x, np.full_like(x, y), z + c])
    parts.append(_tube(rings))
    # head: ellipsoid
    t, p = np.meshgrid(np.linspace(0, np.pi, 18), np.linspace(0, 2 * np.pi, 24, endpoint=False), indexing="ij")
    head = [np.c_[8 * np.sin(tt) * np.sin(pp), 166 + 11 * np.cos(tt), -1 + 9.5 * np.sin(tt) * np.cos(pp)]
            for tt, pp in zip(t, p)]
    parts.append(_tube(head))
    for s in (1, -1):
        parts.append(_limb(np.array([s * 16.5, 139, -0.5]), np.array([s * 33, 88, 1.0]), 5.0, 3.0))   # A-pose arm
        parts.append(_limb(np.array([s * 9.0, 92, 0.0]), np.array([s * 11, 4, 0.0]), 8.5, 4.0))       # leg
    return parts


# ---------------------------------------------------------------------------
#  Orthographic z-buffer renderer
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    """Orthographic camera looking along -z. Image row 0 = top_cm."""
    px_per_cm: float = 4.0
    width: int = 520
    height: int = 740
    top_cm: float = 182.0
    yaw: float = 0.0

    def project(self, p):
        q = rotate_y(p, self.yaw)
        return np.c_[self.width / 2 + q[:, 0] * self.px_per_cm, (self.top_cm - q[:, 1]) * self.px_per_cm, q[:, 2]]


def render(parts, cam: Camera, light=(0.3, 0.5, 1.0), colour=(0.78, 0.74, 0.70)):
    """Returns (rgb float 0..1, mask bool, depth)."""
    h, w = cam.height, cam.width
    zbuf = np.full((h, w), -np.inf)
    shade = np.zeros((h, w))
    l = np.array(light) / np.linalg.norm(light)
    for verts, faces in parts:
        P = cam.project(verts)
        W = rotate_y(verts, cam.yaw)
        for tri in faces:
            p = P[tri]
            x0, x1 = int(max(p[:, 0].min(), 0)), int(min(np.ceil(p[:, 0].max()), w - 1))
            y0, y1 = int(max(p[:, 1].min(), 0)), int(min(np.ceil(p[:, 1].max()), h - 1))
            if x1 < x0 or y1 < y0:
                continue
            n = np.cross(W[tri[1]] - W[tri[0]], W[tri[2]] - W[tri[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = p
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-9:
                continue
            l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
            l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
            l3 = 1 - l1 - l2
            inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            z = l1 * az + l2 * bz + l3 * cz
            zb = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (z > zb)
            zb[upd] = z[upd]
            shade[y0:y1 + 1, x0:x1 + 1][upd] = abs(float(n @ l))
    mask = np.isfinite(zbuf)
    rgb = np.ones((h, w, 3)) * 0.96
    rgb[mask] = np.array(colour) * (0.35 + 0.65 * shade[mask])[:, None]
    return rgb, mask, zbuf


def save_obj(path, parts):
    with open(path, "w") as f:
        off = 1
        for v, fc in parts:
            for p in v:
                f.write(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
            for t in fc:
                f.write(f"f {t[0] + off} {t[1] + off} {t[2] + off}\n")
            off += len(v)
