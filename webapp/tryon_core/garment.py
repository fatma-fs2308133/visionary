"""Garment cut-out and automatic anchor points
(port of prepareGarment.m and autoGarmentAnchors.m).

Anchors are stored as normalised [x, y] in 0..1. Wearer's LEFT = image RIGHT
(the garment is photographed from the front).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ANCHOR_NAMES = ("leftShoulder", "rightShoulder", "leftHip", "rightHip",
                "leftArmpit", "rightArmpit", "leftSleeveEnd", "rightSleeveEnd")


@dataclass
class Garment:
    id: str
    name: str
    rgb: np.ndarray        # H x W x 3 float32 0..1
    alpha: np.ndarray      # H x W float32 0..1
    anchors: dict          # name -> (x, y) normalised
    sleeves: str = ""      # "long" / "short" / "none" if known (catalogue), else ""

    def anchors_px(self) -> dict:
        h, w = self.alpha.shape
        return {k: np.array([v[0] * (w - 1), v[1] * (h - 1)]) for k, v in self.anchors.items()}


# ---------------------------------------------------------------------------
#  Preparation: any garment photo -> RGBA cut-out + anchors
# ---------------------------------------------------------------------------
def prepare_garment(image_bgra: np.ndarray, max_side: int = 512, margin: int = 8,
                    threshold: float = 0.10) -> tuple[np.ndarray, dict]:
    """image_bgra: image as read by cv2.IMREAD_UNCHANGED (BGR or BGRA).
    Returns (RGBA uint8 cut-out, anchors)."""
    if image_bgra.ndim == 2:
        image_bgra = cv2.cvtColor(image_bgra, cv2.COLOR_GRAY2BGR)
    has_alpha = image_bgra.shape[2] == 4 and image_bgra[:, :, 3].min() < 250
    bgr = image_bgra[:, :, :3]
    if has_alpha:
        mask = image_bgra[:, :, 3] > 127
    else:
        mask = _remove_background_ai(bgr)
        if mask is None or mask.sum() < 0.02 * mask.size:
            mask = _remove_background(bgr, threshold)
    # Light parts of a print can be cut out as background; anything fully
    # enclosed by the garment is garment (the neck opening reaches the edge).
    mask = _fill_holes(_largest_component(mask))
    if mask.sum() < 0.02 * mask.size:
        raise ValueError("Could not find the garment. Use a flat, front-facing photo on a plain background.")

    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    rgba = np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (mask * 255).astype(np.uint8)])[y0:y1, x0:x1]

    scale = max_side / max(rgba.shape[:2])
    if scale < 1:
        rgba = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # Pre-multiply: transparent pixels carry no colour (no fringe when warped)
    rgba[rgba[:, :, 3] < 10, :3] = 0
    return rgba, auto_anchors(rgba[:, :, 3] > 127)


_REMBG_SESSION = None


def _remove_background_ai(bgr):
    """Salient-object segmentation (IS-Net via rembg): works on busy or
    shadowed backgrounds where the colour-distance method fails. None if
    rembg is not installed."""
    global _REMBG_SESSION
    try:
        from rembg import new_session, remove
    except ImportError:
        return None
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session("isnet-general-use")
    rgba = remove(bgr, session=_REMBG_SESSION)
    return rgba[:, :, 3] > 127


def _remove_background(bgr, threshold):
    """Fallback: colour distance to the median border colour."""
    img = cv2.GaussianBlur(bgr.astype(np.float32) / 255, (5, 5), 1.0)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(img - bg, axis=2) / np.sqrt(3)
    mask = (dist > threshold).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = _fill_holes(mask > 0).astype(np.uint8)
    # The blur above smears the background colour into the edge pixels:
    # drop them, or the garment gets a light fringe.
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0


def _fill_holes(mask):
    m = mask.astype(np.uint8)
    flood = np.pad(m, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return mask | holes


def _largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


# ---------------------------------------------------------------------------
#  Automatic anchors from the silhouette (see autoGarmentAnchors.m)
# ---------------------------------------------------------------------------
def auto_anchors(m: np.ndarray) -> dict:
    """Anchor points of a front-facing top from its silhouette.

    Product photos are close to left/right symmetric, so everything is
    measured from the garment's centre line:
      - centre    median of the row centres in the lower half (the hem area,
                  where collars, ties and asymmetric necklines do not pull it)
      - torso     20th percentile of the width of the run through the centre
                  over 40-85% of the height (sleeves hanging alongside the
                  body make some rows wider, never narrower). If even that
                  spans the whole garment, long sleeves are merged with the
                  body and the torso is taken as the central 64%.
      - shoulder  92% of the torso half-width out, on the top outline
      - hip       84% of the half-width out, 5% above the median hem
      - armpit    torso edge, just below the lowest row where a sleeve still
                  sticks out (else 30% of the height)
      - sleeve    centroid of the sleeve pixels farthest from the shoulder
    """
    m = _largest_component(m)
    h_img, w_img = m.shape
    rows = np.flatnonzero(m.any(axis=1))
    top, bottom = rows[0], rows[-1]
    h = bottom - top + 1
    cols = np.flatnonzero(m.any(axis=0))
    full = cols[-1] - cols[0] + 1

    lower = np.arange(top + int(0.5 * h), top + int(0.9 * h))
    centres = [np.flatnonzero(m[y]).mean() for y in lower if m[y].any()]
    cx = float(np.median(centres)) if centres else float(cols.mean())
    xc = int(round(cx))

    band = np.arange(top + int(0.4 * h), top + int(0.85 * h))
    widths = []
    for y in band:
        lft, rgt = _run_around(m[y], xc)
        if not np.isnan(lft):
            widths.append(rgt - lft + 1)
    torso_w = float(np.percentile(widths, 20)) if widths else 0.64 * full
    if torso_w > 0.8 * full:
        torso_w = 0.64 * full
    half = torso_w / 2

    # Shoulders on the top outline
    def top_at(x):
        x = int(np.clip(round(x), 0, w_img - 1))
        col = np.flatnonzero(m[:, x])
        return col[0] if len(col) else top

    # Both shoulders at one height, at most 12% below the top (straps and
    # dropped shoulders would otherwise put them low or uneven)
    y_sh = min((top_at(cx - 0.92 * half) + top_at(cx + 0.92 * half)) / 2, top + 0.12 * h) + 0.03 * h
    sh_r = (cx - 0.92 * half, y_sh)              # image left = wearer's right
    sh_l = (cx + 0.92 * half, y_sh)

    # Hips above the median hem of the torso columns
    tcols = np.arange(int(cx - 0.6 * half), int(cx + 0.6 * half) + 1)
    tcols = tcols[(tcols >= 0) & (tcols < w_img) & m.any(axis=0)[np.clip(tcols, 0, w_img - 1)]]
    hem = np.median([np.flatnonzero(m[:, x])[-1] for x in tcols]) if len(tcols) else bottom
    y_hip = hem - 0.05 * h
    hip_r = (cx - 0.84 * half, y_hip)
    hip_l = (cx + 0.84 * half, y_hip)

    # Armpits: lowest row in the upper part where a sleeve sticks out
    def armpit(side):
        edge = cx + side * half
        y_arm = np.nan
        for y in range(top, top + int(0.45 * h)):
            xs = np.flatnonzero(m[y])
            if len(xs) and (side < 0 and xs[0] < edge - 0.1 * half or side > 0 and xs[-1] > edge + 0.1 * half):
                y_arm = y
        if np.isnan(y_arm) or y_arm < top + 0.15 * h:
            y_arm = top + 0.30 * h
        else:
            y_arm = y_arm + 0.02 * h
        return (edge, y_arm)

    arm_r, arm_l = armpit(-1), armpit(+1)

    xs, ys = np.meshgrid(np.arange(w_img), np.arange(h_img))
    end_r = _sleeve_end(_sleeve_region(m & (xs < cx - 1.04 * half), arm_r[1] + 0.05 * h),
                        xs, ys, sh_r, arm_r, -1, torso_w)
    end_l = _sleeve_end(_sleeve_region(m & (xs > cx + 1.04 * half), arm_l[1] + 0.05 * h),
                        xs, ys, sh_l, arm_l, +1, torso_w)

    def n(p):
        return [float(p[0] / (w_img - 1)), float(p[1] / (h_img - 1))]

    return {
        "rightShoulder": n(sh_r), "leftShoulder": n(sh_l),
        "rightHip": n(hip_r), "leftHip": n(hip_l),
        "rightArmpit": n(arm_r), "leftArmpit": n(arm_l),
        "rightSleeveEnd": n(end_r), "leftSleeveEnd": n(end_l),
    }


def _sleeve_region(outside, y_top):
    """Garment pixels outside the torso band that belong to a sleeve: the
    connected pieces that reach up to armpit height. A hem that flares wider
    than the torso is outside the band too, but only near the bottom, so it
    is not counted (it made short sleeves look as long as the garment)."""
    n, labels = cv2.connectedComponents(outside.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(outside)
    for k in range(1, n):
        comp = labels == k
        if np.flatnonzero(comp.any(axis=1))[0] <= y_top:
            keep |= comp
    return keep


def _sleeve_end(s, xs, ys, shoulder, arm, side, w):
    if s.sum() < 0.002 * s.size:
        return (arm[0] + side * 0.05 * w, arm[1])
    px, py = xs[s], ys[s]
    dist = np.hypot(px - shoulder[0], py - shoulder[1])
    far = dist >= np.quantile(dist, 0.85)
    return (float(px[far].mean()), float(py[far].mean()))


def _run_around(row, xc):
    if not row[xc]:
        return np.nan, np.nan
    before = np.flatnonzero(~row[:xc + 1])
    after = np.flatnonzero(~row[xc:])
    left = before[-1] + 1 if len(before) else 0
    right = xc + after[0] - 1 if len(after) else len(row) - 1
    return left, right


# ---------------------------------------------------------------------------
#  Library on disk: <id>.png (RGBA) + <id>.json ({"name", "anchors"})
# ---------------------------------------------------------------------------
def save_garment(folder: Path, gid: str, name: str, rgba: np.ndarray, anchors: dict, sleeves: str = ""):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{gid}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    meta = {"name": name, "anchors": anchors}
    if sleeves:
        meta["sleeves"] = sleeves
    (folder / f"{gid}.json").write_text(json.dumps(meta, indent=2))


def load_garment(folder: Path, gid: str) -> Garment:
    bgra = cv2.imread(str(folder / f"{gid}.png"), cv2.IMREAD_UNCHANGED)
    meta = json.loads((folder / f"{gid}.json").read_text())
    rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    alpha = bgra[:, :, 3].astype(np.float32) / 255
    return Garment(gid, meta["name"], rgb, alpha, meta["anchors"], meta.get("sleeves", ""))
