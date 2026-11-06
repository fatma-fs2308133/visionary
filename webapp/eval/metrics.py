"""Evaluation helpers shared by every method in the benchmark.

Clothing parsing: SegFormer-B2 fine-tuned on ATR clothes
(mattmdjaga/segformer_b2_clothes, 18 labels: background, hat, hair,
sunglasses, upper-clothes, skirt, pants, dress, belt, shoes, face, legs,
arms, bag, scarf). Used only to MEASURE results, never inside a method.
"""


import cv2
import numpy as np

from tryon_core.parsing import default_cloth_parser, face_above_chin, forearms  # noqa: F401  (forearms re-exported)


def parse(rgb: np.ndarray) -> dict:
    """rgb uint8 -> dict of boolean masks (same size as rgb), plus face_hair."""
    p = dict(default_cloth_parser()(rgb))
    p["face_hair"] = p["face"] | p["hair"]
    return p


def coverage_metrics(alpha: np.ndarray, parsing: dict, keypoints: dict) -> dict:
    """For methods that composite a garment layer with a known alpha."""
    g = alpha > 0.5
    n = max(g.sum(), 1)
    upper = parsing["upper"]
    fa = forearms(parsing, keypoints)
    return {
        "old_visible": float((upper & ~g).sum() / max(upper.sum(), 1)),
        "spill_bg": float((g & parsing["background"]).sum() / n),
        "spill_face": float((g & parsing["face_hair"]).sum() / n),
        "forearm_covered": float((g & fa).sum() / max(fa.sum(), 1)),
        "iou_upper": float((g & upper).sum() / max((g | upper).sum(), 1)),
    }


def output_metrics(person: np.ndarray, output: np.ndarray, garment_rgba: np.ndarray, parsing_in: dict,
                   keypoints: dict | None = None) -> dict:
    """For any method (also black-box ones): measured on the output image."""
    if output.shape != person.shape:
        output = cv2.resize(output, (person.shape[1], person.shape[0]), interpolation=cv2.INTER_AREA)
    parsing_out = parse(output)
    # identity / background preservation: SSIM outside the clothing + arm area
    keep = ~(parsing_in["upper"] | parsing_in["arms"] | parsing_out["upper"] | parsing_out["arms"])
    keep = cv2.erode(keep.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ssim_keep = _masked_ssim(person, output, keep)
    face = parsing_in["face_hair"]
    ssim_face = _masked_ssim(person, output, face) if face.sum() > 200 else float("nan")
    # the face alone (the parser's "face" label includes the neck, which a new
    # neckline legitimately changes)
    face_only = face_above_chin(parsing_in, keypoints) if keypoints else parsing_in["face"]
    face_only = cv2.erode(face_only.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ssim_face_only = _masked_ssim(person, output, face_only) if face_only.sum() > 200 else float("nan")
    # garment fidelity: colour histogram match between product and output top (Lab a/b + L)
    new_top = parsing_out["upper"]
    fidelity = _hist_similarity(garment_rgba, output, new_top)
    # how much of the ORIGINAL top survives: old-top pixels that still look like the old top
    old = parsing_in["upper"] & parsing_out["upper"]
    diff = np.abs(cv2.cvtColor(person, cv2.COLOR_RGB2LAB).astype(int) - cv2.cvtColor(output, cv2.COLOR_RGB2LAB).astype(int)).sum(2)
    old_left = float(((diff < 20) & old).sum() / max(parsing_in["upper"].sum(), 1))
    return {"ssim_keep": ssim_keep, "ssim_face": ssim_face, "ssim_face_only": ssim_face_only, "garment_hist": fidelity,
            "old_unchanged": old_left, "edge_seam": _seam_strength(output, new_top)}


def _masked_ssim(a, b, mask):
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(ga, (11, 11), 1.5), cv2.GaussianBlur(gb, (11, 11), 1.5)
    va = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s[mask].mean()) if mask.any() else float("nan")


def _hist_similarity(garment_rgba, output, region):
    if region.sum() < 200:
        return 0.0
    g_lab = cv2.cvtColor(garment_rgba[:, :, :3], cv2.COLOR_RGB2LAB)
    o_lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
    gm = (garment_rgba[:, :, 3] > 127).astype(np.uint8)
    om = region.astype(np.uint8)
    hg = cv2.calcHist([g_lab], [0, 1, 2], gm, [8, 12, 12], [0, 256, 0, 256, 0, 256])
    ho = cv2.calcHist([o_lab], [0, 1, 2], om, [8, 12, 12], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hg, hg, 1, 0, cv2.NORM_L1)
    cv2.normalize(ho, ho, 1, 0, cv2.NORM_L1)
    return float(cv2.compareHist(hg, ho, cv2.HISTCMP_INTERSECT))


def _seam_strength(output, region):
    """Mean gradient on the garment border / mean gradient inside it.
    ~1 = border looks like the rest of the image; >>1 = visible cut-out seam."""
    if region.sum() < 200:
        return float("nan")
    g = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    r = region.astype(np.uint8)
    band = (cv2.dilate(r, np.ones((5, 5), np.uint8)) - cv2.erode(r, np.ones((5, 5), np.uint8))) > 0
    inner = cv2.erode(r, np.ones((15, 15), np.uint8)) > 0
    return float(mag[band].mean() / max(mag[inner].mean(), 1e-3)) if inner.any() else float("nan")
