"""3D proof of concept: multi-view body reconstruction -> measurements -> garment placement.

    .venv/bin/python -m experiments.body3d.run_poc

Writes experiments/body3d/out/: figures, OBJ meshes and results.json.
Experimental only: the app does not use any of this.
"""

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tryon_core.garment import load_garment  # noqa: E402
from tryon_core.homography import apply_homography, estimate_homography  # noqa: E402

from .body import Camera, body_mesh, make_body, render, save_obj, superellipse, rotate_y  # noqa: E402
from .reconstruct import (estimate_yaw, landmarks, perimeter, projected_width, reconstruct,  # noqa: E402
                          single_view, torso_parts, _runs, _scale)
from .wrap import metric_panel, place_2d, place_3d  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
ROOT = Path(__file__).resolve().parents[2]
VIEWS = (("front", 0), ("side", 90), ("back", 180))


def u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def measurements(names=("A", "B")):
    rows, sheets, torsos = [], [], {}
    for name in names:
        body = make_body(name)
        parts = body_mesh(body)
        views = {v: render(parts, Camera(yaw=np.radians(d))) for v, d in VIEWS}
        masks = {v: r[1] for v, r in views.items()}
        multi = reconstruct(masks, body.height)
        single = single_view(masks["front"], body.height)
        torsos[name] = (body, parts, multi)
        for lm, y in landmarks(multi).items():
            a, b, _, _ = body.section(y)
            rows.append({"body": name, "landmark": lm, "height_cm": y,
                         "true_girth": body.girth(y), "multi_view": multi.girth(y), "single_view": single.girth(y),
                         "true_width": 2 * a, "measured_width": 2 * multi.section(y)[0],
                         "true_depth": 2 * b, "multi_depth": 2 * multi.section(y)[1], "single_depth": 2 * single.section(y)[1]})
        save_obj(OUT / f"body_{name}_true.obj", parts)
        save_obj(OUT / f"body_{name}_reconstructed_torso.obj", torso_parts(multi))
        cam34 = Camera(yaw=np.radians(35))
        gt34 = render(parts, cam34)[0]
        rec34 = render(torso_parts(multi), cam34, colour=(0.55, 0.68, 0.80))[0]
        sheets.append(np.hstack([label(u8(views[v][0]), f"{name}: {v}") for v, _ in VIEWS]
                                + [label(u8(gt34), "true body, 35 deg"), label(u8(rec34), "reconstructed torso")]))
    cv2.imwrite(str(OUT / "reconstruction.jpg"), cv2.cvtColor(np.vstack(sheets), cv2.COLOR_RGB2BGR))
    return rows, torsos


def profile_plot(torsos):
    fig, axes = plt.subplots(1, len(torsos), figsize=(5 * len(torsos), 4.2), sharey=True)
    for ax, (name, (body, _, multi)) in zip(np.atleast_1d(axes), torsos.items()):
        ys = multi.ys
        ax.plot([body.girth(y) for y in ys], ys, "k-", lw=2, label="true girth")
        ax.plot([multi.girth(y) for y in ys], ys, "C0-", label="multi-view (front+side+back)")
        single_g = [perimeter(a, 0.7 * a) for a in multi.a]
        ax.plot(single_g, ys, "C3--", label="single front view (depth assumed)")
        ax.set_title(f"body {name}")
        ax.set_xlabel("girth (cm)")
    np.atleast_1d(axes)[0].set_ylabel("height above floor (cm)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "girth_profiles.png", dpi=130)


def orientation(torsos, yaws=(10, 20, 30), keypoint_noise=0.01):
    """|yaw| of a new photo. Two cues:
    silhouette widths + reconstructed cross-sections (estimate_yaw), and the
    shoulder keypoint distance / the true shoulder width known from the
    front capture: cos(yaw) = seen / true. Sensitivity of the second: a
    keypoint error of `keypoint_noise` x shoulder width moves yaw by
    keypoint_noise / sin(yaw)."""
    body, parts, multi = torsos["A"]
    res = []
    for deg in yaws:
        mask = render(parts, Camera(yaw=np.radians(deg)))[1]
        top, ppc = _scale(mask, body.height)
        widths = {}
        for y in (104, 112, 120):
            r = int(top + (body.height - y) * ppc)
            s, e = _runs(mask, [r], mask.shape[1] / 2)[0]
            widths[y] = (e - s) / ppc
        front_w = {y: projected_width(*multi.section(y)[:2], multi.n, 0.0) for y in widths}
        sh = body.joints(np.radians(deg))
        ratio = abs(sh["leftShoulder"][0] - sh["rightShoulder"][0]) / (2 * 15.5)
        res.append({"true_yaw": deg,
                    "silhouette_width_change_pct": round(100 * (np.mean(list(widths.values())) / np.mean(list(front_w.values())) - 1), 1),
                    "yaw_from_silhouette_widths": round(float(np.degrees(estimate_yaw(multi, widths))), 1),
                    "yaw_from_shoulder_ratio": round(float(np.degrees(np.arccos(min(ratio, 1)))), 1),
                    "yaw_uncertainty_per_1pct_keypoint_error": round(float(np.degrees(keypoint_noise / np.sin(np.radians(deg)))), 1)})
    return res


def _row_extent(alpha_row):
    xs = np.flatnonzero(alpha_row > 0.5)
    return (xs[0], xs[-1]) if len(xs) else (np.nan, np.nan)


def placement(torsos, garment_dir, gid, sizes=(("S", 42.0), ("L", 50.0)), yaws=(0, 25)):
    """Same body render, three placements: 2D (current app), 3D on the
    reconstructed torso, and a geometric REFERENCE: the same wrap on the true
    body. The reference has no drape either; it only tells whether the
    placement is geometrically right for that body and garment size."""
    body, parts, multi = torsos["A"]
    g = load_garment(garment_dir, gid)
    a = g.anchors_px()
    names = ("leftShoulder", "rightShoulder", "rightHip", "leftHip")
    chest_px = abs(a["leftArmpit"][0] - a["rightArmpit"][0])
    gx_c = (a["leftShoulder"][0] + a["rightShoulder"][0]) / 2
    rows, numbers = [], []
    chest_y = landmarks(multi)["chest"]
    for deg in yaws:
        yaw = np.radians(deg)
        cam = Camera(yaw=yaw)
        base, mask, depth = render(parts, cam)
        r_chest = int((cam.top_cm - chest_y) * cam.px_per_cm)
        body_l, body_r = _runs(mask, [r_chest], cam.width / 2 + rotate_y(np.array([[0, 0, 0.0]]), yaw)[0, 0])[0]
        two_d, info = place_2d(base, cam, body.joints(), g, yaw)
        hmat = estimate_homography(np.array([a[k] for k in names]), np.array(info["torso"]))
        # where the 2D warp puts the garment's centre line and how big its print is, at chest height
        gy = np.linspace(0, g.alpha.shape[0], 400)
        line = apply_homography(hmat, np.c_[np.full_like(gy, gx_c), gy])
        k = np.argmin(abs(line[:, 1] - r_chest))
        x_cf_2d = line[k, 0]
        p0, p1 = apply_homography(hmat, np.array([[gx_c - 5, gy[k]], [gx_c + 5, gy[k]]]))
        scale_2d = np.linalg.norm(p1 - p0) / 10                        # image px per garment px
        row = [label(u8(base), f"body A, yaw {deg}"), label(u8(two_d), "2D (current app)")]
        a_, b_, c_, _ = body.section(chest_y)
        x_cf_true = cam.width / 2 + rotate_y(np.array([[0.0, chest_y, c_ + b_]]), yaw)[0, 0] * cam.px_per_cm
        for size, flat in sizes:
            panel, origin = metric_panel(g, flat)
            gpx = flat / chest_px * 8.0                                  # panel px per garment px
            res = {}
            for kind, torso in (("3d", multi), ("ref", body)):
                out, alpha, extra = place_3d(base, depth, cam, torso, panel, origin, flat, 140.0, yaw)
                mxr = extra["mx"][r_chest]
                ok = np.flatnonzero(mxr >= 0)
                c = ok[np.argmin(abs(mxr[ok] - origin[0]))]
                slope = (mxr[c + 2] - mxr[c - 2]) / 4                    # panel px per image px
                res[kind] = {"out": out, "x_cf": c, "scale": gpx / slope, "extent": _row_extent(alpha[r_chest]),
                             "strain": extra["strain"].get(round(chest_y), 0.0)}
            row.append(label(u8(res["3d"]["out"]), f"3D, size {size} ({2 * flat:.0f} cm girth)"))
            ref = res["ref"]
            numbers.append({
                "yaw": deg, "size": size, "garment_girth_cm": 2 * flat,
                "body_chest_girth_cm (measured)": round(multi.girth(chest_y), 1),
                "ease_cm": round(2 * flat - multi.girth(chest_y), 1),
                "strain_3d_pct": round(100 * res["3d"]["strain"], 1),
                "print_scale_vs_ref_2d": round(scale_2d / ref["scale"], 3),
                "print_scale_vs_ref_3d": round(res["3d"]["scale"] / ref["scale"], 3),
                "centre_front_error_cm_2d": round((x_cf_2d - x_cf_true) / cam.px_per_cm, 1),
                "centre_front_error_cm_3d": round((res["3d"]["x_cf"] - x_cf_true) / cam.px_per_cm, 1),
                "chest_width_cm_body": round((body_r - body_l) / cam.px_per_cm, 1),
                "chest_width_cm_3d": round((res["3d"]["extent"][1] - res["3d"]["extent"][0]) / cam.px_per_cm, 1),
                "chest_width_cm_ref": round((ref["extent"][1] - ref["extent"][0]) / cam.px_per_cm, 1)})
        rows.append(np.hstack(row))
    cv2.imwrite(str(OUT / f"placement_{gid}.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    return numbers


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, torsos = measurements()
    profile_plot(torsos)
    yaw = orientation(torsos)
    place = {gid: placement(torsos, d, gid) for d, gid in ((ROOT / "eval" / "garments_short", "fila_crop_tee"),
                                                            (ROOT / "data" / "garments", "colour_block_polo"))}
    (OUT / "results.json").write_text(json.dumps({"measurements": rows, "orientation": yaw, "placement": place},
                                                 indent=1, default=float))
    print(f"{'body':4s} {'landmark':8s} {'true':>6s} {'multi':>6s} {'single':>6s}   (girth, cm)")
    for r in rows:
        print(f"{r['body']:4s} {r['landmark']:8s} {r['true_girth']:6.1f} {r['multi_view']:6.1f} {r['single_view']:6.1f}"
              f"   depth true {r['true_depth']:.1f} multi {r['multi_depth']:.1f} single {r['single_depth']:.1f}")
    print("orientation:", yaw)
    for gid, nb in place.items():
        for n in nb:
            print(gid, n)


if __name__ == "__main__":
    main()
