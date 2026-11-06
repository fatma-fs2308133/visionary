"""Reconstruct a real person's torso from three photos (NOT yet validated).

    .venv/bin/python -m experiments.body3d.capture front.jpg side.jpg back.jpg --height 170 [--side-faces left]

Capture protocol (as body-scanning apps use):
  - whole body in frame, head to feet, camera at chest height, 2-3 m away
    (far enough that the photo is close to orthographic);
  - arms slightly away from the body (A-pose), fitted clothes;
  - side photo: a quarter turn, facing image right (or pass --side-faces left).

Silhouettes come from MediaPipe's person mask (pose landmarker), falling
back to the selfie segmenter when no pose is found (common in side views).
The known body height scales every view to centimetres.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tryon_core.parsing import LiveParser
from tryon_core.pose import PoseDetector

from .body import Camera, render, save_obj
from .reconstruct import landmarks, reconstruct, torso_parts


def silhouette(rgb, det, seg):
    pose = det.detect(rgb)
    if pose.mask is not None and pose.keypoints:
        return pose.mask > 0.5, "pose"
    return ~seg(rgb)["background"], "segmenter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("front")
    ap.add_argument("side")
    ap.add_argument("back", nargs="?")
    ap.add_argument("--height", type=float, required=True, help="standing height in cm")
    ap.add_argument("--side-faces", default="right", choices=("right", "left"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "capture"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det, seg = PoseDetector(video=False), LiveParser(video=False)
    masks, sources = {}, {}
    for view in ("front", "side", "back"):
        path = getattr(args, view)
        if not path:
            continue
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        m, src = silhouette(rgb, det, seg)
        if view == "side" and args.side_faces == "left":
            m = m[:, ::-1]
        masks[view], sources[view] = m, src
        cv2.imwrite(str(out / f"silhouette_{view}.png"), m.astype(np.uint8) * 255)
    torso = reconstruct(masks, args.height)
    report = {"silhouettes_from": sources}
    for name, y in landmarks(torso).items():
        a, b, _, _ = torso.section(y)
        report[name] = {"height_cm": round(y, 1), "girth_cm": round(torso.girth(y), 1),
                        "width_cm": round(2 * a, 1), "depth_cm": round(2 * b, 1)}
    (out / "measurements.json").write_text(json.dumps(report, indent=1))
    save_obj(out / "torso.obj", torso_parts(torso))
    view = render(torso_parts(torso), Camera(yaw=np.radians(35)), colour=(0.55, 0.68, 0.80))[0]
    cv2.imwrite(str(out / "torso_35deg.jpg"), cv2.cvtColor((view * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
