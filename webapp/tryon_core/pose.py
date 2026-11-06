"""Body keypoints + person mask with MediaPipe Pose Landmarker.

Replaces HRNet + background subtraction from the MATLAB version: one model
gives the shoulders, elbows and hips AND a person segmentation mask, so the
body-shape fit works without an empty-background photo.
"""

from dataclasses import dataclass
from pathlib import Path

import httpx
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URLS = {
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}
# MediaPipe landmark indices (person's left/right)
LANDMARKS = {"leftShoulder": 11, "rightShoulder": 12, "leftElbow": 13, "rightElbow": 14,
             "leftWrist": 15, "rightWrist": 16, "leftHip": 23, "rightHip": 24,
             "nose": 0, "leftEye": 2, "rightEye": 5, "mouthLeft": 9, "mouthRight": 10}


@dataclass
class PoseResult:
    keypoints: dict          # name -> (x, y, visibility) in pixels
    mask: np.ndarray | None  # H x W float32 person probability, or None
    world: dict | None = None  # name -> (x, y, z) in metres, hip-centred (MediaPipe world landmarks)

    def torso_yaw(self) -> float:
        """Rotation of the shoulder line about the vertical axis, in radians
        (0 = facing the camera), from the 3D world landmarks."""
        if not self.world:
            return 0.0
        l, r = np.array(self.world["leftShoulder"]), np.array(self.world["rightShoulder"])
        d = l - r
        return float(np.arctan2(d[2], d[0]))

    @property
    def confidence(self) -> float:
        torso = ("leftShoulder", "rightShoulder", "leftHip", "rightHip")
        if not self.keypoints:
            return 0.0
        return float(np.mean([self.keypoints[k][2] for k in torso]))


def _model_path(variant: str) -> Path:
    path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(MODEL_URLS[variant], follow_redirects=True, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


class PoseDetector:
    """video=False: independent photos. video=True: a live stream (uses
    tracking between frames, so it is faster and steadier)."""

    def __init__(self, video: bool = False, variant: str | None = None):
        variant = variant or ("lite" if video else "full")
        self.video = video
        self._t = 0
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(_model_path(variant))),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> PoseResult:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            self._t += 33
            res = self._landmarker.detect_for_video(image, self._t)
        else:
            res = self._landmarker.detect(image)
        if not res.pose_landmarks:
            return PoseResult({}, None)
        h, w = rgb.shape[:2]
        lms = res.pose_landmarks[0]
        kps = {name: (lms[i].x * w, lms[i].y * h, float(lms[i].visibility or 0.0))
               for name, i in LANDMARKS.items()}
        mask = res.segmentation_masks[0].numpy_view().astype(np.float32) if res.segmentation_masks else None
        if mask is not None and mask.ndim == 3:
            mask = mask[:, :, 0]
        world = None
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks[0]
            world = {name: (wl[i].x, wl[i].y, wl[i].z) for name, i in LANDMARKS.items()}
        return PoseResult(kps, mask, world)

    def close(self):
        self._landmarker.close()
