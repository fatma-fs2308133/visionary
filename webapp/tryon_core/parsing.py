"""Human / clothing parsing: which pixels are the old top, arms, face, hair,
background and lower body. Two back ends with the same output dict:

  ClothParser  SegFormer-B2 fine-tuned on ATR clothes
               (mattmdjaga/segformer_b2_clothes, 18 labels). ~1 s on CPU,
               used for photos and the studio render.
  LiveParser   MediaPipe multiclass selfie segmenter (hair, body-skin,
               face-skin, clothes, others). A few ms per frame, used live.

Output (all H x W bool): background, upper (top/dress), arms (skin of the
arms; LiveParser: all body skin), face (face skin only), hair, lower
(trousers/skirt/legs). SegFormer's "face" label includes the neck; see
face_above_chin() for how that is handled.
"""

from functools import lru_cache

import cv2
import httpx
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .pose import MODEL_DIR

SEGFORMER = "mattmdjaga/segformer_b2_clothes"
MULTICLASS_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                  "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")


class ClothParser:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.proc = SegformerImageProcessor.from_pretrained(SEGFORMER)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER).eval().to(self.device)
        self.ids = {v.lower().replace("-", "_").replace(" ", "_"): int(k)
                    for k, v in self.model.config.id2label.items()}

    def __call__(self, rgb: np.ndarray) -> dict:
        torch = self.torch
        with torch.no_grad():
            x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
            logits = self.model(pixel_values=x).logits
            up = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
            lab = up.argmax(1)[0].cpu().numpy()

        def any_of(*names):
            return np.isin(lab, [self.ids[n] for n in names if n in self.ids])

        return {"background": lab == self.ids["background"],
                "upper": any_of("upper_clothes", "dress"),
                "arms": any_of("left_arm", "right_arm"),
                "face": any_of("face", "sunglasses"),
                "hair": any_of("hair", "hat"),
                "lower": any_of("pants", "skirt", "left_leg", "right_leg", "belt"),
                "labels": lab}


class LiveParser:
    """MediaPipe multiclass segmenter. "upper" = clothes above the hip line."""

    def __init__(self, video: bool = True):
        path = MODEL_DIR / "selfie_multiclass_256x256.tflite"
        if not path.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            r = httpx.get(MULTICLASS_URL, follow_redirects=True, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        self.video = video
        self._t = 0
        opts = vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            output_category_mask=True, output_confidence_masks=False)
        self._seg = vision.ImageSegmenter.create_from_options(opts)

    def __call__(self, rgb: np.ndarray, keypoints: dict | None = None) -> dict:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            self._t += 33
            res = self._seg.segment_for_video(image, self._t)
        else:
            res = self._seg.segment(image)
        lab = res.category_mask.numpy_view()
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        clothes = lab == 4
        hip_y = rgb.shape[0]
        if keypoints and "leftHip" in keypoints:
            hip_y = int((keypoints["leftHip"][1] + keypoints["rightHip"][1]) / 2 + 0.08 * rgb.shape[0])
        rows = np.arange(rgb.shape[0])[:, None]
        return {"background": lab == 0, "upper": clothes & (rows < hip_y), "arms": lab == 2,
                "face": lab == 3, "hair": lab == 1, "lower": clothes & (rows >= hip_y), "labels": lab}

    def close(self):
        self._seg.close()


def face_above_chin(parsing: dict, keypoints: dict) -> np.ndarray:
    """The face without the neck. SegFormer labels the neck as "face"; the chin
    line is estimated from the pose (mouth + 0.9 x eye-to-mouth distance)."""
    face = parsing["face"]
    need = ("leftEye", "rightEye", "mouthLeft", "mouthRight")
    if not all(k in keypoints for k in need):
        return face
    eye_y = (keypoints["leftEye"][1] + keypoints["rightEye"][1]) / 2
    mouth_y = (keypoints["mouthLeft"][1] + keypoints["mouthRight"][1]) / 2
    chin = int(mouth_y + 0.9 * max(mouth_y - eye_y, 1))
    out = face.copy()
    out[max(chin, 0):] = False
    return out


@lru_cache(maxsize=1)
def default_cloth_parser() -> ClothParser:
    return ClothParser()


def forearms(parsing: dict, keypoints: dict) -> np.ndarray:
    """Arm pixels below the elbows: a top may cover the upper arm, never these."""
    arms = parsing["arms"].copy()
    ys = [keypoints[k][1] for k in ("leftElbow", "rightElbow") if k in keypoints and keypoints[k][2] > 0.5]
    if ys:
        arms[: int(min(ys)), :] = False
    return arms


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
