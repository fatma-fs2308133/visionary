"""HD try-on with a diffusion model (IDM-VTON).

Takes a person photo and a garment photo and generates a realistic picture of
the person wearing the garment (folds, occlusion, old clothes replaced).
Not real time: one image takes ~20 s to a few minutes.

Providers, first one configured wins:
  REPLICATE_API_TOKEN  -> Replicate (cuuupid/idm-vton), paid, fast, reliable
  otherwise            -> the free Hugging Face Space yisol/IDM-VTON
                          (HF_TOKEN optional: a free account gets more GPU quota)
"""

import base64
import os
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

REPLICATE_MODEL = "cuuupid/idm-vton"
HF_SPACE = "yisol/IDM-VTON"


class AITryOnError(RuntimeError):
    pass


def provider_name() -> str:
    return "Replicate" if os.environ.get("REPLICATE_API_TOKEN") else "Hugging Face (free)"


def hd_try_on(person_rgb: np.ndarray, garment_rgba: np.ndarray, description: str = "a shirt") -> np.ndarray:
    """person_rgb: H x W x 3 uint8. garment_rgba: RGBA uint8 cut-out.
    Returns the generated image as RGB uint8."""
    garment_rgb = _flatten_on_white(garment_rgba)
    if os.environ.get("REPLICATE_API_TOKEN"):
        return _replicate(person_rgb, garment_rgb, description)
    return _hf_space(person_rgb, garment_rgb, description)


def _flatten_on_white(rgba):
    a = rgba[:, :, 3:4].astype(np.float32) / 255
    rgb = rgba[:, :, :3].astype(np.float32) * a + 255 * (1 - a)
    # The model expects a garment roughly centred on a 3:4 white canvas
    h, w = rgb.shape[:2]
    side_h = max(h, int(w * 4 / 3)) + 40
    side_w = int(side_h * 3 / 4)
    canvas = np.full((side_h, side_w, 3), 255, np.float32)
    y, x = (side_h - h) // 2, (side_w - w) // 2
    canvas[y:y + h, x:x + w] = rgb
    return canvas.astype(np.uint8)


def _jpeg(rgb, max_side=1024) -> bytes:
    s = max_side / max(rgb.shape[:2])
    if s < 1:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise AITryOnError("The model returned an unreadable image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
def _hf_space(person, garment, description):
    from gradio_client import Client, handle_file

    with tempfile.TemporaryDirectory() as tmp:
        p, g = Path(tmp, "person.jpg"), Path(tmp, "garment.jpg")
        p.write_bytes(_jpeg(person))
        g.write_bytes(_jpeg(garment))
        try:
            client = Client(HF_SPACE, token=os.environ.get("HF_TOKEN") or None, verbose=False)
            out, _mask = client.predict(
                {"background": handle_file(str(p)), "layers": [], "composite": None},
                handle_file(str(g)), description,
                True,    # auto-generate the clothing mask
                True,    # auto-crop the person (works with full-body photos)
                30, 42, api_name="/tryon")
        except Exception as err:   # quota, queue, Space asleep, ...
            msg = str(err)
            if "quota" in msg.lower() or "gpu" in msg.lower():
                msg = ("The free GPU quota of the Hugging Face Space is used up. Wait a while, set HF_TOKEN "
                       "(free account), or set REPLICATE_API_TOKEN.")
            raise AITryOnError(msg) from err
        return _decode(Path(out).read_bytes())


def _replicate(person, garment, description):
    token = os.environ["REPLICATE_API_TOKEN"]
    headers = {"Authorization": f"Bearer {token}"}

    def data_uri(rgb):
        return "data:image/jpeg;base64," + base64.b64encode(_jpeg(rgb)).decode()

    with httpx.Client(base_url="https://api.replicate.com/v1", headers=headers, timeout=60) as c:
        model = c.get(f"/models/{REPLICATE_MODEL}")
        model.raise_for_status()
        version = model.json()["latest_version"]["id"]
        r = c.post("/predictions", json={"version": version, "input": {
            "human_img": data_uri(person), "garm_img": data_uri(garment),
            "garment_des": description, "category": "upper_body", "crop": True,
            "steps": 30, "seed": 42}})
        if r.status_code >= 400:
            raise AITryOnError(f"Replicate: {r.text}")
        pred = r.json()
        deadline = time.time() + 300
        while pred["status"] not in ("succeeded", "failed", "canceled"):
            if time.time() > deadline:
                raise AITryOnError("Replicate took longer than 5 minutes.")
            time.sleep(2)
            pred = c.get(f"/predictions/{pred['id']}").json()
        if pred["status"] != "succeeded":
            raise AITryOnError(f"Replicate: {pred.get('error') or pred['status']}")
        url = pred["output"] if isinstance(pred["output"], str) else pred["output"][0]
        return _decode(c.get(url, headers={}).content)
