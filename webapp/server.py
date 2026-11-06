"""Visionary try-on web app.

CMAI480 course project (Fall 2026), Augmented Reality & Virtual Try-On.
Team: Habiba Mohamed, Ashley Danoy, Supichsha Mokdad, Fatma Salama, Mahajuba Rahman.

    .venv/bin/uvicorn server:app --port 8000      then open http://localhost:8000

REST
  GET  /api/garments                  garment library
  POST /api/garments        (file)    add a garment photo (cut out + anchors)
  POST /api/photos          (file)    upload a person photo -> id + pose
  POST /api/tryon           (json)    instant (classical) try-on of a stored photo
  POST /api/studio          (json)    start a studio render (local hybrid) -> job id
  GET  /api/studio/{job}              progress / result of a studio render
  POST /api/studio/render   (json)    blocking studio render with base64 images
                                      (used by the MATLAB app via webwrite)
  POST /api/hd-tryon        (json)    hosted IDM-VTON (only if TRYON_HOSTED=1)
WebSocket
  /ws/live   client sends a JSON settings message, then JPEG frames (binary);
             the server answers every frame with a JSON stats message and
             the processed JPEG.
"""

import asyncio
import base64
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ai_tryon
from tryon_core.garment import load_garment, prepare_garment, save_garment
from tryon_core.parsing import LiveParser, default_cloth_parser
from tryon_core.pipeline import TryOnOptions, try_on
from tryon_core.pose import PoseDetector

ROOT = Path(__file__).resolve().parent
GARMENT_DIR = ROOT / "data" / "garments"
CATALOG_DIR = ROOT / "data" / "catalog"
MAX_PHOTO_SIDE = 1024
BODY_POINTS = ("leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
               "leftHip", "rightHip")
STUDIO_QUALITY = {"standard": ("vitonhd", 25),       # 512x384, ~35-60 s on an M1 Pro (default)
                  # Experimental, API only: 768x576 with the 1024-trained "mix" checkpoint,
                  # ~2.2 min. Sharper texture and cleaner necklines, but in our benchmark it
                  # shifted garment colours and once changed the garment type (eval/RESULTS.md).
                  "detail": ("mix", 25)}

app = FastAPI(title="Visionary try-on")
_photo_detector: PoseDetector | None = None
_photo_lock = threading.Lock()
_photos: OrderedDict[str, dict] = OrderedDict()     # last few uploaded photos
_garments: dict = {}
_parse_lock = threading.Lock()
_studio_lock = threading.Lock()                     # one diffusion job at a time
_studio_model = None
_jobs: OrderedDict[str, dict] = OrderedDict()


# ---------------------------------------------------------------------------
#  Garment library
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "garment"


def _seed_library():
    """Prepare every catalogue garment (data/catalog) that is not in the library
    yet. The H&M photos are not in git: python data/fetch_hm.py downloads them."""
    if not (CATALOG_DIR / "catalog.json").exists():
        return
    names = json.loads((CATALOG_DIR / "catalog.json").read_text())
    for file, entry in names.items():
        path = CATALOG_DIR / file
        if file.startswith("_") or not path.exists():
            continue
        name, sleeves = (entry, "") if isinstance(entry, str) else (entry["name"], entry.get("sleeves", ""))
        gid = _slug(name)
        if (GARMENT_DIR / f"{gid}.json").exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        try:
            rgba, anchors = prepare_garment(img)
        except ValueError:
            continue
        save_garment(GARMENT_DIR, gid, name, rgba, anchors, sleeves)
        if isinstance(entry, dict) and entry.get("hidden"):
            meta = GARMENT_DIR / f"{gid}.json"
            meta.write_text(json.dumps(dict(json.loads(meta.read_text()), hidden=True)))
        cv2.imwrite(str(GARMENT_DIR / f"{gid}_source.jpg"), img[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 95])


def _garment(gid: str):
    if gid not in _garments:
        if not (GARMENT_DIR / f"{gid}.json").exists():
            raise HTTPException(404, "Unknown garment")
        _garments[gid] = load_garment(GARMENT_DIR, gid)
    return _garments[gid]


@app.on_event("startup")
def _startup():
    global _photo_detector
    GARMENT_DIR.mkdir(parents=True, exist_ok=True)
    _seed_library()
    _photo_detector = PoseDetector(video=False)


@app.get("/api/garments")
def list_garments():
    out = []
    for meta in sorted(GARMENT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        info = json.loads(meta.read_text())
        if info.get("hidden"):              # benchmark garments (VITON-HD): kept on disk, not in the picker
            continue
        out.append({"id": meta.stem, "name": info["name"], "image": f"/garments/{meta.stem}.png"})
    return {"garments": out, "aiProvider": "CatVTON, local (Apple GPU)"}


@app.post("/api/garments")
async def add_garment(file: UploadFile = File(...)):
    img = _read_upload(await file.read(), unchanged=True)
    try:
        rgba, anchors = await asyncio.to_thread(prepare_garment, img)
    except ValueError as err:
        raise HTTPException(422, str(err))
    name = Path(file.filename or "garment").stem
    gid = f"{_slug(name)}_{uuid.uuid4().hex[:4]}"
    save_garment(GARMENT_DIR, gid, name.replace("_", " ").capitalize(), rgba, anchors)
    # The AI model works best on the untouched product photo
    bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(GARMENT_DIR / f"{gid}_source.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"id": gid, "name": name, "image": f"/garments/{gid}.png"}


@app.get("/garments/{gid}.png")
def garment_png(gid: str):
    path = GARMENT_DIR / f"{_slug(gid)}.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


# ---------------------------------------------------------------------------
#  Photos
# ---------------------------------------------------------------------------
def _read_upload(data: bytes, unchanged=False) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Could not read the image. Use a JPG or PNG (iPhone HEIC photos: export as JPG).")
    return img


def _to_data_url(rgb: np.ndarray, quality=90) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _detect_photo(rgb):
    with _photo_lock:                    # one MediaPipe graph, one photo at a time
        return _photo_detector.detect(rgb)


def _parse_photo(rgb):
    with _parse_lock:
        return default_cloth_parser()(rgb)


def _body_points(pose):
    return {k: [round(v[0], 1), round(v[1], 1), round(v[2], 2)] for k, v in pose.keypoints.items() if k in BODY_POINTS}


@app.post("/api/photos")
async def upload_photo(file: UploadFile = File(...)):
    bgr = _read_upload(await file.read())
    s = MAX_PHOTO_SIDE / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pose = await asyncio.to_thread(_detect_photo, rgb)
    parsing = await asyncio.to_thread(_parse_photo, rgb) if pose.keypoints else None
    pid = uuid.uuid4().hex[:12]
    _photos[pid] = {"rgb": rgb, "pose": pose, "parsing": parsing}
    while len(_photos) > 20:
        _photos.popitem(last=False)
    return {"id": pid, "width": rgb.shape[1], "height": rgb.shape[0],
            "personFound": bool(pose.keypoints), "confidence": round(pose.confidence, 2),
            "keypoints": _body_points(pose)}


def _photo(pid: str) -> dict:
    if pid not in _photos:
        raise HTTPException(404, "Photo expired, upload it again.")
    return _photos[pid]


class TryOnRequest(BaseModel):
    photoId: str
    garmentId: str
    options: dict = {}


@app.post("/api/tryon")
async def tryon_photo(req: TryOnRequest):
    photo, garment = _photo(req.photoId), _garment(req.garmentId)
    opts = TryOnOptions.from_dict({**req.options, "debug": True})
    out, info = await asyncio.to_thread(try_on, photo["rgb"], photo["pose"], garment, opts, photo["parsing"])
    stages = info.pop("stages", {})
    info.pop("_alpha", None)
    info.pop("_layer", None)
    views = {"pose": _to_data_url(_draw_pose(photo["rgb"], photo["pose"], info.get("torso")), 80)}
    views.update({k: _to_data_url(_fit_height(v, 360), 80) for k, v in stages.items()})
    return {"image": _to_data_url(out), "stages": views, **info}


def _fit_height(img, height):
    s = height / img.shape[0]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def _draw_pose(rgb, pose, torso):
    """Photo with the body points, upper arms and the torso quad."""
    out = rgb.copy()
    lw = max(3, rgb.shape[0] // 120)    # still visible after scaling to 360 px
    kp = pose.keypoints
    if torso:
        cv2.polylines(out, [np.int32(torso)], True, (255, 255, 255), lw, cv2.LINE_AA)
    for a, b in (("leftShoulder", "leftElbow"), ("rightShoulder", "rightElbow")):
        if a in kp and kp[a][2] > 0.5 and kp[b][2] > 0.5:
            cv2.line(out, np.int32(kp[a][:2]), np.int32(kp[b][:2]), (255, 255, 255), lw, cv2.LINE_AA)
    for x, y, v in (kp[k] for k in BODY_POINTS if k in kp):
        if v > 0.5:
            cv2.circle(out, (int(x), int(y)), 2 * lw, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), 2 * lw, (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
    return _fit_height(out, 360)


# ---------------------------------------------------------------------------
#  Studio render: local hybrid (our geometry + CatVTON on Apple MPS)
# ---------------------------------------------------------------------------
def _garment_source(gid: str) -> np.ndarray:
    """The untouched product photo if we have it, else the cut-out on white."""
    source = GARMENT_DIR / f"{gid}_source.jpg"
    if source.exists():
        return cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
    rgba = cv2.cvtColor(cv2.imread(str(GARMENT_DIR / f"{gid}.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    a = rgba[:, :, 3:4].astype(np.float32) / 255
    return (rgba[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)


def _studio(quality: str):
    """Load (or reuse) the CatVTON model for this quality level."""
    global _studio_model
    from hybrid.catvton import CatVTON
    ckpt, steps = STUDIO_QUALITY[quality]
    if _studio_model is None or _studio_model.checkpoint != ckpt:
        _studio_model = None
        _studio_model = CatVTON(checkpoint=ckpt)      # sliced attention: ~5.5 GB peak
    return _studio_model, steps


def _run_studio(rgb, pose, parsing, garment, source, quality, job=None):
    from hybrid.pipeline import studio_render

    def progress(f):
        if job is not None:
            job.update(stage="Rendering", progress=round(0.1 + 0.85 * f, 3))

    with _studio_lock:
        if job is not None:
            job.update(stage="Loading the model" if _studio_model is None else "Preparing", progress=0.05)
        model, steps = _studio(quality)
        if parsing is None:
            parsing = _parse_photo(rgb)
        out, info = studio_render(rgb, pose, parsing, garment, source, model, steps=steps,
                                  parser=default_cloth_parser(), progress=progress)
    return out, info


class StudioRequest(BaseModel):
    photoId: str
    garmentId: str
    quality: str = "standard"


@app.post("/api/studio")
async def studio_start(req: StudioRequest):
    photo, garment = _photo(req.photoId), _garment(req.garmentId)
    if not photo["pose"].keypoints:
        raise HTTPException(422, "No person found in this photo.")
    if req.quality not in STUDIO_QUALITY:
        raise HTTPException(422, f"Unknown quality {req.quality!r}.")
    jid = uuid.uuid4().hex[:10]
    job = {"status": "running", "stage": "Queued", "progress": 0.0, "started": time.time()}
    _jobs[jid] = job
    while len(_jobs) > 30:
        _jobs.popitem(last=False)

    def work():
        try:
            out, info = _run_studio(photo["rgb"], photo["pose"], photo["parsing"], garment,
                                    _garment_source(_slug(req.garmentId)), req.quality, job)
            job.update(status="done", progress=1.0, stage="Done", image=_to_data_url(out, 93),
                       seconds=info["seconds"], modelSeconds=info["model_seconds"])
        except Exception as err:            # report, never crash the server
            job.update(status="error", error=f"Studio render failed: {err}")

    threading.Thread(target=work, daemon=True).start()
    return {"jobId": jid}


@app.get("/api/studio/{jid}")
def studio_status(jid: str):
    if jid not in _jobs:
        raise HTTPException(404, "Unknown render job.")
    job = _jobs[jid]
    return {k: v for k, v in job.items() if k != "started"} | {"elapsed": round(time.time() - job["started"], 1)}


class StudioRenderRequest(BaseModel):
    person: str                 # base64 JPEG/PNG
    garmentId: str = ""         # a garment of the library, or
    garment: str = ""           # base64 product photo (PNG with alpha or JPEG)
    quality: str = "standard"


def _b64_image(data: str, unchanged=False) -> np.ndarray:
    if "," in data[:100]:
        data = data.split(",", 1)[1]
    return _read_upload(base64.b64decode(data), unchanged)


@app.post("/api/studio/render")
async def studio_render_sync(req: StudioRenderRequest):
    """For the MATLAB app: one blocking call, base64 in, base64 JPEG out."""
    bgr = _b64_image(req.person)
    s = MAX_PHOTO_SIDE / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pose = await asyncio.to_thread(_detect_photo, rgb)
    if not pose.keypoints:
        raise HTTPException(422, "No person found in the photo.")
    if req.garmentId and (GARMENT_DIR / f"{_slug(req.garmentId)}.json").exists():
        garment, source = _garment(_slug(req.garmentId)), _garment_source(_slug(req.garmentId))
    elif req.garment:
        img = _b64_image(req.garment, unchanged=True)
        try:
            rgba, anchors = await asyncio.to_thread(prepare_garment, img)
        except ValueError as err:
            raise HTTPException(422, str(err))
        from tryon_core.garment import Garment
        garment = Garment("upload", "upload", rgba[:, :, :3].astype(np.float32) / 255,
                          rgba[:, :, 3].astype(np.float32) / 255, anchors)
        source = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 4:          # transparent PNG: flatten on white
            a = img[:, :, 3:4].astype(np.float32) / 255
            source = (source * a + 255 * (1 - a)).astype(np.uint8)
    else:
        raise HTTPException(422, "Send garmentId (library) or garment (base64 image).")
    try:
        out, info = await asyncio.to_thread(_run_studio, rgb, pose, None, garment, source, req.quality)
    except Exception as err:
        raise HTTPException(500, f"Studio render failed: {err}")
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
    return {"image": base64.b64encode(buf.tobytes()).decode(), "seconds": info["seconds"]}


# ---------------------------------------------------------------------------
#  Hosted alternative (opt-in): IDM-VTON on Hugging Face / Replicate
# ---------------------------------------------------------------------------
class HDRequest(BaseModel):
    photoId: str
    garmentId: str
    description: str = ""


@app.post("/api/hd-tryon")
async def hd_tryon(req: HDRequest):
    if os.environ.get("TRYON_HOSTED") != "1":
        raise HTTPException(403, "Hosted try-on is off. The studio render runs locally (set TRYON_HOSTED=1 to enable).")
    photo = _photo(req.photoId)
    gid = _slug(req.garmentId)
    source = GARMENT_DIR / f"{gid}_source.jpg"
    if source.exists():                   # original product photo: opaque RGBA
        rgb = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    else:
        rgba = cv2.cvtColor(cv2.imread(str(GARMENT_DIR / f"{gid}.png"), cv2.IMREAD_UNCHANGED),
                            cv2.COLOR_BGRA2RGBA)
    desc = req.description or _garment(req.garmentId).name
    t = time.perf_counter()
    try:
        out = await asyncio.to_thread(ai_tryon.hd_try_on, photo["rgb"], rgba, desc)
    except ai_tryon.AITryOnError as err:
        raise HTTPException(502, str(err))
    return {"image": _to_data_url(out, 93), "seconds": round(time.perf_counter() - t, 1),
            "provider": ai_tryon.provider_name()}


# ---------------------------------------------------------------------------
#  Live video
# ---------------------------------------------------------------------------
@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    detector = await asyncio.to_thread(PoseDetector, True)
    parser = await asyncio.to_thread(LiveParser, True)
    settings = {"garmentId": None, "options": {}}
    # Body parsing (~130 ms on the M1's CPU; MediaPipe has no GPU delegate on
    # macOS) runs in the background: every frame uses the newest finished
    # parse, so the frame rate is not limited by it (masks lag ~1 parse).
    parsing, parse_task = None, None
    try:
        while True:
            msg = await ws.receive()
            if msg.get("text"):
                settings.update(json.loads(msg["text"]))
                continue
            data = msg.get("bytes")
            if data is None:
                if msg.get("type") == "websocket.disconnect":
                    break
                continue
            t0 = time.perf_counter()
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = time.perf_counter()
            pose = await asyncio.to_thread(detector.detect, rgb)
            pose_ms = (time.perf_counter() - t) * 1000
            if settings["garmentId"]:
                opts = TryOnOptions.from_dict(settings["options"])
                use = None
                if pose.keypoints and (opts.occlusion or opts.tps):
                    if parse_task is not None and parse_task.done():
                        if parse_task.exception() is None:
                            parsing = parse_task.result()
                        parse_task = None
                    if parse_task is None:
                        parse_task = asyncio.create_task(asyncio.to_thread(parser, rgb.copy(), dict(pose.keypoints)))
                    if parsing is not None and parsing["labels"].shape == rgb.shape[:2]:
                        use = parsing
                out, info = await asyncio.to_thread(try_on, rgb, pose, _garment(settings["garmentId"]), opts, use)
                info.pop("_layer", None)
                info.pop("_alpha", None)
            else:
                out, info = rgb, {"ok": False, "reason": "Pick a garment", "confidence": pose.confidence}
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            info["poseMs"] = round(pose_ms, 1)
            info["serverMs"] = round((time.perf_counter() - t0) * 1000, 1)
            info["keypoints"] = _body_points(pose)
            await ws.send_text(json.dumps(info))
            await ws.send_bytes(buf.tobytes())
    except WebSocketDisconnect:
        pass
    finally:
        if parse_task is not None:
            await asyncio.gather(parse_task, return_exceptions=True)
        detector.close()
        parser.close()


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
