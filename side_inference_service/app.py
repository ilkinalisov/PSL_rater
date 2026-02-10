import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

# Allow importing backend modules when this service is run from repo root.
ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from side_profile_analyzer import SideProfileAnalyzer  # noqa: E402
from v2.calibration.calibrator import LandmarkCalibrator  # noqa: E402
from v2.local_side_v2_adapter import LocalSideV2Adapter  # noqa: E402
from side_inference_service.engines.three_ddfa_engine import ThreeDDFAEngine  # noqa: E402


MAX_FILE_SIZE = int(os.environ.get("SIDE_SERVICE_MAX_FILE_SIZE", str(10 * 1024 * 1024)))
MAX_IMAGE_DIMENSION = int(os.environ.get("SIDE_SERVICE_MAX_IMAGE_DIMENSION", "4096"))
MODEL_PATH = os.environ.get("SIDE_SERVICE_CALIBRATION_MODEL_PATH", "").strip()

app = FastAPI(title="PSL Side Inference Service", version="1.0.0")

analyzer = SideProfileAnalyzer()
calibrator = LandmarkCalibrator()
if MODEL_PATH:
    try:
        calibrator.load(MODEL_PATH)
        print(f"[SideService] Loaded calibrator: {MODEL_PATH}")
    except Exception as exc:
        print(f"[SideService] Failed to load calibrator: {exc}")
adapter = LocalSideV2Adapter(analyzer, calibrator=calibrator if calibrator.is_ready else None)
remote_engine = ThreeDDFAEngine(side_analyzer=analyzer)


def _sanitize(obj):
    return json.loads(json.dumps(obj))


async def _read_image(file: UploadFile) -> np.ndarray:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    chunks = []
    total = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large.")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    img = cv2.imdecode(np.frombuffer(b"".join(chunks), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / float(max(h, w))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "side-inference",
        "engine": remote_engine.status,
    }


@app.post("/infer/side")
async def infer_side(image: UploadFile = File(...)):
    try:
        img = await _read_image(image)
        result = adapter.analyze(img)
        result = remote_engine.fuse(img, result)

        overlay = result.get("overlay_image")
        overlay_b64 = ""
        if overlay is not None:
            ok, buffer = cv2.imencode(".png", overlay)
            if ok:
                overlay_b64 = base64.b64encode(buffer).decode("utf-8")

        payload = {
            "score": result.get("score"),
            "breakdown": result.get("breakdown"),
            "measurements": result.get("measurements"),
            "landmarks_v2": result.get("landmarks_v2"),
            "quality_v2": result.get("quality_v2"),
            "debug": result.get("debug"),
            "overlay_image": overlay_b64,
            "method_source": result.get("method_source", "local_fallback_legacy"),
        }
        return _sanitize({"success": True, "result": payload})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Side inference failed: {exc}")
