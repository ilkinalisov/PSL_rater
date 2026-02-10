# app.py - COMPLETE VERSION with all endpoints

import base64
import json
import os
import re
import traceback
import uuid
from datetime import datetime
from typing import Dict
from urllib.parse import urlparse

# Force CPU path in headless/container deploys before importing MediaPipe modules.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from facial_analysis import EnhancedFacialAnalyzer
from side_profile_analyzer import SideProfileAnalyzer
try:
    from v2.calibration.calibrator import LandmarkCalibrator
    from v2.edge_contour_tracer import trace_side_contours
    from v2.local_side_v2_adapter import LocalSideV2Adapter
    from v2.overlay_renderer import OVERLAY_RENDERER_VERSION
    from v2.scoring import compute_front_reliability, compute_scores_v2
    from v2.side_inference_client import SideInferenceClient, compare_side_v1_v2
except ImportError:  # pragma: no cover - supports package-style imports in tests/tools.
    from backend.v2.calibration.calibrator import LandmarkCalibrator
    from backend.v2.edge_contour_tracer import trace_side_contours
    from backend.v2.local_side_v2_adapter import LocalSideV2Adapter
    from backend.v2.overlay_renderer import OVERLAY_RENDERER_VERSION
    from backend.v2.scoring import compute_front_reliability, compute_scores_v2
    from backend.v2.side_inference_client import SideInferenceClient, compare_side_v1_v2

# SECURITY: PRIORITY 1
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_BASE64_SIZE = 15 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_BODY_SIZE = 25 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# SECURITY: PRIORITY 3
API_KEY = os.environ.get("API_KEY")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# V2 SERVICE CONFIG
SIDE_INFERENCE_URL = os.environ.get("SIDE_INFERENCE_URL", "").strip()
SIDE_INFERENCE_TIMEOUT_SECONDS = float(os.environ.get("V2_SIDE_TIMEOUT_SECONDS", "3.2"))
SIDE_INFERENCE_CB_FAILURE_THRESHOLD = int(os.environ.get("V2_CB_FAILURE_THRESHOLD", "3"))
SIDE_INFERENCE_CB_RECOVERY_SECONDS = float(os.environ.get("V2_CB_RECOVERY_SECONDS", "20"))
ENABLE_V2_SHADOW = os.environ.get("ENABLE_V2_SHADOW", "").strip().lower() in {"1", "true", "yes", "on"}
V2_CALIBRATION_MODEL_PATH = os.environ.get("V2_CALIBRATION_MODEL_PATH", "").strip()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def sanitize_for_json(obj):
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


def _build_vercel_preview_regex(frontend_url: str) -> str | None:
    """Build a tight preview regex from the configured Vercel host."""
    if not frontend_url:
        return None

    host = urlparse(frontend_url).hostname or ""
    if not host.endswith(".vercel.app"):
        return None

    project = host.split(".")[0]
    if not project:
        return None
    return rf"https://{re.escape(project)}.*\.vercel\.app"


# SECURITY: PRIORITY 6
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


# SECURITY: PRIORITY 7
class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_BODY_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"success": False, "error": "Request body too large."}
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Invalid Content-Length header."}
                )
        return await call_next(request)


class AnalysisNoStoreMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/analyze") or request.url.path.startswith("/v2/analyze"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response


# SECURITY: PRIORITY 2
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    storage_uri="memory://",
)

app = FastAPI(
    title="Facial PSL Analyzer API",
    description="API for analyzing facial proportions and calculating PSL scores",
    version="3.0.0"
)
app.state.limiter = limiter

# SECURITY: PRIORITY 4
_frontend_url = os.environ.get("FRONTEND_URL", "").strip()
_vercel_preview_regex = _build_vercel_preview_regex(_frontend_url)
_allow_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]
if _frontend_url:
    _allow_origins.append(_frontend_url)

# Middleware order (added): CORS -> Security headers -> Body limit -> SlowAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=_vercel_preview_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Client-Mode", "X-Client-Build"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MaxBodySizeMiddleware)
app.add_middleware(AnalysisNoStoreMiddleware)
app.add_middleware(SlowAPIMiddleware)

# Initialize analyzers
front_analyzer = EnhancedFacialAnalyzer()
side_analyzer = SideProfileAnalyzer()

# Initialize V2 inference adapter/client
v2_calibrator = LandmarkCalibrator()
if V2_CALIBRATION_MODEL_PATH:
    try:
        v2_calibrator.load(V2_CALIBRATION_MODEL_PATH)
        print(f"[V2] Loaded calibration model: {V2_CALIBRATION_MODEL_PATH}")
    except Exception as exc:
        print(f"[V2] Failed to load calibration model '{V2_CALIBRATION_MODEL_PATH}': {exc}")

v2_local_side_adapter = LocalSideV2Adapter(
    side_analyzer=side_analyzer,
    calibrator=v2_calibrator if v2_calibrator.is_ready else None,
)
v2_side_inference_client = SideInferenceClient(
    remote_url=SIDE_INFERENCE_URL,
    timeout_seconds=SIDE_INFERENCE_TIMEOUT_SECONDS,
    failure_threshold=SIDE_INFERENCE_CB_FAILURE_THRESHOLD,
    recovery_seconds=SIDE_INFERENCE_CB_RECOVERY_SECONDS,
    local_adapter=v2_local_side_adapter,
)


# SECURITY: PRIORITY 1
async def validate_and_read_image(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> np.ndarray:
    """Safely read and validate an uploaded image file."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP."
        )

    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_size // (1024 * 1024)}MB."
            )
        chunks.append(chunk)

    if total_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    contents = b"".join(chunks)
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image. File may be corrupted.")

    h, w = image.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / float(max(h, w))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    if h < 100 or w < 100:
        raise HTTPException(status_code=400, detail="Image too small. Minimum 100x100 pixels.")

    return image


# SECURITY: PRIORITY 3
async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key for non-public endpoints."""
    if API_KEY is None or API_KEY == "":
        return True
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return True


def _overlay_to_base64(overlay_image) -> str:
    if overlay_image is None:
        return ""
    ok, buffer = cv2.imencode(".png", overlay_image)
    if not ok:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def _pipeline_debug(endpoint_variant: str) -> Dict:
    return {
        "endpoint_variant": endpoint_variant,
        "overlay_renderer_version": OVERLAY_RENDERER_VERSION,
    }


def _run_side_v2(image: np.ndarray) -> Dict:
    """Run V2 side inference with remote->fallback handling."""
    result = v2_side_inference_client.analyze_side(image)
    if not isinstance(result, dict):
        raise ValueError("V2 side inference returned an invalid response type.")
    return result


def _extract_anchor_points_from_debug(side_debug: Dict) -> Dict:
    anchors = {}
    points = (side_debug or {}).get("landmark_points", {}) or {}
    for key in (
        "trichion",
        "glabella",
        "nasion",
        "pronasale",
        "subnasale",
        "menton",
        "gonion",
        "articulare",
        "tragion",
        "pogonion",
    ):
        val = points.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            anchors[key] = (int(val[0]), int(val[1]))
    return anchors


def _build_legacy_contours(image: np.ndarray, side_debug: Dict, measurements) -> Dict:
    anchors = _extract_anchor_points_from_debug(side_debug or {})
    if not anchors:
        return {
            "method": "landmark_fallback_v1",
            "confidence": 0.0,
            "silhouette": [],
            "jaw_ramus": [],
            "debug": {
                "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
                "was_mirrored": bool(getattr(measurements, "was_mirrored", False)),
                "processing_mode": "color",
                "roi": None,
                "fallback_reason": "missing_anchors",
            },
        }
    return trace_side_contours(
        image=image,
        anchor_points=anchors,
        jaw_solver_debug=(side_debug or {}).get("v2_diagnostics", {}).get("jawline_solver", {}),
        was_mirrored=bool(getattr(measurements, "was_mirrored", False)),
    )


def _normalize_v2_side_payload(v2_result: Dict) -> Dict:
    """
    Normalize local/remote V2 shape into endpoint-ready fields.
    Keeps behavior deterministic if remote service schema differs slightly.
    """
    landmarks_v2 = v2_result.get("landmarks_v2", {}) or {}
    quality_v2 = v2_result.get("quality_v2", {}) or {}
    measurements = v2_result.get("measurements", {}) or {}
    try:
        side_score = float(v2_result.get("score", 5.0))
    except Exception:
        side_score = 5.0
    if not np.isfinite(side_score):
        side_score = 5.0
    side_breakdown = v2_result.get("breakdown", {}) or {}
    overlay = v2_result.get("overlay_image")
    transport = v2_result.get("transport", {}) or {}
    debug = v2_result.get("debug", {}) or {}

    # Remote service may return base64 overlay directly.
    overlay_base64 = ""
    if isinstance(overlay, str):
        overlay_base64 = overlay
    else:
        overlay_base64 = _overlay_to_base64(overlay)

    global_conf = ((landmarks_v2.get("confidence", {}) or {}).get("global"))
    try:
        global_conf = float(global_conf)
    except Exception:
        global_conf = 0.35

    quality_v2.setdefault("uncertain_side_landmarks", bool(global_conf < 0.50))
    quality_v2.setdefault("pose_yaw", 80.0)
    quality_v2.setdefault("pose_pitch", 0.0)
    quality_v2.setdefault("occlusion_score", 0.5)
    quality_v2.setdefault("lighting_score", 0.5)
    quality_v2.setdefault("jawline_visibility_score", 0.0)
    quality_v2.setdefault("monochrome_score", 0.0)

    landmarks_v2.setdefault("points", {})
    landmarks_v2.setdefault("jaw_contour", [])
    contours = v2_result.get("contours") or landmarks_v2.get("contours") or {
        "method": "landmark_fallback_v1",
        "confidence": 0.0,
        "silhouette": [],
        "jaw_ramus": [],
        "debug": {
            "image_size": {"width": 0, "height": 0},
            "was_mirrored": False,
            "processing_mode": "color",
            "roi": None,
            "fallback_reason": "missing_contours",
        },
    }
    landmarks_v2.setdefault("contours", contours)
    landmarks_v2.setdefault("method", "local_legacy_mp_hybrid")
    landmarks_v2.setdefault("method_source", v2_result.get("method_source", transport.get("source", "local_fallback_legacy")))
    landmarks_v2.setdefault("overlay_source", "unknown")
    gonial_debug = landmarks_v2.setdefault("gonial_debug", {})
    gonial_debug.setdefault("raw_angle", measurements.get("raw_gonial_angle"))
    gonial_debug.setdefault("source", measurements.get("gonial_source", "default"))
    gonial_debug.setdefault("fallback_reason", measurements.get("gonial_fallback_reason", "default"))
    gonial_debug.setdefault("pre_validation_angle", measurements.get("pre_validation_gonial_angle"))
    gonial_debug.setdefault("pitch_deg", 0.0)
    gonial_debug.setdefault("processing_mode", "color")
    gonial_debug.setdefault("overlay_geometry_source", "straight_fallback")
    gonial_debug.setdefault("overlay_snap_mode", "off")
    gonial_debug.setdefault("overlay_path_quality", 0.0)
    gonial_debug.setdefault("ramus_display_mode", "straight_fallback")
    acceptance_gate = gonial_debug.setdefault("acceptance_gate", {})
    if isinstance(acceptance_gate, dict):
        acceptance_gate.setdefault("residual", None)
    landmarks_v2.setdefault("confidence", {
        "global": round(global_conf, 3),
        "articulare": round(global_conf, 3),
        "gonion": round(global_conf, 3),
        "menton": round(global_conf, 3),
    })

    return {
        "score": side_score,
        "breakdown": side_breakdown,
        "measurements": measurements,
        "landmarks_v2": landmarks_v2,
        "contours": contours,
        "quality_v2": quality_v2,
        "overlay_base64": overlay_base64,
        "transport": transport,
        "debug": debug,
        "method_source": landmarks_v2.get("method_source", v2_result.get("method_source", "local_fallback_legacy")),
        "global_confidence": float(np.clip(global_conf, 0.0, 1.0)),
    }


# SECURITY: PRIORITY 2
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    detail = getattr(exc, "detail", None)
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": "Too many requests. Please wait before trying again.",
            "retry_after": str(detail) if detail is not None else "rate_limit_exceeded"
        }
    )


# SECURITY: PRIORITY 5
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    print(f"[Error] {request.url.path}: {type(exc).__name__}: {str(exc)}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An internal error occurred. Please try again."
        }
    )


# SECURITY: PRIORITY 8
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up analyzer resources on shutdown."""
    try:
        front_analyzer.face_mesh.close()
    except Exception:
        pass
    try:
        side_analyzer.face_mesh.close()
    except Exception:
        pass

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "name": "Facial PSL Analyzer API",
        "version": "3.0.0",
        "status": "running"
    }

@app.get("/health")
@limiter.limit("30/minute")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "service": "facial-psl-analyzer",
        "v2": {
            "side_inference_remote_configured": bool(SIDE_INFERENCE_URL),
            "shadow_mode": ENABLE_V2_SHADOW,
            "circuit_open": bool(v2_side_inference_client.state.is_open),
            "runtime_mode": "gateway_remote_enabled" if bool(SIDE_INFERENCE_URL) else "local_fallback_only",
            "native_runtime_active": None,
            "overlay_renderer_version": OVERLAY_RENDERER_VERSION,
        }
    }

# LEGACY ENDPOINT - For backward compatibility with existing frontend
@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_legacy(
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """
    Legacy single image analysis endpoint
    Kept for backward compatibility
    """
    try:
        image = await validate_and_read_image(file)
        
        measurements, overlay_image, debug_info = front_analyzer.analyze_face(image)
        
        if measurements is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        psl_score, score_breakdown = front_analyzer.calculate_psl_score(measurements)
        
        _, buffer = cv2.imencode('.png', overlay_image)
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')
        
        measurements_dict = measurements_to_dict(measurements)
        
        response = {
            "success": True,
            "score": {
                "psl": psl_score,
                "interpretation": get_psl_interpretation(psl_score),
                "breakdown": score_breakdown
            },
            "gender": {
                "label": getattr(measurements, "gender", "unknown"),
                "confidence": float(getattr(measurements, "gender_confidence", 0.0))
            },
            "measurements": measurements_dict,
            "overlay": overlay_base64,
            "debug": {
                **(debug_info or {}),
                "pipeline": _pipeline_debug("v1_legacy"),
            },
            "image_info": {
                "dimensions": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                },
                "filename": file.filename
            },
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
        return sanitize_for_json(response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Legacy analyze failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed. Please try a different image.")

@app.post("/analyze/pair")
@limiter.limit("6/minute")
async def analyze_pair(
    request: Request,
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """Analyze both front and side images with overall PSL score"""
    try:
        front_img = await validate_and_read_image(front_image)
        side_img = await validate_and_read_image(side_image)
        
        # Analyze front
        front_measurements, front_overlay, front_debug = front_analyzer.analyze_face(front_img)
        if front_measurements is None:
            raise HTTPException(status_code=400, detail="No face in front image")
        
        # Analyze side (with mirroring)
        side_measurements, side_overlay, side_debug = side_analyzer.analyze_side_profile(side_img)
        if side_measurements is None:
            raise HTTPException(status_code=400, detail="No face in side image")
        side_contours = _build_legacy_contours(side_img, side_debug, side_measurements)
        
        # Calculate scores
        front_psl, front_breakdown = front_analyzer.calculate_psl_score(front_measurements)
        side_psl, side_breakdown = side_analyzer.calculate_side_profile_score(side_measurements)

        gender_label, gender_confidence = resolve_gender(front_measurements, side_measurements)

        shadow_debug = None
        if ENABLE_V2_SHADOW:
            try:
                shadow_v2_raw = _run_side_v2(side_img)
                shadow_v2 = _normalize_v2_side_payload(shadow_v2_raw)
                shadow_debug = compare_side_v1_v2(
                    legacy_measurements=measurements_to_dict(side_measurements),
                    v2_result={
                        "measurements": shadow_v2.get("measurements", {}),
                        "landmarks_v2": shadow_v2.get("landmarks_v2", {}),
                        "transport": shadow_v2.get("transport", {}),
                    },
                )
            except Exception as shadow_exc:
                shadow_debug = {"error": str(shadow_exc)}
        
        # Calculate overall PSL
        overall_psl, overall_breakdown = calculate_overall_psl(
            front_psl, front_breakdown,
            side_psl, side_breakdown,
            front_measurements, side_measurements
        )
        
        # Encode overlays
        _, front_buffer = cv2.imencode('.png', front_overlay)
        front_overlay_base64 = base64.b64encode(front_buffer).decode('utf-8')
        
        _, side_buffer = cv2.imencode('.png', side_overlay)
        side_overlay_base64 = base64.b64encode(side_buffer).decode('utf-8')
        
        response = {
            "success": True,
            "overall_score": {
                "psl": overall_psl,
                "interpretation": get_psl_interpretation(overall_psl),
                "category": get_psl_category(overall_psl),
                "breakdown": overall_breakdown
            },
            "gender": {
                "label": gender_label,
                "confidence": gender_confidence
            },
            "front_analysis": {
                "psl": front_psl,
                "interpretation": get_psl_interpretation(front_psl),
                "breakdown": front_breakdown,
                "measurements": measurements_to_dict(front_measurements),
                "gender": {
                    "label": getattr(front_measurements, "gender", "unknown"),
                    "confidence": float(getattr(front_measurements, "gender_confidence", 0.0))
                }
            },
            "side_analysis": {
                "psl": side_psl,
                "interpretation": get_psl_interpretation(side_psl),
                "breakdown": side_breakdown,
                "measurements": measurements_to_dict(side_measurements),
                "contours": side_contours,
                "was_mirrored": getattr(side_measurements, 'was_mirrored', False),
                "is_estimated": getattr(side_measurements, 'is_estimated', False),
                "gender": {
                    "label": getattr(side_measurements, "gender", "unknown"),
                    "confidence": float(getattr(side_measurements, "gender_confidence", 0.0))
                }
            },
            "overlays": {
                "front": front_overlay_base64,
                "side": side_overlay_base64
            },
            "debug": {
                "front": front_debug,
                "side": side_debug,
                "v2_shadow": shadow_debug,
                "pipeline": _pipeline_debug("v1_legacy"),
            },
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
        return sanitize_for_json(response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Pair analysis failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed. Please try a different image.")

@app.post("/analyze/front")
@limiter.limit("10/minute")
async def analyze_front(
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """Analyze front view only"""
    try:
        image = await validate_and_read_image(file)
        
        measurements, overlay_image, debug_info = front_analyzer.analyze_face(image)
        
        if measurements is None:
            raise HTTPException(status_code=400, detail="No face detected")
        
        psl_score, score_breakdown = front_analyzer.calculate_psl_score(measurements)
        
        _, buffer = cv2.imencode('.png', overlay_image)
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            "success": True,
            "score": {
                "psl": psl_score,
                "interpretation": get_psl_interpretation(psl_score),
                "category": get_psl_category(psl_score),
                "breakdown": score_breakdown
            },
            "gender": {
                "label": getattr(measurements, "gender", "unknown"),
                "confidence": float(getattr(measurements, "gender_confidence", 0.0))
            },
            "measurements": measurements_to_dict(measurements),
            "overlay": overlay_base64,
            "debug": {
                **(debug_info or {}),
                "pipeline": _pipeline_debug("v1_legacy"),
            },
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
        return sanitize_for_json(response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Front analysis failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed. Please try a different image.")

@app.post("/analyze/side")
@limiter.limit("10/minute")
async def analyze_side(
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """Analyze side profile only"""
    try:
        image = await validate_and_read_image(file)
        
        measurements, overlay_image, debug_info = side_analyzer.analyze_side_profile(image)
        
        if measurements is None:
            raise HTTPException(status_code=400, detail="No face detected")
        
        side_score, score_breakdown = side_analyzer.calculate_side_profile_score(measurements)
        side_contours = _build_legacy_contours(image, debug_info, measurements)
        shadow_debug = None
        if ENABLE_V2_SHADOW:
            try:
                shadow_v2_raw = _run_side_v2(image)
                shadow_v2 = _normalize_v2_side_payload(shadow_v2_raw)
                shadow_debug = compare_side_v1_v2(
                    legacy_measurements=measurements_to_dict(measurements),
                    v2_result={
                        "measurements": shadow_v2.get("measurements", {}),
                        "landmarks_v2": shadow_v2.get("landmarks_v2", {}),
                        "transport": shadow_v2.get("transport", {}),
                    },
                )
            except Exception as shadow_exc:
                shadow_debug = {"error": str(shadow_exc)}
        
        _, buffer = cv2.imencode('.png', overlay_image)
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            "success": True,
            "score": {
                "psl": side_score,
                "interpretation": get_psl_interpretation(side_score),
                "category": get_psl_category(side_score),
                "breakdown": score_breakdown,
                "was_mirrored": getattr(measurements, 'was_mirrored', False),
                "is_estimated": getattr(measurements, 'is_estimated', False)
            },
            "gender": {
                "label": getattr(measurements, "gender", "unknown"),
                "confidence": float(getattr(measurements, "gender_confidence", 0.0))
            },
            "measurements": measurements_to_dict(measurements),
            "contours": side_contours,
            "overlay": overlay_base64,
            "debug": {
                **(debug_info or {}),
                "v2_shadow": shadow_debug,
                "pipeline": _pipeline_debug("v1_legacy"),
            },
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
        return sanitize_for_json(response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Side analysis failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed. Please try a different image.")


@app.post("/v2/analyze/side")
@limiter.limit("10/minute")
async def analyze_side_v2(
    request: Request,
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """V2 side analysis using remote side-inference service with deterministic fallback."""
    try:
        image = await validate_and_read_image(file)
        v2_raw = _run_side_v2(image)
        v2 = _normalize_v2_side_payload(v2_raw)

        side_score = float(v2["score"])
        interpretation = get_psl_interpretation(side_score)
        category = get_psl_category(side_score)
        side_measurements = v2.get("measurements", {})

        scores_v2 = {
            "front_primary": None,
            "side_primary": round(side_score, 2),
            "overall_optional": round(side_score, 2),
            "overall_reliability": round(v2["global_confidence"], 3),
            "reliable": not bool(v2["quality_v2"].get("uncertain_side_landmarks", False)),
        }

        response = {
            "success": True,
            "score": {
                "psl": round(side_score, 1),
                "interpretation": interpretation,
                "category": category,
                "breakdown": v2["breakdown"],
            },
            "side_analysis": {
                "psl": round(side_score, 1),
                "interpretation": interpretation,
                "category": category,
                "breakdown": v2["breakdown"],
                "measurements": side_measurements,
                "landmarks_v2": v2["landmarks_v2"],
                "contours": v2["contours"],
                "quality_v2": v2["quality_v2"],
                "method_source": v2.get("method_source", "local_fallback_legacy"),
                "transport": v2["transport"],
                "was_mirrored": bool(side_measurements.get("was_mirrored", False)),
                "is_estimated": bool(side_measurements.get("is_estimated", False)),
            },
            "scores_v2": scores_v2,
            "overlay": v2["overlay_base64"],
            "debug": {
                "v2": v2["debug"],
                "transport": v2["transport"],
                "pipeline": _pipeline_debug("v2_side_service"),
            },
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
        }
        return sanitize_for_json(response)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] V2 side analysis failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="V2 side analysis failed. Please try a different image.")


@app.post("/v2/analyze/pair")
@limiter.limit("6/minute")
async def analyze_pair_v2(
    request: Request,
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    _: bool = Depends(verify_api_key)
):
    """V2 pair analysis with side landmarking from dedicated side inference service."""
    try:
        front_img = await validate_and_read_image(front_image)
        side_img = await validate_and_read_image(side_image)

        # Front branch (existing analyzer)
        front_measurements, front_overlay, front_debug = front_analyzer.analyze_face(front_img)
        if front_measurements is None:
            raise HTTPException(status_code=400, detail="No face in front image")
        front_psl, front_breakdown = front_analyzer.calculate_psl_score(front_measurements)
        front_overlay_base64 = _overlay_to_base64(front_overlay)

        # V2 side branch
        v2_raw = _run_side_v2(side_img)
        v2 = _normalize_v2_side_payload(v2_raw)
        side_psl = float(v2["score"])

        front_rel = compute_front_reliability(front_debug)
        scores_v2 = compute_scores_v2(
            front_primary=front_psl,
            side_primary=side_psl,
            front_reliability=front_rel,
            side_reliability=v2["global_confidence"],
        )

        overall_optional = scores_v2["overall_optional"]
        overall_interpretation = get_psl_interpretation(overall_optional)
        overall_category = get_psl_category(overall_optional)

        side_measurements_dict = v2.get("measurements", {})
        side_gender_label = str(side_measurements_dict.get("gender", "unknown"))
        side_gender_conf = float(side_measurements_dict.get("gender_confidence", 0.0) or 0.0)
        front_gender_label = str(getattr(front_measurements, "gender", "unknown"))
        front_gender_conf = float(getattr(front_measurements, "gender_confidence", 0.0) or 0.0)
        if front_gender_label == side_gender_label and front_gender_label != "unknown":
            gender_label = front_gender_label
            gender_confidence = round(min(1.0, (front_gender_conf + side_gender_conf) / 2.0 + 0.08), 3)
        elif front_gender_label != "unknown" and front_gender_conf >= side_gender_conf:
            gender_label = front_gender_label
            gender_confidence = round(front_gender_conf, 3)
        elif side_gender_label != "unknown":
            gender_label = side_gender_label
            gender_confidence = round(side_gender_conf, 3)
        else:
            gender_label = "unknown"
            gender_confidence = round(max(front_gender_conf, side_gender_conf), 3)

        response = {
            "success": True,
            "overall_score": {
                "psl": round(overall_optional, 1),
                "interpretation": overall_interpretation,
                "category": overall_category,
                "reliability": scores_v2["overall_reliability"],
                "reliable": scores_v2["reliable"],
                "breakdown": {
                    "front_psl": round(float(front_psl), 1),
                    "side_psl": round(float(side_psl), 1),
                    "combined_base": round(float(overall_optional), 1),
                    "harmony_bonus": 0.0,
                    "front_components": front_breakdown,
                    "side_components": v2["breakdown"],
                },
            },
            "scores_v2": scores_v2,
            "gender": {
                "label": gender_label,
                "confidence": gender_confidence,
            },
            "front_analysis": {
                "psl": round(front_psl, 1),
                "interpretation": get_psl_interpretation(front_psl),
                "breakdown": front_breakdown,
                "measurements": measurements_to_dict(front_measurements),
                "gender": {
                    "label": getattr(front_measurements, "gender", "unknown"),
                    "confidence": float(getattr(front_measurements, "gender_confidence", 0.0)),
                },
            },
            "side_analysis": {
                "psl": round(side_psl, 1),
                "interpretation": get_psl_interpretation(side_psl),
                "breakdown": v2["breakdown"],
                "measurements": v2["measurements"],
                "landmarks_v2": v2["landmarks_v2"],
                "contours": v2["contours"],
                "quality_v2": v2["quality_v2"],
                "method_source": v2.get("method_source", "local_fallback_legacy"),
                "transport": v2["transport"],
                "was_mirrored": bool(v2["measurements"].get("was_mirrored", False)),
                "is_estimated": bool(v2["measurements"].get("is_estimated", False)),
            },
            "overlays": {
                "front": front_overlay_base64,
                "side": v2["overlay_base64"],
            },
            "debug": {
                "front": front_debug,
                "side_v2": v2["debug"],
                "side_transport": v2["transport"],
                "pipeline": _pipeline_debug("v2_side_service"),
            },
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
        }
        return sanitize_for_json(response)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] V2 pair analysis failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="V2 pair analysis failed. Please try different images.")

@app.post("/analyze/capture")
@limiter.limit("10/minute")
async def analyze_camera_capture(
    request: Request,
    image_data: str = Form(...),
    _: bool = Depends(verify_api_key)
):
    """Analyze base64 image from camera capture"""
    try:
        # SECURITY: PRIORITY 1
        if len(image_data) > MAX_BASE64_SIZE:
            raise HTTPException(status_code=413, detail="Image data too large.")

        # Strip data URL prefix if present
        if image_data.startswith('data:image'):
            allowed_prefixes = ("data:image/jpeg", "data:image/png", "data:image/webp")
            if not image_data.startswith(allowed_prefixes):
                raise HTTPException(status_code=400, detail="Invalid image format in data URL.")
            parts = image_data.split(',', 1)
            if len(parts) != 2:
                raise HTTPException(status_code=400, detail="Invalid data URL payload.")
            image_data = parts[1]

        try:
            img_bytes = base64.b64decode(image_data, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        h, w = image.shape[:2]
        if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / float(max(h, w))
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        h, w = image.shape[:2]
        if h < 100 or w < 100:
            raise HTTPException(status_code=400, detail="Image too small. Minimum 100x100 pixels.")

        measurements, overlay_image, debug_info = front_analyzer.analyze_face(image)

        if measurements is None:
            raise HTTPException(status_code=400, detail="No face detected in captured image")

        psl_score, score_breakdown = front_analyzer.calculate_psl_score(measurements)

        _, buffer = cv2.imencode('.png', overlay_image)
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')

        response = {
            "success": True,
            "score": {
                "psl": psl_score,
                "interpretation": get_psl_interpretation(psl_score),
                "category": get_psl_category(psl_score),
                "breakdown": score_breakdown
            },
            "gender": {
                "label": getattr(measurements, "gender", "unknown"),
                "confidence": float(getattr(measurements, "gender_confidence", 0.0))
            },
            "measurements": measurements_to_dict(measurements),
            "overlay": overlay_base64,
            "debug": {
                **(debug_info or {}),
                "pipeline": _pipeline_debug("v1_legacy"),
            },
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }

        return sanitize_for_json(response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Camera capture analysis failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Analysis failed. Please try a different image.")

# ========== HELPER FUNCTIONS ==========

def calculate_overall_psl(front_psl, front_breakdown, side_psl, side_breakdown, 
                         front_measurements, side_measurements):
    """Calculate overall PSL combining front and side"""
    
    # Jawline is side-profile dependent; remove it from front-view aggregation.
    adjusted_front_breakdown = front_breakdown.copy()
    FRONT_WEIGHTS = {
        "symmetry": 0.30,
        "proportions": 0.26,
        "eyes": 0.20,
        "harmony": 0.14,
        "golden_ratio": 0.10
    }
    
    adjusted_front_psl = sum(
        adjusted_front_breakdown.get(comp, 5.0) * weight 
        for comp, weight in FRONT_WEIGHTS.items()
    )
    
    # Combine front and side more evenly, then apply a mild (not aggressive) range expansion.
    combined_psl = (adjusted_front_psl * 0.52) + (side_psl * 0.48)

    deviation = combined_psl - 5.5
    combined_psl = max(1.0, min(10.0, 5.5 + (deviation * 1.08)))
    
    # Harmony bonus
    score_diff = abs(adjusted_front_psl - side_psl)
    if score_diff < 0.4:
        harmony_bonus = 0.12
    elif score_diff < 0.8:
        harmony_bonus = 0.06
    else:
        harmony_bonus = 0
    
    final_psl = min(10.0, combined_psl + harmony_bonus)
    
    overall_breakdown = {
        "front_psl": round(adjusted_front_psl, 1),
        "side_psl": round(side_psl, 1),
        "combined_base": round(combined_psl, 1),
        "harmony_bonus": round(harmony_bonus, 2),
        "front_components": adjusted_front_breakdown,
        "side_components": side_breakdown,
        "key_metrics": {
            "gonial_angle": getattr(side_measurements, 'gonial_angle', None),
            "canthal_tilt": getattr(front_measurements, 'canthal_tilt_avg', None),
            "facial_symmetry": getattr(front_measurements, 'facial_symmetry_score', None),
            "profile_harmony": getattr(side_measurements, 'profile_harmony_score', None),
            "forward_growth_score": getattr(side_measurements, 'forward_growth_score', None),
            "facial_angle": getattr(side_measurements, 'facial_angle', None),
            "maxillary_prominence": getattr(side_measurements, 'maxillary_prominence', None),
            "mandibular_prominence": getattr(side_measurements, 'mandibular_prominence', None),
            "recession_type": getattr(side_measurements, 'recession_type', None),
            "gender": getattr(front_measurements, 'gender', getattr(side_measurements, 'gender', 'unknown'))
        }
    }
    
    return round(final_psl, 1), overall_breakdown

def resolve_gender(front_measurements, side_measurements):
    """Resolve a single gender estimate from front+side analyzers."""
    front_gender = getattr(front_measurements, "gender", "unknown")
    side_gender = getattr(side_measurements, "gender", "unknown")
    front_conf = float(getattr(front_measurements, "gender_confidence", 0.0))
    side_conf = float(getattr(side_measurements, "gender_confidence", 0.0))

    # If both agree on known label, boost confidence.
    if front_gender == side_gender and front_gender != "unknown":
        return front_gender, round(min(1.0, (front_conf + side_conf) / 2.0 + 0.08), 3)

    # Prefer known over unknown with higher confidence.
    candidates = []
    if front_gender != "unknown":
        candidates.append((front_gender, front_conf))
    if side_gender != "unknown":
        candidates.append((side_gender, side_conf))

    if not candidates:
        return "unknown", round(max(front_conf, side_conf), 3)

    candidates.sort(key=lambda x: x[1], reverse=True)
    label, conf = candidates[0]
    return label, round(conf, 3)

def get_psl_interpretation(score: float) -> str:
    """Get interpretation of PSL score"""
    if score >= 8.8:
        return "Exceptional - Model/Celebrity Tier"
    elif score >= 8.3:
        return "Outstanding - Top 1%"
    elif score >= 7.8:
        return "Excellent - Very Attractive"
    elif score >= 7.3:
        return "Very Good - Attractive"
    elif score >= 6.8:
        return "Good - Above Average"
    elif score >= 6.2:
        return "Decent - Slightly Above Average"
    elif score >= 5.6:
        return "Average - Normal"
    elif score >= 5.0:
        return "Below Average"
    else:
        return "Significantly Below Average"

def get_psl_category(score: float) -> str:
    """Get PSL category"""
    if score >= 8.2:
        return "Model Tier"
    elif score >= 7.2:
        return "High Tier Normie"
    elif score >= 6.2:
        return "Mid-High Tier Normie"
    elif score >= 5.3:
        return "Mid Tier Normie"
    elif score >= 4.4:
        return "Low Tier Normie"
    else:
        return "Below Average"

def measurements_to_dict(measurements):
    """Convert measurements to dict"""
    if hasattr(measurements, 'to_dict'):
        return measurements.to_dict()
    elif hasattr(measurements, '__dict__'):
        result = {}
        for key, value in measurements.__dict__.items():
            if isinstance(value, (np.float32, np.float64)):
                result[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                result[key] = int(value)
            elif isinstance(value, dict):
                result[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else 
                    int(v) if isinstance(v, (np.int32, np.int64)) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        return result
    else:
        return dict(measurements)




# if __name__ == "__main__":
#     import uvicorn
    
#     ssl_keyfile = "ssl/key.pem" if os.path.exists("ssl/key.pem") else None
#     ssl_certfile = "ssl/cert.pem" if os.path.exists("ssl/cert.pem") else None
    
#     uvicorn.run(
#         "app:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         ssl_keyfile=ssl_keyfile,
#         ssl_certfile=ssl_certfile
#     )
