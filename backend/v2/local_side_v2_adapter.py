"""Local fallback adapter that converts existing analyzer output into V2 schema."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .calibration.calibrator import LandmarkCalibrator
from .geometry import (
    build_default_jaw_contour,
    check_landmark_invariants,
    contour_curvature_score,
    gonial_angle_from_points,
    subsample_polyline,
)
from .jawline_solver import solve_jawline_contour
from .overlay_renderer import OVERLAY_RENDERER_VERSION, render_side_overlay


def _as_plain_dict(obj: Any) -> Dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _lighting_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    exposure = 1.0 - min(1.0, abs(mean_val - 128.0) / 128.0)
    contrast = min(1.0, std_val / 64.0)
    return float(np.clip((0.55 * exposure) + (0.45 * contrast), 0.0, 1.0))


def _pose_yaw(points: Dict[str, Tuple[int, int]]) -> float:
    # Side-focused estimate from profile depth vs face height.
    if not all(k in points for k in ("tragion", "pronasale", "nasion", "menton")):
        return 80.0
    depth = abs(float(points["tragion"][0]) - float(points["pronasale"][0]))
    height = abs(float(points["menton"][1]) - float(points["nasion"][1]))
    ratio = depth / max(height, 1.0)
    yaw = 45.0 + (ratio * 45.0)
    return float(np.clip(yaw, 35.0, 95.0))


def _pose_pitch(points: Dict[str, Tuple[int, int]]) -> float:
    if not all(k in points for k in ("subnasale", "menton")):
        return 0.0
    sn = points["subnasale"]
    me = points["menton"]
    vec = np.array([float(me[0] - sn[0]), float(me[1] - sn[1])], dtype=float)
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return float(np.clip(angle - 72.0, -25.0, 25.0))


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _safe_float(value: Any, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    if not _is_finite_number(value):
        val = float(default)
    else:
        val = float(value)
    if lo is not None:
        val = max(float(lo), val)
    if hi is not None:
        val = min(float(hi), val)
    return float(val)


def _blend_xy(base: Tuple[int, int], refined: Tuple[int, int], alpha: float) -> Tuple[int, int]:
    a = float(np.clip(alpha, 0.0, 1.0))
    x = (float(base[0]) * (1.0 - a)) + (float(refined[0]) * a)
    y = (float(base[1]) * (1.0 - a)) + (float(refined[1]) * a)
    return (int(round(x)), int(round(y)))


class LocalSideV2Adapter:
    def __init__(
        self,
        side_analyzer,
        calibrator: Optional[LandmarkCalibrator] = None,
        uncertainty_threshold: float = 0.45,
    ):
        self.side_analyzer = side_analyzer
        self.calibrator = calibrator
        self.uncertainty_threshold = uncertainty_threshold

    def analyze(self, image: np.ndarray) -> Dict:
        measurements, overlay, debug = self.side_analyzer.analyze_side_profile(image)
        if measurements is None:
            raise ValueError("No face detected for side profile.")

        m = _as_plain_dict(measurements)
        debug = debug or {}

        points = {}
        point_map = debug.get("landmark_points", {})
        for key in (
            "articulare",
            "gonion",
            "menton",
            "nasion",
            "subnasale",
            "pogonion",
            "tragion",
            "pronasale",
            "trichion",
            "glabella",
            "labrale_superius",
            "labrale_inferius",
        ):
            if key in point_map and point_map[key] is not None and len(point_map[key]) >= 2:
                points[key] = (int(point_map[key][0]), int(point_map[key][1]))

        # Enforce required points.
        if "pogonion" not in points and "menton" in points:
            points["pogonion"] = points["menton"]
        if "articulare" not in points and "tragion" in points and "gonion" in points:
            points["articulare"] = (int(points["gonion"][0]), int(points["tragion"][1]))

        raw_jaw = debug.get("jaw_contour", [])
        jaw_contour = subsample_polyline(raw_jaw, step=2) if raw_jaw else build_default_jaw_contour(points)

        gonion_conf = _safe_float(debug.get("gonion_confidence", 0.35), 0.35, 0.0, 1.0)
        missing_ratio = _safe_float(debug.get("missing_landmark_ratio", 0.35), 0.35, 0.0, 1.0)
        menton_conf = _safe_float(debug.get("landmark_confidences", {}).get("menton", gonion_conf), gonion_conf, 0.0, 1.0)
        articulare_conf = _safe_float(debug.get("landmark_confidences", {}).get("tragion", gonion_conf), gonion_conf, 0.0, 1.0)

        gonial_debug = {
            "raw_angle": m.get("raw_gonial_angle"),
            "pre_validation_angle": m.get("pre_validation_gonial_angle"),
            "source": m.get("gonial_source", debug.get("gonion_detection_method", "legacy")),
            "fallback_reason": m.get("gonial_fallback_reason", "default"),
            "pitch_deg": 0.0,
            "processing_mode": "color",
            "overlay_geometry_source": "straight_fallback",
            "overlay_snap_mode": "off",
            "overlay_path_quality": 0.0,
            "ramus_display_mode": "straight_fallback",
            "acceptance_gate": {},
        }

        jawline_result = solve_jawline_contour(image, points)
        jawline_visibility = float(np.clip(jawline_result.visibility_score, 0.0, 1.0))
        pitch_deg = _safe_float((jawline_result.debug or {}).get("pitch_deg"), _pose_pitch(points), -30.0, 30.0)
        processing_mode = str((jawline_result.debug or {}).get("processing_mode", "color"))
        monochrome_score = _safe_float((jawline_result.debug or {}).get("monochrome_score"), 0.0, 0.0, 1.0)
        fit_residual = _safe_float((jawline_result.debug or {}).get("fit_residual"), 99.0, 0.0, 999.0)
        gonial_debug["pitch_deg"] = pitch_deg
        gonial_debug["processing_mode"] = processing_mode

        acceptance_threshold = 0.42
        residual_threshold = 12.0
        if processing_mode == "mono":
            residual_threshold = 14.0
        elif processing_mode == "hybrid":
            residual_threshold = 13.0
        if processing_mode in ("mono", "hybrid") and monochrome_score >= 0.58 and fit_residual <= 8.5:
            acceptance_threshold = 0.36
        candidate_points = dict(points)
        if jawline_result.points:
            candidate_points.update(jawline_result.points)
        candidate_invariants = check_landmark_invariants(candidate_points)
        candidate_gonial = None
        candidate_gonial_valid = False
        if all(k in candidate_points for k in ("articulare", "gonion", "menton")):
            candidate_gonial = gonial_angle_from_points(
                candidate_points["articulare"],
                candidate_points["gonion"],
                candidate_points["menton"],
            )
            candidate_gonial_valid = (
                candidate_gonial is not None and
                _is_finite_number(candidate_gonial) and
                95.0 <= float(candidate_gonial) <= 150.0
            )

        residual_ok = fit_residual <= residual_threshold
        jawline_accepted = (
            bool(jawline_result.points) and
            jawline_visibility >= acceptance_threshold and
            bool(candidate_invariants.get("go_posterior_to_me", False)) and
            bool(candidate_invariants.get("ar_superior_to_go", False)) and
            bool(candidate_gonial_valid) and
            bool(residual_ok)
        )

        gonial_debug["acceptance_gate"] = {
            "threshold": acceptance_threshold,
            "threshold_used": acceptance_threshold,
            "visibility": round(jawline_visibility, 3),
            "pitch_deg": round(pitch_deg, 3),
            "processing_mode": processing_mode,
            "monochrome_score": round(monochrome_score, 3),
            "residual": round(float(fit_residual), 3),
            "residual_threshold": round(float(residual_threshold), 3),
            "residual_ok": bool(residual_ok),
            "invariants_ok": bool(
                candidate_invariants.get("go_posterior_to_me", False) and
                candidate_invariants.get("ar_superior_to_go", False)
            ),
            "gonial_valid": bool(candidate_gonial_valid),
            "accepted": bool(jawline_accepted),
        }

        fallback_reason = "low_confidence"
        fallback_blended = False
        if jawline_accepted:
            points = candidate_points
            if "menton" in points:
                points["pogonion"] = points.get("pogonion", points["menton"])
            if jawline_result.jaw_contour:
                jaw_contour = subsample_polyline(jawline_result.jaw_contour, step=2)
            gonial_debug["source"] = jawline_result.source
            gonial_debug["fallback_reason"] = "none"
            if candidate_gonial is not None:
                gonial_debug["pre_validation_angle"] = float(candidate_gonial)
                gonial_debug["raw_angle"] = float(candidate_gonial)
            gonion_conf = float(np.clip(max(gonion_conf, 0.45 + (0.45 * jawline_visibility)), 0.0, 1.0))
            menton_conf = float(np.clip(max(menton_conf, 0.40 + (0.40 * jawline_visibility)), 0.0, 1.0))
            articulare_conf = float(np.clip(max(articulare_conf, 0.38 + (0.38 * jawline_visibility)), 0.0, 1.0))
        else:
            if processing_mode == "mono" and (jawline_visibility < 0.36 or monochrome_score >= 0.72):
                fallback_reason = "mono_low_texture"
            if not residual_ok:
                fallback_reason = "contour_residual_high"
            if candidate_gonial is None or (not candidate_gonial_valid):
                fallback_reason = "mp_invalid" if fallback_reason != "mono_low_texture" else fallback_reason

            can_blend_fallback = (
                bool(jawline_result.points) and
                bool(candidate_invariants.get("go_posterior_to_me", False)) and
                bool(candidate_invariants.get("ar_superior_to_go", False))
            )
            if can_blend_fallback:
                alpha = 0.22 if not residual_ok else 0.30
                for key in ("menton", "gonion", "articulare"):
                    if key in points and key in candidate_points:
                        points[key] = _blend_xy(points[key], candidate_points[key], alpha)
                if "menton" in points and "gonion" in points and points["gonion"][0] < points["menton"][0]:
                    points["gonion"] = (points["menton"][0] + 2, points["gonion"][1])
                if "articulare" in points and "gonion" in points and points["articulare"][1] > points["gonion"][1]:
                    points["articulare"] = (points["articulare"][0], points["gonion"][1] - 6)
                if jawline_result.jaw_contour:
                    jaw_contour = subsample_polyline(jawline_result.jaw_contour, step=2)
                fallback_blended = True

            if gonial_debug["fallback_reason"] in (None, "none", "default"):
                gonial_debug["fallback_reason"] = fallback_reason
            if fallback_blended:
                gonial_debug["source"] = "contour_tangent_fallback"

        jaw_curvature = contour_curvature_score(jaw_contour)
        quality = {
            "pose_yaw": round(_pose_yaw(points), 2),
            "lighting_score": round(_lighting_score(image), 3),
            "occlusion_score": round(float(np.clip(1.0 - ((0.75 * missing_ratio) + (0.25 * (1.0 - gonion_conf))), 0.0, 1.0)), 3),
            "jawline_visibility_score": round(jawline_visibility, 3),
            "monochrome_score": round(monochrome_score, 3),
        }
        pose_pitch = _pose_pitch(points)
        quality["pose_pitch"] = round(pose_pitch, 2)

        context = {
            "pose_yaw": quality["pose_yaw"],
            "pose_pitch": quality["pose_pitch"],
            "occlusion_score": quality["occlusion_score"],
            "lighting_score": quality["lighting_score"],
            "global_confidence": gonion_conf,
            "articulare_confidence": articulare_conf,
            "gonion_confidence": gonion_conf,
            "menton_confidence": menton_conf,
            "jaw_curvature": jaw_curvature,
            "jawline_visibility_score": jawline_visibility,
            "monochrome_score": monochrome_score,
        }

        point_conf = {
            "articulare": float(np.clip(articulare_conf, 0.0, 1.0)),
            "gonion": float(np.clip(gonion_conf, 0.0, 1.0)),
            "menton": float(np.clip(menton_conf, 0.0, 1.0)),
        }
        global_conf = float(np.clip(
            (0.38 * point_conf["gonion"]) +
            (0.22 * point_conf["menton"]) +
            (0.18 * point_conf["articulare"]) +
            (0.12 * (1.0 - missing_ratio)) +
            (0.10 * jawline_visibility),
            0.0,
            1.0
        ))

        if self.calibrator and self.calibrator.is_ready:
            corrected, conf_adj = self.calibrator.apply(points, context)
            points.update(corrected)
            for key, val in conf_adj.items():
                if key in point_conf:
                    point_conf[key] = float(np.clip(point_conf[key] * val, 0.0, 1.0))
            global_conf = float(np.clip(np.mean(list(point_conf.values())), 0.0, 1.0))

        invariants = check_landmark_invariants(points)
        if not invariants.get("go_posterior_to_me", True):
            global_conf = max(0.0, global_conf - 0.12)
        if not invariants.get("ar_superior_to_go", True):
            global_conf = max(0.0, global_conf - 0.10)

        # Recompute gonial from corrected points where possible.
        gonial_v2 = None
        gonial_valid = False
        if all(k in points for k in ("articulare", "gonion", "menton")):
            gonial_v2 = gonial_angle_from_points(points["articulare"], points["gonion"], points["menton"])
            gonial_debug["pre_validation_angle"] = float(gonial_v2) if gonial_v2 is not None else gonial_debug["pre_validation_angle"]
            if gonial_v2 is not None and 95.0 <= float(gonial_v2) <= 150.0:
                gonial_valid = True
                m["gonial_angle"] = float(gonial_v2)
                gonial_debug["raw_angle"] = float(gonial_v2)
                if jawline_accepted and gonial_debug["source"] in ("default", "mediapipe", "shape_fallback"):
                    gonial_debug["source"] = "jawline_contour"
                gonial_debug["fallback_reason"] = "none"
                m["is_estimated"] = bool(m.get("is_estimated", False))
            else:
                if gonial_debug["fallback_reason"] in (None, "none", "default"):
                    gonial_debug["fallback_reason"] = "mp_invalid"
                m["is_estimated"] = True

        high_pitch = abs(float(pose_pitch)) >= 12.0
        uncertain = (
            global_conf < self.uncertainty_threshold or
            (jawline_visibility < 0.20 and gonial_valid is False) or
            (high_pitch and jawline_visibility < 0.52) or
            (processing_mode == "mono" and jawline_visibility < 0.40) or
            (not gonial_valid)
        )
        if uncertain:
            m["is_estimated"] = True
            global_conf = float(np.clip(global_conf * 0.88, 0.0, 1.0))

        quality.update({
            "uncertain_side_landmarks": uncertain,
        })

        # Required v2 point contract.
        required_points = {
            "articulare": list(points.get("articulare", (0, 0))),
            "gonion": list(points.get("gonion", (0, 0))),
            "menton": list(points.get("menton", (0, 0))),
            "nasion": list(points.get("nasion", (0, 0))),
            "subnasale": list(points.get("subnasale", (0, 0))),
            "pogonion": list(points.get("pogonion", points.get("menton", (0, 0)))),
        }

        m["gonial_source"] = str(gonial_debug.get("source", m.get("gonial_source", "default")))
        m["gonial_fallback_reason"] = str(gonial_debug.get("fallback_reason", m.get("gonial_fallback_reason", "default")))
        m["raw_gonial_angle"] = gonial_debug.get("raw_angle")
        m["pre_validation_gonial_angle"] = gonial_debug.get("pre_validation_angle")
        m["gonial_angle"] = _safe_float(m.get("gonial_angle"), 120.0, 95.0, 150.0)
        m["profile_harmony_score"] = _safe_float(m.get("profile_harmony_score"), 5.0, 0.0, 10.0)
        m["forward_growth_score"] = _safe_float(m.get("forward_growth_score"), 5.0, 0.0, 10.0)

        # Keep scoring aligned with finalized landmark/angle outputs.
        for field in (
            "gonial_angle",
            "is_estimated",
            "gonial_source",
            "gonial_fallback_reason",
            "raw_gonial_angle",
            "pre_validation_gonial_angle",
            "gender",
            "gender_confidence",
        ):
            if field in m:
                setattr(measurements, field, m[field])
        side_score, side_breakdown = self.side_analyzer.calculate_side_profile_score(measurements)
        if not _is_finite_number(side_score):
            side_score = 5.0
            side_breakdown = {
                "gonial_angle": 5.0,
                "nasolabial": 5.0,
                "facial_convexity": 5.0,
                "vertical_balance": 5.0,
                "profile_harmony": 5.0,
                "forward_growth": 5.0,
            }
        m = _as_plain_dict(measurements)

        method_name = "jawline_contour_fused_local" if jawline_accepted and gonial_valid else "local_legacy_mp_hybrid"
        method_source = "local_fallback_legacy"
        overlay_source = "v2_landmarks_renderer"
        overlay_image = overlay
        try:
            overlay_image = render_side_overlay(
                image.copy(),
                points,
                jaw_contour,
                gonial_debug=gonial_debug,
                processing_mode=processing_mode,
                monochrome_score=monochrome_score,
            )
        except Exception:
            overlay_source = "legacy_overlay_fallback"
            gonial_debug.setdefault("overlay_geometry_source", "straight_fallback")
            gonial_debug.setdefault("overlay_snap_mode", "off")
            gonial_debug.setdefault("overlay_path_quality", 0.0)
            gonial_debug.setdefault("ramus_display_mode", "straight_fallback")

        return {
            "score": float(side_score),
            "breakdown": side_breakdown,
            "measurements": m,
            "overlay_image": overlay_image,
            "debug": {
                **debug,
                "v2_diagnostics": {
                    "jaw_curvature": round(float(jaw_curvature), 3),
                    "invariants": invariants,
                    "context": context,
                    "global_confidence": round(float(global_conf), 3),
                    "jawline_solver": jawline_result.debug,
                    "gonial_debug": gonial_debug,
                    "overlay_source": overlay_source,
                    "overlay_renderer_version": OVERLAY_RENDERER_VERSION,
                },
            },
            "landmarks_v2": {
                "points": required_points,
                "jaw_contour": jaw_contour,
                "confidence": {
                    "global": round(global_conf, 3),
                    "articulare": round(point_conf["articulare"], 3),
                    "gonion": round(point_conf["gonion"], 3),
                    "menton": round(point_conf["menton"], 3),
                },
                "gonial_debug": gonial_debug,
                "method": method_name,
                "engine": "local_fallback_mp_hybrid",
                "method_source": method_source,
                "overlay_source": overlay_source,
            },
            "quality_v2": {
                "pose_yaw": quality["pose_yaw"],
                "pose_pitch": quality["pose_pitch"],
                "occlusion_score": quality["occlusion_score"],
                "lighting_score": quality["lighting_score"],
                "jawline_visibility_score": quality["jawline_visibility_score"],
                "monochrome_score": quality["monochrome_score"],
                "uncertain_side_landmarks": uncertain,
            },
            "method_source": method_source,
        }
