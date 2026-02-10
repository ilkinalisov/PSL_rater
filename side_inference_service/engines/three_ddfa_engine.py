"""3DDFA-oriented side engine with jawline fusion.

The 3DDFA runtime is optional in this repository. When unavailable, this engine
still performs jawline contour fusion and marks method_source accordingly.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Dict, Tuple

import numpy as np

from v2.geometry import check_landmark_invariants, gonial_angle_from_points
from v2.edge_contour_tracer import trace_side_contours
from v2.jawline_solver import solve_jawline_contour
from v2.overlay_renderer import OVERLAY_RENDERER_VERSION, render_side_overlay


Point = Tuple[int, int]


class ThreeDDFAEngine:
    def __init__(self, side_analyzer=None):
        self.available = False
        self._load_error = ""
        self._runtime = None
        self._runtime_mode = "stub_jawline_fusion"
        self._native_runtime_active = False
        self.side_analyzer = side_analyzer
        self._try_load_runtime()

    def _try_load_runtime(self):
        # Optional runtime: repository can run without 3DDFA installed.
        try:
            # Common 3DDFA_V2 public entrypoints; loaded lazily/defensively.
            from TDDFA import TDDFA  # type: ignore
            from FaceBoxes import FaceBoxes  # type: ignore
            self._runtime = {
                "TDDFA": TDDFA,
                "FaceBoxes": FaceBoxes,
            }
            self.available = True
            self._runtime_mode = "native_runtime_detected_not_wired"
        except Exception as exc:  # pragma: no cover - runtime dependent
            self._load_error = str(exc)
            self.available = False
            self._runtime_mode = "stub_jawline_fusion"

    @property
    def status(self) -> Dict:
        return {
            "available": self.available,
            "load_error": self._load_error or None,
            "runtime_mode": self._runtime_mode,
            "native_runtime_active": self._native_runtime_active,
        }

    def _read_points(self, result: Dict) -> Dict[str, Point]:
        points = ((result.get("landmarks_v2") or {}).get("points") or {})
        out = {}
        for key, val in points.items():
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                out[key] = (int(val[0]), int(val[1]))
        return out

    def _write_points(self, result: Dict, points: Dict[str, Point]):
        landmarks = result.setdefault("landmarks_v2", {})
        point_map = landmarks.setdefault("points", {})
        for key, val in points.items():
            point_map[key] = [int(val[0]), int(val[1])]

    def _set_method_defaults(self, result: Dict):
        landmarks = result.setdefault("landmarks_v2", {})
        quality = result.setdefault("quality_v2", {})
        measurements = result.setdefault("measurements", {})
        landmarks.setdefault("gonial_debug", {})
        landmarks.setdefault("overlay_source", "unknown")
        quality.setdefault("jawline_visibility_score", 0.0)
        quality.setdefault("monochrome_score", 0.0)
        measurements.setdefault("raw_gonial_angle", None)
        measurements.setdefault("pre_validation_gonial_angle", None)
        measurements.setdefault("gonial_source", "default")
        measurements.setdefault("gonial_fallback_reason", "default")

    def _rescore(self, result: Dict):
        if self.side_analyzer is None:
            return
        measurements = result.get("measurements")
        if not isinstance(measurements, dict):
            return
        try:
            from side_profile_analyzer import SideProfileMeasurements  # type: ignore

            field_names = {f.name for f in fields(SideProfileMeasurements)}
            payload = {}
            for key in field_names:
                if key in measurements:
                    payload[key] = measurements[key]
            required = {
                "facial_convexity_angle",
                "nasofrontal_angle",
                "nasolabial_angle",
                "gonial_angle",
                "nasofacial_angle",
                "forehead_slope",
                "nasal_projection",
                "chin_projection",
                "lip_projection",
                "profile_harmony_score",
                "vertical_profile_balance",
            }
            if not required.issubset(payload.keys()):
                return
            obj = SideProfileMeasurements(**payload)
            side_score, side_breakdown = self.side_analyzer.calculate_side_profile_score(obj)
            result["score"] = float(side_score)
            result["breakdown"] = side_breakdown
        except Exception:
            return

    def fuse(self, image: np.ndarray, base_result: Dict) -> Dict:
        """
        Fuse jawline contour with base side-result.
        If 3DDFA runtime is available, method metadata reflects remote-3ddfa path.
        """
        result = dict(base_result or {})
        self._set_method_defaults(result)

        points = self._read_points(result)
        was_mirrored = bool((result.get("measurements") or {}).get("was_mirrored", False))
        working_image = np.fliplr(image).copy() if was_mirrored else image
        jaw = solve_jawline_contour(working_image, points)
        visibility = float(np.clip(jaw.visibility_score, 0.0, 1.0))
        processing_mode = str((jaw.debug or {}).get("processing_mode", "color"))
        monochrome_score = float(np.clip((jaw.debug or {}).get("monochrome_score", 0.0), 0.0, 1.0))
        fit_residual = float(np.clip((jaw.debug or {}).get("fit_residual", 99.0), 0.0, 999.0))

        quality = result.setdefault("quality_v2", {})
        quality["jawline_visibility_score"] = round(visibility, 3)
        quality["monochrome_score"] = round(monochrome_score, 3)

        landmarks = result.setdefault("landmarks_v2", {})
        gonial_debug = landmarks.setdefault("gonial_debug", {})
        gonial_debug.setdefault("raw_angle", (result.get("measurements") or {}).get("raw_gonial_angle"))
        gonial_debug.setdefault("pre_validation_angle", (result.get("measurements") or {}).get("pre_validation_gonial_angle"))
        gonial_debug.setdefault("source", (result.get("measurements") or {}).get("gonial_source", "default"))
        gonial_debug.setdefault("fallback_reason", (result.get("measurements") or {}).get("gonial_fallback_reason", "default"))
        pitch_deg = float((jaw.debug or {}).get("pitch_deg", 0.0))
        gonial_debug["processing_mode"] = processing_mode
        gonial_debug.setdefault("overlay_geometry_source", "straight_fallback")
        gonial_debug.setdefault("overlay_snap_mode", "off")
        gonial_debug.setdefault("overlay_path_quality", 0.0)
        gonial_debug.setdefault("ramus_display_mode", "straight_fallback")
        acceptance_threshold = 0.42
        residual_threshold = 12.0
        if processing_mode == "mono":
            residual_threshold = 14.0
        elif processing_mode == "hybrid":
            residual_threshold = 13.0
        if processing_mode in ("mono", "hybrid") and monochrome_score >= 0.58 and fit_residual <= 8.5:
            acceptance_threshold = 0.36
        candidate_points = dict(points)
        if jaw.points:
            candidate_points.update(jaw.points)
        candidate_invariants = check_landmark_invariants(candidate_points)
        candidate_gonial = None
        candidate_gonial_valid = False
        if all(k in candidate_points for k in ("articulare", "gonion", "menton")):
            candidate_gonial = gonial_angle_from_points(
                candidate_points["articulare"],
                candidate_points["gonion"],
                candidate_points["menton"],
            )
            candidate_gonial_valid = candidate_gonial is not None and 95.0 <= float(candidate_gonial) <= 150.0
        residual_ok = fit_residual <= residual_threshold

        accepted = (
            bool(jaw.points) and
            visibility >= acceptance_threshold and
            bool(candidate_invariants.get("go_posterior_to_me", False)) and
            bool(candidate_invariants.get("ar_superior_to_go", False)) and
            bool(candidate_gonial_valid) and
            bool(residual_ok)
        )
        gonial_debug.setdefault("pitch_deg", pitch_deg)
        gonial_debug.setdefault("acceptance_gate", {})
        gonial_debug["acceptance_gate"] = {
            "threshold": acceptance_threshold,
            "threshold_used": acceptance_threshold,
            "visibility": round(visibility, 3),
            "pitch_deg": round(pitch_deg, 3),
            "processing_mode": processing_mode,
            "monochrome_score": round(monochrome_score, 3),
            "residual": round(fit_residual, 3),
            "residual_threshold": round(residual_threshold, 3),
            "residual_ok": bool(residual_ok),
            "invariants_ok": bool(
                candidate_invariants.get("go_posterior_to_me", False) and
                candidate_invariants.get("ar_superior_to_go", False)
            ),
            "gonial_valid": bool(candidate_gonial_valid),
            "accepted": bool(accepted),
        }

        if accepted:
            points = candidate_points
            self._write_points(result, points)
            if jaw.jaw_contour:
                landmarks["jaw_contour"] = jaw.jaw_contour
            gonial_debug["source"] = "jawline_contour"
            gonial_debug["fallback_reason"] = "none"
        else:
            if processing_mode == "mono" and (visibility < 0.36 or monochrome_score >= 0.72):
                gonial_debug["fallback_reason"] = "mono_low_texture"
            elif not residual_ok:
                gonial_debug["fallback_reason"] = "contour_residual_high"
            if gonial_debug.get("fallback_reason") in (None, "none", "default"):
                gonial_debug["fallback_reason"] = jaw.fallback_reason

        gonial = None
        if all(k in points for k in ("articulare", "gonion", "menton")):
            gonial = gonial_angle_from_points(points["articulare"], points["gonion"], points["menton"])
            gonial_debug["pre_validation_angle"] = float(gonial) if gonial is not None else gonial_debug.get("pre_validation_angle")

        measurements = result.setdefault("measurements", {})
        if gonial is not None and 95.0 <= float(gonial) <= 150.0:
            measurements["gonial_angle"] = float(gonial)
            measurements["raw_gonial_angle"] = float(gonial)
            measurements["pre_validation_gonial_angle"] = float(gonial)
            measurements["gonial_source"] = str(gonial_debug.get("source", "jawline_contour"))
            measurements["gonial_fallback_reason"] = "none"
            measurements["is_estimated"] = bool(measurements.get("is_estimated", False))
            gonial_debug["raw_angle"] = float(gonial)
            gonial_debug["fallback_reason"] = "none"
        else:
            measurements["is_estimated"] = True
            measurements["gonial_source"] = str(gonial_debug.get("source", measurements.get("gonial_source", "default")))
            measurements["gonial_fallback_reason"] = str(gonial_debug.get("fallback_reason", "mp_invalid"))
            if gonial_debug.get("fallback_reason") in (None, "none", "default"):
                gonial_debug["fallback_reason"] = "mp_invalid"

        uncertain = bool(result.get("quality_v2", {}).get("uncertain_side_landmarks", False))
        if visibility < 0.20 or gonial is None or not (95.0 <= float(gonial) <= 150.0):
            uncertain = True
        if processing_mode == "mono" and visibility < 0.40:
            uncertain = True
        quality["uncertain_side_landmarks"] = uncertain

        # Truthful method metadata.
        if self.available and self._native_runtime_active and accepted:
            landmarks["method"] = "3ddfa_v2_jawline_fused"
            landmarks["engine"] = "remote_3ddfa_v2"
            landmarks["method_source"] = "remote_3ddfa_jaw_v2"
            result["method_source"] = "remote_3ddfa_jaw_v2"
        else:
            landmarks["method"] = "local_legacy_mp_hybrid"
            landmarks["engine"] = "local_fallback_mp_hybrid"
            landmarks["method_source"] = "local_fallback_legacy"
            result["method_source"] = "local_fallback_legacy"

        # Ensure overlay matches finalized landmarks (post-fusion points).
        overlay_source = "v2_landmarks_renderer"
        overlay_jaw_contour = (result.get("landmarks_v2") or {}).get("jaw_contour") or []
        safe_overlay_mode = (
            (not accepted) or
            bool(gonial_debug.get("fallback_reason") not in (None, "none")) or
            (processing_mode == "mono" and monochrome_score >= 0.72)
        )
        if safe_overlay_mode:
            safe_contour = []
            points_for_safe = (result.get("landmarks_v2") or {}).get("points") or points
            for key in ("menton", "gonion", "articulare"):
                val = points_for_safe.get(key)
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    safe_contour.append([int(val[0]), int(val[1])])
            overlay_jaw_contour = safe_contour if len(safe_contour) >= 2 else []
        try:
            points_map = (result.get("landmarks_v2") or {}).get("points") or {}
            result["overlay_image"] = render_side_overlay(
                working_image.copy(),
                points_map,
                overlay_jaw_contour,
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
        landmarks["overlay_source"] = overlay_source

        contours = trace_side_contours(
            image=working_image,
            anchor_points=(result.get("landmarks_v2") or {}).get("points") or points,
            jaw_solver_debug=jaw.debug if isinstance(jaw.debug, dict) else {},
            was_mirrored=was_mirrored,
        )
        landmarks["contours"] = contours
        result["contours"] = contours

        result.setdefault("debug", {})
        result["debug"]["remote_engine"] = {
            "three_ddfa": self.status,
            "jawline_solver": jaw.debug,
            "jawline_visibility_score": round(visibility, 3),
            "overlay_renderer_version": OVERLAY_RENDERER_VERSION,
            "overlay_source": overlay_source,
        }
        landmarks["gonial_debug"] = gonial_debug
        self._rescore(result)
        return result
