import unittest
from dataclasses import dataclass
from unittest.mock import patch

import cv2
import numpy as np

from backend.v2.jawline_solver import JawlineSolveResult
from backend.v2.local_side_v2_adapter import LocalSideV2Adapter
from backend.v2.overlay_renderer import (
    _build_overlay_edge_map,
    _snap_polyline_to_edges,
    build_gonial_overlay_paths,
    render_side_overlay,
)


@dataclass
class _FakeMeasurements:
    gonial_angle: float = 118.0
    is_estimated: bool = False
    gonial_source: str = "mediapipe"
    gonial_fallback_reason: str = "none"
    raw_gonial_angle: float = 118.0
    pre_validation_gonial_angle: float = 118.0
    profile_harmony_score: float = 6.0
    forward_growth_score: float = 6.0
    gender: str = "unknown"
    gender_confidence: float = 0.0


class _FakeSideAnalyzer:
    def analyze_side_profile(self, image):
        debug = {
            "landmark_points": {
                "articulare": (100, 78),
                "gonion": (100, 118),
                "menton": (70, 140),
                "nasion": (48, 58),
                "subnasale": (56, 96),
                "pogonion": (72, 138),
                "tragion": (112, 84),
                "pronasale": (44, 92),
                "trichion": (60, 26),
                "glabella": (52, 54),
                "labrale_superius": (62, 114),
                "labrale_inferius": (64, 124),
            },
            "jaw_contour": [[72, 138], [88, 130], [100, 118]],
            "gonion_confidence": 0.44,
            "missing_landmark_ratio": 0.10,
            "landmark_confidences": {
                "menton": 0.53,
                "tragion": 0.56,
            },
        }
        legacy_overlay = np.zeros_like(image)
        return _FakeMeasurements(), legacy_overlay, debug

    def calculate_side_profile_score(self, _measurements):
        return 7.2, {
            "gonial_angle": 7.4,
            "nasolabial": 7.0,
            "facial_convexity": 7.1,
            "vertical_balance": 7.3,
            "profile_harmony": 7.0,
            "forward_growth": 6.8,
        }


class TestV2OverlayRenderer(unittest.TestCase):
    def test_renderer_draws_points_and_contour(self):
        image = np.zeros((180, 180, 3), dtype=np.uint8)
        points = {
            "trichion": (58, 24),
            "glabella": (52, 52),
            "nasion": (48, 60),
            "pronasale": (42, 88),
            "subnasale": (56, 98),
            "labrale_superius": (62, 112),
            "labrale_inferius": (64, 124),
            "menton": (72, 144),
            "gonion": (106, 118),
            "articulare": (118, 76),
            "tragion": (114, 84),
        }
        contour = [[72, 144], [90, 132], [106, 118], [114, 96]]

        out = render_side_overlay(image.copy(), points, contour)
        self.assertEqual(out.shape, image.shape)
        self.assertGreater(int(np.sum(out)), 0)
        self.assertGreater(int(np.sum(out[118, 106])), 0)  # gonion circle/line region

    def test_local_adapter_overlay_uses_finalized_v2_points(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        adapter = LocalSideV2Adapter(_FakeSideAnalyzer())

        jaw_result = JawlineSolveResult(
            points={
                "articulare": (118, 76),
                "gonion": (106, 118),
                "menton": (72, 144),
            },
            jaw_contour=[[72, 144], [90, 132], [106, 118], [114, 96]],
            visibility_score=0.74,
            source="jawline_contour",
            fallback_reason="none",
            debug={
                "pitch_deg": 1.5,
                "processing_mode": "mono",
                "monochrome_score": 0.86,
                "fit_residual": 6.8,
            },
        )

        with patch("backend.v2.local_side_v2_adapter.solve_jawline_contour", return_value=jaw_result):
            result = adapter.analyze(image)

        self.assertEqual(result["landmarks_v2"]["overlay_source"], "v2_landmarks_renderer")
        self.assertEqual(result["landmarks_v2"]["gonial_debug"]["processing_mode"], "mono")
        self.assertGreater(result["quality_v2"]["monochrome_score"], 0.80)
        self.assertEqual(result["landmarks_v2"]["gonial_debug"]["fallback_reason"], "none")
        self.assertIn(
            result["landmarks_v2"]["gonial_debug"]["ramus_display_mode"],
            {"contour", "hybrid_vertical", "straight_fallback"},
        )
        self.assertEqual(result["landmarks_v2"]["method"], "jawline_contour_fused_local")
        self.assertGreater(int(np.sum(result["overlay_image"])), 0)
        self.assertGreater(int(np.sum(result["overlay_image"][118, 106])), 0)

    def test_contour_segment_is_non_colinear_for_curved_jaw(self):
        points = {
            "articulare": (126, 76),
            "gonion": (108, 116),
            "menton": (70, 146),
        }
        contour = [
            [70, 146],
            [78, 142],
            [88, 136],
            [98, 128],
            [108, 116],
            [116, 102],
            [124, 86],
        ]
        info = build_gonial_overlay_paths(points, contour, (200, 200, 3), processing_mode="color")
        mandibular = info["mandibular_path"]
        self.assertGreaterEqual(len(mandibular), 4)
        self.assertNotEqual(mandibular[0][1], mandibular[len(mandibular) // 2][1])
        self.assertIn(info["overlay_geometry_source"], {"contour", "hybrid"})

    def test_hybrid_ramus_is_generated_when_contour_is_partial(self):
        points = {
            "articulare": (126, 76),
            "gonion": (108, 116),
            "menton": (70, 146),
        }
        # Contour intentionally contains mandibular part only; no robust ramus section near articulare.
        contour = [[70, 146], [82, 140], [94, 132], [108, 116], [112, 114]]
        info = build_gonial_overlay_paths(points, contour, (200, 200, 3), processing_mode="hybrid")
        ramus = info["ramus_path"]
        self.assertGreaterEqual(len(ramus), 3)
        self.assertEqual(ramus[0], (108, 116))
        self.assertEqual(info.get("ramus_display_mode"), "hybrid_vertical")
        self.assertIn(info["overlay_geometry_source"], {"hybrid", "straight_fallback"})

    def test_mono_edge_snap_moves_path_toward_boundary(self):
        image = np.full((180, 180, 3), 40, dtype=np.uint8)
        # Synthetic jaw boundary edge.
        cv2.line(image, (60, 140), (130, 110), (220, 220, 220), 2, lineType=cv2.LINE_AA)
        edge_map = _build_overlay_edge_map(image, mode="mono")
        poly = [(60, 148), (78, 144), (98, 136), (118, 126), (130, 118)]
        snapped = _snap_polyline_to_edges(poly, edge_map, radius_px=4)
        self.assertEqual(len(snapped), len(poly))
        before = np.mean([p[1] for p in poly])
        after = np.mean([p[1] for p in snapped])
        self.assertLess(after, before)

    def test_overlay_source_falls_back_when_contour_is_weak(self):
        image = np.zeros((180, 180, 3), dtype=np.uint8)
        points = {
            "articulare": (120, 78),
            "gonion": (106, 118),
            "menton": (72, 146),
        }
        gonial_debug = {"processing_mode": "color", "monochrome_score": 0.0}
        out = render_side_overlay(image.copy(), points, [[106, 118]], gonial_debug=gonial_debug)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(gonial_debug.get("overlay_geometry_source"), "straight_fallback")
        self.assertEqual(gonial_debug.get("ramus_display_mode"), "straight_fallback")
        self.assertEqual(gonial_debug.get("overlay_snap_mode"), "off")


if __name__ == "__main__":
    unittest.main()
