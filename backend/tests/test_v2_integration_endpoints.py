import base64
import os
import unittest
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi.testclient import TestClient


@unittest.skipUnless(os.environ.get("RUN_V2_INTEGRATION_TESTS") == "1", "Integration tests are opt-in.")
class TestV2Endpoints(unittest.TestCase):
    def setUp(self):
        import backend.main as main  # local import to avoid eager import in normal unit runs

        self.main = main
        self.client = TestClient(main.app)

        @dataclass
        class _Front:
            gender: str = "unknown"
            gender_confidence: float = 0.0

            def to_dict(self):
                return {"gender": self.gender, "gender_confidence": self.gender_confidence}

        class _FrontAnalyzer:
            def analyze_face(self, image):
                return _Front(), image, {"detected_landmarks": 50, "landmark_count": 468}

            def calculate_psl_score(self, _):
                return 7.2, {"symmetry": 7.0}

        class _SideClient:
            def analyze_side(self, image):
                return {
                    "score": 7.4,
                    "breakdown": {"gonial_angle": 7.1},
                    "measurements": {
                        "gonial_angle": 118.0,
                        "gender": "unknown",
                        "gender_confidence": 0.0,
                    },
                    "landmarks_v2": {
                        "points": {
                            "articulare": [100, 80],
                            "gonion": [100, 120],
                            "menton": [60, 150],
                            "nasion": [40, 60],
                            "subnasale": [52, 100],
                            "pogonion": [65, 145],
                        },
                        "jaw_contour": [[60, 150], [100, 120]],
                        "confidence": {"global": 0.7, "articulare": 0.7, "gonion": 0.7, "menton": 0.7},
                        "method": "local_legacy_mp_hybrid",
                        "method_source": "local_fallback_legacy",
                        "gonial_debug": {
                            "raw_angle": 118.0,
                            "source": "mediapipe",
                            "fallback_reason": "none",
                            "pre_validation_angle": 118.0,
                            "pitch_deg": 0.0,
                            "processing_mode": "color",
                            "overlay_geometry_source": "hybrid",
                            "overlay_snap_mode": "off",
                            "overlay_path_quality": 0.72,
                            "ramus_display_mode": "hybrid_vertical",
                            "acceptance_gate": {},
                        },
                        "overlay_source": "v2_landmarks_renderer",
                    },
                    "contours": {
                        "method": "edge_trace_dijkstra_v1",
                        "confidence": 0.72,
                        "silhouette": [{"x": 30, "y": 20}, {"x": 45, "y": 40}],
                        "jaw_ramus": [{"x": 65, "y": 145}, {"x": 100, "y": 120}],
                        "debug": {
                            "image_size": {"width": 120, "height": 160},
                            "was_mirrored": False,
                            "processing_mode": "color",
                            "roi": {"x1": 10, "y1": 10, "x2": 110, "y2": 150},
                            "fallback_reason": "none",
                        },
                    },
                    "quality_v2": {
                        "pose_yaw": 82.0,
                        "occlusion_score": 0.8,
                        "lighting_score": 0.7,
                        "jawline_visibility_score": 0.41,
                        "monochrome_score": 0.1,
                        "uncertain_side_landmarks": False,
                    },
                    "overlay_image": image,
                    "debug": {},
                    "transport": {"source": "local_fallback", "latency_ms": 10},
                    "method_source": "local_fallback_legacy",
                }

        self.main.front_analyzer = _FrontAnalyzer()
        self.main.v2_side_inference_client = _SideClient()

    def _fake_image_bytes(self):
        image = np.zeros((160, 120, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        return buf.tobytes()

    def test_v2_side_endpoint(self):
        payload = self._fake_image_bytes()
        response = self.client.post(
            "/v2/analyze/side",
            files={"file": ("side.jpg", payload, "image/jpeg")},
            headers={"X-API-Key": os.environ.get("API_KEY", "")} if os.environ.get("API_KEY") else {},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body.get("success"))
        self.assertIn("landmarks_v2", body["side_analysis"])
        self.assertIn("contours", body["side_analysis"])
        self.assertIn("method", body["side_analysis"]["contours"])
        self.assertIn("silhouette", body["side_analysis"]["contours"])
        self.assertIn("jaw_ramus", body["side_analysis"]["contours"])
        self.assertIn("method_source", body["side_analysis"]["landmarks_v2"])
        self.assertIn("gonial_debug", body["side_analysis"]["landmarks_v2"])
        self.assertIn("pitch_deg", body["side_analysis"]["landmarks_v2"]["gonial_debug"])
        self.assertIn("processing_mode", body["side_analysis"]["landmarks_v2"]["gonial_debug"])
        self.assertIn("overlay_geometry_source", body["side_analysis"]["landmarks_v2"]["gonial_debug"])
        self.assertIn("overlay_snap_mode", body["side_analysis"]["landmarks_v2"]["gonial_debug"])
        self.assertIn("overlay_path_quality", body["side_analysis"]["landmarks_v2"]["gonial_debug"])
        self.assertIn("ramus_display_mode", body["side_analysis"]["landmarks_v2"]["gonial_debug"])
        self.assertIn("acceptance_gate", body["side_analysis"]["landmarks_v2"]["gonial_debug"])
        self.assertIn("overlay_source", body["side_analysis"]["landmarks_v2"])
        self.assertIn("monochrome_score", body["side_analysis"]["quality_v2"])
        self.assertIn("pipeline", body["debug"])
        self.assertIn("endpoint_variant", body["debug"]["pipeline"])
        self.assertIn("overlay_renderer_version", body["debug"]["pipeline"])

    def test_v2_pair_endpoint(self):
        payload = self._fake_image_bytes()
        response = self.client.post(
            "/v2/analyze/pair",
            files={
                "front_image": ("front.jpg", payload, "image/jpeg"),
                "side_image": ("side.jpg", payload, "image/jpeg"),
            },
            headers={"X-API-Key": os.environ.get("API_KEY", "")} if os.environ.get("API_KEY") else {},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body.get("success"))
        self.assertIn("scores_v2", body)
        self.assertIn("contours", body["side_analysis"])
        self.assertIn("method_source", body["side_analysis"])
        self.assertIn("pipeline", body["debug"])
        self.assertIn("endpoint_variant", body["debug"]["pipeline"])


if __name__ == "__main__":
    unittest.main()
