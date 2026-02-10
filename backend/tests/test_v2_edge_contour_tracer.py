import unittest

import cv2
import numpy as np

from backend.v2.edge_contour_tracer import trace_side_contours


class TestEdgeContourTracer(unittest.TestCase):
    def _synthetic_profile(self):
        img = np.full((260, 220, 3), 35, dtype=np.uint8)
        face_poly = np.array([
            [120, 34],
            [102, 62],
            [82, 90],
            [86, 112],
            [92, 128],
            [94, 150],
            [92, 188],
            [145, 170],
            [162, 154],
            [165, 112],
            [158, 72],
        ], dtype=np.int32)
        cv2.fillPoly(img, [face_poly], color=(170, 195, 225))
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        points = {
            "trichion": (120, 34),
            "glabella": (106, 58),
            "nasion": (100, 68),
            "pronasale": (82, 90),
            "subnasale": (86, 112),
            "menton": (92, 188),
            "gonion": (161, 154),
            "articulare": (162, 116),
            "tragion": (164, 108),
        }
        return img, points

    def test_trace_contours_success(self):
        image, points = self._synthetic_profile()
        contours = trace_side_contours(image, points)
        self.assertIn(contours.get("method"), {"edge_trace_dijkstra_v1", "contour_walk_fallback_v1"})
        self.assertGreaterEqual(len(contours.get("silhouette", [])), 2)
        self.assertGreaterEqual(len(contours.get("jaw_ramus", [])), 2)
        self.assertGreater(float(contours.get("confidence", 0.0)), 0.05)

    def test_trace_contours_fallback_on_missing_anchors(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        contours = trace_side_contours(image, {"gonion": (60, 70)})
        self.assertEqual(contours.get("method"), "landmark_fallback_v1")
        self.assertEqual(float(contours.get("confidence", 0.0)), 0.0)
        self.assertIn("fallback_reason", contours.get("debug", {}))


if __name__ == "__main__":
    unittest.main()
