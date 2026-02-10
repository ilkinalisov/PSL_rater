import unittest

import cv2
import numpy as np

from backend.v2.jawline_solver import solve_jawline_contour


class TestJawlineSolver(unittest.TestCase):
    def _synthetic_profile(self):
        img = np.full((240, 220, 3), 35, dtype=np.uint8)
        # Face polygon (left-facing profile) with visible jaw corner.
        face_poly = np.array([
            [120, 38],   # forehead
            [100, 70],   # bridge
            [78, 92],    # nose
            [86, 112],   # subnasale
            [92, 126],   # upper lip
            [94, 146],   # lower lip
            [92, 178],   # menton
            [145, 166],  # mandibular body
            [162, 152],  # gonial region
            [164, 115],  # ramus near ear
            [158, 78],   # temple
        ], dtype=np.int32)
        cv2.fillPoly(img, [face_poly], color=(165, 190, 220))
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        anchors = {
            "menton": (93, 178),
            "subnasale": (86, 112),
            "pronasale": (78, 92),
            "gonion": (159, 152),
            "tragion": (164, 116),
            "articulare": (159, 126),
            "pogonion": (95, 174),
            "nasion": (102, 72),
        }
        return img, anchors

    def _rotate_profile(self, image, anchors, angle_deg):
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated_img = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rotated_anchors = {}
        for key, (x, y) in anchors.items():
            vec = np.array([x, y, 1.0], dtype=float)
            out = mat @ vec
            rotated_anchors[key] = (int(round(out[0])), int(round(out[1])))
        return rotated_img, rotated_anchors

    def test_solver_finds_jaw_points(self):
        image, anchors = self._synthetic_profile()
        result = solve_jawline_contour(image, anchors)
        self.assertGreater(result.visibility_score, 0.05)
        self.assertTrue(result.jaw_contour)
        self.assertIn("menton", result.points)
        self.assertIn("gonion", result.points)
        self.assertIn("articulare", result.points)

        me = result.points["menton"]
        go = result.points["gonion"]
        ar = result.points["articulare"]
        self.assertGreaterEqual(go[0], me[0])  # posterior to menton in left-facing frame
        self.assertLessEqual(ar[1], go[1])      # superior to gonion

    def test_solver_handles_pitch_down(self):
        image, anchors = self._synthetic_profile()
        # Rotate clockwise to simulate head-down posture.
        tilted_img, tilted_anchors = self._rotate_profile(image, anchors, -14.0)
        result = solve_jawline_contour(tilted_img, tilted_anchors)
        self.assertGreater(result.visibility_score, 0.03)
        self.assertTrue(result.jaw_contour)
        self.assertIn("pitch_deg", result.debug)
        self.assertIn("normalization_angle", result.debug)
        me = result.points["menton"]
        go = result.points["gonion"]
        ar = result.points["articulare"]
        self.assertGreaterEqual(go[0], me[0])
        self.assertLessEqual(ar[1], go[1])

    def test_solver_suppresses_neck_drift(self):
        image, anchors = self._synthetic_profile()
        # Add strong neck-like edge below chin; solver should still keep Go close to jaw corner.
        cv2.line(image, (150, 168), (186, 226), (230, 230, 230), 6)
        result = solve_jawline_contour(image, anchors)
        self.assertIn("gonion", result.points)
        go = result.points["gonion"]
        # Go should not collapse far below chin/neck edge region.
        self.assertLessEqual(go[1], anchors["menton"][1] + 6)

    def test_solver_handles_missing_anchors(self):
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        result = solve_jawline_contour(img, {"gonion": (70, 80)})
        self.assertEqual(result.points, {})
        self.assertEqual(result.jaw_contour, [])
        self.assertIn(result.fallback_reason, {"default", "shape_weak"})


if __name__ == "__main__":
    unittest.main()
