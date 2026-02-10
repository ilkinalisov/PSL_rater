import unittest

from backend.v2.geometry import check_landmark_invariants, gonial_angle_from_points


class TestV2Geometry(unittest.TestCase):
    def test_invariants_happy_path(self):
        points = {
            "articulare": (420, 210),
            "gonion": (420, 280),
            "menton": (290, 340),
        }
        inv = check_landmark_invariants(points)
        self.assertTrue(inv["has_required_points"])
        self.assertTrue(inv["go_posterior_to_me"])
        self.assertTrue(inv["ar_superior_to_go"])

    def test_invariants_failures(self):
        points = {
            "articulare": (310, 350),
            "gonion": (300, 290),
            "menton": (320, 300),
        }
        inv = check_landmark_invariants(points)
        self.assertFalse(inv["go_posterior_to_me"])
        self.assertFalse(inv["ar_superior_to_go"])

    def test_gonial_angle_stability_small_noise(self):
        ar = (410, 215)
        go = (408, 288)
        me = (302, 344)
        base = gonial_angle_from_points(ar, go, me)
        noisy = gonial_angle_from_points((411, 214), (407, 289), (303, 343))
        self.assertIsNotNone(base)
        self.assertIsNotNone(noisy)
        self.assertLess(abs(base - noisy), 2.0)


if __name__ == "__main__":
    unittest.main()
