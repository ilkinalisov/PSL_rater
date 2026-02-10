import unittest

from backend.v2.scoring import compute_scores_v2


class TestV2Scoring(unittest.TestCase):
    def test_reliable_overall(self):
        scores = compute_scores_v2(
            front_primary=7.8,
            side_primary=8.2,
            front_reliability=0.82,
            side_reliability=0.79,
        )
        self.assertTrue(scores["reliable"])
        self.assertGreaterEqual(scores["overall_reliability"], 0.45)
        self.assertGreater(scores["overall_optional"], 7.5)

    def test_low_side_confidence_downweights_overall(self):
        scores = compute_scores_v2(
            front_primary=8.2,
            side_primary=8.4,
            front_reliability=0.88,
            side_reliability=0.20,
        )
        self.assertFalse(scores["reliable"])
        self.assertLess(scores["overall_optional"], 8.3)

    def test_non_finite_inputs_are_sanitized(self):
        scores = compute_scores_v2(
            front_primary=float("nan"),
            side_primary=float("inf"),
            front_reliability=float("nan"),
            side_reliability=float("-inf"),
        )
        self.assertGreaterEqual(scores["overall_optional"], 1.0)
        self.assertLessEqual(scores["overall_optional"], 10.0)


if __name__ == "__main__":
    unittest.main()
