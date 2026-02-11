import unittest
from dataclasses import replace

from backend.side_profile_analyzer import SideProfileAnalyzer, SideProfileMeasurements


def _build_measurements(profile_harmony_score: float, *, is_estimated: bool) -> SideProfileMeasurements:
    return SideProfileMeasurements(
        facial_convexity_angle=165.0,
        nasofrontal_angle=130.0,
        nasolabial_angle=110.01876039378396,
        gonial_angle=97.71227295079954,
        nasofacial_angle=36.0,
        forehead_slope=45.0,
        nasal_projection=0.04666666666666667,
        chin_projection=0.04666666666666667,
        lip_projection=0.005,
        profile_harmony_score=profile_harmony_score,
        vertical_profile_balance={
            "upper_third": 0.27102803738317754,
            "middle_third": 0.32242990654205606,
            "lower_third": 0.40654205607476634,
        },
        forward_growth_score=5.246023529411765,
        facial_angle=82.0,
        maxillary_prominence=0.008333333333333333,
        mandibular_prominence=-0.15714285714285714,
        recession_type="severe",
        gender="male",
        gender_confidence=0.24,
        gonion_confidence=1.0,
        raw_gonial_angle=97.71227295079954,
        pre_validation_gonial_angle=97.71227295079954,
        gonial_source="jawline_contour",
        gonial_fallback_reason="none",
        is_estimated=is_estimated,
        was_mirrored=True,
    )


class TestSideProfileScoringGuardrails(unittest.TestCase):
    def setUp(self):
        # Avoid heavy MediaPipe runtime init; scoring method is pure.
        self.analyzer = SideProfileAnalyzer.__new__(SideProfileAnalyzer)

    def test_high_harmony_100_scale_does_not_collapse(self):
        m = _build_measurements(85.64316082833518, is_estimated=True)
        score, breakdown = SideProfileAnalyzer.calculate_side_profile_score(self.analyzer, m)

        self.assertGreaterEqual(score, 5.5)
        self.assertGreater(breakdown["profile_harmony"], 7.5)
        self.assertGreater(breakdown["gonial_angle"], 2.7)

    def test_harmony_scale_compatibility_0_10_and_0_100(self):
        m_100 = _build_measurements(85.0, is_estimated=False)
        m_10 = _build_measurements(8.5, is_estimated=False)

        score_100, breakdown_100 = SideProfileAnalyzer.calculate_side_profile_score(self.analyzer, m_100)
        score_10, breakdown_10 = SideProfileAnalyzer.calculate_side_profile_score(self.analyzer, m_10)

        self.assertAlmostEqual(breakdown_100["profile_harmony"], breakdown_10["profile_harmony"], places=3)
        self.assertAlmostEqual(score_100, score_10, places=2)

    def test_estimated_gonial_blends_toward_neutral_not_collapse(self):
        m_reliable = _build_measurements(85.0, is_estimated=False)
        m_estimated = replace(m_reliable, is_estimated=True)

        _, b_reliable = SideProfileAnalyzer.calculate_side_profile_score(self.analyzer, m_reliable)
        _, b_estimated = SideProfileAnalyzer.calculate_side_profile_score(self.analyzer, m_estimated)

        self.assertGreater(b_estimated["gonial_angle"], b_reliable["gonial_angle"])


if __name__ == "__main__":
    unittest.main()
