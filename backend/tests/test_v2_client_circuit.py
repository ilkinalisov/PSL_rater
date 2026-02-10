import unittest

from backend.v2.side_inference_client import SideInferenceClient


class _DummyAdapter:
    def analyze(self, image):
        return {
            "score": 7.1,
            "breakdown": {},
            "measurements": {},
            "landmarks_v2": {
                "points": {},
                "jaw_contour": [],
                "confidence": {"global": 0.5, "articulare": 0.5, "gonion": 0.5, "menton": 0.5},
                "method": "local_legacy_mp_hybrid",
                "method_source": "local_fallback_legacy",
                "gonial_debug": {"raw_angle": None, "source": "default", "fallback_reason": "default", "pre_validation_angle": None},
            },
            "quality_v2": {
                "pose_yaw": 80.0,
                "occlusion_score": 0.7,
                "lighting_score": 0.7,
                "jawline_visibility_score": 0.0,
                "uncertain_side_landmarks": False,
            },
            "method_source": "local_fallback_legacy",
        }


class TestSideInferenceCircuit(unittest.TestCase):
    def test_circuit_opens_and_fallback_used(self):
        client = SideInferenceClient(
            remote_url="http://invalid-host",
            timeout_seconds=0.5,
            failure_threshold=2,
            recovery_seconds=30.0,
            local_adapter=_DummyAdapter(),
        )
        # Force remote call failure without network dependency.
        client._call_remote = lambda image: (_ for _ in ()).throw(RuntimeError("forced remote failure"))

        img = object()
        res1 = client.analyze_side(img)
        self.assertEqual(res1["transport"]["source"], "local_fallback")
        self.assertFalse(client.state.is_open)

        res2 = client.analyze_side(img)
        self.assertEqual(res2["transport"]["source"], "local_fallback")
        if client.remote_enabled:
            self.assertTrue(client.state.is_open)
        else:
            self.assertEqual(client.state.failure_count, 0)


if __name__ == "__main__":
    unittest.main()
