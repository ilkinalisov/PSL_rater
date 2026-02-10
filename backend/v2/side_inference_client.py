"""Remote side inference client with timeout and circuit-breaker fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


@dataclass
class CircuitState:
    failure_count: int = 0
    open_until_ts: float = 0.0
    last_error: str = ""

    @property
    def is_open(self) -> bool:
        return self.open_until_ts > time.time()


class SideInferenceClient:
    def __init__(
        self,
        remote_url: str,
        timeout_seconds: float,
        failure_threshold: int,
        recovery_seconds: float,
        local_adapter,
    ):
        self.remote_url = (remote_url or "").strip().rstrip("/")
        self.timeout_seconds = float(max(0.5, timeout_seconds))
        self.failure_threshold = int(max(1, failure_threshold))
        self.recovery_seconds = float(max(1.0, recovery_seconds))
        self.local_adapter = local_adapter
        self.state = CircuitState()

    @property
    def remote_enabled(self) -> bool:
        return bool(self.remote_url) and requests is not None

    def _record_success(self):
        self.state.failure_count = 0
        self.state.open_until_ts = 0.0
        self.state.last_error = ""

    def _record_failure(self, message: str):
        self.state.failure_count += 1
        self.state.last_error = message
        if self.state.failure_count >= self.failure_threshold:
            self.state.open_until_ts = time.time() + self.recovery_seconds

    def _call_remote(self, image: np.ndarray) -> Dict:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("Failed to encode side image for remote inference.")

        files = {"image": ("side.jpg", encoded.tobytes(), "image/jpeg")}
        resp = requests.post(
            f"{self.remote_url}/infer/side",
            files=files,
            timeout=self.timeout_seconds,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Remote inference HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        if not payload.get("success", False):
            raise RuntimeError(f"Remote inference failed: {payload.get('error', 'unknown')}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Remote inference returned invalid result.")
        return result

    def analyze_side(self, image: np.ndarray) -> Dict:
        """
        Returns V2 side-analysis dictionary.
        Always deterministic: remote if healthy, otherwise local fallback.
        """
        started = time.time()
        remote_error = None

        if self.remote_enabled and not self.state.is_open:
            try:
                result = self._call_remote(image)
                self._record_success()
                latency_ms = int((time.time() - started) * 1000)
                result["transport"] = {
                    "source": "remote",
                    "latency_ms": latency_ms,
                    "circuit_open": False,
                }
                return result
            except Exception as exc:
                remote_error = str(exc)
                self._record_failure(remote_error)

        # Fallback path (also used when remote disabled).
        result = self.local_adapter.analyze(image)
        latency_ms = int((time.time() - started) * 1000)
        result["transport"] = {
            "source": "local_fallback",
            "latency_ms": latency_ms,
            "circuit_open": bool(self.state.is_open),
            "remote_error": remote_error or self.state.last_error or None,
        }
        return result


def compare_side_v1_v2(legacy_measurements: Dict, v2_result: Dict) -> Dict:
    """Compute lightweight v1-v2 drift diagnostics for shadow mode."""
    v1_gonial = None
    if isinstance(legacy_measurements, dict):
        v1_gonial = legacy_measurements.get("gonial_angle")
    elif hasattr(legacy_measurements, "gonial_angle"):
        v1_gonial = getattr(legacy_measurements, "gonial_angle", None)

    v2_gonial = (v2_result.get("measurements") or {}).get("gonial_angle")
    v2_conf = ((v2_result.get("landmarks_v2") or {}).get("confidence") or {}).get("global")
    v2_gonial_debug = ((v2_result.get("landmarks_v2") or {}).get("gonial_debug") or {})
    delta = None
    if v1_gonial is not None and v2_gonial is not None:
        try:
            delta = round(float(v2_gonial) - float(v1_gonial), 3)
        except Exception:
            delta = None

    return {
        "v1_gonial": v1_gonial,
        "v2_gonial": v2_gonial,
        "gonial_delta": delta,
        "v2_global_confidence": v2_conf,
        "v2_gonial_source": v2_gonial_debug.get("source"),
        "v2_gonial_fallback_reason": v2_gonial_debug.get("fallback_reason"),
        "v2_method_source": (v2_result.get("landmarks_v2") or {}).get("method_source"),
        "transport": v2_result.get("transport", {}),
    }
