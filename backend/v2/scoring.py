"""V2 scoring and reliability helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_float(value, default: float, low: float, high: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    return float(np.clip(out, low, high))


def compute_front_reliability(front_debug: Dict) -> float:
    """Estimate front reliability from available front debug fields."""
    if not front_debug:
        return 0.75
    landmark_count = float(front_debug.get("detected_landmarks", 0))
    face_landmarks = float(front_debug.get("landmark_count", 468))
    coverage = landmark_count / max(1.0, face_landmarks * 0.08)
    base = np.clip(coverage, 0.0, 1.0)
    return float(np.clip(0.55 + (0.40 * base), 0.0, 1.0))


def compute_scores_v2(
    front_primary: float,
    side_primary: float,
    front_reliability: float,
    side_reliability: float,
    reliability_threshold: float = 0.45,
) -> Dict:
    """
    Reliability-weighted overall summary.
    If reliability is low, overall remains available but down-weighted and flagged.
    """
    front_primary = _safe_float(front_primary, 5.0, 1.0, 10.0)
    side_primary = _safe_float(side_primary, 5.0, 1.0, 10.0)
    front_rel = _safe_float(front_reliability, 0.75, 0.0, 1.0)
    side_rel = _safe_float(side_reliability, 0.35, 0.0, 1.0)

    overall_reliability = float(np.clip(min(front_rel, side_rel), 0.0, 1.0))
    base = (front_primary * 0.50) + (side_primary * 0.50)

    if overall_reliability >= reliability_threshold:
        overall_optional = float(np.clip(base, 1.0, 10.0))
        reliable = True
    else:
        # Deterministic down-weight for uncertain side landmarks.
        attenuation = 0.62 + (0.33 * overall_reliability)
        attenuation *= 0.94 if side_rel < 0.40 else 1.0
        overall_optional = float(np.clip(base * attenuation, 1.0, 10.0))
        reliable = False

    return {
        "front_primary": round(front_primary, 2),
        "side_primary": round(side_primary, 2),
        "overall_optional": round(overall_optional, 2),
        "overall_reliability": round(overall_reliability, 3),
        "reliable": reliable,
    }
