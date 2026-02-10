"""Landmark calibration runtime for V2 side-profile points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


Point = Tuple[int, int]
DEFAULT_FEATURE_NAMES = [
    "bias",
    "pose_yaw",
    "occlusion_score",
    "lighting_score",
    "global_confidence",
    "articulare_confidence",
    "gonion_confidence",
    "menton_confidence",
    "jaw_curvature",
]


class LandmarkCalibrator:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration model not found: {model_path}")
        self.model = json.loads(path.read_text())
        self.model_path = model_path

    @property
    def is_ready(self) -> bool:
        return self.model is not None and "weights" in self.model

    def _feature_vector(self, context: Dict) -> np.ndarray:
        feature_names: List[str] = self.model.get("feature_names", DEFAULT_FEATURE_NAMES) if self.model else DEFAULT_FEATURE_NAMES
        values = []
        for name in feature_names:
            if name == "bias":
                values.append(1.0)
            else:
                values.append(float(context.get(name, 0.0)))
        return np.array(values, dtype=float)

    def apply(self, points: Dict[str, Point], context: Dict) -> Tuple[Dict[str, Point], Dict[str, float]]:
        """
        Returns:
        - corrected points
        - confidence adjustment per point (0..1 multiplier)
        """
        if not self.is_ready:
            return points, {}

        x = self._feature_vector(context)
        corrected = dict(points)
        confidence_adj: Dict[str, float] = {}
        max_delta = float(self.model.get("max_pixel_shift", 18.0))
        weights = self.model.get("weights", {})

        for point_name, params in weights.items():
            if point_name not in corrected:
                continue
            wdx = np.array(params.get("dx", []), dtype=float)
            wdy = np.array(params.get("dy", []), dtype=float)
            if len(wdx) != len(x) or len(wdy) != len(x):
                continue
            dx = float(np.dot(wdx, x))
            dy = float(np.dot(wdy, x))
            dx = float(np.clip(dx, -max_delta, max_delta))
            dy = float(np.clip(dy, -max_delta, max_delta))
            px, py = corrected[point_name]
            corrected[point_name] = (int(round(px + dx)), int(round(py + dy)))

            mag = float(np.hypot(dx, dy))
            confidence_adj[point_name] = float(np.clip(1.0 - (mag / max(1.0, max_delta * 1.6)), 0.55, 1.0))

        return corrected, confidence_adj

