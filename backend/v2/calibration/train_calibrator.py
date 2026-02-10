"""Train a lightweight landmark correction regressor from labeled data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


FEATURE_NAMES = [
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

POINTS = ("articulare", "gonion", "menton")


def _feature_vector(row: Dict) -> np.ndarray:
    quality = row.get("quality", {})
    conf = row.get("confidences", {})
    features = [
        1.0,  # bias
        float(quality.get("pose_yaw", 80.0)),
        float(quality.get("occlusion_score", 0.5)),
        float(quality.get("lighting_score", 0.5)),
        float(conf.get("global", 0.5)),
        float(conf.get("articulare", 0.5)),
        float(conf.get("gonion", 0.5)),
        float(conf.get("menton", 0.5)),
        float(row.get("jaw_curvature", 0.0)),
    ]
    return np.array(features, dtype=float)


def _fit_linear(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """Ridge-style closed-form linear regression."""
    xtx = X.T @ X
    reg = np.eye(xtx.shape[0], dtype=float) * l2
    inv = np.linalg.inv(xtx + reg)
    return inv @ X.T @ y


def train(rows: List[Dict]) -> Dict:
    x_rows = []
    targets = {p: {"dx": [], "dy": []} for p in POINTS}

    for row in rows:
        raw = row.get("raw_points", {})
        target = row.get("target_points", {})
        if not raw or not target:
            continue
        if not all(k in raw and k in target for k in POINTS):
            continue

        x = _feature_vector(row)
        x_rows.append(x)
        for p in POINTS:
            dx = float(target[p][0]) - float(raw[p][0])
            dy = float(target[p][1]) - float(raw[p][1])
            targets[p]["dx"].append(dx)
            targets[p]["dy"].append(dy)

    if not x_rows:
        raise ValueError("No valid training rows found. Expected rows with raw_points + target_points.")

    X = np.vstack(x_rows)
    model = {
        "version": "1.0",
        "feature_names": FEATURE_NAMES,
        "points": list(POINTS),
        "max_pixel_shift": 18.0,
        "weights": {},
        "sample_count": int(X.shape[0]),
    }

    for p in POINTS:
        ydx = np.array(targets[p]["dx"], dtype=float)
        ydy = np.array(targets[p]["dy"], dtype=float)
        if len(ydx) != X.shape[0] or len(ydy) != X.shape[0]:
            raise ValueError(f"Target mismatch for point: {p}")
        wdx = _fit_linear(X, ydx).tolist()
        wdy = _fit_linear(X, ydy).tolist()
        model["weights"][p] = {"dx": wdx, "dy": wdy}

    return model


def main():
    parser = argparse.ArgumentParser(description="Train V2 side landmark calibrator.")
    parser.add_argument("--input", required=True, help="Path to JSON training rows")
    parser.add_argument("--output", required=True, help="Path to save calibration model JSON")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text())
    model = train(rows)
    Path(args.output).write_text(json.dumps(model, indent=2))
    print(f"Trained calibrator on {model['sample_count']} samples -> {args.output}")


if __name__ == "__main__":
    main()

