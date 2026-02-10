"""Offline evaluation gates for V2 side-profile landmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .geometry import gonial_angle_from_points, point_distance


def _nme(pred: Dict, gt: Dict, key: str) -> float:
    if key not in pred or key not in gt:
        return float("nan")
    if "nasion" not in gt or "menton" not in gt:
        return float("nan")
    norm = point_distance(gt["nasion"], gt["menton"])
    if norm <= 1e-6:
        return float("nan")
    return point_distance(pred[key], gt[key]) / norm


def evaluate(rows: List[Dict]) -> Dict:
    ar_nme = []
    go_nme = []
    me_nme = []
    angle_abs_err = []
    uncertain_count = 0

    for row in rows:
        pred = row.get("pred_points", {})
        gt = row.get("gt_points", {})
        quality = row.get("quality_v2", {})

        ar = _nme(pred, gt, "articulare")
        go = _nme(pred, gt, "gonion")
        me = _nme(pred, gt, "menton")
        for value, bucket in ((ar, ar_nme), (go, go_nme), (me, me_nme)):
            if np.isfinite(value):
                bucket.append(value)

        pred_ang = gonial_angle_from_points(pred.get("articulare"), pred.get("gonion"), pred.get("menton")) \
            if all(k in pred for k in ("articulare", "gonion", "menton")) else None
        gt_ang = gonial_angle_from_points(gt.get("articulare"), gt.get("gonion"), gt.get("menton")) \
            if all(k in gt for k in ("articulare", "gonion", "menton")) else None
        if pred_ang is not None and gt_ang is not None:
            angle_abs_err.append(abs(pred_ang - gt_ang))

        if bool(quality.get("uncertain_side_landmarks", False)):
            uncertain_count += 1

    n = max(1, len(rows))
    results = {
        "sample_count": len(rows),
        "ar_nme_pct": float(np.nanmean(ar_nme) * 100.0) if ar_nme else None,
        "go_nme_pct": float(np.nanmean(go_nme) * 100.0) if go_nme else None,
        "me_nme_pct": float(np.nanmean(me_nme) * 100.0) if me_nme else None,
        "gonial_mae_deg": float(np.nanmean(angle_abs_err)) if angle_abs_err else None,
        "uncertain_rate_pct": float((uncertain_count / n) * 100.0),
    }
    return results


def gate_pass(results: Dict) -> Dict:
    gates = {
        "ar_nme_pass": results["ar_nme_pct"] is not None and results["ar_nme_pct"] <= 3.0,
        "go_nme_pass": results["go_nme_pct"] is not None and results["go_nme_pct"] <= 2.5,
        "me_nme_pass": results["me_nme_pct"] is not None and results["me_nme_pct"] <= 2.0,
        "gonial_mae_pass": results["gonial_mae_deg"] is not None and results["gonial_mae_deg"] <= 3.5,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def main():
    parser = argparse.ArgumentParser(description="Evaluate V2 holdout metrics against release gates.")
    parser.add_argument("--input", required=True, help="JSON file with rows containing pred_points + gt_points")
    parser.add_argument("--output", required=False, help="Optional output JSON path")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text())
    results = evaluate(rows)
    gates = gate_pass(results)
    payload = {"results": results, "gates": gates}
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

