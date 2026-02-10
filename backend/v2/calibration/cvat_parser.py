"""Parse CVAT exports into a normalized training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


TARGET_POINTS = {"articulare", "gonion", "menton", "nasion", "subnasale", "pogonion"}


def _point_from_shape(shape: Dict):
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None
    return [int(round(float(pts[0]))), int(round(float(pts[1])))]


def parse_cvat_json(path: str) -> List[Dict]:
    """
    Supports common CVAT JSON structures and returns normalized records:
    {
      image_id,
      points: {...},
      jaw_contour: [[x,y], ...],
      metadata: {...}
    }
    """
    data = json.loads(Path(path).read_text())
    records: List[Dict] = []

    # CVAT tasks format with items list.
    items = data.get("items")
    if isinstance(items, list):
        for item in items:
            image_id = str(item.get("id", item.get("name", "unknown")))
            points = {}
            jaw_contour = []
            metadata = dict(item.get("attributes", {}))

            for ann in item.get("annotations", []):
                label = str(ann.get("label") or ann.get("label_id") or "").lower()
                ann_type = ann.get("type", "").lower()
                if ann_type in {"points", "point"} and label in TARGET_POINTS:
                    point = _point_from_shape(ann)
                    if point is not None:
                        points[label] = point
                elif ann_type in {"polyline", "polygon"} and "jaw" in label:
                    raw = ann.get("points", [])
                    jaw_contour = [[int(round(raw[i])), int(round(raw[i + 1]))] for i in range(0, len(raw) - 1, 2)]

            if points:
                records.append({
                    "image_id": image_id,
                    "points": points,
                    "jaw_contour": jaw_contour,
                    "metadata": metadata,
                })
        return records

    # COCO-style keypoints fallback.
    if "images" in data and "annotations" in data:
        image_by_id = {img["id"]: img for img in data.get("images", [])}
        categories = {cat["id"]: cat for cat in data.get("categories", [])}
        for ann in data.get("annotations", []):
            image_id = ann.get("image_id")
            img = image_by_id.get(image_id, {})
            category = categories.get(ann.get("category_id"), {})
            kpt_names = category.get("keypoints", [])
            kpts = ann.get("keypoints", [])
            points = {}
            for idx, name in enumerate(kpt_names):
                base = idx * 3
                if base + 2 >= len(kpts):
                    continue
                x, y, v = kpts[base], kpts[base + 1], kpts[base + 2]
                name = str(name).lower()
                if v > 0 and name in TARGET_POINTS:
                    points[name] = [int(round(x)), int(round(y))]
            if points:
                records.append({
                    "image_id": str(img.get("file_name", image_id)),
                    "points": points,
                    "jaw_contour": [],
                    "metadata": {},
                })
        return records

    raise ValueError("Unsupported CVAT JSON structure.")


def main():
    parser = argparse.ArgumentParser(description="Parse CVAT export into normalized side-landmark records.")
    parser.add_argument("--input", required=True, help="Path to CVAT JSON export")
    parser.add_argument("--output", required=True, help="Path to write normalized JSON records")
    args = parser.parse_args()

    rows = parse_cvat_json(args.input)
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()

