"""Geometry helpers for V2 side-profile landmarking."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Point = Tuple[int, int]


def point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def angle_3pt(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> Optional[float]:
    """Return angle ABC in degrees."""
    ba = np.array([float(a[0]) - float(b[0]), float(a[1]) - float(b[1])], dtype=float)
    bc = np.array([float(c[0]) - float(b[0]), float(c[1]) - float(b[1])], dtype=float)
    mba = np.linalg.norm(ba)
    mbc = np.linalg.norm(bc)
    if mba < 1e-8 or mbc < 1e-8:
        return None
    cosv = np.clip(np.dot(ba, bc) / (mba * mbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def gonial_angle_from_points(articulare: Point, gonion: Point, menton: Point) -> Optional[float]:
    raw = angle_3pt(articulare, gonion, menton)
    if raw is None:
        return None
    # Keep obtuse anatomical interpretation where needed.
    if raw < 90.0:
        raw = 180.0 - raw
    return float(raw)


def subsample_polyline(points: Iterable[Sequence[float]], step: int = 4) -> List[List[int]]:
    out: List[List[int]] = []
    if step <= 0:
        step = 1
    for idx, pt in enumerate(points):
        if idx % step != 0:
            continue
        out.append([int(round(float(pt[0]))), int(round(float(pt[1])))])
    return out


def build_default_jaw_contour(points: Dict[str, Point]) -> List[List[int]]:
    ordered = []
    for key in ("menton", "pogonion", "gonion", "tragion", "articulare"):
        if key in points and points[key] is not None:
            ordered.append([int(points[key][0]), int(points[key][1])])
    dedup: List[List[int]] = []
    seen = set()
    for pt in ordered:
        key = (pt[0], pt[1])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(pt)
    return dedup


def contour_curvature_score(contour: Sequence[Sequence[float]]) -> float:
    """Simple curvature proxy based on directional changes."""
    if contour is None or len(contour) < 3:
        return 0.0
    turns = []
    for i in range(1, len(contour) - 1):
        p0 = np.array(contour[i - 1], dtype=float)
        p1 = np.array(contour[i], dtype=float)
        p2 = np.array(contour[i + 1], dtype=float)
        v1 = p0 - p1
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        turns.append(float(np.degrees(np.arccos(cosang))))
    if not turns:
        return 0.0
    return float(np.mean(turns))


def check_landmark_invariants(points: Dict[str, Point]) -> Dict[str, bool]:
    """
    In left-facing normalized view:
    - Go should be posterior to Me (x larger)
    - Ar should be superior to Go (y smaller)
    """
    out = {
        "has_required_points": all(k in points for k in ("articulare", "gonion", "menton")),
        "go_posterior_to_me": False,
        "ar_superior_to_go": False,
    }
    if not out["has_required_points"]:
        return out
    go = points["gonion"]
    me = points["menton"]
    ar = points["articulare"]
    out["go_posterior_to_me"] = int(go[0]) >= int(me[0])
    out["ar_superior_to_go"] = int(ar[1]) <= int(go[1])
    return out

