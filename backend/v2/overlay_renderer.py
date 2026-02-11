"""V2 side overlay renderer from finalized landmarks."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Polyline = List[Point]
RegressionModel = Tuple[str, float, float]
OVERLAY_RENDERER_VERSION = "v2_landmarks_renderer@4"


def _to_point(val: Sequence[float]) -> Point:
    return (int(round(float(val[0]))), int(round(float(val[1]))))


def _extract_points(points: Dict[str, Sequence[float]]) -> Dict[str, Point]:
    out: Dict[str, Point] = {}
    for key, val in (points or {}).items():
        if val is None or len(val) < 2:
            continue
        out[key] = _to_point(val)
    return out


def _as_polyline(polyline: Optional[Iterable[Sequence[float]]]) -> Polyline:
    pts: Polyline = []
    for pt in polyline or []:
        if pt is None or len(pt) < 2:
            continue
        p = _to_point(pt)
        if not pts or pts[-1] != p:
            pts.append(p)
    return pts


def _nearest_point_index(polyline: Polyline, pt: Point, max_dist_px: float) -> Optional[int]:
    if not polyline:
        return None
    arr = np.array(polyline, dtype=float)
    q = np.array(pt, dtype=float)
    d = np.linalg.norm(arr - q, axis=1)
    idx = int(np.argmin(d))
    if float(d[idx]) <= float(max_dist_px):
        return idx
    return None


def _extract_poly_segment(polyline: Polyline, idx_a: int, idx_b: int) -> Polyline:
    if not polyline:
        return []
    idx_a = int(np.clip(idx_a, 0, len(polyline) - 1))
    idx_b = int(np.clip(idx_b, 0, len(polyline) - 1))
    if idx_a <= idx_b:
        return list(polyline[idx_a : idx_b + 1])
    return list(reversed(polyline[idx_b : idx_a + 1]))


def _densify_polyline(polyline: Polyline, step_px: float = 2.5) -> Polyline:
    if len(polyline) < 2:
        return list(polyline)
    out: Polyline = [polyline[0]]
    step = max(0.8, float(step_px))
    for i in range(len(polyline) - 1):
        p0 = np.array(polyline[i], dtype=float)
        p1 = np.array(polyline[i + 1], dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        n = max(1, int(math.ceil(dist / step)))
        for j in range(1, n + 1):
            t = float(j) / float(n)
            p = p0 + ((p1 - p0) * t)
            pi = (int(round(float(p[0]))), int(round(float(p[1]))))
            if out[-1] != pi:
                out.append(pi)
    return out


def _smooth_polyline(polyline: Polyline, window: int = 5) -> Polyline:
    if len(polyline) < 3 or window <= 1:
        return list(polyline)
    w = int(window)
    if w % 2 == 0:
        w += 1
    arr = np.array(polyline, dtype=float)
    pad = w // 2
    x = np.pad(arr[:, 0], (pad, pad), mode="edge")
    y = np.pad(arr[:, 1], (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    sx = np.convolve(x, kernel, mode="valid")
    sy = np.convolve(y, kernel, mode="valid")
    out: Polyline = []
    for i in range(len(sx)):
        p = (int(round(float(sx[i]))), int(round(float(sy[i]))))
        if not out or out[-1] != p:
            out.append(p)
    return out


def _polyline_len(polyline: Polyline) -> float:
    if len(polyline) < 2:
        return 0.0
    arr = np.array(polyline, dtype=float)
    seg = arr[1:] - arr[:-1]
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + (ab * t)
    return float(np.linalg.norm(point - proj))


def _residual_to_line(polyline: Polyline, a: Point, b: Point) -> float:
    if len(polyline) < 2:
        return 1e6
    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    d = []
    for p in polyline:
        d.append(_point_segment_distance(np.array(p, dtype=float), aa, bb))
    return float(np.median(d)) if d else 1e6


def _polyline_curvature(polyline: Polyline) -> float:
    if len(polyline) < 3:
        return 0.0
    arr = np.array(polyline, dtype=float)
    turns = []
    for i in range(1, len(arr) - 1):
        v1 = arr[i] - arr[i - 1]
        v2 = arr[i + 1] - arr[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        turns.append(float(np.degrees(np.arccos(cosv))))
    return float(np.mean(turns)) if turns else 0.0


def _quadratic_bezier(p0: Point, p1: Point, p2: Point, samples: int = 22) -> Polyline:
    out: Polyline = []
    n = max(8, int(samples))
    a = np.array(p0, dtype=float)
    b = np.array(p1, dtype=float)
    c = np.array(p2, dtype=float)
    for i in range(n + 1):
        t = float(i) / float(n)
        pt = ((1.0 - t) ** 2) * a + (2.0 * (1.0 - t) * t * b) + ((t ** 2) * c)
        p = (int(round(float(pt[0]))), int(round(float(pt[1]))))
        if not out or out[-1] != p:
            out.append(p)
    return out


def _line_quality(polyline: Polyline, a: Point, b: Point) -> float:
    if len(polyline) < 2:
        return 0.0
    ab = float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))
    if ab < 1e-3:
        return 0.0
    plen = _polyline_len(polyline)
    residual = _residual_to_line(polyline, a, b)
    curvature = _polyline_curvature(polyline)
    len_ratio = float(np.clip(plen / max(ab, 1.0), 1.0, 2.6))
    shape_score = float(np.clip((len_ratio - 1.0) / 1.2, 0.0, 1.0))
    residual_score = float(np.clip(1.0 - (residual / max(8.0, 0.32 * ab)), 0.0, 1.0))
    curvature_score = float(np.clip(curvature / 24.0, 0.0, 1.0))
    return float(np.clip((0.50 * residual_score) + (0.30 * shape_score) + (0.20 * curvature_score), 0.0, 1.0))


def _fit_segment_polyfit(points: Polyline) -> Optional[RegressionModel]:
    if len(points) < 2:
        return None
    arr = np.array(points, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    try:
        if var_x >= var_y:
            if float(np.ptp(x)) < 1e-6:
                return None
            m, b = np.polyfit(x, y, deg=1)
            return ("xy", float(m), float(b))
        if float(np.ptp(y)) < 1e-6:
            return None
        m, b = np.polyfit(y, x, deg=1)
        return ("yx", float(m), float(b))
    except Exception:
        return None


def _segment_sse(points: Polyline, model: RegressionModel) -> float:
    if len(points) < 2:
        return 1e9
    axis, m, b = model
    arr = np.array(points, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    if axis == "xy":
        pred = (m * x) + b
        resid = y - pred
    else:
        pred = (m * y) + b
        resid = x - pred
    return float(np.sum(resid * resid))


def _model_to_implicit_line(model: RegressionModel) -> Tuple[float, float, float]:
    axis, m, b = model
    if axis == "xy":
        return (float(m), -1.0, float(b))
    return (1.0, float(-m), float(-b))


def _intersect_lines(
    l1: Tuple[float, float, float],
    l2: Tuple[float, float, float],
) -> Optional[Tuple[float, float]]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = (a1 * b2) - (a2 * b1)
    if abs(det) < 1e-6:
        return None
    x = ((b1 * c2) - (b2 * c1)) / det
    y = ((c1 * a2) - (c2 * a1)) / det
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return (float(x), float(y))


def _point_to_polyline_distance(point: Point, polyline: Polyline) -> float:
    if len(polyline) < 2:
        return 1e6
    p = np.array(point, dtype=float)
    best = 1e9
    for i in range(len(polyline) - 1):
        a = np.array(polyline[i], dtype=float)
        b = np.array(polyline[i + 1], dtype=float)
        best = min(best, _point_segment_distance(p, a, b))
    return float(best)


def _fit_piecewise_jawline_regression(
    contour: Polyline,
    menton: Point,
    articulare: Point,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "regression_ok": False,
        "mandibular_path": [],
        "ramus_path": [],
        "gonion_intersection": None,
        "split_idx": -1,
        "sse_total": 0.0,
    }
    if len(contour) < 8:
        return out

    arr = np.array(contour, dtype=float)
    min_x = float(np.min(arr[:, 0]))
    max_x = float(np.max(arr[:, 0]))
    min_y = float(np.min(arr[:, 1]))
    max_y = float(np.max(arr[:, 1]))
    diag = float(np.hypot(max_x - min_x, max_y - min_y))
    max_anchor_dist = max(18.0, 0.22 * diag)

    idx_me = _nearest_point_index(contour, menton, max_anchor_dist)
    idx_ar = _nearest_point_index(contour, articulare, max_anchor_dist)
    if idx_me is None or idx_ar is None:
        return out

    seg = _extract_poly_segment(contour, idx_me, idx_ar)
    seg = _smooth_polyline(_densify_polyline(seg, step_px=2.2), window=5)
    if len(seg) < 8:
        return out

    min_pts = 4
    best = None
    for split_idx in range(min_pts - 1, len(seg) - min_pts):
        left = seg[: split_idx + 1]
        right = seg[split_idx:]
        if len(left) < min_pts or len(right) < min_pts:
            continue
        left_model = _fit_segment_polyfit(left)
        right_model = _fit_segment_polyfit(right)
        if left_model is None or right_model is None:
            continue
        sse_total = _segment_sse(left, left_model) + _segment_sse(right, right_model)
        if best is None or sse_total < best["sse_total"]:
            best = {
                "split_idx": split_idx,
                "left_model": left_model,
                "right_model": right_model,
                "sse_total": float(sse_total),
            }

    if best is None:
        return out

    intersection = _intersect_lines(
        _model_to_implicit_line(best["left_model"]),
        _model_to_implicit_line(best["right_model"]),
    )
    if intersection is None:
        return out
    inter_pt = (int(round(intersection[0])), int(round(intersection[1])))

    margin = max(12.0, 0.10 * diag)
    if not (min_x - margin <= inter_pt[0] <= max_x + margin and min_y - margin <= inter_pt[1] <= max_y + margin):
        return out
    if _point_to_polyline_distance(inter_pt, seg) > max(24.0, 0.18 * diag):
        return out
    if float(np.linalg.norm(np.array(inter_pt, dtype=float) - np.array(menton, dtype=float))) < 6.0:
        return out
    if float(np.linalg.norm(np.array(inter_pt, dtype=float) - np.array(articulare, dtype=float))) < 6.0:
        return out

    out.update(
        {
            "regression_ok": True,
            "mandibular_path": [menton, inter_pt],
            "ramus_path": [inter_pt, articulare],
            "gonion_intersection": inter_pt,
            "split_idx": int(best["split_idx"]),
            "sse_total": float(best["sse_total"]),
        }
    )
    return out


def _build_overlay_edge_map(image: np.ndarray, mode: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0.9)
    median_val = float(np.median(blur))
    if mode == "mono":
        lo = int(max(0.0, 0.45 * median_val))
        hi = int(min(255.0, 1.30 * median_val))
    elif mode == "hybrid":
        lo = int(max(0.0, 0.50 * median_val))
        hi = int(min(255.0, 1.40 * median_val))
    else:
        lo = int(max(0.0, 0.55 * median_val))
        hi = int(min(255.0, 1.55 * median_val))
    canny = cv2.Canny(blur, lo, hi)

    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, grad_bin = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grad_edges = cv2.Canny(grad_bin, 30, 120)

    if mode in ("mono", "hybrid"):
        edges = cv2.bitwise_or(canny, grad_edges)
    else:
        edges = canny
    edges = cv2.dilate(edges, None, iterations=1)
    return edges


def _snap_polyline_to_edges(polyline: Polyline, edge_map: np.ndarray, radius_px: int = 2) -> Polyline:
    if len(polyline) < 2:
        return list(polyline)
    h, w = edge_map.shape[:2]
    r = max(1, int(radius_px))
    snapped: Polyline = []
    for p in polyline:
        x, y = int(p[0]), int(p[1])
        x0 = max(0, x - r)
        y0 = max(0, y - r)
        x1 = min(w - 1, x + r)
        y1 = min(h - 1, y + r)
        patch = edge_map[y0 : y1 + 1, x0 : x1 + 1]
        yy, xx = np.where(patch > 0)
        if len(xx) == 0:
            cand = (x, y)
        else:
            dx = xx.astype(float) + float(x0) - float(x)
            dy = yy.astype(float) + float(y0) - float(y)
            idx = int(np.argmin((dx * dx) + (dy * dy)))
            cand = (int(xx[idx] + x0), int(yy[idx] + y0))
        if not snapped or snapped[-1] != cand:
            snapped.append(cand)
    return _smooth_polyline(snapped, window=3)


def _maybe_snap_path(path: Polyline, edge_map: np.ndarray, radius_px: int, a: Point, b: Point) -> Tuple[Polyline, bool]:
    if len(path) < 2:
        return path, False
    snapped = _snap_polyline_to_edges(path, edge_map, radius_px=radius_px)
    if len(snapped) < 2:
        return path, False
    q_before = _line_quality(path, a, b)
    q_after = _line_quality(snapped, a, b)
    curv_before = _polyline_curvature(path)
    curv_after = _polyline_curvature(snapped)
    shift = float(np.linalg.norm(np.array(path[0], dtype=float) - np.array(snapped[0], dtype=float)))
    shift += float(np.linalg.norm(np.array(path[-1], dtype=float) - np.array(snapped[-1], dtype=float)))
    if q_after + 0.03 < q_before:
        return path, False
    if curv_after > ((curv_before * 1.65) + 6.0):
        return path, False
    if shift > (radius_px * 4.2):
        return path, False
    return snapped, True


def _draw_polyline(
    overlay: np.ndarray,
    contour: Iterable[Sequence[float]],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    pts = _as_polyline(contour)
    if len(pts) < 2:
        return
    cv2.polylines(
        overlay,
        [np.array(pts, dtype=np.int32)],
        False,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _build_legacy_gonial_overlay_paths(
    points: Dict[str, Point],
    jaw_contour: Optional[List[Sequence[float]]],
    image_shape: Tuple[int, int, int],
    processing_mode: str = "color",
) -> Dict[str, object]:
    _ = processing_mode  # kept explicit for stable signature
    if not all(k in points for k in ("articulare", "gonion", "menton")):
        return {
            "mandibular_path": [],
            "ramus_path": [],
            "overlay_geometry_source": "straight_fallback",
            "overlay_path_quality": 0.0,
            "ramus_display_mode": "straight_fallback",
            "gonion_overlay_point": None,
            "overlay_regression_split_idx": None,
            "overlay_regression_sse": None,
        }

    h = int(image_shape[0])
    w = int(image_shape[1])
    ar = points["articulare"]
    go = points["gonion"]
    me = points["menton"]
    contour = _as_polyline(jaw_contour)

    mandibular_path: Polyline = [me, go]
    ramus_path: Polyline = [go, ar]
    use_mandibular_contour = False
    use_ramus_contour = False
    ramus_display_mode = "straight_fallback"

    if len(contour) >= 6:
        max_me_dist = max(16.0, min(float(w), float(h)) * 0.08)
        max_go_dist = max(14.0, min(float(w), float(h)) * 0.07)
        idx_me = _nearest_point_index(contour, me, max_me_dist)
        idx_go = _nearest_point_index(contour, go, max_go_dist)
        if idx_me is not None and idx_go is not None:
            seg = _extract_poly_segment(contour, idx_me, idx_go)
            seg = _smooth_polyline(_densify_polyline(seg, step_px=2.5), window=5)
            seg_len = _polyline_len(seg)
            base = float(np.linalg.norm(np.array(me, dtype=float) - np.array(go, dtype=float)))
            seg_res = _residual_to_line(seg, me, go)
            if len(seg) >= 4 and seg_len >= max(10.0, 0.68 * base) and seg_res <= max(18.0, 0.42 * base):
                mandibular_path = seg
                use_mandibular_contour = True

        idx_go_r = _nearest_point_index(contour, go, max_go_dist)
        idx_ar = _nearest_point_index(contour, ar, max(18.0, min(float(w), float(h)) * 0.09))
        if idx_go_r is not None and idx_ar is not None:
            rseg = _extract_poly_segment(contour, idx_go_r, idx_ar)
            rseg = _smooth_polyline(_densify_polyline(rseg, step_px=2.3), window=5)
            rlen = _polyline_len(rseg)
            base = float(np.linalg.norm(np.array(go, dtype=float) - np.array(ar, dtype=float)))
            rres = _residual_to_line(rseg, go, ar)
            if len(rseg) >= 4 and rlen >= max(8.0, 0.52 * base) and rres <= max(16.0, 0.46 * base):
                ramus_path = rseg
                use_ramus_contour = True
                ramus_display_mode = "contour"

    if not use_ramus_contour:
        allow_hybrid = use_mandibular_contour or len(contour) >= 4
        if allow_hybrid:
            go_arr = np.array(go, dtype=float)
            ar_arr = np.array(ar, dtype=float)
            me_arr = np.array(me, dtype=float)
            target = ar_arr - go_arr
            dist = float(np.linalg.norm(target))
            if dist < 6.0:
                dist = max(18.0, float(np.linalg.norm(go_arr - me_arr)) * 0.62)

            # Enforce near-vertical posterior ramus display to match intended contour behavior.
            max_dx = max(4.0, min(float(w), float(h)) * 0.045)
            dx = float(np.clip(target[0], -max_dx, max_dx))
            dy = float(target[1])
            if dy >= -6.0:
                dy = -max(14.0, dist * 0.75)
            end = go_arr + np.array([dx, dy], dtype=float)

            # Slight curvature toward posterior while keeping mostly vertical orientation.
            ctrl_dx = float(np.clip(dx * 0.45, -max_dx * 0.75, max_dx * 0.75))
            ctrl = go_arr + np.array([ctrl_dx, dy * 0.58], dtype=float)
            ramus_path = _smooth_polyline(
                _quadratic_bezier(
                    go,
                    (int(round(float(ctrl[0]))), int(round(float(ctrl[1])))),
                    (int(round(float(end[0]))), int(round(float(end[1])))),
                    samples=20,
                ),
                window=5,
            )
            ramus_display_mode = "hybrid_vertical"
        else:
            ramus_path = [go, ar]
            ramus_display_mode = "straight_fallback"

    if mandibular_path:
        mandibular_path[0] = me
        mandibular_path[-1] = go
    if ramus_path:
        ramus_path[0] = go
        if ramus_display_mode == "contour":
            ramus_path[-1] = ar

    source = "straight_fallback"
    if use_mandibular_contour and use_ramus_contour:
        source = "contour"
    elif use_mandibular_contour or ramus_display_mode == "hybrid_vertical":
        source = "hybrid"

    ramus_ref_end = ar if ramus_display_mode != "hybrid_vertical" else ramus_path[-1]
    q_m = _line_quality(mandibular_path, me, go)
    q_r = _line_quality(ramus_path, go, ramus_ref_end)
    path_quality = float(np.clip((0.55 * q_m) + (0.45 * q_r), 0.0, 1.0))
    if source == "straight_fallback":
        path_quality = min(path_quality, 0.34)

    return {
        "mandibular_path": mandibular_path,
        "ramus_path": ramus_path,
        "overlay_geometry_source": source,
        "overlay_path_quality": round(path_quality, 3),
        "ramus_display_mode": ramus_display_mode,
        "gonion_overlay_point": go,
        "overlay_regression_split_idx": None,
        "overlay_regression_sse": None,
    }


def _build_gonial_overlay_paths(
    points: Dict[str, Point],
    jaw_contour: Optional[List[Sequence[float]]],
    image_shape: Tuple[int, int, int],
    processing_mode: str = "color",
) -> Dict[str, object]:
    if not all(k in points for k in ("articulare", "gonion", "menton")):
        return _build_legacy_gonial_overlay_paths(points, jaw_contour, image_shape, processing_mode=processing_mode)

    ar = points["articulare"]
    me = points["menton"]
    contour = _as_polyline(jaw_contour)
    if len(contour) >= 8:
        regression = _fit_piecewise_jawline_regression(contour, menton=me, articulare=ar)
        if bool(regression.get("regression_ok")):
            gonion_overlay = regression.get("gonion_intersection")
            mandibular_path = list(regression.get("mandibular_path") or [])
            ramus_path = list(regression.get("ramus_path") or [])
            if (
                isinstance(gonion_overlay, tuple) and
                len(mandibular_path) >= 2 and
                len(ramus_path) >= 2
            ):
                q_m = _line_quality(mandibular_path, me, gonion_overlay)
                q_r = _line_quality(ramus_path, gonion_overlay, ar)
                path_quality = float(np.clip((0.55 * q_m) + (0.45 * q_r), 0.0, 1.0))
                return {
                    "mandibular_path": mandibular_path,
                    "ramus_path": ramus_path,
                    "overlay_geometry_source": "piecewise_regression",
                    "overlay_path_quality": round(path_quality, 3),
                    "ramus_display_mode": "piecewise_regression",
                    "gonion_overlay_point": gonion_overlay,
                    "overlay_regression_split_idx": int(regression.get("split_idx", -1)),
                    "overlay_regression_sse": round(float(regression.get("sse_total", 0.0)), 3),
                }

    return _build_legacy_gonial_overlay_paths(points, jaw_contour, image_shape, processing_mode=processing_mode)


def build_gonial_overlay_paths(
    points: Dict[str, Sequence[float]],
    jaw_contour: Optional[List[Sequence[float]]],
    image_shape: Tuple[int, int, int],
    processing_mode: str = "color",
) -> Dict[str, object]:
    return _build_gonial_overlay_paths(
        points=_extract_points(points),
        jaw_contour=jaw_contour,
        image_shape=image_shape,
        processing_mode=processing_mode,
    )


def draw_gonial_overlay(
    overlay: np.ndarray,
    points: Dict[str, Sequence[float]],
    jaw_contour: Optional[List[Sequence[float]]] = None,
    *,
    gonial_debug: Optional[Dict] = None,
    processing_mode: str = "color",
    monochrome_score: float = 0.0,
    color: Tuple[int, int, int] = (255, 140, 0),
    thickness: int = 2,
    point_radius: int = 3,
) -> Dict[str, object]:
    pts = _extract_points(points)
    if not all(k in pts for k in ("articulare", "gonion", "menton")):
        meta = {
            "overlay_geometry_source": "straight_fallback",
            "overlay_snap_mode": "off",
            "overlay_path_quality": 0.0,
            "ramus_display_mode": "straight_fallback",
            "overlay_regression_split_idx": None,
            "overlay_regression_sse": None,
        }
        if isinstance(gonial_debug, dict):
            gonial_debug.update(meta)
        return meta

    ar = pts["articulare"]
    go = pts["gonion"]
    me = pts["menton"]
    path_info = _build_gonial_overlay_paths(pts, jaw_contour, overlay.shape, processing_mode=processing_mode)
    gonion_overlay = path_info.get("gonion_overlay_point")
    if not isinstance(gonion_overlay, tuple):
        gonion_overlay = go

    mandibular_path: Polyline = list(path_info["mandibular_path"]) if path_info.get("mandibular_path") else [me, gonion_overlay]
    ramus_path: Polyline = list(path_info["ramus_path"]) if path_info.get("ramus_path") else [gonion_overlay, ar]
    source = str(path_info.get("overlay_geometry_source", "straight_fallback"))
    path_quality = float(path_info.get("overlay_path_quality", 0.0))
    ramus_display_mode = str(path_info.get("ramus_display_mode", "straight_fallback"))
    overlay_regression_split_idx = path_info.get("overlay_regression_split_idx")
    overlay_regression_sse = path_info.get("overlay_regression_sse")
    ramus_ref_end = ar if ramus_display_mode != "hybrid_vertical" else ramus_path[-1]

    snap_mode = "off"
    want_snap = processing_mode in ("mono", "hybrid") or float(monochrome_score) >= 0.62
    if want_snap:
        effective_mode = processing_mode if processing_mode in ("mono", "hybrid", "color") else "hybrid"
        edge_map = _build_overlay_edge_map(overlay, effective_mode)
        radius = 3 if effective_mode == "mono" else 2
        snapped_m, used_m = _maybe_snap_path(mandibular_path, edge_map, radius, me, gonion_overlay)
        snapped_r, used_r = _maybe_snap_path(ramus_path, edge_map, radius, gonion_overlay, ramus_ref_end)
        if used_m:
            mandibular_path = snapped_m
        if used_r:
            ramus_path = snapped_r
        if used_m or used_r:
            snap_mode = effective_mode
            q_m = _line_quality(mandibular_path, me, gonion_overlay)
            q_r = _line_quality(ramus_path, gonion_overlay, ramus_ref_end)
            path_quality = float(np.clip((0.55 * q_m) + (0.45 * q_r), 0.0, 1.0))

    if len(mandibular_path) < 2:
        mandibular_path = [me, gonion_overlay]
    if len(ramus_path) < 2:
        ramus_path = [gonion_overlay, ar]
    mandibular_path[0] = me
    mandibular_path[-1] = gonion_overlay
    ramus_path[0] = gonion_overlay
    if ramus_display_mode == "hybrid_vertical":
        end = ramus_path[-1]
        max_dx = max(4, int(min(overlay.shape[0], overlay.shape[1]) * 0.05))
        x = int(np.clip(end[0], gonion_overlay[0] - max_dx, gonion_overlay[0] + max_dx))
        y = int(min(end[1], gonion_overlay[1] - 8))
        ramus_path[-1] = (x, y)
        ramus_ref_end = ramus_path[-1]
    elif ramus_display_mode == "contour":
        ramus_path[-1] = ar
        ramus_ref_end = ar
    else:
        ramus_path[-1] = ar
        ramus_ref_end = ar

    _draw_polyline(overlay, mandibular_path, color, thickness=thickness)
    _draw_polyline(overlay, ramus_path, color, thickness=thickness)

    vertex = np.array(gonion_overlay, dtype=float)
    v1 = np.array(ar, dtype=float) - vertex
    v2 = np.array(me, dtype=float) - vertex
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 > 1e-6 and n2 > 1e-6:
        t1 = math.atan2(float(v1[1]), float(v1[0]))
        t2 = math.atan2(float(v2[1]), float(v2[0]))
        sweep = t2 - t1
        while sweep <= -math.pi:
            sweep += 2.0 * math.pi
        while sweep > math.pi:
            sweep -= 2.0 * math.pi
        radius = max(16, int(min(n1, n2) * 0.26))
        count = max(12, int(abs(sweep) * radius / 3.0))
        arc = []
        for i in range(count + 1):
            t = float(i) / float(max(1, count))
            th = t1 + (sweep * t)
            arc.append(
                (
                    int(round(float(vertex[0]) + (radius * math.cos(th)))),
                    int(round(float(vertex[1]) + (radius * math.sin(th)))),
                )
            )
        cv2.polylines(
            overlay,
            [np.array(arc, dtype=np.int32)],
            False,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    pr = max(2, int(point_radius))
    for p in (ar, gonion_overlay, me):
        cv2.circle(overlay, p, pr, color, -1, lineType=cv2.LINE_AA)

    meta = {
        "overlay_geometry_source": source,
        "overlay_snap_mode": snap_mode,
        "overlay_path_quality": round(float(path_quality), 3),
        "ramus_display_mode": ramus_display_mode,
        "overlay_regression_split_idx": overlay_regression_split_idx,
        "overlay_regression_sse": overlay_regression_sse,
    }
    if isinstance(gonial_debug, dict):
        gonial_debug.update(meta)
    return meta


def _compute_naso_frontal_overlay(points: Dict[str, Point]) -> Optional[Dict[str, object]]:
    if "nasion" not in points:
        return None

    nasion = points["nasion"]
    forehead = None
    forehead_source = None
    for key in ("trichion", "glabella"):
        if key in points:
            forehead = points[key]
            forehead_source = key
            break
    if forehead is None or forehead_source is None:
        return None

    nose_ref = None
    nose_source = None
    for key in ("pronasale", "subnasale"):
        if key in points:
            nose_ref = points[key]
            nose_source = key
            break
    if nose_ref is None or nose_source is None:
        return None

    forehead_vec = np.array([float(forehead[0] - nasion[0]), float(forehead[1] - nasion[1])], dtype=float)
    nose_vec = np.array([float(nose_ref[0] - nasion[0]), float(nose_ref[1] - nasion[1])], dtype=float)
    nf = float(np.linalg.norm(forehead_vec))
    nn = float(np.linalg.norm(nose_vec))
    if nf < 1e-6 or nn < 1e-6:
        return None

    cosv = float(np.clip(np.dot(forehead_vec, nose_vec) / (nf * nn), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosv)))
    return {
        "nasion": nasion,
        "forehead": forehead,
        "angle": angle,
        "source": f"{forehead_source}->{nose_source}",
    }


def render_side_overlay(
    image: np.ndarray,
    points: Dict[str, Sequence[float]],
    jaw_contour: Optional[List[Sequence[float]]] = None,
    *,
    gonial_debug: Optional[Dict] = None,
    processing_mode: Optional[str] = None,
    monochrome_score: Optional[float] = None,
) -> np.ndarray:
    """
    Draw side overlay from finalized v2 landmarks.
    - Green: profile/jaw contour
    - Blue: contour-first Ar-Go-Me gonial geometry
    """
    if image is None:
        return image

    overlay = image.copy()
    pts = _extract_points(points)

    if jaw_contour:
        _draw_polyline(overlay, jaw_contour, (0, 255, 0), thickness=2)
    else:
        fallback = []
        for key in ("subnasale", "labrale_superius", "labrale_inferius", "menton", "gonion", "tragion"):
            if key in pts:
                fallback.append(pts[key])
        _draw_polyline(overlay, fallback, (0, 255, 0), thickness=2)

    profile_keys = [
        "trichion",
        "glabella",
        "nasion",
        "pronasale",
        "subnasale",
        "labrale_superius",
        "labrale_inferius",
        "menton",
    ]
    profile_pts = [pts[k] for k in profile_keys if k in pts]
    if len(profile_pts) >= 2:
        _draw_polyline(overlay, profile_pts, (0, 255, 0), thickness=2)

    nf_overlay = _compute_naso_frontal_overlay(pts)
    if nf_overlay is not None:
        nasion = nf_overlay["nasion"]
        forehead = nf_overlay["forehead"]
        angle = float(nf_overlay["angle"])
        _draw_polyline(overlay, [nasion, forehead], (0, 255, 0), thickness=2)

        tx = int(np.clip(nasion[0] - 44, 4, max(4, overlay.shape[1] - 92)))
        ty = int(np.clip(nasion[1] - 12, 16, max(16, overlay.shape[0] - 4)))
        cv2.putText(
            overlay,
            f"NF:{angle:.1f}deg",
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
        if isinstance(gonial_debug, dict):
            gonial_debug["naso_frontal_angle"] = round(angle, 3)
            gonial_debug["naso_frontal_source"] = str(nf_overlay.get("source", "unknown"))
            gonial_debug["naso_frontal_segment"] = {
                "nasion": [int(nasion[0]), int(nasion[1])],
                "forehead": [int(forehead[0]), int(forehead[1])],
            }
    elif isinstance(gonial_debug, dict):
        gonial_debug.pop("naso_frontal_angle", None)
        gonial_debug.pop("naso_frontal_source", None)
        gonial_debug.pop("naso_frontal_segment", None)

    mode = str(processing_mode or (gonial_debug or {}).get("processing_mode", "color"))
    mono = float(monochrome_score if monochrome_score is not None else (gonial_debug or {}).get("monochrome_score", 0.0))
    draw_gonial_overlay(
        overlay=overlay,
        points=pts,
        jaw_contour=jaw_contour,
        gonial_debug=gonial_debug,
        processing_mode=mode,
        monochrome_score=mono,
        color=(255, 140, 0),
        thickness=2,
        point_radius=3,
    )

    cv2.addWeighted(overlay, 0.56, image, 0.44, 0, image)
    return image
