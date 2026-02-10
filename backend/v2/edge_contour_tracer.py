"""Edge-traced side contour extraction (silhouette + jaw/ramus)."""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Path = List[Point]


def _to_point(value: Sequence[float]) -> Optional[Point]:
    if value is None or len(value) < 2:
        return None
    return (int(round(float(value[0]))), int(round(float(value[1]))))


def _in_bounds(pt: Point, w: int, h: int) -> Point:
    return (
        int(np.clip(pt[0], 0, max(0, w - 1))),
        int(np.clip(pt[1], 0, max(0, h - 1))),
    )


def _distance(a: Point, b: Point) -> float:
    return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def _sample_line(a: Point, b: Point, step: float = 2.0) -> Path:
    dist = _distance(a, b)
    if dist < 1.0:
        return [a, b] if a != b else [a]
    count = max(2, int(math.ceil(dist / max(1.0, step))))
    out: Path = []
    for i in range(count + 1):
        t = float(i) / float(count)
        x = int(round((1.0 - t) * a[0] + (t * b[0])))
        y = int(round((1.0 - t) * a[1] + (t * b[1])))
        pt = (x, y)
        if not out or out[-1] != pt:
            out.append(pt)
    return out


def _simplify(path: Path, epsilon_ratio: float = 0.008) -> Path:
    if len(path) < 3:
        return list(path)
    arr = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
    peri = cv2.arcLength(arr, False)
    if peri <= 0.0:
        return list(path)
    eps = float(max(1.0, epsilon_ratio * peri))
    approx = cv2.approxPolyDP(arr, eps, False)
    out: Path = []
    for p in approx.reshape((-1, 2)).tolist():
        pt = (int(p[0]), int(p[1]))
        if not out or out[-1] != pt:
            out.append(pt)
    if out and path:
        out[0] = path[0]
        out[-1] = path[-1]
    return out if len(out) >= 2 else list(path)


def _combine_paths(paths: List[Path]) -> Path:
    out: Path = []
    for p in paths:
        if not p:
            continue
        if not out:
            out.extend(p)
            continue
        if out[-1] == p[0]:
            out.extend(p[1:])
        else:
            out.extend(p)
    return out


def _preprocess_edges(image: np.ndarray, roi: Tuple[int, int, int, int]) -> Dict[str, np.ndarray]:
    x1, y1, x2, y2 = roi
    roi_img = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 7, 45, 45)

    median = float(np.median(denoised))
    lower = int(max(0.0, 0.58 * median))
    upper = int(min(255.0, 1.45 * median))
    canny = cv2.Canny(denoised, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1)

    grad_x = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, grad_bin = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grad_edges = cv2.Canny(grad_bin, 30, 120)

    merged = cv2.bitwise_or(closed, grad_edges)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=1)

    return {
        "edges": merged,
        "strength": mag_u8,
    }


def _nearest_edge_point(
    edges: np.ndarray,
    pt: Point,
    roi: Tuple[int, int, int, int],
    max_radius: int = 14,
) -> Tuple[Point, float, bool]:
    x1, y1, _, _ = roi
    lx = int(pt[0] - x1)
    ly = int(pt[1] - y1)
    h, w = edges.shape[:2]

    if 0 <= lx < w and 0 <= ly < h and edges[ly, lx] > 0:
        return pt, 0.0, True

    r = max(2, int(max_radius))
    x0 = max(0, lx - r)
    y0 = max(0, ly - r)
    x2 = min(w - 1, lx + r)
    y2 = min(h - 1, ly + r)
    patch = edges[y0 : y2 + 1, x0 : x2 + 1]
    yy, xx = np.where(patch > 0)
    if len(xx) == 0:
        return pt, float(max_radius), False

    dx = xx.astype(float) + float(x0) - float(lx)
    dy = yy.astype(float) + float(y0) - float(ly)
    idx = int(np.argmin((dx * dx) + (dy * dy)))
    snapped_local = (int(xx[idx] + x0), int(yy[idx] + y0))
    snapped = (snapped_local[0] + x1, snapped_local[1] + y1)
    return snapped, float(math.hypot(dx[idx], dy[idx])), True


def _segment_corridor(
    shape: Tuple[int, int],
    a: Point,
    b: Point,
    roi: Tuple[int, int, int, int],
    half_width: int,
) -> np.ndarray:
    h, w = shape
    x1, y1, _, _ = roi
    mask = np.zeros((h, w), dtype=np.uint8)
    aa = (int(a[0] - x1), int(a[1] - y1))
    bb = (int(b[0] - x1), int(b[1] - y1))
    cv2.line(mask, aa, bb, 255, max(2, int(2 * half_width)))
    if half_width > 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (half_width, half_width))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _dijkstra_path(
    edges: np.ndarray,
    strength: np.ndarray,
    start: Point,
    end: Point,
    roi: Tuple[int, int, int, int],
    corridor: np.ndarray,
    max_nodes: int = 32000,
) -> Optional[Path]:
    x1, y1, _, _ = roi
    sx, sy = int(start[0] - x1), int(start[1] - y1)
    ex, ey = int(end[0] - x1), int(end[1] - y1)
    h, w = edges.shape[:2]
    if not (0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h):
        return None

    inf = 1e18
    dist = np.full((h, w), inf, dtype=np.float64)
    prev = np.full((h, w, 2), -1, dtype=np.int32)

    dist[sy, sx] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, sx, sy)]
    visited = 0

    neighbors = [
        (-1, -1, math.sqrt(2.0)), (0, -1, 1.0), (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),                             (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),  (0, 1, 1.0),  (1, 1, math.sqrt(2.0)),
    ]

    while heap:
        cur_cost, x, y = heapq.heappop(heap)
        if cur_cost > dist[y, x]:
            continue

        visited += 1
        if visited > max_nodes:
            return None

        if x == ex and y == ey:
            break

        for dx, dy, step in neighbors:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue

            edge_bonus = float(strength[ny, nx]) / 255.0
            off_edge_penalty = 0.0 if edges[ny, nx] > 0 else 1.45
            corridor_penalty = 0.0 if corridor[ny, nx] > 0 else 2.35
            weight = step * (1.0 + (1.5 * (1.0 - edge_bonus))) + off_edge_penalty + corridor_penalty

            nd = cur_cost + weight
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx] = (x, y)
                heapq.heappush(heap, (nd, nx, ny))

    if not np.isfinite(dist[ey, ex]) or dist[ey, ex] >= inf:
        return None

    path_local: Path = []
    cx, cy = ex, ey
    limit = h * w
    while limit > 0:
        limit -= 1
        path_local.append((cx, cy))
        if cx == sx and cy == sy:
            break
        px, py = prev[cy, cx]
        if px < 0 or py < 0:
            return None
        cx, cy = int(px), int(py)

    if not path_local:
        return None

    path_local.reverse()
    path: Path = [(int(px + x1), int(py + y1)) for px, py in path_local]
    return path


def _contour_walk_fallback(edges: np.ndarray, start: Point, end: Point, roi: Tuple[int, int, int, int]) -> Optional[Path]:
    x1, y1, _, _ = roi
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    s_local = np.array([float(start[0] - x1), float(start[1] - y1)], dtype=float)
    e_local = np.array([float(end[0] - x1), float(end[1] - y1)], dtype=float)

    best = None
    best_cost = 1e12

    for contour in contours:
        if contour is None or len(contour) < 8:
            continue
        pts = contour.reshape((-1, 2)).astype(float)
        ds = np.linalg.norm(pts - s_local, axis=1)
        de = np.linalg.norm(pts - e_local, axis=1)
        is_idx = int(np.argmin(ds))
        ie_idx = int(np.argmin(de))
        cost = float(ds[is_idx] + de[ie_idx])
        if cost < best_cost:
            best_cost = cost
            best = (pts, is_idx, ie_idx)

    if best is None or best_cost > 32.0:
        return None

    pts, is_idx, ie_idx = best
    n = len(pts)
    if is_idx <= ie_idx:
        seg1 = pts[is_idx : ie_idx + 1]
        seg2 = np.concatenate([pts[ie_idx:], pts[: is_idx + 1]], axis=0)
    else:
        seg1 = pts[ie_idx : is_idx + 1][::-1]
        seg2 = np.concatenate([pts[is_idx:], pts[: ie_idx + 1]], axis=0)[::-1]

    def seg_cost(seg: np.ndarray) -> float:
        if len(seg) < 2:
            return 1e9
        length = float(np.sum(np.linalg.norm(seg[1:] - seg[:-1], axis=1)))
        return length

    chosen = seg1 if seg_cost(seg1) <= seg_cost(seg2) else seg2
    out: Path = []
    for p in chosen.tolist():
        pt = (int(round(float(p[0] + x1))), int(round(float(p[1] + y1))))
        if not out or out[-1] != pt:
            out.append(pt)

    if len(out) < 2:
        return None
    out[0] = start
    out[-1] = end
    return out


def _trace_segment(
    edges: np.ndarray,
    strength: np.ndarray,
    start: Point,
    end: Point,
    roi: Tuple[int, int, int, int],
    half_width: int,
) -> Dict[str, object]:
    s_snapped, s_snap, s_ok = _nearest_edge_point(edges, start, roi)
    e_snapped, e_snap, e_ok = _nearest_edge_point(edges, end, roi)

    if not (s_ok and e_ok):
        line = _sample_line(start, end)
        return {
            "path": line,
            "mode": "line_fallback",
            "snap_distance": float(max(s_snap, e_snap)),
            "success": False,
        }

    corridor = _segment_corridor(edges.shape[:2], s_snapped, e_snapped, roi, half_width=half_width)
    traced = _dijkstra_path(
        edges=edges,
        strength=strength,
        start=s_snapped,
        end=e_snapped,
        roi=roi,
        corridor=corridor,
    )
    if traced and len(traced) >= 2:
        traced[0] = start
        traced[-1] = end
        return {
            "path": traced,
            "mode": "dijkstra",
            "snap_distance": float((s_snap + e_snap) * 0.5),
            "success": True,
        }

    contour_path = _contour_walk_fallback(edges, start, end, roi)
    if contour_path and len(contour_path) >= 2:
        return {
            "path": contour_path,
            "mode": "contour_walk",
            "snap_distance": float((s_snap + e_snap) * 0.5),
            "success": False,
        }

    line = _sample_line(start, end)
    return {
        "path": line,
        "mode": "line_fallback",
        "snap_distance": float((s_snap + e_snap) * 0.5),
        "success": False,
    }


def _build_roi(points: List[Point], w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_span = max(1, max(xs) - min(xs))
    y_span = max(1, max(ys) - min(ys))
    pad_x = max(24, int(0.18 * x_span))
    pad_y = max(24, int(0.18 * y_span))

    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(w, max(xs) + pad_x)
    y2 = min(h, max(ys) + pad_y)

    if (x2 - x1) < 64:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - 32)
        x2 = min(w, cx + 32)
    if (y2 - y1) < 64:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - 32)
        y2 = min(h, cy + 32)

    return (x1, y1, x2, y2)


def _normalize_points(anchor_points: Dict[str, Sequence[float]], w: int, h: int) -> Dict[str, Point]:
    points: Dict[str, Point] = {}
    for key, value in (anchor_points or {}).items():
        pt = _to_point(value)
        if pt is None:
            continue
        points[key] = _in_bounds(pt, w, h)
    return points


def _fallback_polyline(points: Dict[str, Point], keys: List[str]) -> Path:
    out: Path = []
    for key in keys:
        pt = points.get(key)
        if pt is None:
            continue
        if not out or out[-1] != pt:
            out.append(pt)
    return out


def _to_xy_list(path: Path) -> List[Dict[str, int]]:
    return [{"x": int(p[0]), "y": int(p[1])} for p in path]


def trace_side_contours(
    image: np.ndarray,
    anchor_points: Dict[str, Sequence[float]],
    jaw_solver_debug: Optional[Dict] = None,
    was_mirrored: bool = False,
) -> Dict[str, object]:
    h, w = image.shape[:2]
    points = _normalize_points(anchor_points, w, h)

    silhouette_keys = ["trichion", "glabella", "nasion", "pronasale", "subnasale", "menton"]
    jaw_keys = ["menton", "gonion", "articulare"]

    silhouette_anchors = [points[k] for k in silhouette_keys if k in points]
    jaw_anchors = [points[k] for k in jaw_keys if k in points]

    fallback_reason = "none"
    if len(silhouette_anchors) < 2 or len(jaw_anchors) < 2:
        silhouette_fb = _fallback_polyline(points, silhouette_keys)
        jaw_fb = _fallback_polyline(points, jaw_keys)
        return {
            "method": "landmark_fallback_v1",
            "confidence": 0.0,
            "silhouette": _to_xy_list(silhouette_fb),
            "jaw_ramus": _to_xy_list(jaw_fb),
            "debug": {
                "image_size": {"width": int(w), "height": int(h)},
                "was_mirrored": bool(was_mirrored),
                "processing_mode": str((jaw_solver_debug or {}).get("processing_mode", "color")),
                "roi": None,
                "fallback_reason": "missing_anchors",
            },
        }

    roi_points = silhouette_anchors + jaw_anchors
    roi = _build_roi(roi_points, w, h)
    maps = _preprocess_edges(image, roi)
    edges = maps["edges"]
    strength = maps["strength"]

    edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))

    all_modes: List[str] = []
    all_snap_dist: List[float] = []
    dijkstra_success = 0
    segment_count = 0

    silhouette_segments: List[Path] = []
    for i in range(len(silhouette_anchors) - 1):
        segment_count += 1
        seg = _trace_segment(
            edges=edges,
            strength=strength,
            start=silhouette_anchors[i],
            end=silhouette_anchors[i + 1],
            roi=roi,
            half_width=10,
        )
        path = list(seg["path"])
        silhouette_segments.append(path)
        all_modes.append(str(seg["mode"]))
        all_snap_dist.append(float(seg["snap_distance"]))
        if bool(seg["success"]):
            dijkstra_success += 1

    jaw_segments: List[Path] = []
    for i in range(len(jaw_anchors) - 1):
        segment_count += 1
        seg = _trace_segment(
            edges=edges,
            strength=strength,
            start=jaw_anchors[i],
            end=jaw_anchors[i + 1],
            roi=roi,
            half_width=12,
        )
        path = list(seg["path"])
        jaw_segments.append(path)
        all_modes.append(str(seg["mode"]))
        all_snap_dist.append(float(seg["snap_distance"]))
        if bool(seg["success"]):
            dijkstra_success += 1

    silhouette = _simplify(_combine_paths(silhouette_segments), epsilon_ratio=0.007)
    jaw_ramus = _simplify(_combine_paths(jaw_segments), epsilon_ratio=0.006)

    if len(silhouette) < 2:
        silhouette = _fallback_polyline(points, silhouette_keys)
        fallback_reason = "path_disconnected"
    if len(jaw_ramus) < 2:
        jaw_ramus = _fallback_polyline(points, jaw_keys)
        fallback_reason = "path_disconnected"

    mode_set = set(all_modes)
    if mode_set == {"dijkstra"}:
        method = "edge_trace_dijkstra_v1"
    elif "contour_walk" in mode_set:
        method = "contour_walk_fallback_v1"
        if fallback_reason == "none":
            fallback_reason = "path_disconnected"
    elif "line_fallback" in mode_set:
        method = "landmark_fallback_v1"
        if fallback_reason == "none":
            fallback_reason = "edge_failed"
    else:
        method = "edge_trace_dijkstra_v1"

    completion = float(dijkstra_success) / float(max(1, segment_count))
    avg_snap = float(np.mean(all_snap_dist)) if all_snap_dist else 16.0
    snap_score = float(np.clip(1.0 - (avg_snap / 16.0), 0.0, 1.0))
    density_score = float(np.clip(edge_density * 90.0, 0.0, 1.0))

    confidence = float(np.clip((0.58 * completion) + (0.24 * snap_score) + (0.18 * density_score), 0.0, 1.0))
    if method == "contour_walk_fallback_v1":
        confidence = min(confidence, 0.62)
    if method == "landmark_fallback_v1":
        confidence = min(confidence, 0.30)

    if confidence < 0.26 and method != "landmark_fallback_v1":
        method = "landmark_fallback_v1"
        fallback_reason = "low_texture"

    processing_mode = str((jaw_solver_debug or {}).get("processing_mode", "color"))

    return {
        "method": method,
        "confidence": round(float(confidence), 3),
        "silhouette": _to_xy_list(silhouette),
        "jaw_ramus": _to_xy_list(jaw_ramus),
        "debug": {
            "image_size": {"width": int(w), "height": int(h)},
            "was_mirrored": bool(was_mirrored),
            "processing_mode": processing_mode,
            "roi": {
                "x1": int(roi[0]),
                "y1": int(roi[1]),
                "x2": int(roi[2]),
                "y2": int(roi[3]),
            },
            "fallback_reason": fallback_reason,
            "edge_density": round(edge_density, 4),
            "segment_completion": round(completion, 3),
            "avg_anchor_snap": round(avg_snap, 3),
        },
    }
