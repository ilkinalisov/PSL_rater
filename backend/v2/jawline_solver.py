"""Jawline contour solver for side-profile landmark refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import gonial_angle_from_points


Point = Tuple[int, int]


@dataclass
class JawlineSolveResult:
    points: Dict[str, Point]
    jaw_contour: List[List[int]]
    visibility_score: float
    source: str
    fallback_reason: str
    debug: Dict


def _clip_point(pt: Sequence[float], w: int, h: int) -> Point:
    return (
        int(np.clip(round(float(pt[0])), 0, max(0, w - 1))),
        int(np.clip(round(float(pt[1])), 0, max(0, h - 1))),
    )


def _safe_get(points: Dict[str, Point], key: str) -> Optional[Point]:
    val = points.get(key)
    if val is None or len(val) < 2:
        return None
    return (int(val[0]), int(val[1]))


def _skin_mask(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    hsv_mask = cv2.inRange(hsv, (0, 20, 45), (35, 255, 255))
    ycc_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin = cv2.bitwise_or(hsv_mask, ycc_mask)
    kernel = np.ones((5, 5), np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    return skin


def _monochrome_score(roi_bgr: np.ndarray) -> float:
    """Estimate how close the ROI is to monochrome (1.0 => nearly grayscale)."""
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    b = roi_bgr[:, :, 0].astype(np.float32)
    g = roi_bgr[:, :, 1].astype(np.float32)
    r = roi_bgr[:, :, 2].astype(np.float32)
    chroma = np.mean((np.abs(r - g) + np.abs(g - b) + np.abs(b - r)) / 3.0) / 255.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]) / 255.0)
    score = 1.0 - float(np.clip((0.65 * chroma) + (0.35 * sat), 0.0, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def _mono_edge_map(enhanced_gray: np.ndarray, denoised_gray: np.ndarray) -> np.ndarray:
    """Monochrome-robust edge map using gradients + adaptive silhouette boundaries."""
    grad_x = cv2.Sobel(denoised_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(denoised_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, grad_bin = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive = cv2.adaptiveThreshold(
        enhanced_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        3,
    )
    mono_mix = cv2.bitwise_or(grad_bin, adaptive)
    mono_edges = cv2.Canny(mono_mix, 35, 120)
    mono_edges = cv2.dilate(mono_edges, None, iterations=1)
    return mono_edges


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(float)


def _compute_curvature(points: np.ndarray, idx: int, stride: int = 3) -> float:
    lo = idx - stride
    hi = idx + stride
    if lo < 0 or hi >= len(points):
        return 0.0
    p0 = points[lo].astype(float)
    p1 = points[idx].astype(float)
    p2 = points[hi].astype(float)
    v1 = p0 - p1
    v2 = p2 - p1
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def _blend_point(base: Optional[Point], refined: Point, alpha: float) -> Point:
    if base is None:
        return refined
    ax = (float(base[0]) * (1.0 - alpha)) + (float(refined[0]) * alpha)
    ay = (float(base[1]) * (1.0 - alpha)) + (float(refined[1]) * alpha)
    return (int(round(ax)), int(round(ay)))


def _rotate_points(points: np.ndarray, center: Sequence[float], angle_deg: float) -> np.ndarray:
    if points is None or len(points) == 0:
        return points
    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    c = np.array([float(center[0]), float(center[1])], dtype=float)
    p = points.astype(float) - c
    rotated = np.empty_like(p, dtype=float)
    rotated[:, 0] = (p[:, 0] * cos_t) - (p[:, 1] * sin_t)
    rotated[:, 1] = (p[:, 0] * sin_t) + (p[:, 1] * cos_t)
    return rotated + c


def _rotate_point(point: Sequence[float], center: Sequence[float], angle_deg: float) -> np.ndarray:
    arr = np.array([[float(point[0]), float(point[1])]], dtype=float)
    return _rotate_points(arr, center, angle_deg)[0]


def _unit_vector(vec: np.ndarray, default: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return default.astype(float)
    return vec / n


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + (ab * t)
    return float(np.linalg.norm(point - proj))


def _estimate_profile_pitch_deg(menton: Point, subnasale: Point, tragion: Optional[Point]) -> float:
    """
    Estimate pitch for normalized left-facing profile.
    Positive => chin looks rotated downward (head pitched down).
    """
    me = np.array(menton, dtype=float)
    sn = np.array(subnasale, dtype=float)
    v = me - sn
    raw_angle = float(np.degrees(np.arctan2(v[1], v[0])))
    # Typical side-profile sn->me slope in image space is around 72 degrees.
    pitch = raw_angle - 72.0

    if tragion is not None:
        tr = np.array(tragion, dtype=float)
        ear_vec = tr - sn
        ear_angle = float(np.degrees(np.arctan2(ear_vec[1], ear_vec[0])))
        pitch += 0.22 * (ear_angle - 8.0)

    return float(np.clip(pitch, -24.0, 24.0))


def solve_jawline_contour(image: np.ndarray, anchor_points: Dict[str, Point]) -> JawlineSolveResult:
    """
    Refine Ar/Go/Me from visible mandibular contour.
    Assumes normalized left-facing profile.
    """
    h, w = image.shape[:2]
    menton = _safe_get(anchor_points, "menton") or _safe_get(anchor_points, "pogonion")
    tragion = _safe_get(anchor_points, "tragion")
    subnasale = _safe_get(anchor_points, "subnasale")
    pronasale = _safe_get(anchor_points, "pronasale")
    gonion_seed = _safe_get(anchor_points, "gonion")
    articulare_seed = _safe_get(anchor_points, "articulare")

    if menton is None or subnasale is None:
        return JawlineSolveResult(
            points={},
            jaw_contour=[],
            visibility_score=0.0,
            source="jawline_contour",
            fallback_reason="default",
            debug={"reason": "missing_required_anchor"},
        )

    if tragion is None:
        tragion = (
            min(w - 1, menton[0] + max(20, int(w * 0.18))),
            max(0, menton[1] - max(20, int(h * 0.16))),
        )

    pitch_deg = _estimate_profile_pitch_deg(menton, subnasale, tragion)
    # Normalize solving frame against pitch without over-rotating hard cases.
    normalization_angle = float(np.clip(-0.72 * pitch_deg, -18.0, 18.0))

    profile_points = [menton, tragion, subnasale]
    if pronasale is not None:
        profile_points.append(pronasale)
    if gonion_seed is not None:
        profile_points.append(gonion_seed)

    xs = [p[0] for p in profile_points]
    ys = [p[1] for p in profile_points]
    pad_x = max(24, int((max(xs) - min(xs)) * 0.35))
    pad_y = max(20, int((max(ys) - min(ys)) * 0.40))
    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(w, max(xs) + pad_x)
    y2 = min(h, max(ys) + pad_y)

    if (x2 - x1) < 60 or (y2 - y1) < 60:
        return JawlineSolveResult(
            points={},
            jaw_contour=[],
            visibility_score=0.0,
            source="jawline_contour",
            fallback_reason="shape_weak",
            debug={"reason": "small_roi"},
        )

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 7, 45, 45)
    mono_score = _monochrome_score(roi)

    median_val = float(np.median(denoised))
    lower = int(max(0.0, 0.55 * median_val))
    upper = int(min(255.0, 1.45 * median_val))
    edges = cv2.Canny(denoised, lower, upper)
    skin = _skin_mask(roi)
    skin_boundary = cv2.Canny(skin, 40, 120)
    color_edges = cv2.bitwise_or(edges, cv2.dilate(skin_boundary, None, iterations=1))
    mono_edges = _mono_edge_map(enhanced, denoised)

    if mono_score >= 0.78:
        processing_mode = "mono"
        fused_edges = mono_edges
    elif mono_score >= 0.58:
        processing_mode = "hybrid"
        fused_edges = cv2.bitwise_or(color_edges, mono_edges)
    else:
        processing_mode = "color"
        fused_edges = color_edges

    raw_points = np.column_stack(np.where(fused_edges > 0))
    if raw_points.size == 0:
        return JawlineSolveResult(
            points={},
            jaw_contour=[],
            visibility_score=0.0,
            source="jawline_contour",
            fallback_reason="mono_low_texture" if processing_mode in ("mono", "hybrid") else "edge_failed",
            debug={
                "reason": "no_edges",
                "pitch_deg": round(pitch_deg, 3),
                "processing_mode": processing_mode,
                "monochrome_score": round(mono_score, 3),
            },
        )

    # Convert from (row, col) to (x, y), then offset to full-image coordinates.
    edge_points = np.stack([raw_points[:, 1], raw_points[:, 0]], axis=1).astype(float)
    edge_points[:, 0] += float(x1)
    edge_points[:, 1] += float(y1)

    zone_x_min = max(0, min(menton[0], subnasale[0], tragion[0]) - int(w * 0.05))
    zone_x_max = min(w - 1, max(menton[0], subnasale[0], tragion[0]) + int(w * 0.07))
    zone_y_min = max(0, min(subnasale[1], tragion[1]) - int(h * 0.03))

    pitch_down = max(0.0, pitch_deg)
    zone_cap_ratio = max(0.04, 0.08 - min(0.04, pitch_down * 0.002))
    zone_y_max = min(h - 1, menton[1] + int(h * zone_cap_ratio))

    posterior_dx = np.maximum(0.0, edge_points[:, 0] - float(menton[0]))
    neck_allowance = max(8.0, (0.028 * float(h)) + (0.15 * pitch_down))
    chin_envelope = float(menton[1]) + neck_allowance + (0.06 * posterior_dx)

    mask = (
        (edge_points[:, 0] >= zone_x_min) &
        (edge_points[:, 0] <= zone_x_max) &
        (edge_points[:, 1] >= zone_y_min) &
        (edge_points[:, 1] <= zone_y_max) &
        (edge_points[:, 1] <= chin_envelope)
    )
    edge_points = edge_points[mask]
    if len(edge_points) < 32:
        return JawlineSolveResult(
            points={},
            jaw_contour=[],
            visibility_score=0.0,
            source="jawline_contour",
            fallback_reason="mono_low_texture" if processing_mode in ("mono", "hybrid") else "edge_failed",
            debug={
                "reason": "insufficient_edge_points",
                "count": int(len(edge_points)),
                "pitch_deg": round(pitch_deg, 3),
                "processing_mode": processing_mode,
                "monochrome_score": round(mono_score, 3),
            },
        )

    rotation_center = np.array([float(menton[0]), float(menton[1])], dtype=float)
    edge_points_norm = _rotate_points(edge_points, rotation_center, normalization_angle)

    menton_norm = _rotate_point(menton, rotation_center, normalization_angle)
    tragion_norm = _rotate_point(tragion, rotation_center, normalization_angle) if tragion is not None else None
    gonion_seed_norm = _rotate_point(gonion_seed, rotation_center, normalization_angle) if gonion_seed is not None else None

    # Build constrained mandibular polyline: choose one y per x via candidate scoring.
    x_bins = np.round(edge_points_norm[:, 0]).astype(int)
    x_unique = np.unique(x_bins)
    if len(x_unique) < 24:
        return JawlineSolveResult(
            points={},
            jaw_contour=[],
            visibility_score=0.0,
            source="jawline_contour",
            fallback_reason="mono_low_texture" if processing_mode in ("mono", "hybrid") else "shape_weak",
            debug={
                "reason": "insufficient_polyline_bins",
                "count": int(len(x_unique)),
                "pitch_deg": round(pitch_deg, 3),
                "processing_mode": processing_mode,
                "monochrome_score": round(mono_score, 3),
            },
        )

    y_all = edge_points_norm[:, 1]
    y_low = float(np.percentile(y_all, 8))
    y_high = float(np.percentile(y_all, 95))
    y_span = max(1.0, y_high - y_low)

    ref_a = np.array(menton_norm, dtype=float)
    ref_b = np.array(gonion_seed_norm if gonion_seed_norm is not None else tragion_norm, dtype=float) if (gonion_seed_norm is not None or tragion_norm is not None) else None

    jaw_poly_norm: List[Tuple[float, float]] = []
    for xv in sorted(x_unique.tolist()):
        idxs = np.where(x_bins == xv)[0]
        if len(idxs) == 0:
            continue

        candidates = edge_points_norm[idxs]
        best_score = -1e9
        best_point = None
        for cand in candidates:
            inferior = float((cand[1] - y_low) / y_span)
            posterior = max(0.0, float(cand[0] - menton_norm[0]))
            neck_cap_norm = float(menton_norm[1]) + max(6.0, 0.02 * float(h)) + (0.05 * posterior)
            neck_penalty = max(0.0, float(cand[1] - neck_cap_norm)) / max(1.0, 0.06 * float(h))

            seed_score = 0.0
            if ref_b is not None:
                dref = _point_segment_distance(cand, ref_a, ref_b)
                seed_score = max(0.0, 1.0 - (dref / max(18.0, 0.12 * float(max(h, w)))))

            score = (1.20 * inferior) + (0.55 * seed_score) - (1.30 * neck_penalty)
            if score > best_score:
                best_score = score
                best_point = (float(cand[0]), float(cand[1]))

        if best_point is not None:
            jaw_poly_norm.append(best_point)

    if len(jaw_poly_norm) < 24:
        return JawlineSolveResult(
            points={},
            jaw_contour=[],
            visibility_score=0.0,
            source="jawline_contour",
            fallback_reason="mono_low_texture" if processing_mode in ("mono", "hybrid") else "shape_weak",
            debug={
                "reason": "insufficient_polyline_points",
                "count": int(len(jaw_poly_norm)),
                "pitch_deg": round(pitch_deg, 3),
                "processing_mode": processing_mode,
                "monochrome_score": round(mono_score, 3),
            },
        )

    jaw_poly_norm_arr = np.array(sorted(jaw_poly_norm, key=lambda p: p[0]), dtype=float)
    jaw_poly_norm_arr[:, 1] = _moving_average(jaw_poly_norm_arr[:, 1], window=7)

    x_span = float(max(1.0, jaw_poly_norm_arr[-1, 0] - jaw_poly_norm_arr[0, 0]))
    y_cap_norm = float(menton_norm[1] + max(6.0, (0.06 - min(0.02, pitch_down * 0.001)) * float(h)))
    jaw_poly_norm_arr[:, 1] = np.minimum(jaw_poly_norm_arr[:, 1], y_cap_norm)

    jaw_points_norm = np.round(jaw_poly_norm_arr).astype(int)

    n = len(jaw_points_norm)
    anterior_end = max(4, int(n * 0.38))
    posterior_start = max(anterior_end + 1, int(n * 0.52))

    anterior = jaw_points_norm[:anterior_end]
    me_idx_local = int(np.argmax(anterior[:, 1]))
    me_idx = me_idx_local
    me_refined_norm = np.array([int(anterior[me_idx_local, 0]), int(anterior[me_idx_local, 1])], dtype=float)

    y_min_band = float(np.min(jaw_points_norm[:, 1]))
    y_max_band = float(np.max(jaw_points_norm[:, 1]))
    y_band_span = max(1.0, y_max_band - y_min_band)

    corridor_x_min = float(me_refined_norm[0] + max(4, int(w * 0.01)))
    corridor_x_max = float(jaw_points_norm[-1, 0])
    if tragion_norm is not None:
        corridor_x_max = min(corridor_x_max, float(tragion_norm[0] + max(6, int(w * 0.04))))

    jaw_band_y_min = float(me_refined_norm[1] - max(14, int(h * 0.16)))
    jaw_band_y_max = float(me_refined_norm[1] + max(6, int(h * 0.05)))

    best_score = -1.0
    best_idx = None
    best_curv = 0.0
    for i in range(posterior_start, n - 3):
        pt = jaw_points_norm[i]
        if float(pt[0]) < corridor_x_min or float(pt[0]) > corridor_x_max:
            continue
        if float(pt[1]) < jaw_band_y_min or float(pt[1]) > jaw_band_y_max:
            continue

        curv = _compute_curvature(jaw_points_norm, i, stride=3)
        if curv < 8.0:
            continue

        posteriority = float(np.clip((float(pt[0]) - float(me_refined_norm[0])) / max(1.0, x_span), 0.0, 1.0))
        inferior = float(np.clip((float(pt[1]) - y_min_band) / y_band_span, 0.0, 1.0))

        seed_bonus = 0.0
        if gonion_seed_norm is not None:
            dist = np.hypot(float(pt[0] - gonion_seed_norm[0]), float(pt[1] - gonion_seed_norm[1]))
            seed_bonus = max(0.0, 1.0 - (dist / max(36.0, 0.20 * max(h, w))))

        score = (1.15 * curv) + (14.0 * seed_bonus) + (9.0 * inferior) + (5.0 * posteriority)
        if score > best_score:
            best_score = score
            best_idx = i
            best_curv = curv

    if best_idx is None:
        # Deterministic backup in posterior section nearest to seed/corridor.
        candidates = []
        for i in range(posterior_start, n):
            pt = jaw_points_norm[i]
            if float(pt[0]) < corridor_x_min or float(pt[0]) > corridor_x_max:
                continue
            if float(pt[1]) < jaw_band_y_min or float(pt[1]) > jaw_band_y_max:
                continue
            d = 1e6
            if gonion_seed_norm is not None:
                d = float(np.hypot(float(pt[0] - gonion_seed_norm[0]), float(pt[1] - gonion_seed_norm[1])))
            candidates.append((d, i))
        if not candidates:
            return JawlineSolveResult(
                points={},
                jaw_contour=[],
                visibility_score=0.0,
                source="jawline_contour",
                fallback_reason="mono_low_texture" if processing_mode in ("mono", "hybrid") else "shape_weak",
                debug={
                    "reason": "no_gonion_candidate",
                    "pitch_deg": round(pitch_deg, 3),
                    "processing_mode": processing_mode,
                    "monochrome_score": round(mono_score, 3),
                },
            )
        best_idx = sorted(candidates, key=lambda x: x[0])[0][1]
        best_curv = _compute_curvature(jaw_points_norm, best_idx, stride=3)

    go_refined_norm = np.array([float(jaw_points_norm[best_idx, 0]), float(jaw_points_norm[best_idx, 1])], dtype=float)

    # Ar constrained by local ramus direction: posterior contour tangent + tragion anchor.
    if 2 <= best_idx < (n - 2):
        tangent = jaw_points_norm[best_idx + 2].astype(float) - jaw_points_norm[best_idx - 2].astype(float)
    else:
        tangent = go_refined_norm - me_refined_norm
    tangent = _unit_vector(tangent, np.array([1.0, 0.0], dtype=float))

    n1 = np.array([-tangent[1], tangent[0]], dtype=float)
    n2 = -n1
    normal_candidates = [n1, n2]
    normal_candidates.sort(key=lambda v: (v[1] < 0.0, v[0] >= -0.30), reverse=True)
    ramus_dir = normal_candidates[0]
    if ramus_dir[1] > -0.05:
        ramus_dir = np.array([max(ramus_dir[0], 0.20), -abs(ramus_dir[1]) - 0.20], dtype=float)
    ramus_dir = _unit_vector(ramus_dir, np.array([0.25, -0.97], dtype=float))

    if tragion_norm is not None:
        vgt = np.array([float(tragion_norm[0] - go_refined_norm[0]), float(tragion_norm[1] - go_refined_norm[1])], dtype=float)
        vgt = _unit_vector(vgt, ramus_dir)
        ramus_dir = _unit_vector((ramus_dir * 0.58) + (vgt * 0.42), ramus_dir)

    if tragion_norm is not None:
        go_tr_dist = float(np.linalg.norm(np.array(tragion_norm, dtype=float) - go_refined_norm))
        ramus_len = np.clip(go_tr_dist * 0.72, 0.10 * float(h), 0.24 * float(h))
    else:
        ramus_len = np.clip(0.16 * float(h), 14.0, 0.22 * float(h))

    ar_candidate_norm = go_refined_norm + (ramus_dir * ramus_len)
    if tragion_norm is not None:
        ar_candidate_norm[0] = np.clip(
            ar_candidate_norm[0],
            go_refined_norm[0] - max(4.0, 0.02 * float(w)),
            float(tragion_norm[0]) + max(5.0, 0.04 * float(w)),
        )
    ar_candidate_norm[1] = min(ar_candidate_norm[1], go_refined_norm[1] - max(8.0, 0.03 * float(h)))

    # Convert solved contour/points back to image coordinates.
    jaw_points_img = _rotate_points(jaw_points_norm.astype(float), rotation_center, -normalization_angle)
    jaw_points_img[:, 0] = np.clip(jaw_points_img[:, 0], 0, max(0, w - 1))
    jaw_points_img[:, 1] = np.clip(jaw_points_img[:, 1], 0, max(0, h - 1))
    jaw_points = np.round(jaw_points_img).astype(int)

    me_refined = _clip_point(_rotate_point(me_refined_norm, rotation_center, -normalization_angle), w, h)
    go_refined = _clip_point(_rotate_point(go_refined_norm, rotation_center, -normalization_angle), w, h)
    ar_refined = _clip_point(_rotate_point(ar_candidate_norm, rotation_center, -normalization_angle), w, h)

    mandibular_pts = jaw_points_norm[: max(2, best_idx + 1)].astype(float)
    d_mand = []
    for pt in mandibular_pts[::2]:
        d_mand.append(_point_segment_distance(pt, me_refined_norm, go_refined_norm))
    mandibular_residual = float(np.median(d_mand)) if d_mand else 99.0

    ramus_target = np.array(tragion_norm if tragion_norm is not None else ar_candidate_norm, dtype=float)
    ramus_pts = jaw_points_norm[max(0, best_idx - 2): min(n, best_idx + 8)].astype(float)
    d_ram = []
    for pt in ramus_pts:
        d_ram.append(_point_segment_distance(pt, go_refined_norm, ramus_target))
    ramus_residual = float(np.median(d_ram)) if d_ram else mandibular_residual
    fit_residual = float((0.75 * mandibular_residual) + (0.25 * ramus_residual))

    density = float(len(edge_points)) / float(max(1.0, (x2 - x1) * (y2 - y1)))
    coverage = float(len(jaw_points_norm)) / max(8.0, x_span)
    span_norm = float(np.clip(x_span / max(1.0, (x2 - x1)), 0.0, 1.0))
    curvature_score = float(np.clip(best_curv / 75.0, 0.0, 1.0))
    density_score = float(np.clip(density * 95.0, 0.0, 1.0))
    coverage_score = float(np.clip(coverage * 2.4, 0.0, 1.0))
    pitch_penalty = float(np.clip(abs(pitch_deg) / 28.0, 0.0, 0.35))
    residual_penalty = float(np.clip((fit_residual - 8.0) / 24.0, 0.0, 0.35))
    visibility = float(np.clip(
        (0.31 * span_norm) +
        (0.23 * coverage_score) +
        (0.30 * curvature_score) +
        (0.16 * density_score) -
        (0.12 * pitch_penalty) -
        (0.10 * residual_penalty),
        0.0,
        1.0,
    ))

    # Blend with seeds so we keep temporal stability while shifting toward visible contour.
    blend_alpha = float(np.clip(0.30 + (0.42 * visibility), 0.30, 0.78))
    me_out = _blend_point(menton, me_refined, blend_alpha)
    go_out = _blend_point(gonion_seed, go_refined, blend_alpha)
    ar_out = _blend_point(articulare_seed, ar_refined, blend_alpha * 0.92)

    jaw_contour = [[int(pt[0]), int(pt[1])] for pt in jaw_points[::2]]
    gonial = gonial_angle_from_points(ar_out, go_out, me_out)
    fallback_reason = "none"
    if processing_mode in ("mono", "hybrid") and (mono_score >= 0.70 and visibility < 0.34):
        fallback_reason = "mono_low_texture"
    if fit_residual > 15.0:
        fallback_reason = "contour_residual_high"
    if gonial is None or not (95.0 <= float(gonial) <= 150.0):
        fallback_reason = "mp_invalid"

    return JawlineSolveResult(
        points={"articulare": ar_out, "gonion": go_out, "menton": me_out},
        jaw_contour=jaw_contour,
        visibility_score=visibility,
        source="jawline_contour",
        fallback_reason=fallback_reason,
        debug={
            "roi_bbox": [int(x1), int(y1), int(x2), int(y2)],
            "edge_points": int(len(edge_points)),
            "polyline_points": int(len(jaw_points)),
            "curvature_peak": round(float(best_curv), 3),
            "blend_alpha": round(float(blend_alpha), 3),
            "pitch_deg": round(float(pitch_deg), 3),
            "normalization_angle": round(float(normalization_angle), 3),
            "processing_mode": processing_mode,
            "monochrome_score": round(float(mono_score), 3),
            "fit_residual": round(float(fit_residual), 3),
            "raw_refined": {
                "menton": [int(me_refined[0]), int(me_refined[1])],
                "gonion": [int(go_refined[0]), int(go_refined[1])],
                "articulare": [int(ar_refined[0]), int(ar_refined[1])],
            },
        },
    )
