# side_profile_analyzer.py - Improved gonial angle and continuous scoring

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import numpy as np
import mediapipe as mp
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

try:
    from v2.overlay_renderer import draw_gonial_overlay
except Exception:  # pragma: no cover - import path differs by runtime entrypoint
    try:
        from backend.v2.overlay_renderer import draw_gonial_overlay  # type: ignore
    except Exception:  # pragma: no cover - keep legacy fallback if v2 module unavailable
        draw_gonial_overlay = None

@dataclass
class SideProfileMeasurements:
    """Side profile measurements"""
    facial_convexity_angle: float
    nasofrontal_angle: float
    nasolabial_angle: float
    gonial_angle: float
    nasofacial_angle: float
    forehead_slope: float
    nasal_projection: float
    chin_projection: float
    lip_projection: float
    profile_harmony_score: float
    vertical_profile_balance: Dict[str, float]
    forward_growth_score: float = 5.0
    facial_angle: float = 90.0
    maxillary_prominence: float = 0.0
    mandibular_prominence: float = 0.0
    recession_type: str = "unknown"
    gender: str = "unknown"
    gender_confidence: float = 0.0
    gonion_confidence: float = 0.0
    raw_gonial_angle: Optional[float] = None
    pre_validation_gonial_angle: Optional[float] = None
    gonial_source: str = "default"
    gonial_fallback_reason: str = "default"
    is_estimated: bool = False
    was_mirrored: bool = False

    def to_dict(self):
        return {
            "facial_convexity_angle": float(self.facial_convexity_angle),
            "nasofrontal_angle": float(self.nasofrontal_angle),
            "nasolabial_angle": float(self.nasolabial_angle),
            "gonial_angle": float(self.gonial_angle),
            "nasofacial_angle": float(self.nasofacial_angle),
            "forehead_slope": float(self.forehead_slope),
            "nasal_projection": float(self.nasal_projection),
            "chin_projection": float(self.chin_projection),
            "lip_projection": float(self.lip_projection),
            "profile_harmony_score": float(self.profile_harmony_score),
            "vertical_profile_balance": self.vertical_profile_balance,
            "forward_growth_score": float(self.forward_growth_score),
            "facial_angle": float(self.facial_angle),
            "maxillary_prominence": float(self.maxillary_prominence),
            "mandibular_prominence": float(self.mandibular_prominence),
            "recession_type": self.recession_type,
            "gender": self.gender,
            "gender_confidence": float(self.gender_confidence),
            "gonion_confidence": float(self.gonion_confidence),
            "raw_gonial_angle": float(self.raw_gonial_angle) if self.raw_gonial_angle is not None else None,
            "pre_validation_gonial_angle": float(self.pre_validation_gonial_angle) if self.pre_validation_gonial_angle is not None else None,
            "gonial_source": self.gonial_source,
            "gonial_fallback_reason": self.gonial_fallback_reason,
            "is_estimated": self.is_estimated,
            "was_mirrored": self.was_mirrored
        }

class SideProfileAnalyzer:
    def __init__(self, static_image_mode=True, max_num_faces=1,
                 min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

        self.SIDE_LANDMARKS = {
            "trichion": 10,
            "glabella": 168,
            "nasion": 6,
            "pronasale": 4,
            "subnasale": 2,
            "columella": 1,
            "labrale_superius": 13,
            "labrale_inferius": 14,
            "upper_lip_mid": 0,
            "pogonion": 152,
            "menton": 199,           # True lowest chin point
            "tragion": 234,          # Ear/TMJ reference for articulare estimation

            # Jaw and cheek landmarks
            "left_cheek": 205,       # Lateral cheek point (not ear)
            "left_jaw_angle": 172,   # For ramus reference
            "jaw_low": 172,
            "jaw_mid": 136,
        }

        self.IDEAL_ANGLES = {
            "gonial": 120,
            "nasolabial": 100,
            "facial_convexity": 165,
            "nasofrontal": 130
        }

        # Minimum landmark confidence threshold
        self.MIN_CONFIDENCE = 0.5
        self.enable_hybrid_debug_overlay = os.environ.get("SIDE_OVERLAY_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")

    def _create_face_mesh(self, min_detection_confidence: float):
        return self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def _run_face_mesh(self, image: np.ndarray, min_detection_confidence: float):
        mesh = self._create_face_mesh(min_detection_confidence)
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)
        finally:
            mesh.close()

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def _adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        if gamma <= 0:
            return image
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)],
            dtype=np.uint8
        )
        return cv2.LUT(image, table)

    def _detect_face_landmarks_with_retries(self, image: np.ndarray):
        """Robust side-face detection using multiple preprocessing passes."""
        upscaled = cv2.resize(image, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        stronger_upscaled = cv2.resize(image, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)
        downscaled_85 = cv2.resize(image, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
        downscaled_70 = cv2.resize(image, None, fx=0.70, fy=0.70, interpolation=cv2.INTER_AREA)
        enhanced = self._preprocess_image(image)
        enhanced_upscaled = cv2.resize(enhanced, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
        enhanced_downscaled = cv2.resize(enhanced, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        contrast = cv2.convertScaleAbs(image, alpha=1.15, beta=10)

        attempts = [
            ("raw", image, 0.45),
            ("downscaled_85", downscaled_85, 0.40),
            ("downscaled_70", downscaled_70, 0.38),
            ("clahe", enhanced, 0.35),
            ("clahe_downscaled", enhanced_downscaled, 0.33),
            ("denoised", denoised, 0.33),
            ("contrast", contrast, 0.32),
            ("gamma_bright", self._adjust_gamma(image, 1.25), 0.32),
            ("gamma_dark", self._adjust_gamma(image, 0.80), 0.32),
            ("upscaled_raw", upscaled, 0.30),
            ("upscaled_clahe", enhanced_upscaled, 0.25),
            ("upscaled_raw_strong", stronger_upscaled, 0.22),
        ]

        attempted = []
        for step_name, candidate, det_conf in attempts:
            attempted.append(step_name)
            landmarks = self._run_face_mesh(candidate, det_conf)
            if landmarks is not None:
                return landmarks, {
                    "detector_step": step_name,
                    "detector_confidence": det_conf,
                    "attempts": attempted
                }

        return None, {"detector_step": "failed", "attempts": attempted}

    def _detect_face_landmarks_with_crop_fallback(self, image: np.ndarray):
        """Fallback detector: face-detection bbox -> local face-mesh -> remap to full image."""
        h, w = image.shape[:2]
        attempted = []
        detector_inputs = [
            ("fd_raw", image),
            ("fd_clahe", self._preprocess_image(image)),
            ("fd_contrast", cv2.convertScaleAbs(image, alpha=1.10, beta=8)),
            ("fd_gamma", self._adjust_gamma(image, 1.20)),
        ]

        for model_selection in (1, 0):
            face_detector = self.mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=0.22
            )
            try:
                for step_name, candidate in detector_inputs:
                    attempted.append(f"{step_name}:m{model_selection}")
                    rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
                    det_results = face_detector.process(rgb)
                    if not det_results.detections:
                        continue

                    detections = sorted(
                        det_results.detections,
                        key=lambda d: float(d.score[0]) if d.score else 0.0,
                        reverse=True
                    )

                    for det in detections[:2]:
                        rel = det.location_data.relative_bounding_box
                        x = int(rel.xmin * w)
                        y = int(rel.ymin * h)
                        bw = int(rel.width * w)
                        bh = int(rel.height * h)
                        if bw < 50 or bh < 50:
                            continue

                        pad_x = int(bw * 0.45)
                        pad_y = int(bh * 0.55)
                        x0 = max(0, x - pad_x)
                        y0 = max(0, y - pad_y)
                        x1 = min(w, x + bw + pad_x)
                        y1 = min(h, y + bh + pad_y)
                        if (x1 - x0) < 80 or (y1 - y0) < 80:
                            continue

                        roi = image[y0:y1, x0:x1]
                        if roi.size == 0:
                            continue

                        roi_variants = [
                            ("roi_raw", roi, 0.32),
                            ("roi_clahe", self._preprocess_image(roi), 0.30),
                            ("roi_upscaled", cv2.resize(roi, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC), 0.28),
                        ]

                        for roi_name, roi_img, det_conf in roi_variants:
                            attempted.append(f"{step_name}:{roi_name}:m{model_selection}")
                            local_landmarks = self._run_face_mesh(roi_img, det_conf)
                            if local_landmarks is None:
                                continue

                            roi_h, roi_w = roi_img.shape[:2]
                            scale_x = (x1 - x0) / float(max(1, roi_w))
                            scale_y = (y1 - y0) / float(max(1, roi_h))

                            for lm in local_landmarks.landmark:
                                abs_x = x0 + (lm.x * roi_w * scale_x)
                                abs_y = y0 + (lm.y * roi_h * scale_y)
                                lm.x = float(np.clip(abs_x / max(1, w), 0.0, 1.0))
                                lm.y = float(np.clip(abs_y / max(1, h), 0.0, 1.0))
                                lm.z = float(lm.z * ((roi_w * scale_x) / max(1.0, float(w))))

                            return local_landmarks, {
                                "detector_step": "crop_fallback",
                                "detector_confidence": det_conf,
                                "attempts": attempted
                            }
            finally:
                face_detector.close()

        return None, {"detector_step": "crop_failed", "attempts": attempted}

    def _collect_landmark_points(self, landmarks, w: int, h: int) -> List[Tuple[int, int]]:
        points = []
        try:
            total = len(landmarks.landmark)
        except:
            total = 0

        for idx in range(total):
            pt = self._landmark_to_pixel(landmarks, idx, w, h)
            if pt is not None:
                points.append(pt)
        return points

    def _landmark_to_pixel(self, landmarks, index: int, w: int, h: int) -> Optional[Tuple[int, int]]:
        """Convert normalized landmark coordinates to pixel coordinates."""
        try:
            lm = landmarks.landmark[index]
            return (int(lm.x * w), int(lm.y * h))
        except:
            return None

    def _select_posterior_tragion(self, coords: Dict, landmarks, w: int, h: int) -> Optional[Tuple[int, int]]:
        """Pick the most posterior tragion candidate (left/right) after orientation normalization."""
        candidates = []
        for idx in (234, 454):
            pt = self._landmark_to_pixel(landmarks, idx, w, h)
            if pt is not None:
                candidates.append(pt)

        if not candidates:
            return coords.get("tragion")

        trichion_y = coords.get("trichion", (0, int(h * 0.2)))[1]
        chin_y = coords.get("menton", coords.get("pogonion", (0, int(h * 0.8))))[1]
        valid = [p for p in candidates if trichion_y <= p[1] <= chin_y]
        if valid:
            candidates = valid

        # Analyzer mirrors right-facing profiles to left-facing, so posterior is larger X.
        return max(candidates, key=lambda p: p[0])

    def _estimate_gonion(self, coords: Dict, landmarks, w: int, h: int) -> Tuple[Optional[Tuple[int, int]], float]:
        """Estimate gonion and return a confidence score in [0, 1]."""
        gonion = self._estimate_gonion_improved(coords, landmarks, w, h)
        if gonion is None:
            return None, 0.0

        chin_key = "menton" if "menton" in coords else "pogonion"
        if chin_key not in coords:
            return gonion, 0.25

        chin = coords[chin_key]
        tragion = coords.get("tragion")
        trichion_y = coords.get("trichion", (0, int(h * 0.2)))[1]
        full_face_h = max(20, abs(chin[1] - trichion_y))
        full_face_w = max(30, abs(coords.get("pronasale", chin)[0] - (tragion[0] if tragion else chin[0] + full_face_h // 2)))

        posterior_ok = gonion[0] >= (chin[0] + int(full_face_w * 0.20))
        inferior_ok = (chin[1] - int(full_face_h * 0.35)) <= gonion[1] <= (chin[1] + int(full_face_h * 0.12))

        confidence = 0.25
        if posterior_ok:
            confidence += 0.35
        if inferior_ok:
            confidence += 0.25

        if tragion is not None:
            # Cornerness against the ear→chin baseline.
            abx = float(chin[0] - tragion[0])
            aby = float(chin[1] - tragion[1])
            apx = float(gonion[0] - tragion[0])
            apy = float(gonion[1] - tragion[1])
            denom = math.hypot(abx, aby)
            if denom > 1e-6:
                cornerness = abs(abx * apy - aby * apx) / denom
                confidence += min(0.15, cornerness / max(15.0, full_face_h * 0.08) * 0.15)

        return gonion, max(0.0, min(1.0, confidence))

    # HYBRID EDGE DETECTION
    def _get_jaw_roi(self, coords: Dict, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        """Define a padded jaw ROI using menton/tragion/gonion estimates."""
        menton = coords.get("menton", coords.get("pogonion"))
        tragion = coords.get("tragion")
        gonion = coords.get("gonion")

        if menton is None:
            return None

        if tragion is None:
            tragion = (min(w - 1, menton[0] + int(w * 0.30)), max(0, menton[1] - int(h * 0.20)))

        points = [menton, tragion]
        if gonion is not None:
            points.append(gonion)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        span_x = max(20, max(xs) - min(xs))
        span_y = max(20, max(ys) - min(ys))
        pad_x = int(span_x * 0.35)
        pad_y = int(span_y * 0.35)

        x1 = max(0, min(xs) - pad_x)
        y1 = max(0, min(ys) - pad_y)
        x2 = min(w, max(xs) + pad_x)
        y2 = min(h, max(ys) + pad_y)

        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return None

        return (x1, y1, x2, y2)

    # HYBRID EDGE DETECTION
    def _preprocess_jaw_roi(self, image: np.ndarray, roi_bbox: Tuple[int, int, int, int]):
        """Enhance jaw ROI for contour extraction."""
        x1, y1, x2, y2 = roi_bbox
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
        return blurred, roi

    # HYBRID EDGE DETECTION
    def _create_skin_mask(self, roi: np.ndarray) -> np.ndarray:
        """Create a coarse skin mask to suppress background/hair edges."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        hsv_mask = cv2.inRange(hsv, (0, 20, 50), (35, 255, 255))
        ycrcb_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        combined = cv2.bitwise_or(hsv_mask, ycrcb_mask)
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        return combined

    # HYBRID EDGE DETECTION
    def _detect_jaw_edges(self, preprocessed_roi: np.ndarray, roi_img: np.ndarray) -> np.ndarray:
        """Detect jaw edges using adaptive Canny + skin-boundary filtering."""
        median_val = np.median(preprocessed_roi)
        lower = int(max(10, 0.5 * median_val))
        upper = int(min(255, 1.5 * median_val))
        edges = cv2.Canny(preprocessed_roi, lower, upper)

        skin_mask = self._create_skin_mask(roi_img)
        skin_boundary = cv2.Canny(skin_mask, 50, 150)
        skin_boundary = cv2.dilate(skin_boundary, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.bitwise_and(edges, skin_boundary)

        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

    # HYBRID EDGE DETECTION
    def _extract_jaw_contour_points(self, edges: np.ndarray, roi_bbox: Tuple[int, int, int, int], mediapipe_coords: Dict):
        """Extract edge points likely belonging to mandibular boundary."""
        edge_points = np.column_stack(np.where(edges > 0))
        if len(edge_points) == 0:
            return None

        edge_points = edge_points[:, ::-1]
        x_offset, y_offset = roi_bbox[0], roi_bbox[1]
        edge_points_global = edge_points + np.array([x_offset, y_offset])

        menton = mediapipe_coords.get("menton")
        tragion = mediapipe_coords.get("tragion")
        gonion_est = mediapipe_coords.get("gonion")

        if not (menton and tragion and gonion_est):
            return edge_points_global

        nose_tip = mediapipe_coords.get("pronasale")
        x_min = min(menton[0], tragion[0], gonion_est[0]) - 30
        x_max = max(menton[0], tragion[0], gonion_est[0]) + 30
        y_min = min(tragion[1], gonion_est[1]) - 20
        y_max = menton[1] + 25
        if nose_tip is not None:
            y_min = max(y_min, nose_tip[1] - 15)

        mask = (
            (edge_points_global[:, 1] >= y_min) &
            (edge_points_global[:, 1] <= y_max) &
            (edge_points_global[:, 0] >= x_min) &
            (edge_points_global[:, 0] <= x_max)
        )
        filtered = edge_points_global[mask]
        if len(filtered) > 12:
            return filtered
        return edge_points_global

    # HYBRID EDGE DETECTION
    def _ransac_fit_line(self, points: np.ndarray):
        """Fit line with lightweight custom RANSAC. Returns (mean_point, direction)."""
        if points is None or len(points) < 5:
            return None

        best_inliers = 0
        best_line = None
        n_iterations = 120
        threshold = 4.5

        for _ in range(n_iterations):
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm

            vecs = points - p1
            dists = np.abs(vecs[:, 0] * direction[1] - vecs[:, 1] * direction[0])
            inlier_mask = dists < threshold
            inlier_count = int(np.sum(inlier_mask))

            if inlier_count > best_inliers:
                inlier_points = points[inlier_mask]
                mean_pt = np.mean(inlier_points, axis=0)
                centered = inlier_points - mean_pt
                cov = np.cov(centered.T)
                if cov.shape == (2, 2):
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    direction_vec = eigvecs[:, int(np.argmax(eigvals))]
                    best_line = (mean_pt, direction_vec / (np.linalg.norm(direction_vec) + 1e-8))
                    best_inliers = inlier_count

        return best_line

    # HYBRID EDGE DETECTION
    def _line_intersection(self, line1, line2) -> Optional[Tuple[int, int]]:
        """Find intersection of two 2D parametric lines."""
        if line1 is None or line2 is None:
            return None
        p1, d1 = line1
        p2, d2 = line2

        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=float)
        b = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return None

        params = np.linalg.solve(A, b)
        intersection = p1 + (params[0] * d1)
        return (int(round(intersection[0])), int(round(intersection[1])))

    # HYBRID EDGE DETECTION
    def _fit_jaw_lines_ransac(self, jaw_points: np.ndarray, mediapipe_coords: Dict):
        """Split jaw points into mandibular/ramus sets and fit two robust lines."""
        if jaw_points is None or len(jaw_points) < 15:
            return None, None, None, None, None

        menton = np.array(mediapipe_coords.get("menton", (0, 0)), dtype=float)
        gonion_est = np.array(mediapipe_coords.get("gonion", (0, 0)), dtype=float)
        tragion = np.array(mediapipe_coords.get("tragion", (0, 0)), dtype=float)

        if np.linalg.norm(gonion_est - menton) < 1e-3 or np.linalg.norm(tragion - gonion_est) < 1e-3:
            return None, None, None, None, None

        def point_to_line_dist(points: np.ndarray, line_start: np.ndarray, line_end: np.ndarray):
            line_vec = line_end - line_start
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-6:
                return np.full(len(points), 1e6)
            line_unit = line_vec / line_len
            point_vecs = points - line_start
            cross = np.abs(point_vecs[:, 0] * line_unit[1] - point_vecs[:, 1] * line_unit[0])
            return cross

        dist_to_mand = point_to_line_dist(jaw_points, menton, gonion_est)
        dist_to_ramus = point_to_line_dist(jaw_points, gonion_est, tragion)

        mandibular_mask = dist_to_mand <= dist_to_ramus
        mandibular_points = jaw_points[mandibular_mask]
        ramus_points = jaw_points[~mandibular_mask]

        if len(mandibular_points) < 5 or len(ramus_points) < 5:
            x_split = gonion_est[0]
            mandibular_points = jaw_points[jaw_points[:, 0] <= x_split]
            ramus_points = jaw_points[jaw_points[:, 0] > x_split]

        mandibular_line = self._ransac_fit_line(mandibular_points) if len(mandibular_points) >= 5 else None
        ramus_line = self._ransac_fit_line(ramus_points) if len(ramus_points) >= 5 else None
        refined_gonion = self._line_intersection(mandibular_line, ramus_line)
        return mandibular_line, ramus_line, refined_gonion, mandibular_points, ramus_points

    @staticmethod
    def _sort_contour_points(points: np.ndarray, start_point: np.ndarray, end_point: np.ndarray) -> np.ndarray:
        """Order scattered edge pixels into a drawable polyline via nearest-neighbor walk."""
        if points is None or len(points) < 2:
            return points
        pts = np.array(points, dtype=float)
        start_pt = np.array(start_point, dtype=float)

        # Begin from the point nearest to start_point
        dists_to_start = np.linalg.norm(pts - start_pt, axis=1)
        current_idx = int(np.argmin(dists_to_start))

        visited = np.zeros(len(pts), dtype=bool)
        ordered = [pts[current_idx]]
        visited[current_idx] = True

        for _ in range(len(pts) - 1):
            dists = np.linalg.norm(pts - pts[current_idx], axis=1)
            dists[visited] = np.inf
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] > 15.0:
                break
            ordered.append(pts[nearest_idx])
            visited[nearest_idx] = True
            current_idx = nearest_idx

        return np.array(ordered, dtype=np.int32)

    # HYBRID EDGE DETECTION
    def _calculate_refined_gonial_angle(self, mandibular_line, ramus_line) -> Optional[float]:
        """Calculate obtuse gonial angle from mandibular + ramus lines."""
        if mandibular_line is None or ramus_line is None:
            return None

        _, d1 = mandibular_line
        _, d2 = ramus_line

        d1_norm = np.linalg.norm(d1)
        d2_norm = np.linalg.norm(d2)
        if d1_norm < 1e-6 or d2_norm < 1e-6:
            return None

        cos_angle = np.clip(np.dot(d1, d2) / (d1_norm * d2_norm), -1.0, 1.0)
        acute = np.degrees(np.arccos(abs(cos_angle)))
        gonial_angle = 180.0 - acute

        if 100.0 <= gonial_angle <= 145.0:
            return float(gonial_angle)
        if 100.0 <= acute <= 145.0:
            return float(acute)
        return None

    # HYBRID EDGE DETECTION
    def _estimate_gonion_hybrid(self, image: np.ndarray, coords: Dict, landmarks, w: int, h: int):
        """
        Hybrid gonion estimation:
        1) edge-refined contour method
        2) MediaPipe geometric method fallback
        3) default fallback
        """
        overlay_debug = {
            "roi_bbox": None,
            "jaw_points": None,
            "mandibular_line": None,
            "ramus_line": None,
            "refined_gonion": None,
            "mandibular_points": None,
            "ramus_points": None,
        }

        base_gonion, base_conf = self._estimate_gonion(coords, landmarks, w, h)
        hybrid_coords = dict(coords)
        if base_gonion is not None:
            hybrid_coords["gonion"] = base_gonion

        try:
            roi_bbox = self._get_jaw_roi(hybrid_coords, w, h)
            if roi_bbox is not None:
                preprocessed_roi, roi_img = self._preprocess_jaw_roi(image, roi_bbox)
                edges = self._detect_jaw_edges(preprocessed_roi, roi_img)
                jaw_points = self._extract_jaw_contour_points(edges, roi_bbox, hybrid_coords)

                if jaw_points is not None and len(jaw_points) >= 20:
                    mand_line, ramus_line, refined_gonion, mandibular_pts, ramus_pts = self._fit_jaw_lines_ransac(jaw_points, hybrid_coords)
                    refined_angle = self._calculate_refined_gonial_angle(mand_line, ramus_line)

                    if refined_angle is not None and refined_gonion is not None:
                        refined_gonion = (
                            max(0, min(w - 1, int(refined_gonion[0]))),
                            max(0, min(h - 1, int(refined_gonion[1]))),
                        )
                        overlay_debug.update({
                            "roi_bbox": roi_bbox,
                            "jaw_points": jaw_points,
                            "mandibular_line": mand_line,
                            "ramus_line": ramus_line,
                            "refined_gonion": refined_gonion,
                            "mandibular_points": mandibular_pts,
                            "ramus_points": ramus_pts,
                        })
                        confidence = float(np.clip(max(base_conf, 0.85), 0.0, 1.0))
                        print(f"[Gonion] Edge-based: angle={refined_angle:.1f}, point={refined_gonion}, conf={confidence:.2f}")
                        return refined_gonion, refined_angle, confidence, "edge_based", overlay_debug

        except Exception as e:
            print(f"[Gonion] Edge-based detection failed: {type(e).__name__}: {e}")

        if base_gonion is not None:
            coords_with_gonion = {**coords, "gonion": base_gonion}
            coords_with_gonion["articulare"] = self._estimate_ramus_top(coords_with_gonion, w, h)

            if all(k in coords_with_gonion for k in ["articulare", "gonion", "menton"]):
                ar = np.array(coords_with_gonion["articulare"], dtype=float)
                go = np.array(coords_with_gonion["gonion"], dtype=float)
                me = np.array(coords_with_gonion["menton"], dtype=float)
                ba = ar - go
                bc = me - go
                denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
                cos_a = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
                mp_angle = float(np.degrees(np.arccos(cos_a)))
                if 100.0 <= mp_angle <= 145.0:
                    confidence = float(np.clip(max(base_conf, 0.45), 0.0, 1.0))
                    print(f"[Gonion] MediaPipe fallback: angle={mp_angle:.1f}, point={base_gonion}, conf={confidence:.2f}")
                    return base_gonion, mp_angle, confidence, "mediapipe", overlay_debug

        gonion_est = coords.get("gonion", base_gonion)
        default_conf = float(np.clip(max(base_conf, 0.10), 0.0, 1.0))
        print(f"[Gonion] Default fallback: angle=120.0, point={gonion_est}, conf={default_conf:.2f}")
        return gonion_est, 120.0, default_conf, "default", overlay_debug

    def _estimate_ramus_top(self, coords: Dict, w: int, h: int) -> Tuple[int, int]:
        """Estimate ramus superior point (Ar) as TMJ-area point above gonion."""
        return self._estimate_articulare(coords, w, h)

    def _estimate_nasolabial_angle(self, coords: Dict) -> Tuple[Optional[float], Dict]:
        """Estimate nasolabial angle at subnasale using multiple nose/lip candidates."""
        debug = {"candidates": []}
        subnasale = coords.get("subnasale")
        if subnasale is None:
            return None, debug

        nose_candidates = []
        for key in ("columella", "pronasale", "nasion"):
            if key in coords:
                nose_candidates.append((key, coords[key]))
        if "columella" in coords and "pronasale" in coords:
            c = coords["columella"]
            p = coords["pronasale"]
            nose_candidates.append(("col_prona_mid", ((c[0] + p[0]) // 2, (c[1] + p[1]) // 2)))

        lip_candidates = []
        for key in ("labrale_superius", "upper_lip_mid", "labrale_inferius"):
            if key in coords:
                lip_candidates.append((key, coords[key]))

        if not nose_candidates or not lip_candidates:
            return None, debug

        def angle_3pt(a, b, c):
            ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
            bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
            mba = np.linalg.norm(ba)
            mbc = np.linalg.norm(bc)
            if mba <= 1e-6 or mbc <= 1e-6:
                return None
            cosv = np.clip(np.dot(ba, bc) / (mba * mbc), -1.0, 1.0)
            return float(np.degrees(np.arccos(cosv)))

        best_angle = None
        best_cost = float("inf")
        for nose_name, nose_pt in nose_candidates:
            for lip_name, lip_pt in lip_candidates:
                angle = angle_3pt(nose_pt, subnasale, lip_pt)
                if angle is None:
                    continue
                debug["candidates"].append({
                    "nose": nose_name,
                    "lip": lip_name,
                    "angle": round(angle, 2)
                })

                # Prefer values in 85-115 and near 100.
                range_penalty = 0.0
                if angle < 85:
                    range_penalty = (85 - angle) * 3.0
                elif angle > 115:
                    range_penalty = (angle - 115) * 3.0
                cost = abs(angle - 100.0) + range_penalty
                if cost < best_cost:
                    best_cost = cost
                    best_angle = angle

        debug["selected_raw"] = None if best_angle is None else round(best_angle, 2)
        return best_angle, debug

    def _calculate_forward_growth(self, coords: Dict, w: int, h: int) -> Dict:
        """Compute forward-growth related measurements from side profile points."""
        result = {
            "forward_growth_score": 5.0,
            "facial_angle": 90.0,
            "maxillary_prominence": 0.0,
            "mandibular_prominence": 0.0,
            "recession_type": "unknown"
        }

        required = ("nasion", "glabella", "subnasale")
        if not all(k in coords for k in required):
            return result

        nasion = coords["nasion"]
        glabella = coords["glabella"]
        subnasale = coords["subnasale"]
        pog = coords.get("pogonion", coords.get("menton"))
        if pog is None:
            return result
        menton = coords.get("menton", pog)

        face_h = max(20.0, float(abs(coords.get("menton", pog)[1] - coords.get("trichion", nasion)[1])))

        # Stabilize with blended references (raw pog can be noisy on casual photos).
        ref_x = (0.65 * nasion[0]) + (0.35 * glabella[0])
        pog_x = (0.70 * pog[0]) + (0.30 * menton[0])
        pog_y = (0.70 * pog[1]) + (0.30 * menton[1])

        # Left-facing normalized: smaller x is more anterior.
        maxillary_prom = (ref_x - subnasale[0]) / face_h
        mandibular_prom = (subnasale[0] - pog_x) / face_h

        dx = float(ref_x - pog_x)  # positive means pog is anterior to reference vertical
        dy = float(max(1.0, pog_y - nasion[1]))
        raw_facial_angle = 90.0 + math.degrees(math.atan2(dx, dy))
        facial_angle = float(np.clip(raw_facial_angle, 82.0, 98.0))

        # Score components with tolerant ranges (avoid collapsing to zero on noisy landmarks).
        facial_score = max(0.0, 10.0 - (abs(facial_angle - 90.0) * 0.48))
        maxillary_score = max(0.0, 10.0 - (abs(maxillary_prom - 0.018) / 0.085) * 10.0)
        mandibular_score = max(0.0, 10.0 - (abs(mandibular_prom + 0.001) / 0.095) * 10.0)
        forward_growth_score = float(np.clip((facial_score * 0.42) + (maxillary_score * 0.30) + (mandibular_score * 0.28), 3.0, 10.0))

        # Recession buckets.
        recession_index = (
            (90.0 - facial_angle) * 0.55
            + max(0.0, (0.008 - maxillary_prom) * 170.0)
            + max(0.0, (-0.015 - mandibular_prom) * 170.0)
        )
        if recession_index < 1.3:
            recession = "good"
        elif recession_index < 2.6:
            recession = "mild"
        elif recession_index < 4.3:
            recession = "moderate"
        else:
            recession = "severe"

        result.update({
            "forward_growth_score": forward_growth_score,
            "facial_angle": facial_angle,
            "maxillary_prominence": float(maxillary_prom),
            "mandibular_prominence": float(mandibular_prom),
            "recession_type": recession
        })
        return result

    def _infer_gender_from_side(self, gonial: float, forward_growth: float, mandibular_prominence: float) -> Tuple[str, float]:
        """Heuristic side-profile gender estimate (male/female/unknown)."""
        score = 0.5
        if gonial <= 118:
            score += 0.22
        elif gonial >= 123:
            score -= 0.22

        if forward_growth >= 6.8:
            score += 0.08
        elif forward_growth <= 5.2:
            score -= 0.08

        if mandibular_prominence >= 0.002:
            score += 0.10
        elif mandibular_prominence <= -0.006:
            score -= 0.10

        score = float(np.clip(score, 0.0, 1.0))
        confidence = float(np.clip(abs(score - 0.5) * 2.0, 0.0, 1.0))

        if score >= 0.62:
            return "male", confidence
        if score <= 0.38:
            return "female", confidence
        return "unknown", confidence

    def detect_profile_orientation(self, image: np.ndarray) -> str:
        """Detect left or right facing"""
        landmarks, _ = self._detect_face_landmarks_with_retries(image)
        if landmarks is None:
            landmarks, _ = self._detect_face_landmarks_with_crop_fallback(image)
        if landmarks is None:
            return 'left'

        w = image.shape[1]
        nose_x = landmarks.landmark[4].x * w

        return 'left' if nose_x < w / 2 else 'right'

    def _extract_side_landmark_coords(self, landmarks, w: int, h: int):
        """Extract configured landmark coordinates and confidence estimates."""
        coords = {}
        confidences = {}
        for name, idx in self.SIDE_LANDMARKS.items():
            try:
                lm = landmarks.landmark[idx]
                confidence = 1.0 - abs(lm.z)
                coords[name] = (int(lm.x * w), int(lm.y * h))
                confidences[name] = confidence
            except Exception:
                continue
        return coords, confidences

    def _is_left_facing_geometry(self, coords: Dict) -> bool:
        """Geometry vote for normalized left-facing profile expectation."""
        left_votes = 0
        right_votes = 0

        pronasale = coords.get("pronasale")
        subnasale = coords.get("subnasale")
        tragion = coords.get("tragion")
        menton = coords.get("menton")
        gonion = coords.get("gonion")

        if pronasale is not None and tragion is not None:
            if pronasale[0] < tragion[0]:
                left_votes += 1
            else:
                right_votes += 1

        if subnasale is not None and tragion is not None:
            if subnasale[0] < tragion[0]:
                left_votes += 1
            else:
                right_votes += 1

        if menton is not None and gonion is not None:
            if gonion[0] >= menton[0]:
                left_votes += 1
            else:
                right_votes += 1

        if (left_votes + right_votes) == 0:
            return True
        return left_votes >= right_votes

    def analyze_side_profile(self, image: np.ndarray):
        """Analyze with auto-mirroring"""
        if image is None:
            return None, None, {"error": "No image"}

        # Auto-mirror if right-facing
        orientation = self.detect_profile_orientation(image)
        was_mirrored = False

        if orientation == 'right':
            image = cv2.flip(image, 1)
            was_mirrored = True
            print("[Profile] Mirrored image")

        landmarks, detection_debug = self._detect_face_landmarks_with_retries(image)
        if landmarks is None:
            crop_landmarks, crop_debug = self._detect_face_landmarks_with_crop_fallback(image)
            if crop_landmarks is not None:
                landmarks = crop_landmarks
                detection_debug = crop_debug

        # Fallback: if detection still fails, try opposite orientation once.
        if landmarks is None:
            fallback_img = cv2.flip(image, 1)
            fallback_landmarks, fallback_debug = self._detect_face_landmarks_with_retries(fallback_img)
            if fallback_landmarks is None:
                fallback_landmarks, fallback_crop_debug = self._detect_face_landmarks_with_crop_fallback(fallback_img)
                if fallback_landmarks is not None:
                    fallback_debug = fallback_crop_debug
            if fallback_landmarks is None:
                return None, None, {
                    "error": "No face detected",
                    "detector_attempts": detection_debug.get("attempts", [])
                }

            image = fallback_img
            landmarks = fallback_landmarks
            was_mirrored = not was_mirrored
            detection_debug = {
                "detector_step": fallback_debug.get("detector_step"),
                "detector_confidence": fallback_debug.get("detector_confidence"),
                "attempts": detection_debug.get("attempts", []) + ["flip_fallback"] + fallback_debug.get("attempts", [])
            }

        h, w = image.shape[:2]

        # Get landmarks with confidence tracking
        coords, confidences = self._extract_side_landmark_coords(landmarks, w, h)
        for name, confidence in confidences.items():
            if confidence < self.MIN_CONFIDENCE:
                print(f"[Landmark] Low confidence for {name}: {confidence:.2f}")

        # Find true menton (lowest chin point)
        menton = self._find_menton(landmarks, w, h)
        if menton:
            coords["menton"] = menton

        # Use whichever tragion candidate is truly posterior in the normalized view
        tragion = self._select_posterior_tragion(coords, landmarks, w, h)
        if tragion:
            coords["tragion"] = tragion

        # Protocol guard: enforce normalized left-facing geometry before hybrid tracing.
        if not self._is_left_facing_geometry(coords):
            forced_image = cv2.flip(image, 1)
            forced_landmarks, forced_debug = self._detect_face_landmarks_with_retries(forced_image)
            if forced_landmarks is None:
                crop_landmarks, crop_debug = self._detect_face_landmarks_with_crop_fallback(forced_image)
                if crop_landmarks is not None:
                    forced_landmarks = crop_landmarks
                    forced_debug = crop_debug
            if forced_landmarks is not None:
                image = forced_image
                landmarks = forced_landmarks
                was_mirrored = not was_mirrored
                coords, confidences = self._extract_side_landmark_coords(landmarks, w, h)
                menton = self._find_menton(landmarks, w, h)
                if menton:
                    coords["menton"] = menton
                tragion = self._select_posterior_tragion(coords, landmarks, w, h)
                if tragion:
                    coords["tragion"] = tragion
                detection_debug = {
                    "detector_step": forced_debug.get("detector_step"),
                    "detector_confidence": forced_debug.get("detector_confidence"),
                    "attempts": detection_debug.get("attempts", []) + ["forced_left_reorientation"] + forced_debug.get("attempts", []),
                }
            else:
                # Preserve protocol even if re-detection fails by flipping known anchors.
                image = forced_image
                was_mirrored = not was_mirrored
                flipped_coords = {}
                for key, pt in coords.items():
                    flipped_coords[key] = (int((w - 1) - pt[0]), int(pt[1]))
                coords = flipped_coords
                detection_debug = {
                    "detector_step": detection_debug.get("detector_step"),
                    "detector_confidence": detection_debug.get("detector_confidence"),
                    "attempts": detection_debug.get("attempts", []) + ["forced_left_reorientation_geom_only"],
                }

        # HYBRID EDGE DETECTION
        gonion, gonial_angle_hybrid, gonion_conf, detection_method, hybrid_debug = self._estimate_gonion_hybrid(
            image, coords, landmarks, w, h
        )
        if gonion:
            coords["gonion"] = gonion
            coords["articulare"] = self._estimate_ramus_top(coords, w, h)

        # Calculate measurements
        measurements = self._calculate_measurements(
            coords,
            w,
            h,
            gonion_confidence=gonion_conf,
            gonial_angle_override=gonial_angle_hybrid if detection_method == "edge_based" else None,
            gonion_detection_method=detection_method
        )
        measurements.was_mirrored = was_mirrored

        # Overlay
        overlay = self._create_overlay(
            image.copy(),
            coords,
            measurements,
            hybrid_debug=hybrid_debug if detection_method == "edge_based" else None
        )

        core_points = {}
        for key in ("articulare", "gonion", "menton", "nasion", "subnasale", "pogonion", "tragion", "pronasale", "glabella"):
            if key in coords and coords[key] is not None:
                core_points[key] = (int(coords[key][0]), int(coords[key][1]))

        jaw_contour = []
        if hybrid_debug and hybrid_debug.get("jaw_points") is not None:
            try:
                # Keep response compact while preserving contour geometry.
                jaw_points = hybrid_debug["jaw_points"]
                jaw_contour = [[int(pt[0]), int(pt[1])] for pt in jaw_points[::4]]
            except Exception:
                jaw_contour = []

        if not jaw_contour:
            fallback_jaw = [coords.get(k) for k in ("menton", "gonion", "tragion", "articulare")]
            jaw_contour = [[int(pt[0]), int(pt[1])] for pt in fallback_jaw if pt is not None]

        missing_ratio = 0.0
        if self.SIDE_LANDMARKS:
            missing_ratio = max(0.0, min(1.0, 1.0 - (len(confidences) / float(len(self.SIDE_LANDMARKS)))))

        debug = {
            "orientation": f"{'right (mirrored)' if was_mirrored else 'left'}",
            "gonion_estimated": gonion is not None,
            "landmark_confidences": {k: round(v, 2) for k, v in confidences.items()},
            "detection": detection_debug,
            "gonion_confidence": round(gonion_conf, 3),
            "gonion_detection_method": detection_method,
            "gonial_angle_used": round(float(gonial_angle_hybrid), 2) if gonial_angle_hybrid is not None else None,
            "gonial_source": measurements.gonial_source,
            "gonial_fallback_reason": measurements.gonial_fallback_reason,
            "raw_gonial_angle": round(float(measurements.raw_gonial_angle), 3) if measurements.raw_gonial_angle is not None else None,
            "pre_validation_gonial_angle": round(float(measurements.pre_validation_gonial_angle), 3) if measurements.pre_validation_gonial_angle is not None else None,
            "landmark_points": core_points,
            "jaw_contour": jaw_contour,
            "missing_landmark_ratio": round(float(missing_ratio), 3)
        }

        return measurements, overlay, debug

    def _preprocess_image(self, image):
        """Enhance contrast to improve face detection on similar-color backgrounds"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _find_menton(self, landmarks, w: int, h: int) -> Tuple[int, int]:
        """Find soft-tissue menton as the most inferior-anterior chin contour point."""
        all_points = self._collect_landmark_points(landmarks, w, h)
        if not all_points:
            return None

        chin_indices = [199, 152, 148, 149, 150, 176, 175, 377, 400, 378, 379, 365, 397]
        points = [self._landmark_to_pixel(landmarks, idx, w, h) for idx in chin_indices]
        points = [p for p in points if p is not None]

        try:
            lower_lip_y = int(landmarks.landmark[14].y * h)
        except:
            lower_lip_y = int(h * 0.45)

        try:
            nose_x = int(landmarks.landmark[4].x * w)
        except:
            nose_x = min(p[0] for p in points)

        ear_candidates = []
        for idx in (234, 454):
            pt = self._landmark_to_pixel(landmarks, idx, w, h)
            if pt is not None:
                ear_candidates.append(pt)
        ear_x = max([p[0] for p in ear_candidates], default=max(p[0] for p in all_points))

        face_depth = max(30, abs(ear_x - nose_x))
        anterior_limit_x = nose_x + int(face_depth * 0.62)

        hull_points = []
        if len(all_points) >= 3:
            hull = cv2.convexHull(np.array(all_points, dtype=np.int32))
            hull_points = [tuple(int(v) for v in p[0]) for p in hull]
        seed_points = points + hull_points

        # Keep points below lower lip and in anterior chin zone; this avoids jaw-angle/posterior drift.
        candidates = [p for p in seed_points if p[1] >= (lower_lip_y + int(h * 0.01)) and p[0] <= anterior_limit_x]
        if not candidates:
            candidates = [p for p in seed_points if p[1] >= (lower_lip_y + int(h * 0.01))]
        if not candidates:
            candidates = seed_points if seed_points else all_points

        def menton_score(pt):
            inferior = (pt[1] - lower_lip_y) / max(1.0, float(h) * 0.55)
            anterior = (ear_x - pt[0]) / max(1.0, float(face_depth))
            posterior_penalty = max(0.0, (pt[0] - anterior_limit_x) / max(1.0, float(face_depth)))
            lip_penalty = max(0.0, ((lower_lip_y + int(h * 0.02)) - pt[1]) / max(1.0, float(h)))
            return (1.55 * inferior) + (0.65 * anterior) - (0.55 * posterior_penalty) - (0.40 * lip_penalty)

        best = max(candidates, key=menton_score)

        chin_low = self._landmark_to_pixel(landmarks, 199, w, h)
        chin_center = self._landmark_to_pixel(landmarks, 152, w, h)
        if chin_low is not None:
            blend_x_ref = chin_center[0] if chin_center is not None else best[0]
            blended = (
                int(round((best[0] * 0.72) + (blend_x_ref * 0.28))),
                int(round((best[1] * 0.68) + (chin_low[1] * 0.32)))
            )
            return (
                max(0, min(w - 1, blended[0])),
                max(0, min(h - 1, blended[1]))
            )

        return best

    def _estimate_gonion_improved(self, coords: Dict, landmarks, w: int, h: int) -> Optional[Tuple[int, int]]:
        """
        Estimate gonion (jaw angle) for a left-facing profile.
        In left-facing: left = anterior (nose), right = posterior (ear).
        Gonion is the most posterior-inferior corner of the mandible.
        """
        # Use menton (true chin tip) if available, fallback to pogonion
        chin_key = "menton" if "menton" in coords else "pogonion"
        if chin_key not in coords:
            return None

        chin = coords[chin_key]
        ear = coords.get("tragion")

        trichion_y = coords.get("trichion", (0, int(h * 0.2)))[1]
        full_face_h = abs(chin[1] - trichion_y)
        if full_face_h < 20:
            full_face_h = int(h * 0.5)

        y_min = chin[1] - int(full_face_h * 0.45)
        y_max = chin[1] + int(full_face_h * 0.05)
        posterior_floor = chin[0] + int(full_face_h * 0.03)
        if ear is not None:
            posterior_floor = max(posterior_floor, ear[0] - int(full_face_h * 0.12))

        nose_x = coords.get("pronasale", coords.get("subnasale", chin))[0]
        profile_depth = max(35, abs((ear[0] if ear is not None else chin[0] + int(full_face_h * 0.4)) - nose_x))
        pred_x = chin[0] + int(profile_depth * 0.50)
        pred_y = chin[1] - int(full_face_h * 0.18)
        if ear is not None:
            # Slightly more posterior/inferior target for gonion (jaw corner).
            pred_x = ear[0] + int(profile_depth * 0.12)
            pred_y = ear[1] + int((chin[1] - ear[1]) * 0.80)

        all_points = self._collect_landmark_points(landmarks, w, h)
        hull_points = []
        if len(all_points) >= 3:
            hull = cv2.convexHull(np.array(all_points, dtype=np.int32))
            hull_points = [tuple(int(v) for v in p[0]) for p in hull]

        jaw_candidate_indices = [58, 172, 136, 150, 149, 148, 152, 377, 400, 378, 379, 365, 397]
        jaw_candidates = [self._landmark_to_pixel(landmarks, idx, w, h) for idx in jaw_candidate_indices]
        jaw_candidates = [p for p in jaw_candidates if p is not None]
        jaw_candidates.extend(hull_points)

        # Include tracked jaw landmarks if present
        for key in ("left_jaw_angle", "jaw_low", "jaw_mid"):
            if key in coords:
                jaw_candidates.append(coords[key])

        y_min = min(y_min, pred_y - int(full_face_h * 0.12))
        y_max = max(y_max, pred_y + int(full_face_h * 0.12))
        posterior_floor = max(posterior_floor, chin[0] + int(profile_depth * 0.26))
        candidates = [p for p in jaw_candidates if y_min <= p[1] <= y_max and p[0] >= posterior_floor]
        if not candidates:
            candidates = [p for p in jaw_candidates if y_min <= p[1] <= y_max]
        if not candidates:
            candidates = jaw_candidates

        if candidates:
            def point_to_line_distance(point, a, b):
                abx = float(b[0] - a[0])
                aby = float(b[1] - a[1])
                apx = float(point[0] - a[0])
                apy = float(point[1] - a[1])
                denom = math.hypot(abx, aby)
                if denom < 1e-6:
                    return 0.0
                return abs(abx * apy - aby * apx) / denom

            def gonion_score(pt):
                posterior = pt[0] / max(w, 1)
                inferior = pt[1] / max(h, 1)
                pred_distance = math.hypot(pt[0] - pred_x, pt[1] - pred_y) / max(full_face_h, 1)
                cornerness = 0.0
                if ear is not None:
                    cornerness = point_to_line_distance(pt, ear, chin) / max(full_face_h, 1)
                return (1.7 * posterior) + (0.8 * inferior) + (1.1 * cornerness) - (1.6 * pred_distance)

            best_candidate = max(candidates, key=gonion_score)
            # If contour candidates are too far from expected mandibular angle, use geometric prediction.
            if math.hypot(best_candidate[0] - pred_x, best_candidate[1] - pred_y) > (full_face_h * 0.22):
                gonion_x, gonion_y = pred_x, pred_y
            else:
                gonion_x, gonion_y = best_candidate

            # Guard against overly anterior gonion picks.
            min_posterior_x = chin[0] + int(profile_depth * 0.32)
            if gonion_x < min_posterior_x:
                gonion_x = int(round((gonion_x * 0.45) + (min_posterior_x * 0.55)))

            # Guard against overly superior gonion picks.
            min_inferior_y = chin[1] - int(full_face_h * 0.30)
            if gonion_y < min_inferior_y:
                gonion_y = int(round((gonion_y * 0.35) + (max(min_inferior_y, pred_y) * 0.65)))
        else:
            # Final fallback if all candidate extraction fails
            gonion_x, gonion_y = pred_x, pred_y

        gonion_x = max(0, min(w - 1, int(gonion_x)))
        gonion_y = max(0, min(h - 1, int(gonion_y)))
        return (gonion_x, gonion_y)

    def _estimate_articulare(self, coords: Dict, w: int, h: int) -> Tuple[int, int]:
        """Estimate articulare (Ar) - junction of posterior ramus border and cranial base.
        Ar sits near the ear (TMJ area), above the gonion by the ramus height."""
        if "gonion" in coords:
            gonion = coords["gonion"]

            if "tragion" in coords:
                tragion = coords["tragion"]
                # Keep ramus plane near-vertical through gonion, with Ar near TMJ level.
                ar_x = gonion[0]
                ar_y = tragion[1] + int(abs(gonion[1] - tragion[1]) * 0.12)
                ar_y = min(ar_y, gonion[1] - max(8, int(h * 0.03)))
            else:
                # Fallback: use full face height to estimate ramus length
                trichion_y = coords.get("trichion", (0, int(h * 0.2)))[1]
                chin_y = coords.get("menton", coords.get("pogonion", (0, int(h * 0.8))))[1]
                full_face_h = abs(chin_y - trichion_y)
                if full_face_h < 20:
                    full_face_h = int(h * 0.5)
                ar_y = gonion[1] - int(full_face_h * 0.28)
                ar_x = gonion[0]

            return (max(0, min(w - 1, ar_x)), max(0, min(h - 1, ar_y)))

        return (int(w * 0.15), int(h * 0.4))

    def _calculate_measurements(
        self,
        coords: Dict,
        w: int,
        h: int,
        gonion_confidence: float = 0.0,
        gonial_angle_override: Optional[float] = None,
        gonion_detection_method: str = "default",
    ) -> SideProfileMeasurements:
        """Calculate all measurements"""

        def angle_3pt(A, B, C):
            """Angle ABC (B is vertex)"""
            if not all([A, B, C]):
                return None

            try:
                BA = np.array([A[0] - B[0], A[1] - B[1]], dtype=float)
                BC = np.array([C[0] - B[0], C[1] - B[1]], dtype=float)

                dot = np.dot(BA, BC)
                mag_BA = np.linalg.norm(BA)
                mag_BC = np.linalg.norm(BC)

                if mag_BA == 0 or mag_BC == 0:
                    return None

                cos_angle = np.clip(dot / (mag_BA * mag_BC), -1.0, 1.0)
                angle = float(np.degrees(np.arccos(cos_angle)))
                return max(0.0, min(180.0, angle))
            except:
                return None

        is_estimated = gonion_confidence < 0.40

        # HYBRID EDGE DETECTION
        # GONIAL ANGLE - Ar-Go-Me (articulare→gonion→menton), with validation-only gating.
        shape_based_gonial = self._estimate_gonial_from_shape(coords, w, h)
        gonial = None
        raw_gonial = None
        pre_validation_gonial = None
        gonial_source = "default"
        fallback_reason = "default"

        if gonion_detection_method != "edge_based":
            fallback_reason = "edge_failed"

        if gonial_angle_override is not None:
            pre_validation_gonial = float(gonial_angle_override)
            raw_gonial = float(gonial_angle_override)
            if 95.0 <= raw_gonial <= 150.0:
                gonial = raw_gonial
                gonial_source = "edge_based"
                fallback_reason = "none"
                is_estimated = False
                print(f"[Gonial] Using edge-based override: {gonial:.1f}")
            else:
                fallback_reason = "mp_invalid"
                print(f"[Gonial] Edge override out of range: {raw_gonial:.1f}")

        if gonial is None and all(k in coords for k in ["articulare", "gonion", "menton"]):
            if gonion_confidence >= 0.40:
                mp_gonial = angle_3pt(
                    coords["articulare"],
                    coords["gonion"],
                    coords["menton"]
                )
                if mp_gonial is not None:
                    pre_validation_gonial = float(mp_gonial)
                    raw_gonial = float(mp_gonial)
                    print(f"[Gonial] Raw calculation: {mp_gonial:.1f}")
                    if 95.0 <= mp_gonial <= 150.0:
                        gonial = float(mp_gonial)
                        gonial_source = "mediapipe"
                        fallback_reason = "none"
                    else:
                        fallback_reason = "mp_invalid"
                else:
                    fallback_reason = "mp_invalid"
            else:
                fallback_reason = "low_confidence"

        if gonial is None:
            if shape_based_gonial is not None and 95.0 <= shape_based_gonial <= 150.0:
                gonial = float(shape_based_gonial)
                gonial_source = "shape_fallback"
                if fallback_reason == "none":
                    fallback_reason = "low_confidence"
                is_estimated = True
            else:
                gonial = 120.0
                gonial_source = "default"
                fallback_reason = "shape_weak" if shape_based_gonial is None else fallback_reason
                is_estimated = True
                print("[Gonial] Falling back to deterministic default: 120.0")

        # Keep measurement bounded without floor-clipping near 110.
        if gonial < 95.0 or gonial > 150.0:
            gonial = float(np.clip(gonial, 95.0, 150.0))
            is_estimated = True
            if fallback_reason == "none":
                fallback_reason = "default"

        # NASOLABIAL
        nasolabial_raw, nasolabial_debug = self._estimate_nasolabial_angle(coords)
        print(f"[Nasolabial] Raw candidates: {nasolabial_debug.get('candidates', [])}")
        print(f"[Nasolabial] Selected raw: {nasolabial_debug.get('selected_raw')}")
        if nasolabial_raw is None:
            nasolabial = 100.0
            is_estimated = True
        else:
            if not (85.0 <= nasolabial_raw <= 115.0):
                is_estimated = True
            nasolabial = float(np.clip(nasolabial_raw, 85.0, 115.0))

        # FACIAL CONVEXITY - use menton (true chin tip) instead of pogonion
        convexity = None
        chin_for_convexity = "menton" if "menton" in coords else "pogonion"
        if all(k in coords for k in ["glabella", "subnasale", chin_for_convexity]):
            convexity = angle_3pt(
                coords["glabella"],
                coords["subnasale"],
                coords[chin_for_convexity]
            )

            if convexity and not (155 <= convexity <= 180):
                convexity = 165.0
                is_estimated = True
        else:
            convexity = 165.0
            is_estimated = True

        if not convexity:
            convexity = 165.0
            is_estimated = True

        # NASOFRONTAL
        nasofrontal = 130

        # Other measurements
        nasofacial = 36
        forehead_slope = 45

        ref_x = coords.get("glabella", (w//2, 0))[0]
        nasal_projection = abs(coords.get("pronasale", (ref_x,0))[0] - ref_x) / w
        chin_pt = coords.get("menton", coords.get("pogonion", (ref_x, 0)))
        chin_projection = abs(chin_pt[0] - ref_x) / w
        lip_projection = abs(coords.get("labrale_superius", (ref_x,0))[0] - ref_x) / w

        # Vertical balance
        trich_y = coords.get("trichion", (0, h//4))[1]
        glab_y = coords.get("glabella", (0, h//2))[1]
        subn_y = coords.get("subnasale", (0, h*2//3))[1]
        pog_y = coords.get("pogonion", (0, h*3//4))[1]

        total_h = abs(pog_y - trich_y)
        if total_h > 0:
            upper = abs(glab_y - trich_y) / total_h
            middle = abs(subn_y - glab_y) / total_h
            lower = abs(pog_y - subn_y) / total_h
        else:
            upper = middle = lower = 0.333

        balance = {
            "upper_third": upper,
            "middle_third": middle,
            "lower_third": lower
        }

        forward_growth = self._calculate_forward_growth(coords, w, h)
        gender, gender_conf = self._infer_gender_from_side(
            gonial=gonial,
            forward_growth=forward_growth["forward_growth_score"],
            mandibular_prominence=forward_growth["mandibular_prominence"]
        )

        # Harmony
        harmony = self._calculate_harmony(
            gonial,
            nasolabial,
            convexity,
            nasofrontal,
            gender=gender,
            forward_growth_score=forward_growth["forward_growth_score"]
        )

        return SideProfileMeasurements(
            facial_convexity_angle=convexity,
            nasofrontal_angle=nasofrontal,
            nasolabial_angle=nasolabial,
            gonial_angle=gonial,
            nasofacial_angle=nasofacial,
            forehead_slope=forehead_slope,
            nasal_projection=nasal_projection,
            chin_projection=chin_projection,
            lip_projection=lip_projection,
            profile_harmony_score=harmony,
            vertical_profile_balance=balance,
            forward_growth_score=forward_growth["forward_growth_score"],
            facial_angle=forward_growth["facial_angle"],
            maxillary_prominence=forward_growth["maxillary_prominence"],
            mandibular_prominence=forward_growth["mandibular_prominence"],
            recession_type=forward_growth["recession_type"],
            gender=gender,
            gender_confidence=gender_conf,
            gonion_confidence=gonion_confidence,
            raw_gonial_angle=float(raw_gonial) if raw_gonial is not None else None,
            pre_validation_gonial_angle=float(pre_validation_gonial) if pre_validation_gonial is not None else None,
            gonial_source=gonial_source,
            gonial_fallback_reason=fallback_reason,
            is_estimated=is_estimated
        )

    def _estimate_gonial_from_shape(self, coords: Dict, w: int, h: int) -> Optional[float]:
        """Estimate gonial angle continuously from mandibular shape."""
        chin_key = "menton" if "menton" in coords else "pogonion"
        if chin_key not in coords or "gonion" not in coords:
            return None

        chin = coords[chin_key]
        gonion = coords["gonion"]

        dx = float(abs(chin[0] - gonion[0]))
        dy = float(abs(chin[1] - gonion[1]))
        jaw_length = float(np.hypot(dx, dy))
        if jaw_length < max(8.0, 0.03 * float(max(w, h))):
            return None

        horizontal_ratio = dx / jaw_length
        vertical_ratio = dy / jaw_length
        # A weak geometry signal should not force an angle.
        if horizontal_ratio < 0.06 and vertical_ratio < 0.30:
            return None

        estimated = 124.0 + (vertical_ratio * 16.0) - (horizontal_ratio * 6.0)
        if not (95.0 <= estimated <= 150.0):
            return None
        return float(estimated)

    def _gaussian_score(self, value, ideal, std_dev):
        """Continuous Gaussian scoring - no sharp cliffs"""
        deviation = abs(value - ideal)
        return 10.0 * math.exp(-(deviation**2) / (2 * std_dev**2))

    def _stretch_score(self, score: float, center: float = 5.5, strength: float = 1.5) -> float:
        # PART A1: Re-expand compressed score distributions.
        deviation = score - center
        stretched = center + (deviation * strength)
        return max(1.0, min(10.0, stretched))

    def _calculate_harmony(self, gonial, nasolabial, convexity, nasofrontal, gender: str = "unknown", forward_growth_score: float = 5.0):
        """Harmony score using continuous Gaussian curves"""
        scores = []

        if gender == "male":
            gonial_ideal = 116
            nasolabial_ideal = 100
            convexity_ideal = 163
        elif gender == "female":
            gonial_ideal = 122
            nasolabial_ideal = 101
            convexity_ideal = 166
        else:
            gonial_ideal = 119
            nasolabial_ideal = 100
            convexity_ideal = 165

        # PART A4 / FIX 9: Wider harmony Gaussians for noisy side landmarks.
        gonial_score = self._gaussian_score(gonial, ideal=gonial_ideal, std_dev=13.0)
        scores.append(gonial_score * 10 * 0.30)  # scale to 0-100 contribution

        # Nasolabial (weight 0.25)
        naso_score = self._gaussian_score(nasolabial, ideal=nasolabial_ideal, std_dev=14.0)
        scores.append(naso_score * 10 * 0.20)

        # Convexity (weight 0.25)
        conv_score = self._gaussian_score(convexity, ideal=convexity_ideal, std_dev=11.0)
        scores.append(conv_score * 10 * 0.20)

        # Nasofrontal (weight 0.1)
        nasof_score = self._gaussian_score(nasofrontal, ideal=130, std_dev=16.0)
        scores.append(nasof_score * 10 * 0.1)

        # Forward growth contribution
        scores.append(np.clip(forward_growth_score, 0.0, 10.0) * 10 * 0.20)

        return sum(scores)

    def calculate_side_profile_score(self, measurements: SideProfileMeasurements) -> Tuple[float, Dict]:
        """Calculate PSL score using continuous Gaussian curves"""
        scores = {}

        def _safe_val(value: float, default: float, low: float = 0.0, high: float = 10.0) -> float:
            try:
                out = float(value)
            except Exception:
                out = float(default)
            if not np.isfinite(out):
                out = float(default)
            return float(np.clip(out, low, high))

        gonial_angle = _safe_val(getattr(measurements, "gonial_angle", 120.0), 120.0, 95.0, 150.0)
        nasolabial_angle = _safe_val(getattr(measurements, "nasolabial_angle", 100.0), 100.0, 70.0, 130.0)
        facial_convexity_angle = _safe_val(getattr(measurements, "facial_convexity_angle", 165.0), 165.0, 135.0, 180.0)
        profile_harmony = _safe_val(getattr(measurements, "profile_harmony_score", 5.0), 5.0, 0.0, 10.0)
        forward_growth = _safe_val(getattr(measurements, "forward_growth_score", 5.0), 5.0, 0.0, 10.0)

        balance = getattr(measurements, "vertical_profile_balance", {}) or {}
        cleaned_balance = {}
        for key in ("upper_third", "middle_third", "lower_third"):
            cleaned_balance[key] = _safe_val(balance.get(key, 1.0 / 3.0), 1.0 / 3.0, 0.0, 1.0)
        balance = cleaned_balance

        gender = getattr(measurements, "gender", "unknown")
        # PART A3 / FIX 1: Further widen std devs with soft floors for side-profile noise.
        if gender == "male":
            gonial_ideal = 116
            gonial_std = 11.0
        elif gender == "female":
            gonial_ideal = 122
            gonial_std = 11.0
        else:
            gonial_ideal = 119
            gonial_std = 12.0

        # FIX 1: Soft floor 2.0 with wider tolerance.
        scores["gonial_angle"] = max(2.0, self._gaussian_score(
            gonial_angle, ideal=gonial_ideal, std_dev=gonial_std
        ))

        # Nasolabial angle
        # PART A3 / FIX 1: Wider std_dev and soft floor.
        scores["nasolabial"] = max(2.0, self._gaussian_score(
            nasolabial_angle, ideal=100, std_dev=12.0
        ))

        # Facial convexity
        # PART A3 / FIX 1: Wider std_dev and soft floor.
        scores["facial_convexity"] = max(2.0, self._gaussian_score(
            facial_convexity_angle, ideal=165, std_dev=11.0
        ))

        # Vertical balance - score based on how close thirds are to 1/3 each
        ideal_third = 1.0 / 3.0
        balance_dev = sum(abs(v - ideal_third) for v in balance.values())
        # PART A3 / FIX 1: More tolerant balance denominator and soft floor.
        scores["vertical_balance"] = max(2.0, 10.0 * math.exp(-(balance_dev**2) / 0.08))

        # Profile harmony
        scores["profile_harmony"] = max(2.0, min(10.0, profile_harmony / 9.6))

        # Forward growth
        scores["forward_growth"] = forward_growth

        # Weighted combination
        if gender == "male":
            WEIGHTS = {
                "gonial_angle": 0.33,
                "forward_growth": 0.22,
                "profile_harmony": 0.20,
                "facial_convexity": 0.12,
                "nasolabial": 0.08,
                "vertical_balance": 0.05
            }
        elif gender == "female":
            WEIGHTS = {
                "gonial_angle": 0.24,
                "forward_growth": 0.16,
                "profile_harmony": 0.24,
                "facial_convexity": 0.16,
                "nasolabial": 0.12,
                "vertical_balance": 0.08
            }
        else:
            WEIGHTS = {
                "gonial_angle": 0.30,
                "forward_growth": 0.18,
                "profile_harmony": 0.23,
                "facial_convexity": 0.14,
                "nasolabial": 0.10,
                "vertical_balance": 0.05
            }

        weighted = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS.keys())
        if not np.isfinite(weighted):
            weighted = 5.0

        if measurements.is_estimated:
            # PART A10 / FIX 10: Very light estimation penalty.
            weighted *= 0.95

        # PART A1: Stretch side score after weighted aggregation.
        weighted = self._stretch_score(weighted, center=5.3, strength=1.5)
        # PART A1 / FIX 2: No additive uplift.
        weighted = float(np.clip(weighted, 1.0, 10.0))
        if not np.isfinite(weighted):
            weighted = 5.0

        sanitized_scores = {}
        for key, value in scores.items():
            try:
                sanitized = float(value)
            except Exception:
                sanitized = 5.0
            if not np.isfinite(sanitized):
                sanitized = 5.0
            sanitized_scores[key] = float(np.clip(sanitized, 0.0, 10.0))

        return round(weighted, 1), sanitized_scores

    # HYBRID EDGE DETECTION
    def _draw_edge_based_overlay(self, overlay: np.ndarray, hybrid_debug: Dict):
        """Draw edge points + fitted lines for hybrid jaw detection debug."""
        if not hybrid_debug:
            return

        jaw_points = hybrid_debug.get("jaw_points")
        mandibular_line = hybrid_debug.get("mandibular_line")
        ramus_line = hybrid_debug.get("ramus_line")
        refined_gonion = hybrid_debug.get("refined_gonion")
        roi_bbox = hybrid_debug.get("roi_bbox")

        if roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 180, 255), 1)

        if jaw_points is not None and len(jaw_points) > 0:
            for pt in jaw_points[::3]:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

        if mandibular_line is not None:
            pt, direction = mandibular_line
            p1 = (int(pt[0] - (direction[0] * 200)), int(pt[1] - (direction[1] * 200)))
            p2 = (int(pt[0] + (direction[0] * 200)), int(pt[1] + (direction[1] * 200)))
            cv2.line(overlay, p1, p2, (0, 200, 255), 2)

        if ramus_line is not None:
            pt, direction = ramus_line
            p1 = (int(pt[0] - (direction[0] * 160)), int(pt[1] - (direction[1] * 160)))
            p2 = (int(pt[0] + (direction[0] * 160)), int(pt[1] + (direction[1] * 160)))
            cv2.line(overlay, p1, p2, (255, 200, 0), 2)

        if refined_gonion is not None:
            cv2.circle(overlay, (int(refined_gonion[0]), int(refined_gonion[1])), 6, (0, 0, 255), -1)

    def _create_overlay(self, image, coords, measurements, hybrid_debug: Optional[Dict] = None):
        """Create overlay"""
        overlay = image.copy()
        h, w = image.shape[:2]

        # PART D1: Contour-first gonial overlay (Ar-Go-Me), shared with v2 renderer logic.
        if all(k in coords for k in ["articulare", "gonion", "menton"]):
            jaw_contour = []
            if hybrid_debug and hybrid_debug.get("jaw_contour"):
                try:
                    jaw_contour = [
                        [int(round(float(pt[0]))), int(round(float(pt[1])))]
                        for pt in (hybrid_debug.get("jaw_contour") or [])
                        if pt is not None and len(pt) >= 2
                    ]
                except Exception:
                    jaw_contour = []
            if not jaw_contour and hybrid_debug and hybrid_debug.get("jaw_points") is not None:
                try:
                    jaw_arr = np.array(hybrid_debug.get("jaw_points"), dtype=float)
                    jaw_contour = [[int(round(float(pt[0]))), int(round(float(pt[1])))] for pt in jaw_arr[::2]]
                except Exception:
                    jaw_contour = []
            if not jaw_contour:
                for key in ("menton", "pogonion", "gonion", "articulare", "tragion"):
                    if key in coords and coords[key] is not None:
                        jaw_contour.append([int(coords[key][0]), int(coords[key][1])])
                if len(jaw_contour) < 3 and "menton" in coords and "gonion" in coords:
                    me = coords["menton"]
                    go = coords["gonion"]
                    mid = (int(round((0.65 * me[0]) + (0.35 * go[0]))), int(round((0.70 * me[1]) + (0.30 * go[1]))))
                    jaw_contour = [[me[0], me[1]], [mid[0], mid[1]], [go[0], go[1]]]

            if callable(draw_gonial_overlay):
                draw_gonial_overlay(
                    overlay=overlay,
                    points=coords,
                    jaw_contour=jaw_contour,
                    processing_mode="hybrid" if hybrid_debug else "color",
                    monochrome_score=0.0,
                    color=(255, 140, 0),
                    thickness=2,
                    point_radius=3,
                )
            else:
                cv2.line(overlay, coords["articulare"], coords["gonion"], (255, 140, 0), 2)
                cv2.line(overlay, coords["gonion"], coords["menton"], (255, 140, 0), 2)
                for key in ["articulare", "gonion", "menton"]:
                    cv2.circle(overlay, coords[key], 3, (255, 140, 0), -1)

        # HYBRID EDGE DETECTION
        if hybrid_debug and self.enable_hybrid_debug_overlay:
            self._draw_edge_based_overlay(overlay, hybrid_debug)

        # Profile line
        profile = ["trichion", "glabella", "nasion", "pronasale", "subnasale",
                   "labrale_superius", "labrale_inferius"]

        for i in range(len(profile) - 1):
            if profile[i] in coords and profile[i+1] in coords:
                cv2.line(overlay, coords[profile[i]], coords[profile[i+1]], (0, 255, 0), 2)

        # Draw lower-lip to chin guide vertically downward (x fixed at lower lip).
        if "labrale_inferius" in coords and "menton" in coords:
            lower_lip = coords["labrale_inferius"]
            menton = coords["menton"]
            chin_guide_end = (lower_lip[0], menton[1])
            cv2.line(overlay, lower_lip, chin_guide_end, (0, 255, 0), 2)

        # PART D1: Remove forward-growth reference clutter.

        # PART D1: More see-through overlay.
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        return image
