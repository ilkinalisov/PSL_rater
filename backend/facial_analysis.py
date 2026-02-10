#facial_analysis.py

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import numpy as np
import mediapipe as mp
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import json

@dataclass
class FacialMeasurements:
    """Enhanced facial measurements with all requested metrics"""
    # Basic measurements
    face_width: float
    face_height: float
    jaw_width: float
    
    # Eye measurements
    left_eye_tilt: float  # in degrees
    right_eye_tilt: float
    interpupillary_distance: float
    eye_separation_ratio: float
    
    # Nose measurements
    nose_width: float
    nose_length: float
    
    # Mouth measurements
    mouth_width: float
    philtrum_length: float
    
    # Jaw measurements
    jaw_to_cheek_ratio: float
    gonial_angle: Optional[float]
    ramus_length: float
    
    # Ratio measurements
    fwhr: float  # Facial Width-to-Height Ratio
    midface_ratio: float
    chin_to_philtrum_ratio: float
    
    # Proportions
    vertical_proportions: Dict[str, float]  # Upper, Middle, Lower
    
    # Symmetry
    facial_symmetry_score: float  # 0-100%
    
    # Golden ratio
    phi_deviation: float
    
    # Additional metrics
    nasal_index: float  # nose width/length
    facial_width_to_height: float
    canthal_tilt_avg: float
    gender: str = "unknown"
    gender_confidence: float = 0.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "face_width": float(self.face_width),
            "face_height": float(self.face_height),
            "jaw_width": float(self.jaw_width),
            "left_eye_tilt": float(self.left_eye_tilt),
            "right_eye_tilt": float(self.right_eye_tilt),
            "interpupillary_distance": float(self.interpupillary_distance),
            "eye_separation_ratio": float(self.eye_separation_ratio),
            "nose_width": float(self.nose_width),
            "nose_length": float(self.nose_length),
            "mouth_width": float(self.mouth_width),
            "philtrum_length": float(self.philtrum_length),
            "jaw_to_cheek_ratio": float(self.jaw_to_cheek_ratio),
            "gonial_angle": float(self.gonial_angle) if self.gonial_angle else None,
            "ramus_length": float(self.ramus_length),
            "fwhr": float(self.fwhr),
            "midface_ratio": float(self.midface_ratio),
            "chin_to_philtrum_ratio": float(self.chin_to_philtrum_ratio),
            "vertical_proportions": self.vertical_proportions,
            "facial_symmetry_score": float(self.facial_symmetry_score),
            "phi_deviation": float(self.phi_deviation),
            "nasal_index": float(self.nasal_index),
            "facial_width_to_height": float(self.facial_width_to_height),
            "canthal_tilt_avg": float(self.canthal_tilt_avg),
            "gender": self.gender,
            "gender_confidence": float(self.gender_confidence)
        }

class EnhancedFacialAnalyzer:
    def __init__(self, static_image_mode=True, max_num_faces=1, 
                 min_detection_confidence=0.5):
        """Initialize MediaPipe Face Mesh for facial landmark detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe Face Mesh has 468 landmarks
        # Define key landmark indices for MediaPipe
        self.LANDMARKS = {
            # Face outline
            "forehead_top": 10,
            "forehead_left": 151,
            "forehead_right": 337,
            
            # Jawline
            "jaw_left": 172,
            "jaw_right": 397,
            "chin": 152,
            "jaw_left_mid": 136,
            "jaw_right_mid": 365,
            
            # Eyes
            "left_eye_outer": 33,
            "left_eye_inner": 133,
            "left_eye_top": 159,
            "left_eye_bottom": 145,
            
            "right_eye_outer": 263,
            "right_eye_inner": 362,
            "right_eye_top": 386,
            "right_eye_bottom": 374,
            
            # Nose
            "nose_tip": 4,
            "nose_bridge_top": 168,
            "nose_bridge_bottom": 6,
            "nose_left_ala": 49,
            "nose_right_ala": 279,
            
            # Mouth
            "mouth_left": 61,
            "mouth_right": 291,
            "mouth_top": 13,
            "mouth_bottom": 14,
            "philtrum": 0,
            
            # Cheekbones
            "left_cheek": 50,
            "right_cheek": 280,
            "left_zygoma": 100,
            "right_zygoma": 329,
            
            # Brow
            "left_brow_outer": 107,
            "left_brow_inner": 105,
            "right_brow_outer": 336,
            "right_brow_inner": 334,
            
            # Facial thirds
            "hairline": 10,
            "browline": 168,
            "nasal_base": 2,
            "menton": 152,
            
            # Ear points (approximate)
            "left_ear_top": 234,
            "right_ear_top": 454
        }
        
        # Golden ratio
        self.PHI = 1.61803398875
        
        # Weights for PSL scoring
        self.WEIGHTS = {
            "symmetry": 0.27,
            "proportions": 0.24,
            "eyes": 0.18,
            "jawline": 0.05,
            "harmony": 0.16,
            "golden_ratio": 0.10
        }

    def _align_face(self, image: np.ndarray) -> np.ndarray:
        """# FIX 8: Rotate image so inter-eye line is horizontal."""
        if image is None:
            return image

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return image

        landmarks = results.multi_face_landmarks[0].landmark
        left_pt = (int(landmarks[33].x * w), int(landmarks[33].y * h))
        right_pt = (int(landmarks[263].x * w), int(landmarks[263].y * h))

        dx = right_pt[0] - left_pt[0]
        dy = right_pt[1] - left_pt[1]
        if abs(dx) < 1:
            return image

        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) > 1.0 and abs(angle) < 30.0:
            center = ((left_pt[0] + right_pt[0]) // 2, (left_pt[1] + right_pt[1]) // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
        return image
    
    def analyze_face(self, image: np.ndarray):
        """Main analysis function"""
        if image is None:
            return None, None, {"error": "No image provided"}

        # FIX 8: Align before measurement extraction.
        image = self._align_face(image)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None, None, {"error": "No face detected"}
        
        # Get landmarks for first face
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmark_coords = {}
        for name, idx in self.LANDMARKS.items():
            try:
                lm = landmarks.landmark[idx]
                landmark_coords[name] = (int(lm.x * w), int(lm.y * h))
            except:
                # Approximate position if landmark doesn't exist
                if "left" in name:
                    landmark_coords[name] = (w // 4, h // 3)
                elif "right" in name:
                    landmark_coords[name] = (w * 3 // 4, h // 3)
                elif "top" in name:
                    landmark_coords[name] = (w // 2, h // 4)
                elif "bottom" in name:
                    landmark_coords[name] = (w // 2, h * 3 // 4)
                else:
                    landmark_coords[name] = (w // 2, h // 2)
        
        # Calculate all measurements
        measurements = self._calculate_measurements(landmark_coords, w, h)
        
        # Create overlay image
        overlay_image = self._create_prominent_overlays(image.copy(), landmark_coords, measurements)
        
        # Prepare debug info
        debug_info = {
            "landmark_count": len(landmarks.landmark),
            "image_dimensions": (w, h),
            "detected_landmarks": len(landmark_coords),
            "gender": measurements.gender,
            "gender_confidence": measurements.gender_confidence
        }
        
        return measurements, overlay_image, debug_info
    
    def _calculate_measurements(self, coords: Dict, img_width: int, img_height: int) -> FacialMeasurements:
        """Calculate all facial measurements from landmarks"""
        
        def dist(p1, p2):
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # 1. Basic facial dimensions
        face_width = dist(coords["jaw_left"], coords["jaw_right"])
        face_height = dist(coords["forehead_top"], coords["chin"])
        
        # 2. Jaw width
        jaw_width = dist(
            (coords["jaw_left"][0], coords["chin"][1]),
            (coords["jaw_right"][0], coords["chin"][1])
        )
        
        # 3. Eye measurements
        left_eye_tilt = self._calculate_canthal_tilt(
            coords["left_eye_inner"], coords["left_eye_outer"]
        )
        right_eye_tilt = self._calculate_canthal_tilt(
            coords["right_eye_inner"], coords["right_eye_outer"]
        )
        
        interpupillary = dist(coords["left_eye_inner"], coords["right_eye_inner"])
        
        # 4. Nose measurements
        nose_width = dist(coords["nose_left_ala"], coords["nose_right_ala"])
        nose_length = dist(coords["nose_bridge_top"], coords["nose_tip"])
        
        # 5. Mouth measurements
        mouth_width = dist(coords["mouth_left"], coords["mouth_right"])
        philtrum_length = dist(coords["nose_tip"], coords["philtrum"])
        
        # 6. fWHR (Facial Width-to-Height Ratio)
        bizygomatic_width = dist(coords["left_zygoma"], coords["right_zygoma"])
        upper_face_height = dist(coords["browline"], coords["nasal_base"])
        fwhr = bizygomatic_width / upper_face_height if upper_face_height > 0 else 0
        
        # 7. Midface ratio
        total_face_height = dist(coords["hairline"], coords["menton"])
        midface_height = dist(coords["browline"], coords["nasal_base"])
        midface_ratio = midface_height / total_face_height if total_face_height > 0 else 0
        
        # 8. Eye separation ratio
        eye_separation_ratio = interpupillary / face_width if face_width > 0 else 0
        
        # 9. Chin-to-philtrum ratio
        chin_to_nose = dist(coords["nasal_base"], coords["menton"])
        chin_to_philtrum_ratio = chin_to_nose / philtrum_length if philtrum_length > 0 else 0
        
        # 10. Jaw-to-cheek ratio
        jaw_to_cheek_ratio = jaw_width / face_width if face_width > 0 else 0
        
        # 11. Vertical proportions
        upper_third = dist(coords["hairline"], coords["browline"])
        middle_third = dist(coords["browline"], coords["nasal_base"])
        lower_third = dist(coords["nasal_base"], coords["menton"])
        
        vertical_proportions = {
            "upper": upper_third / total_face_height if total_face_height > 0 else 0,
            "middle": middle_third / total_face_height if total_face_height > 0 else 0,
            "lower": lower_third / total_face_height if total_face_height > 0 else 0
        }
        
        # 12. Ramus length (approximate)
        ramus_length = dist(coords["jaw_left_mid"], coords["left_ear_top"])
        
        # 13. Symmetry score
        symmetry_score = self._calculate_facial_symmetry(coords)
        
        # 14. Phi deviation
        phi_deviation = self._calculate_golden_ratio_deviation(coords)
        
        # 15. Other ratios
        nasal_index = nose_width / nose_length if nose_length > 0 else 0
        facial_width_to_height = face_width / face_height if face_height > 0 else 0
        canthal_tilt_avg = (left_eye_tilt + right_eye_tilt) / 2
        
        gender, gender_conf = self._infer_gender_from_front(
            fwhr=fwhr,
            jaw_to_cheek_ratio=jaw_to_cheek_ratio,
            canthal_tilt=canthal_tilt_avg,
            lower_third_ratio=vertical_proportions.get("lower", 0.33)
        )

        return FacialMeasurements(
            face_width=face_width,
            face_height=face_height,
            jaw_width=jaw_width,
            left_eye_tilt=left_eye_tilt,
            right_eye_tilt=right_eye_tilt,
            interpupillary_distance=interpupillary,
            eye_separation_ratio=eye_separation_ratio,
            nose_width=nose_width,
            nose_length=nose_length,
            mouth_width=mouth_width,
            philtrum_length=philtrum_length,
            jaw_to_cheek_ratio=jaw_to_cheek_ratio,
            gonial_angle=None,  # Would require side profile
            ramus_length=ramus_length,
            fwhr=fwhr,
            midface_ratio=midface_ratio,
            chin_to_philtrum_ratio=chin_to_philtrum_ratio,
            vertical_proportions=vertical_proportions,
            facial_symmetry_score=symmetry_score,
            phi_deviation=phi_deviation,
            nasal_index=nasal_index,
            facial_width_to_height=facial_width_to_height,
            canthal_tilt_avg=canthal_tilt_avg,
            gender=gender,
            gender_confidence=gender_conf
        )

    def _infer_gender_from_front(self, fwhr: float, jaw_to_cheek_ratio: float, canthal_tilt: float, lower_third_ratio: float) -> Tuple[str, float]:
        """Heuristic front-view gender estimate (male/female/unknown)."""
        score = 0.5

        if fwhr >= 1.90:
            score += 0.22
        elif fwhr <= 1.82:
            score -= 0.22

        if jaw_to_cheek_ratio >= 0.72:
            score += 0.18
        elif jaw_to_cheek_ratio <= 0.64:
            score -= 0.18

        if canthal_tilt >= 6.2:
            score += 0.06
        elif canthal_tilt <= 4.0:
            score -= 0.06

        if lower_third_ratio >= 0.35:
            score += 0.07
        elif lower_third_ratio <= 0.31:
            score -= 0.07

        score = float(np.clip(score, 0.0, 1.0))
        confidence = float(np.clip(abs(score - 0.5) * 2.0, 0.0, 1.0))

        if score >= 0.62:
            return "male", confidence
        if score <= 0.38:
            return "female", confidence
        return "unknown", confidence
    
    def _calculate_canthal_tilt(self, inner_corner: Tuple, outer_corner: Tuple) -> float:
        """Calculate eye tilt in degrees."""
        # FIX 4: Side-invariant tilt by normalizing horizontal direction with abs(dx).
        dx = outer_corner[0] - inner_corner[0]
        dy = inner_corner[1] - outer_corner[1]

        if abs(dx) < 1:
            return 0.0

        angle = math.degrees(math.atan2(dy, abs(dx)))
        return max(-15.0, min(15.0, angle))
    
    def _calculate_facial_symmetry(self, coords: Dict) -> float:
        """Calculate facial symmetry score (0-100%)"""
        symmetric_pairs = [
            ("left_eye_outer", "right_eye_outer"),
            ("left_eye_inner", "right_eye_inner"),
            ("left_cheek", "right_cheek"),
            ("mouth_left", "mouth_right"),
            ("nose_left_ala", "nose_right_ala"),
            ("jaw_left_mid", "jaw_right_mid"),
            ("left_zygoma", "right_zygoma"),
        ]

        # FIX 3: Compute robust midline from bilateral pair averages.
        midline_x_estimates = []
        for left, right in symmetric_pairs:
            if left in coords and right in coords:
                midline_x_estimates.append((coords[left][0] + coords[right][0]) / 2.0)
        if not midline_x_estimates:
            face_center_x = coords.get("forehead_top", (0, 0))[0]
        else:
            face_center_x = sum(midline_x_estimates) / len(midline_x_estimates)

        deviations = []
        for left, right in symmetric_pairs:
            if left in coords and right in coords:
                left_coord = coords[left]
                right_coord = coords[right]

                mirrored_right = (int(2 * face_center_x - right_coord[0]), right_coord[1])
                
                # Distance between left and mirrored right
                dx = left_coord[0] - mirrored_right[0]
                dy = left_coord[1] - mirrored_right[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Normalize by face width
                face_width = math.sqrt(
                    (coords["jaw_right"][0] - coords["jaw_left"][0])**2 +
                    (coords["jaw_right"][1] - coords["jaw_left"][1])**2
                )
                if face_width > 0:
                    normalized_deviation = dist / face_width
                    deviations.append(normalized_deviation)
        
        if not deviations:
            return 50.0
        
        avg_deviation = sum(deviations) / len(deviations)
        # PART A8: More forgiving symmetry scaling for landmark noise.
        symmetry_score = max(0, 100 - (avg_deviation * 350))
        return min(100, symmetry_score)

    def _gaussian_score_static(self, value: float, ideal: float, std_dev: float) -> float:
        # PART A5: Shared Gaussian utility for robust metric scoring.
        return 10.0 * math.exp(-((value - ideal) ** 2) / (2 * (std_dev ** 2)))

    def _stretch_score(self, score: float, center: float = 5.5, strength: float = 1.5) -> float:
        # PART A1: Re-expand compressed score distributions.
        deviation = score - center
        stretched = center + (deviation * strength)
        return max(1.0, min(10.0, stretched))
    
    def _calculate_golden_ratio_deviation(self, coords: Dict) -> float:
        """Calculate overall deviation from golden ratio proportions"""
        ratios_to_check = []
        
        # Face width to height
        face_width = math.sqrt(
            (coords["jaw_right"][0] - coords["jaw_left"][0])**2 +
            (coords["jaw_right"][1] - coords["jaw_left"][1])**2
        )
        face_height = math.sqrt(
            (coords["chin"][0] - coords["forehead_top"][0])**2 +
            (coords["chin"][1] - coords["forehead_top"][1])**2
        )
        
        if face_height > 0:
            ratios_to_check.append(face_width / face_height)
        
        # Facial thirds
        if "hairline" in coords and "browline" in coords:
            forehead = math.sqrt(
                (coords["browline"][0] - coords["hairline"][0])**2 +
                (coords["browline"][1] - coords["hairline"][1])**2
            )
        else:
            forehead = face_height * 0.33
        
        if "browline" in coords and "nasal_base" in coords:
            midface = math.sqrt(
                (coords["nasal_base"][0] - coords["browline"][0])**2 +
                (coords["nasal_base"][1] - coords["browline"][1])**2
            )
        else:
            midface = face_height * 0.33
        
        if "nasal_base" in coords and "menton" in coords:
            lower = math.sqrt(
                (coords["menton"][0] - coords["nasal_base"][0])**2 +
                (coords["menton"][1] - coords["nasal_base"][1])**2
            )
        else:
            lower = face_height * 0.34
        
        if forehead > 0:
            ratios_to_check.append(midface / forehead)
        if midface > 0:
            ratios_to_check.append(lower / midface)
        
        if not ratios_to_check:
            return 0.5
        
        # Calculate average deviation from phi
        deviations = [abs(ratio - self.PHI) / self.PHI for ratio in ratios_to_check]
        avg_deviation = sum(deviations) / len(deviations)
        return avg_deviation
    
    def calculate_psl_score(self, measurements: FacialMeasurements) -> Tuple[float, Dict]:
        """Calculate PSL score (1-10) with improved eye scoring"""
        
        # Calculate individual scores
        scores = {}
        gender = getattr(measurements, "gender", "unknown")

        if gender == "male":
            fwhr_ideal = 1.95
            tilt_low, tilt_high = 5.0, 8.0
            jaw_ideal = 0.74
            weights = {
                "symmetry": 0.25,
                "proportions": 0.23,
                "eyes": 0.18,
                "jawline": 0.07,
                "harmony": 0.17,
                "golden_ratio": 0.10
            }
        elif gender == "female":
            fwhr_ideal = 1.80
            tilt_low, tilt_high = 3.0, 6.0
            jaw_ideal = 0.66
            weights = {
                "symmetry": 0.27,
                "proportions": 0.25,
                "eyes": 0.20,
                "jawline": 0.04,
                "harmony": 0.15,
                "golden_ratio": 0.09
            }
        else:
            fwhr_ideal = 1.87
            tilt_low, tilt_high = 4.0, 7.0
            jaw_ideal = 0.70
            weights = self.WEIGHTS.copy()
        
        # 1. Symmetry score
        scores["symmetry"] = (measurements.facial_symmetry_score / 100) * 10
        
        # PART A6: Gaussian fWHR scoring with wider tolerance + soft floor.
        fwhr_dev = abs(measurements.fwhr - fwhr_ideal)
        fwhr_score = 10.0 * math.exp(-(fwhr_dev ** 2) / (2 * 0.26 ** 2))
        fwhr_score = max(3.0, fwhr_score)

        # PART A6: Gaussian thirds harmony scoring + soft floor.
        thirds_ideal = 0.333
        thirds_dev = (
            abs(measurements.vertical_proportions["upper"] - thirds_ideal) +
            abs(measurements.vertical_proportions["middle"] - thirds_ideal) +
            abs(measurements.vertical_proportions["lower"] - thirds_ideal)
        ) / 3
        thirds_score = 10.0 * math.exp(-(thirds_dev ** 2) / (2 * 0.09 ** 2))
        thirds_score = max(3.0, thirds_score)
        
        scores["proportions"] = (fwhr_score * 0.6 + thirds_score * 0.4)
        
        # 3. Eyes score - IMPROVED CANTHAL TILT SCORING
        avg_tilt = measurements.canthal_tilt_avg

        if tilt_low <= avg_tilt <= tilt_high:
            tilt_score = 10.0
        elif avg_tilt < tilt_low:
            # FIX 4: Steeper penalties away from target canthal tilt range.
            tilt_score = max(0.0, 10.0 - ((tilt_low - avg_tilt) * 1.5))
        else:
            tilt_score = max(0.0, 10.0 - ((avg_tilt - tilt_high) * 1.3))
        
        # Eye separation ratio (ideal: 0.4)
        eye_sep_ideal = 0.4
        eye_sep_score = 10 * (1 - min(abs(measurements.eye_separation_ratio - eye_sep_ideal) / eye_sep_ideal, 1))
        
        scores["eyes"] = (tilt_score * 0.6 + eye_sep_score * 0.4)
        
        # PART A5: Robust jawline scoring from ratio + jaw width signal.
        jaw_deviation = abs(measurements.jaw_to_cheek_ratio - jaw_ideal)
        jaw_ratio_score = 10.0 * math.exp(-(jaw_deviation ** 2) / (2 * 0.15 ** 2))
        jaw_width_ratio = measurements.jaw_width / measurements.face_width if measurements.face_width > 0 else 0.7
        jaw_width_score = self._gaussian_score_static(jaw_width_ratio * 100.0, ideal=75.0, std_dev=12.0)
        jaw_score = (jaw_ratio_score * 0.6) + (jaw_width_score * 0.4)
        jaw_score = max(4.0, jaw_score)
        scores["jawline"] = jaw_score
        
        # PART A2: Harmony should rely on stable front metrics (exclude jawline from front harmony).
        feature_scores = [scores["symmetry"], scores["proportions"], scores["eyes"]]
        feature_mean = float(np.mean(feature_scores))
        feature_std = float(np.std(feature_scores))
        consistency_bonus = max(0.0, 10.0 - (feature_std * 1.1))
        harmony_score = (feature_mean * 0.7) + (consistency_bonus * 0.3)
        scores["harmony"] = max(2.5, min(10.0, harmony_score))
        
        # 6. Golden ratio score
        golden_score = 10 * (1 - min(measurements.phi_deviation, 1))
        scores["golden_ratio"] = golden_score
        
        # PART A1: Weighted average followed by stretch to restore full dynamic range.
        weighted_score = sum(scores.get(component, 5.0) * weight for component, weight in weights.items())
        stretched = self._stretch_score(weighted_score, center=5.5, strength=1.18)
        
        final_score = min(10.0, self._apply_scaling_curve(stretched))
        
        return round(final_score, 1), scores
    
    def _apply_scaling_curve(self, score: float) -> float:
        """Mild S-curve that preserves full 1-10 range."""
        # FIX 2: Don't crush highs or inflate lows.
        if score <= 2:
            return score * 0.88
        elif score >= 8:
            return 8 + (score - 8) * 0.90
        else:
            normalized = (score - 2) / 6
            curved = normalized ** 1.02
            return 2 + curved * 6
    
    def _create_prominent_overlays(self, image: np.ndarray, coords: Dict, measurements: FacialMeasurements) -> np.ndarray:
        """Create highly visible overlays"""
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # PART D2: Cleaner overlay palette.
        colors = {
            "face_outline": (0, 255, 0),        # Bright Green
            "facial_thirds": (255, 140, 0),     # Bright Orange
            "eye_measurements": (0, 200, 255),  # Cyan
            "symmetry": (255, 255, 0),          # Yellow
            "jawline": (255, 165, 0)            # Orange
        }
        
        def draw_bold_line(img, pt1, pt2, color, thickness=2):
            cv2.line(img, pt1, pt2, color, thickness)
        
        # 1. Face outline
        if all(k in coords for k in ["jaw_left", "forehead_top", "jaw_right", "chin"]):
            face_points = np.array([
                coords["jaw_left"],
                coords["forehead_top"],
                coords["jaw_right"],
                coords["chin"],
                coords["jaw_left"]
            ], np.int32)
            
            for i in range(len(face_points)-1):
                draw_bold_line(overlay, tuple(face_points[i]), tuple(face_points[i+1]), 
                             colors["face_outline"], 2)
        
        # PART D2: Short facial-third tick marks instead of full-width lines.
        thirds_points = ["hairline", "browline", "nasal_base", "menton"]
        if all(p in coords for p in thirds_points):
            for point in thirds_points:
                y = coords[point][1]
                cv2.line(overlay, (12, y), (42, y), colors["facial_thirds"], 2)
                cv2.line(overlay, (w - 42, y), (w - 12, y), colors["facial_thirds"], 2)
        
        # 3. Eye measurements
        for side in ["left", "right"]:
            inner = f"{side}_eye_inner"
            outer = f"{side}_eye_outer"
            
            if inner in coords and outer in coords:
                draw_bold_line(overlay, coords[inner], coords[outer], 
                             colors["eye_measurements"], 2)
        
        # 4. Jawline
        jaw_points = ["jaw_left", "jaw_left_mid", "chin", "jaw_right_mid", "jaw_right"]
        if all(p in coords for p in jaw_points):
            jaw_line = [coords[p] for p in jaw_points]
            for i in range(len(jaw_line)-1):
                draw_bold_line(overlay, jaw_line[i], jaw_line[i+1], colors["jawline"], 2)

        # PART D2: Keep symmetry center line, using robust bilateral midline.
        symmetry_pairs = [
            ("left_eye_outer", "right_eye_outer"),
            ("left_eye_inner", "right_eye_inner"),
            ("left_cheek", "right_cheek"),
            ("mouth_left", "mouth_right"),
            ("nose_left_ala", "nose_right_ala"),
            ("jaw_left_mid", "jaw_right_mid"),
            ("left_zygoma", "right_zygoma"),
        ]
        symmetry_estimates = []
        for left, right in symmetry_pairs:
            if left in coords and right in coords:
                symmetry_estimates.append((coords[left][0] + coords[right][0]) / 2.0)
        if symmetry_estimates:
            symmetry_x = int(sum(symmetry_estimates) / len(symmetry_estimates))
        elif "forehead_top" in coords:
            symmetry_x = coords["forehead_top"][0]
        else:
            symmetry_x = w // 2

        if symmetry_x is not None:
            for y in range(0, h, 20):
                cv2.line(overlay, (symmetry_x, y), (symmetry_x, y + 10), colors["symmetry"], 2)
        
        # PART D2: More transparent overlay.
        alpha = 0.55
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image
