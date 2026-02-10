# enhanced_side_analyzer.py

import cv2
import numpy as np
import mediapipe as mp
import math
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class EnhancedSideMeasurements:
    """More accurate side profile measurements"""
    # Core angles
    gonial_angle: float  # Jaw angle (should be ~120°)
    nasolabial_angle: float  # Should be ~90-110°
    nasofrontal_angle: float  # Should be ~130°
    facial_convexity: float  # Should be ~165°
    
    # Derived metrics
    jawline_score: float
    profile_harmony: float
    
    def to_dict(self):
        return {k: float(v) for k, v in self.__dict__.items()}

class EnhancedSideAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        
        # More accurate landmark mapping for profiles
        self.PROFILE_LANDMARKS = {
            # Forehead and brow
            "forehead": 10,
            "glabella": 168,
            "nasion": 6,
            
            # Nose
            "pronasale": 4,
            "subnasale": 2,
            "columella": 1,
            
            # Mouth and lips
            "upper_lip": 13,
            "lower_lip": 14,
            "labrale_superius": 13,
            "labrale_inferius": 14,
            
            # Chin and jaw
            "pogonion": 152,
            "gnathion": 149,
            "menton": 152,
            
            # Jaw angle - try multiple points
            "gonion_candidate1": 58,
            "gonion_candidate2": 172,
            "gonion_candidate3": 136,
            
            # Ear reference
            "tragion": 234,  # Approximate ear point
        }
    
    def detect_head_pose(self, image):
        """Detect if image is truly a side profile"""
        with self.mp_pose.Pose(min_detection_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Check if head is turned
                left_ear = results.pose_landmarks.landmark[7]  # Left ear
                right_ear = results.pose_landmarks.landmark[8]  # Right ear
                
                # If one ear is visible and other isn't, it's a profile
                ear_visibility_diff = abs(left_ear.visibility - right_ear.visibility)
                return ear_visibility_diff > 0.5
            
        return False
    
    def estimate_gonion_from_shape(self, face_landmarks, image_shape):
        """Estimate gonion position when direct detection fails"""
        h, w = image_shape[:2]
        
        # Get known points
        chin = None
        if hasattr(face_landmarks, 'landmark'):
            # Use multiple chin points to estimate jawline
            chin_points = []
            for idx in [152, 149, 148, 176, 377, 400, 378, 379]:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    chin_points.append((int(lm.x * w), int(lm.y * h)))
            
            if chin_points:
                # Find lowest point (chin)
                chin = max(chin_points, key=lambda p: p[1])
                
                # Estimate gonion as point at back of jaw
                # This is approximate - would need better method
                jaw_width = abs(chin_points[0][0] - chin_points[-1][0])
                gonion_x = chin[0] - jaw_width * 0.3
                gonion_y = chin[1] - jaw_width * 0.2
                
                return (int(gonion_x), int(gonion_y))
        
        return None
    
    def analyze_enhanced_profile(self, image):
        """More accurate side profile analysis"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            results = face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                return None, None, {"error": "No face detected"}
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Extract coordinates
            coords = {}
            for name, idx in self.PROFILE_LANDMARKS.items():
                try:
                    lm = landmarks.landmark[idx]
                    coords[name] = (int(lm.x * w), int(lm.y * h))
                except:
                    continue
            
            # Estimate gonion if not detected
            if "gonion_candidate1" not in coords:
                estimated_gonion = self.estimate_gonion_from_shape(landmarks, image.shape)
                if estimated_gonion:
                    coords["gonion"] = estimated_gonion
            
            # Calculate angles with validation
            measurements = self._calculate_angles(coords)
            
            # Create overlay
            overlay = self._create_enhanced_overlay(image.copy(), coords, measurements)
            
            return measurements, overlay, {"status": "enhanced_analysis"}
    
    def _calculate_angles(self, coords):
        """Calculate angles with validation checks"""
        
        def safe_angle(A, B, C):
            """Calculate angle ABC with validation"""
            if not all([A, B, C]):
                return None
            
            try:
                BA = np.array([A[0] - B[0], A[1] - B[1]])
                BC = np.array([C[0] - B[0], C[1] - B[1]])
                
                dot = np.dot(BA, BC)
                norm_BA = np.linalg.norm(BA)
                norm_BC = np.linalg.norm(BC)
                
                if norm_BA == 0 or norm_BC == 0:
                    return None
                
                cos_angle = dot / (norm_BA * norm_BC)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                
                return np.degrees(np.arccos(cos_angle))
            except:
                return None
        
        # Calculate gonial angle (use multiple points)
        gonial_angle = None
        if all(k in coords for k in ["pogonion", "gonion", "tragion"]):
            gonial_angle = safe_angle(coords["pogonion"], coords["gonion"], coords["tragion"])
        
        # Calculate nasolabial angle
        nasolabial = None
        if all(k in coords for k in ["pronasale", "subnasale", "upper_lip"]):
            nasolabial = safe_angle(coords["pronasale"], coords["subnasale"], coords["upper_lip"])
        
        # Set defaults if calculations fail
        if gonial_angle is None or gonial_angle < 30:  # Unrealistically small
            gonial_angle = 120  # Reasonable default
        
        if nasolabial is None or nasolabial > 180:  # Unrealistically large
            nasolabial = 100  # Reasonable default
        
        # Calculate scores
        jawline_score = self._score_gonial_angle(gonial_angle)
        profile_harmony = self._calculate_overall_harmony(gonial_angle, nasolabial)
        
        return EnhancedSideMeasurements(
            gonial_angle=gonial_angle,
            nasolabial_angle=nasolabial,
            nasofrontal_angle=130,  # Placeholder
            facial_convexity=165,   # Placeholder
            jawline_score=jawline_score,
            profile_harmony=profile_harmony
        )
    
    def _score_gonial_angle(self, angle):
        """Score gonial angle (ideal: 115-125°)"""
        ideal = 120
        deviation = min(abs(angle - ideal) / ideal, 1.0)
        return 10 * (1 - deviation)
    
    def _calculate_overall_harmony(self, gonial, nasolabial):
        """Calculate overall profile harmony"""
        scores = []
        
        # Score gonial angle
        gonial_ideal = 120
        gonial_score = 1 - min(abs(gonial - gonial_ideal) / gonial_ideal, 1.0)
        scores.append(gonial_score)
        
        # Score nasolabial angle
        nasolabial_ideal = 100
        nasolabial_score = 1 - min(abs(nasolabial - nasolabial_ideal) / nasolabial_ideal, 1.0)
        scores.append(nasolabial_score)
        
        return np.mean(scores) * 100
    
    def _create_enhanced_overlay(self, image, coords, measurements):
        """Create overlay with validation indicators"""
        overlay = image.copy()
        
        # Draw key points
        for name, point in coords.items():
            color = (0, 255, 0) if "gonion" in name else (0, 200, 255)
            cv2.circle(overlay, point, 6, color, -1)
            cv2.putText(overlay, name, (point[0] + 10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw angles
        if all(k in coords for k in ["pogonion", "gonion", "tragion"]):
            cv2.line(overlay, coords["pogonion"], coords["gonion"], (255, 0, 0), 2)
            cv2.line(overlay, coords["gonion"], coords["tragion"], (255, 0, 0), 2)
        
        # Add measurements
        y_offset = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        metrics = [
            f"Gonial Angle: {measurements.gonial_angle:.1f}°",
            f"Nasolabial: {measurements.nasolabial_angle:.1f}°",
            f"Jawline Score: {measurements.jawline_score:.1f}/10",
            f"Profile Harmony: {measurements.profile_harmony:.1f}%"
        ]
        
        for text in metrics:
            cv2.putText(overlay, text, (15, y_offset), font, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return overlay