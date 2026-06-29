"""
Liveness Detection Module for SmartVision Face Authentication
Detects real person vs photo/video spoof using blink detection and head movement
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from app.core.logger import logger


class LivenessDetector:
    """Detect liveness using facial landmarks and motion."""
    
    def __init__(self):
        self.face_mesh = None
        self.face_cascade = None
        self.mp_face_mesh = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self._initialize_mediapipe()
        
        # Blink detection parameters
        self.EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for blink
        self.CONSECUTIVE_FRAMES = 3  # Frames to confirm blink
        
        # Head movement parameters
        self.HEAD_MOVEMENT_THRESHOLD = 15.0  # Degrees for head rotation
        
        # State tracking
        self.blink_count = 0
        self.head_movements = []
        self.previous_ear_ratios = []
        self.previous_head_angles = None

    @staticmethod
    def _has_legacy_mediapipe_solutions(mp_module) -> bool:
        solutions = getattr(mp_module, "solutions", None)
        return bool(
            solutions
            and hasattr(solutions, "face_mesh")
            and hasattr(solutions, "drawing_utils")
            and hasattr(solutions, "drawing_styles")
        )
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe FaceMesh."""
        try:
            import mediapipe as mp
            if not self._has_legacy_mediapipe_solutions(mp):
                logger.warning("MediaPipe legacy 'solutions' API is unavailable in this environment; using fallback liveness detection")
                self._initialize_fallback_detector()
                return
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe FaceMesh initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe; using fallback liveness detection: {e}")
            self._initialize_fallback_detector()

    def _initialize_fallback_detector(self):
        """Initialize a Haar-cascade fallback when MediaPipe FaceMesh is unavailable."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                logger.warning("OpenCV Haar cascade could not be loaded for fallback liveness detection")
                self.face_cascade = None
                return
            self.face_cascade = cascade
            logger.info("Fallback Haar-cascade liveness detector initialized")
        except Exception as exc:
            logger.warning(f"Failed to initialize fallback liveness detector: {exc}")
            self.face_cascade = None
    
    def detect_liveness(self, frames: List[np.ndarray]) -> Tuple[bool, Dict]:
        """
        Detect if the input is a live person (not photo/video).
        
        Args:
            frames: List of consecutive face frames
            
        Returns:
            Tuple of (is_live: bool, metrics: dict)
        """
        if not frames or len(frames) < 3:
            logger.warning("Need at least 3 frames for liveness detection")
            return False, {"error": "insufficient_frames"}

        if self.face_mesh is None:
            return self._fallback_detect_liveness(frames)
        
        blink_detected = self._detect_blinks(frames)
        head_movement_detected = self._detect_head_movement(frames)
        
        is_live = blink_detected or head_movement_detected
        
        metrics = {
            "blink_detected": blink_detected,
            "blink_count": self.blink_count,
            "head_movement_detected": head_movement_detected,
            "head_movements": len(self.head_movements),
            "is_live": is_live
        }
        
        logger.info(f"Liveness check: is_live={is_live}, blinks={self.blink_count}, head_moves={len(self.head_movements)}")
        
        # Reset state after check
        self._reset_state()
        
        return is_live, metrics

    def _fallback_detect_liveness(self, frames: List[np.ndarray]) -> Tuple[bool, Dict]:
        """Fallback liveness check using frame-to-frame face motion when MediaPipe is unavailable."""
        face_motion_scores = []
        face_centers = []

        for frame in frames:
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = []
            if self.face_cascade is not None:
                boxes = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            if len(boxes) > 0:
                x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
                roi = gray[y:y + h, x:x + w]
                face_centers.append((x + w / 2.0, y + h / 2.0))
            else:
                roi = gray
                face_centers.append(None)

            face_motion_scores.append(roi)

        if len(face_motion_scores) < 2:
            return False, {
                "fallback_used": True,
                "motion_score": 0.0,
                "center_shift": 0.0,
                "is_live": False,
            }

        motion_score = 0.0
        valid_motion_pairs = 0
        for prev_roi, curr_roi in zip(face_motion_scores, face_motion_scores[1:]):
            if prev_roi.size == 0 or curr_roi.size == 0:
                continue
            resized_prev = cv2.resize(prev_roi, (96, 96))
            resized_curr = cv2.resize(curr_roi, (96, 96))
            motion_score += float(np.mean(cv2.absdiff(resized_prev, resized_curr)))
            valid_motion_pairs += 1

        motion_score = motion_score / valid_motion_pairs if valid_motion_pairs else 0.0

        center_shift = 0.0
        valid_center_pairs = 0
        for prev_center, curr_center in zip(face_centers, face_centers[1:]):
            if prev_center is None or curr_center is None:
                continue
            center_shift += float(np.linalg.norm(np.array(curr_center) - np.array(prev_center)))
            valid_center_pairs += 1
        center_shift = center_shift / valid_center_pairs if valid_center_pairs else 0.0

        is_live = motion_score >= 3.0 or center_shift >= 4.0
        metrics = {
            "fallback_used": True,
            "motion_score": motion_score,
            "center_shift": center_shift,
            "blink_detected": False,
            "blink_count": 0,
            "head_movement_detected": center_shift >= 4.0,
            "head_movements": valid_center_pairs,
            "is_live": is_live,
        }
        logger.info(
            "Fallback liveness check: is_live=%s motion_score=%.2f center_shift=%.2f",
            is_live,
            motion_score,
            center_shift,
        )
        return is_live, metrics
    
    def _detect_blinks(self, frames: List[np.ndarray]) -> bool:
        """Detect blinks in frame sequence using Eye Aspect Ratio (EAR)."""
        ear_values = []
        
        for frame in frames:
            ear = self._calculate_eye_aspect_ratio(frame)
            if ear is not None:
                ear_values.append(ear)
        
        if len(ear_values) < 3:
            logger.warning("Not enough EAR values calculated")
            return False
        
        # Detect blinks (EAR below threshold)
        blink_frames = [ear < self.EAR_THRESHOLD for ear in ear_values]
        
        # Count consecutive blinks
        consecutive_blinks = 0
        max_consecutive = 0
        
        for is_blink in blink_frames:
            if is_blink:
                consecutive_blinks += 1
                max_consecutive = max(max_consecutive, consecutive_blinks)
            else:
                consecutive_blinks = 0
        
        # Need at least CONSECUTIVE_FRAMES frames with eyes closed
        has_blink = max_consecutive >= self.CONSECUTIVE_FRAMES
        
        if has_blink:
            self.blink_count = max_consecutive // self.CONSECUTIVE_FRAMES
        
        return has_blink
    
    def _calculate_eye_aspect_ratio(self, frame: np.ndarray) -> Optional[float]:
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        if frame is None or self.face_mesh is None:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Eye landmark indices (MediaPipe FaceMesh)
        # Left eye: [33, 133, 160, 159, 158, 157, 173]
        # Right eye: [362, 263, 385, 386, 387, 388, 466]
        
        left_eye_points = [33, 133, 160, 159, 158, 157, 173]
        right_eye_points = [362, 263, 385, 386, 387, 388, 466]
        
        height, width, _ = frame.shape
        
        # Get left eye landmarks
        left_eye = []
        for idx in left_eye_points:
            point = face_landmarks.landmark[idx]
            left_eye.append([point.x * width, point.y * height])
        left_eye = np.array(left_eye)
        
        # Get right eye landmarks
        right_eye = []
        for idx in right_eye_points:
            point = face_landmarks.landmark[idx]
            right_eye.append([point.x * width, point.y * height])
        right_eye = np.array(right_eye)
        
        # Calculate EAR for left eye
        # Vertical distances
        left_ear_v1 = np.linalg.norm(left_eye[2] - left_eye[5])
        left_ear_v2 = np.linalg.norm(left_eye[3] - left_eye[4])
        # Horizontal distance
        left_ear_h = np.linalg.norm(left_eye[0] - left_eye[1])
        if left_ear_h <= 1e-6:
            return None
        left_ear = (left_ear_v1 + left_ear_v2) / (2.0 * left_ear_h)
        
        # Calculate EAR for right eye
        right_ear_v1 = np.linalg.norm(right_eye[2] - right_eye[5])
        right_ear_v2 = np.linalg.norm(right_eye[3] - right_eye[4])
        right_ear_h = np.linalg.norm(right_eye[0] - right_eye[1])
        if right_ear_h <= 1e-6:
            return None
        right_ear = (right_ear_v1 + right_ear_v2) / (2.0 * right_ear_h)
        
        # Average both eyes
        ear = (left_ear + right_ear) / 2.0
        
        return ear
    
    def _detect_head_movement(self, frames: List[np.ndarray]) -> bool:
        """Detect head rotation/movement across frames."""
        if len(frames) < 2:
            return False
        
        pitch_changes = []
        yaw_changes = []
        roll_changes = []
        
        for i, frame in enumerate(frames):
            angles = self._estimate_head_pose(frame)
            
            if angles is not None:
                pitch, yaw, roll = angles
                
                if i > 0 and self.previous_head_angles is not None:
                    prev_pitch, prev_yaw, prev_roll = self.previous_head_angles
                    
                    pitch_change = abs(pitch - prev_pitch)
                    yaw_change = abs(yaw - prev_yaw)
                    roll_change = abs(roll - prev_roll)
                    
                    pitch_changes.append(pitch_change)
                    yaw_changes.append(yaw_change)
                    roll_changes.append(roll_change)
                
                self.previous_head_angles = (pitch, yaw, roll)
        
        # Check if any significant head movement detected
        total_movement = sum(pitch_changes) + sum(yaw_changes) + sum(roll_changes)
        has_movement = total_movement > self.HEAD_MOVEMENT_THRESHOLD
        
        if has_movement:
            self.head_movements = pitch_changes + yaw_changes + roll_changes
        
        return has_movement
    
    def _estimate_head_pose(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Estimate head pose (pitch, yaw, roll) from facial landmarks."""
        if frame is None or self.face_mesh is None:
            return None
        
        height, width, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get nose tip and chin landmarks for pose estimation
        nose_tip_idx = 1
        chin_idx = 152
        left_eye_inner_idx = 33
        right_eye_inner_idx = 362
        mouth_left_idx = 61
        mouth_right_idx = 291
        
        points_3d = []
        points_2d = []
        
        # Simplified 3D face model (approximate positions)
        face_3d_model = {
            nose_tip_idx: (0, 0, 0),
            chin_idx: (0, -63, -58),
            left_eye_inner_idx: (-15, -15, -20),
            right_eye_inner_idx: (15, -15, -20),
            mouth_left_idx: (-30, -45, -30),
            mouth_right_idx: (30, -45, -30)
        }
        
        for idx in [nose_tip_idx, chin_idx, left_eye_inner_idx, 
                    right_eye_inner_idx, mouth_left_idx, mouth_right_idx]:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points_2d.append(np.array([x, y], dtype=np.float64))
            points_3d.append(face_3d_model[idx])
        
        points_2d = np.array(points_2d, dtype=np.float64)
        points_3d = np.array(points_3d, dtype=np.float64)
        
        # Camera matrix
        focal_length = height
        camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            points_3d, points_2d, camera_matrix, dist_coeffs
        )
        
        if not success:
            return None
        
        # Convert rotation vector to Euler angles
        rot_matrix, _ = cv2.Rodrigues(rot_vec)
        proj_matrix, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)
        
        pitch = (proj_matrix[0] * 180) / np.pi
        yaw = (proj_matrix[1] * 180) / np.pi
        roll = (proj_matrix[2] * 180) / np.pi
        
        return pitch, yaw, roll
    
    def _reset_state(self):
        """Reset internal state for next liveness check."""
        self.blink_count = 0
        self.head_movements = []
        self.previous_ear_ratios = []
        self.previous_head_angles = None
    
    def cleanup(self):
        """Cleanup MediaPipe resources."""
        if self.face_mesh and hasattr(self.face_mesh, "close"):
            self.face_mesh.close()


# Singleton instance
_liveness_detector = None


def get_liveness_detector() -> LivenessDetector:
    """Get or create singleton LivenessDetector instance."""
    global _liveness_detector
    if _liveness_detector is None:
        _liveness_detector = LivenessDetector()
    return _liveness_detector
