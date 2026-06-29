"""
Face Authentication Module for SmartVision
Handles face enrollment and authentication with liveness verification
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from app.core.logger import logger
from app.database.firebase_client import get_db


class _UnavailableLivenessDetector:
    """Fallback used only when the liveness detector cannot be initialized at all.

    Returns is_live=True so that face enrollment and authentication are not
    permanently blocked simply because the optional liveness module failed to
    load.  A warning is logged so the operator knows liveness is not active.
    """

    def detect_liveness(self, frames: List[np.ndarray]) -> Tuple[bool, Dict]:
        logger.warning(
            "Liveness detector unavailable – skipping liveness check and proceeding."
        )
        return True, {
            "warning": "liveness_detector_unavailable",
            "is_live": True,
        }

    def cleanup(self):
        return None


class FaceAuthManager:
    """Manage face-based authentication with anti-spoof protection."""
    DEFAULT_SIMILARITY_THRESHOLDS = {
        128: 0.60,
        512: 0.75,
    }
    
    def __init__(self):
        self.face_embedding_gen = None
        self.liveness_detector = None
        self.similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLDS[512]
        self.min_enrollment_frames = 5
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize face embedding generator and liveness detector."""
        module_errors = []
        try:
            from app.auth.face_embedding import get_face_embedding_generator
            self.face_embedding_gen = get_face_embedding_generator()
        except Exception as e:
            module_errors.append(f"embedding generator: {e}")

        try:
            from app.auth.liveness_detection import get_liveness_detector
            self.liveness_detector = get_liveness_detector()
        except Exception as e:
            logger.warning(f"Failed to initialize liveness detector; using unavailable-detector fallback: {e}")
            self.liveness_detector = _UnavailableLivenessDetector()
            module_errors.append(f"liveness detector: {e}")

        if self.face_embedding_gen is None:
            error_message = "; ".join(module_errors) or "embedding generator unavailable"
            logger.error("Failed to initialize face auth modules: %s", error_message)
            raise RuntimeError(error_message)

        logger.info("Face auth modules initialized")
    
    def enroll_face(self, frames: List[np.ndarray], user_id: str) -> Tuple[bool, str]:
        """
        Enroll user's face into the database.
        
        Args:
            frames: List of face frames (minimum 5)
            user_id: Firebase user ID
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        logger.info(f"[FACE_AUTH] enroll_face called with {len(frames)} frames for user {user_id}")
        
        if not frames or len(frames) < self.min_enrollment_frames:
            return False, f"Need at least {self.min_enrollment_frames} frames for enrollment"
        
        try:
            # Check liveness first
            logger.info("[FACE_AUTH] Starting liveness check...")
            is_live, liveness_metrics = self.liveness_detector.detect_liveness(frames)
            
            if not is_live:
                logger.warning(f"Liveness check failed for user {user_id}. Metrics: {liveness_metrics}")
                return False, "Liveness check failed. Please look at camera naturally and blink or move your head."
            
            logger.info("[FACE_AUTH] Liveness check passed")
            
            # Generate average embedding from multiple frames
            logger.info("[FACE_AUTH] Generating average embedding...")
            avg_embedding = self.face_embedding_gen.generate_average_embedding(frames)
            
            if avg_embedding is None:
                logger.error("[FACE_AUTH] Could not generate face embedding - no clear face detected")
                return False, "Could not generate face embedding. No clear face detected."
            
            logger.info(f"[FACE_AUTH] Generated embedding with shape: {avg_embedding.shape}")
            
            # Validate embedding
            if not self.face_embedding_gen.validate_embedding(avg_embedding):
                logger.error("[FACE_AUTH] Invalid embedding generated")
                return False, "Invalid embedding generated"
            
            logger.info("[FACE_AUTH] Embedding validated, storing in database...")
            
            # Store in Firestore
            success = self._store_face_encoding(user_id, avg_embedding.tolist())
            
            if success:
                logger.info(f"Face enrolled successfully for user {user_id}")
                return True, "Face enrolled successfully!"
            else:
                return False, "Failed to store face encoding in database"
                
        except Exception as e:
            logger.error(f"Face enrollment error: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Enrollment failed: {str(e)}"
    
    def authenticate_face(
        self,
        frame: np.ndarray,
        user_id: str,
        liveness_frames: Optional[List[np.ndarray]] = None,
    ) -> Tuple[bool, str, float]:
        """
        Authenticate user by comparing live face with stored encoding.
        
        Args:
            frame: Single face frame
            user_id: Firebase user ID
            
        Returns:
            Tuple of (authenticated: bool, message: str, similarity_score: float)
        """
        if liveness_frames:
            is_live, _ = self.verify_liveness(liveness_frames)
            if not is_live:
                logger.warning(f"Liveness check failed during authentication for user {user_id}")
                return False, "Liveness check failed. Please blink or move your head naturally.", 0.0

        if frame is None:
            return False, "No face frame provided", 0.0
        
        try:
            # Fetch stored embedding
            stored_embedding = self._get_face_encoding(user_id)
            
            if stored_embedding is None:
                return False, "No face registered for this user", 0.0
             
            # Generate embedding from the whole authentication sequence when possible.
            live_embedding = self._generate_auth_embedding(frame, liveness_frames)
             
            if live_embedding is None:
                return False, "No face detected in frame", 0.0

            if live_embedding.shape != stored_embedding.shape:
                logger.warning(
                    "Stored face embedding dimension mismatch for user %s: live=%s stored=%s",
                    user_id,
                    live_embedding.shape,
                    stored_embedding.shape,
                )
                return False, "Stored face data is outdated. Please enroll your face again.", 0.0

            effective_threshold = self._threshold_for_embedding(stored_embedding)
              
            # Calculate similarity
            similarity = self.face_embedding_gen.cosine_similarity(
                live_embedding, 
                stored_embedding
            )
             
            logger.info(f"Face authentication: similarity={similarity:.4f}, threshold={effective_threshold:.4f}")
             
            if similarity >= effective_threshold:
                logger.info(f"Face authenticated for user {user_id}")
                return True, "Authentication successful", similarity
            else:
                logger.warning(f"Face mismatch: similarity={similarity:.4f}")
                return False, "Face does not match", similarity
                
        except Exception as e:
            logger.error(f"Face authentication error: {e}")
            return False, f"Authentication failed: {str(e)}", 0.0
    
    def verify_liveness(self, frames: List[np.ndarray]) -> Tuple[bool, Dict]:
        """
        Verify liveness of face sequence.
        
        Args:
            frames: List of consecutive face frames
            
        Returns:
            Tuple of (is_live: bool, metrics: dict)
        """
        if not frames or len(frames) < 3:
            return False, {"error": "insufficient_frames"}
        
        return self.liveness_detector.detect_liveness(frames)

    def _threshold_for_embedding(self, embedding: np.ndarray) -> float:
        if embedding is None or len(embedding.shape) != 1:
            return self.similarity_threshold
        return self.DEFAULT_SIMILARITY_THRESHOLDS.get(int(embedding.shape[0]), self.similarity_threshold)

    def _generate_auth_embedding(
        self,
        frame: np.ndarray,
        liveness_frames: Optional[List[np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        candidate_frames = [img for img in (liveness_frames or []) if img is not None]
        if len(candidate_frames) >= 2:
            avg_embedding = self.face_embedding_gen.generate_average_embedding(candidate_frames)
            if avg_embedding is not None and self.face_embedding_gen.validate_embedding(avg_embedding):
                return avg_embedding

        if frame is None:
            return None

        embedding = self.face_embedding_gen.generate_embedding(frame)
        if embedding is None:
            return None
        if not self.face_embedding_gen.validate_embedding(embedding):
            return None
        return embedding
    
    def _store_face_encoding(self, user_id: str, embedding: List[float]) -> bool:
        """Store face encoding in Firestore."""
        try:
            db = get_db()
            if not db:
                logger.error("Firestore database not available")
                return False
            embedding_dim = len(embedding)
             
            data = {
                "face_encoding": embedding,
                "face_enrolled_at": time.time(),
                "embedding_dim": embedding_dim,
                "model": getattr(self.face_embedding_gen, "model_name", "Facenet"),
            }
            
            db.collection("users").document(user_id).set(data, merge=True)
            logger.info(f"Face encoding stored for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store face encoding: {e}")
            return False
    
    def _get_face_encoding(self, user_id: str) -> Optional[np.ndarray]:
        """Fetch face encoding from Firestore."""
        try:
            db = get_db()
            if not db:
                return None
            
            doc_ref = db.collection("users").document(user_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logger.warning(f"User {user_id} not found")
                return None
            
            data = doc.to_dict()
            encoding = data.get("face_encoding")
            
            if encoding is None:
                logger.warning(f"No face encoding found for user {user_id}")
                return None
            
            return np.array(encoding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to fetch face encoding: {e}")
            return None
    
    def has_registered_face(self, user_id: str) -> bool:
        """Check if user has a registered face encoding."""
        try:
            db = get_db()
            if not db:
                return False
            
            doc_ref = db.collection("users").document(user_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return False
            
            data = doc.to_dict()
            return "face_encoding" in data and data["face_encoding"] is not None
            
        except Exception as e:
            logger.error(f"Failed to check registered face: {e}")
            return False
    
    def update_threshold(self, new_threshold: float):
        """Update similarity threshold (0.5 to 0.9 recommended)."""
        if 0.5 <= new_threshold <= 0.9:
            self.similarity_threshold = new_threshold
            logger.info(f"Face auth threshold updated to {new_threshold}")
        else:
            logger.warning(f"Invalid threshold: {new_threshold}. Must be between 0.5 and 0.9")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.liveness_detector:
            self.liveness_detector.cleanup()


# Singleton instance
_face_auth_manager = None


def get_face_auth_manager() -> FaceAuthManager:
    """Get or create singleton FaceAuthManager instance."""
    global _face_auth_manager
    if _face_auth_manager is None:
        _face_auth_manager = FaceAuthManager()
    return _face_auth_manager
