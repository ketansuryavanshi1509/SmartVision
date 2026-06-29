"""
Face Embedding Module for SmartVision
Uses MediaPipe FaceMesh for landmark detection and DeepFace FaceNet for embeddings
"""

import cv2
import inspect
import numpy as np
from typing import Dict, List, Optional, Tuple
from app.core.logger import logger


class FaceEmbeddingGenerator:
    """Generate face embeddings using DeepFace with FaceNet model."""
    SUPPORTED_EMBEDDING_DIMS = {128, 512}
    
    def __init__(self):
        self.model = None
        self.model_name = "Facenet"
        self.embedding_dim = None
        self._normalize_kwarg_supported = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load DeepFace FaceNet model."""
        try:
            from deepface import DeepFace
            # Test model loading
            logger.info("Loading DeepFace FaceNet model...")
            # Model will be loaded on first use to save memory
            self.model = DeepFace
            logger.info("DeepFace FaceNet model ready")
        except Exception as e:
            logger.error(f"Failed to load DeepFace model: {e}")
            raise
    
    def generate_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from a single frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            512-dimensional embedding vector or None if no face detected
        """
        if frame is None:
            return None
            
        try:
            results = self._call_represent(frame)
            
            if not results or len(results) == 0:
                logger.warning("No face detected in frame")
                return None
            
            # Return first face embedding (primary face)
            embedding = np.array(results[0]["embedding"], dtype=np.float32)
            if embedding.ndim != 1 or embedding.shape[0] not in self.SUPPORTED_EMBEDDING_DIMS:
                logger.warning("Unsupported face embedding dimension returned: %s", embedding.shape)
                return None
            if self.embedding_dim != embedding.shape[0]:
                previous_dim = self.embedding_dim
                self.embedding_dim = embedding.shape[0]
                if previous_dim is None:
                    logger.info("Face embedding dimension detected: %d", self.embedding_dim)
                else:
                    logger.warning("Face embedding dimension changed from %d to %d", previous_dim, self.embedding_dim)
             
            # Normalize to unit vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def _call_represent(self, frame: np.ndarray):
        """Call DeepFace.represent with compatibility across DeepFace versions."""
        kwargs = {
            "img_path": frame,
            "model_name": self.model_name,
            "enforce_detection": True,
            "detector_backend": "opencv",
            "align": True,
        }

        if self._normalize_kwarg_supported is None:
            try:
                params = inspect.signature(self.model.represent).parameters
                self._normalize_kwarg_supported = "normalize" in params
            except (TypeError, ValueError):
                self._normalize_kwarg_supported = True

        if self._normalize_kwarg_supported:
            try:
                return self.model.represent(**kwargs, normalize=True)
            except TypeError as exc:
                if "normalize" not in str(exc):
                    raise
                logger.info("DeepFace.represent does not support normalize=...; retrying without it")
                self._normalize_kwarg_supported = False

        return self.model.represent(**kwargs)
    
    def generate_average_embedding(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Generate average embedding from multiple frames for robustness.
        
        Args:
            frames: List of BGR image frames
            
        Returns:
            Averaged face embedding vector or None
        """
        embeddings = []
        
        for i, frame in enumerate(frames):
            embedding = self.generate_embedding(frame)
            if embedding is not None:
                embeddings.append(embedding)
                logger.info(f"Frame {i+1}/{len(frames)}: Embedding generated successfully")
            else:
                logger.warning(f"Frame {i+1}/{len(frames)}: No face detected")
        
        if not embeddings:
            return None
        
        # Average all valid embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Re-normalize after averaging
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        logger.info(f"Generated average embedding from {len(embeddings)} frames")
        return avg_embedding
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0.0 to 1.0, higher = more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        if embedding1.shape != embedding2.shape:
            logger.warning(
                "Cannot compare face embeddings with different dimensions: %s vs %s",
                embedding1.shape,
                embedding2.shape,
            )
            return 0.0
        
        # Ensure vectors are normalized
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        similarity = float(np.dot(e1, e2))
        
        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding vector format."""
        if embedding is None:
            return False
        if not isinstance(embedding, np.ndarray):
            return False
        if len(embedding.shape) != 1:
            return False
        if embedding.shape[0] not in self.SUPPORTED_EMBEDDING_DIMS:
            logger.warning(f"Unsupported face embedding dimension: {embedding.shape[0]}")
            return False
        if self.embedding_dim is None:
            self.embedding_dim = embedding.shape[0]
        elif embedding.shape[0] != self.embedding_dim:
            logger.warning(f"Expected {self.embedding_dim}-dim embedding, got {embedding.shape[0]}")
            return False
        return True


# Singleton instance
_face_embedding_generator = None


def get_face_embedding_generator() -> FaceEmbeddingGenerator:
    """Get or create singleton FaceEmbeddingGenerator instance."""
    global _face_embedding_generator
    if _face_embedding_generator is None:
        _face_embedding_generator = FaceEmbeddingGenerator()
    return _face_embedding_generator
