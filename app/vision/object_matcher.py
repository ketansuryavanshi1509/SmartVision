"""
CLIP-based Personal Object Matcher for SmartVision
Uses OpenAI CLIP ViT-B/32 for semantic object embeddings
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from app.core.logger import logger


class CLIPObjectMatcher:
    """Extract and match object embeddings using CLIP model."""
    
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = None
        self.embedding_dim = 512  # CLIP ViT-B/32 output dimension
        self._initialize_clip()
    
    def _initialize_clip(self):
        """Load CLIP model with caching."""
        try:
            import clip
            import torch
            
            # Check if CUDA is available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"CLIP will run on {self.device}")
            
            # Load CLIP ViT-B/32 model (best speed/accuracy balance)
            logger.info("Loading CLIP ViT-B/32 model...")
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
            self.model.eval()
            
            logger.info("CLIP model loaded successfully")
            
        except ImportError as e:
            logger.error(f"CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
            raise
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def extract_embedding(self, image) -> Optional[np.ndarray]:
        """
        Extract CLIP embedding from an image.
        
        Args:
            image: PIL Image or numpy array (BGR from OpenCV)
            
        Returns:
            512-dimensional normalized embedding vector
        """
        if image is None:
            return None
        
        try:
            import torch
            from PIL import Image
            import cv2
            
            # Convert OpenCV BGR to RGB PIL Image
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # Preprocess image
            input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Extract image embedding (no text encoding needed for object matching)
            with torch.no_grad():
                image_features = self.model.encode_image(input_tensor)
                
                # Normalize to unit vector for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = image_features.cpu().numpy().flatten()
            
            # Validate embedding dimension
            if len(embedding) != self.embedding_dim:
                logger.error(f"Expected {self.embedding_dim}-dim CLIP embedding, got {len(embedding)}")
                return None
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"CLIP embedding extraction failed: {e}")
            return None
    
    def extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Extract CLIP text embedding for zero-shot classification.
        
        Args:
            text: Text description (e.g., "a photo of a wallet")
            
        Returns:
            512-dimensional normalized text embedding
        """
        if not text:
            return None
        
        try:
            import clip
            import torch
            
            # Tokenize text
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            
            # Extract text embedding
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embedding = text_features.cpu().numpy().flatten()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"CLIP text embedding extraction failed: {e}")
            return None
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two CLIP embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Ensure vectors are normalized (CLIP already normalizes)
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        similarity = float(np.dot(e1, e2))
        
        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def match_object(self, query_image, stored_embedding: np.ndarray) -> Tuple[float, bool]:
        """
        Match a query image against a stored embedding.
        
        Args:
            query_image: Current camera frame or cropped object
            stored_embedding: Previously stored CLIP embedding
            
        Returns:
            Tuple of (similarity_score, is_match)
            Threshold: >= 0.85 for positive match
        """
        query_embedding = self.extract_embedding(query_image)
        
        if query_embedding is None:
            return 0.0, False
        
        similarity = self.cosine_similarity(query_embedding, stored_embedding)
        
        # Higher threshold than CNN due to CLIP's semantic understanding
        is_match = similarity >= 0.85
        
        logger.info(f"CLIP match: similarity={similarity:.4f}, threshold=0.85, match={is_match}")
        
        return similarity, is_match
    
    def search_top_k(self, query_image, stored_objects: List[Dict], k: int = 5) -> List[Dict]:
        """
        Search for top-k similar objects from stored collection.
        
        Args:
            query_image: Current camera frame
            stored_objects: List of dicts with 'name', 'embedding', 'blip_caption'
            k: Number of results to return
            
        Returns:
            List of matches sorted by similarity
        """
        query_embedding = self.extract_embedding(query_image)
        
        if query_embedding is None:
            return []
        
        matches = []
        
        for obj in stored_objects:
            stored_embedding = np.array(obj.get('embedding'), dtype=np.float32)
            
            if len(stored_embedding) != self.embedding_dim:
                logger.warning(f"Invalid embedding dimension for {obj.get('name')}")
                continue
            
            similarity = self.cosine_similarity(query_embedding, stored_embedding)
            
            if similarity > 0.5:  # Lower threshold for search results
                matches.append({
                    'object_name': obj.get('name'),
                    'similarity': similarity,
                    'blip_caption': obj.get('blip_caption'),
                    'confidence': 'high' if similarity >= 0.90 else 'medium' if similarity >= 0.80 else 'low'
                })
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:k]
    
    def generate_confidence_description(self, similarity: float, object_name: str) -> str:
        """
        Generate natural language description based on confidence.
        
        Args:
            similarity: CLIP similarity score
            object_name: Name of the matched object
            
        Returns:
            Human-readable description
        """
        if similarity >= 0.92:
            return f"Your {object_name} is right here!"
        elif similarity >= 0.85:
            return f"I can see your {object_name}"
        elif similarity >= 0.75:
            return f"That looks like your {object_name}"
        else:
            return f"I see something similar to your {object_name}"
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Validate CLIP embedding format."""
        if embedding is None:
            return False
        if not isinstance(embedding, np.ndarray):
            return False
        if len(embedding.shape) != 1:
            return False
        if embedding.shape[0] != self.embedding_dim:
            logger.warning(f"Expected {self.embedding_dim}-dim CLIP embedding, got {embedding.shape[0]}")
            return False
        return True


# Singleton instance
_clip_matcher = None


def get_clip_matcher() -> CLIPObjectMatcher:
    """Get or create singleton CLIPObjectMatcher instance."""
    global _clip_matcher
    if _clip_matcher is None:
        _clip_matcher = CLIPObjectMatcher()
    return _clip_matcher
