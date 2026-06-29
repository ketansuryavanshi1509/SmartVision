import numpy as np
import time
from typing import Optional
from app.core.logger import logger
from app.database.firebase_client import get_db

# Support both ResNet (2048) and CLIP (512) embeddings
EMBEDDING_DIM_RESNET = 2048
EMBEDDING_DIM_CLIP = 512
DEFAULT_EMBEDDING_DIM = EMBEDDING_DIM_CLIP  # Default to CLIP for new objects

class PersonalObjectManager:
    """Manages Firestore vector and metadata storage for personal objects."""

    @staticmethod
    def _default_threshold_for_dimension(query_dim: int) -> float:
        if query_dim == EMBEDDING_DIM_CLIP:
            return 0.85
        if query_dim == EMBEDDING_DIM_RESNET:
            return 0.80
        raise ValueError(f"Invalid query embedding dimension: {query_dim}")

    def _resolve_match_threshold(self, query_dim: int, match_threshold: Optional[float]) -> float:
        default_threshold = self._default_threshold_for_dimension(query_dim)
        if match_threshold is None:
            return default_threshold

        try:
            threshold = float(match_threshold)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid match_threshold %r supplied for %d-dim search; using default %.2f",
                match_threshold,
                query_dim,
                default_threshold,
            )
            return default_threshold

        return max(0.0, min(1.0, threshold))
    
    def get_all_objects(self, user_id: str) -> list:
        """Fetch all personal objects for the user, without embeddings."""
        db = get_db()
        if not db:
            return []
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter
            docs = db.collection("users").document(user_id).collection("personal_objects").order_by("created_at").get()
            return [doc.to_dict().get("object_name") for doc in docs]
        except Exception as e:
            logger.error(f"Failed to fetch objects list: {e}")
            return []

    def store_object(self, user_id: str, object_name: str, image_url: str, embedding: list, blip_caption: str) -> dict:
        """Stores object embedding and metadata directly in Firestore.
        
        Args:
            user_id: Firebase user ID
            object_name: Name of the object (e.g., "wallet")
            image_url: URL or path to object image
            embedding: List of floats (512-dim for CLIP, 2048 for ResNet)
            blip_caption: BLIP-generated caption for richer description
        """
        db = get_db()
        if not db:
            raise ValueError("Firestore client not available")
        
        # Validate embedding dimension
        emb_dim = len(embedding)
        if emb_dim not in [EMBEDDING_DIM_CLIP, EMBEDDING_DIM_RESNET]:
            logger.warning(f"Storing object with non-standard embedding dim: {emb_dim}")
        
        payload = {
            "object_name": object_name.lower(),
            "image_url": image_url,
            "blip_caption": blip_caption,
            "embedding": embedding,
            "embedding_dim": emb_dim,  # Store dimension for compatibility
            "model": "CLIP" if emb_dim == EMBEDDING_DIM_CLIP else "ResNet50",
            "created_at": time.time()
        }
        
        _, doc_ref = db.collection("users").document(user_id).collection("personal_objects").add(payload)
        payload["doc_id"] = doc_ref.id
        
        # Omit embedding from return payload to save bandwidth
        ret_payload = {k: v for k, v in payload.items() if k != "embedding"}
        logger.info(f"Stored personal object '{object_name}' with {emb_dim}-dim embedding")
        return ret_payload

    def search_similar(self, user_id: str, query_embedding: list, match_count: int = 5, match_threshold: Optional[float] = None) -> list:
        """
        Searches Firestore for personal objects and computes cosine similarity locally.
        Supports both CLIP (512-dim) and ResNet (2048-dim) embeddings.
        
        Args:
            user_id: Firebase user ID
            query_embedding: Query vector (512 or 2048 dimensions)
            match_count: Number of results to return
            match_threshold: Minimum similarity threshold (0.85 for CLIP, 0.80 for ResNet)
        """
        db = get_db()
        if not db:
            logger.error("Firestore client not available for search_similar")
            return []
            
        # Format query vector
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_dim = int(query_vec.size)
        
        # Determine expected dimension based on query
        try:
            expected_dim = query_dim
            effective_threshold = self._resolve_match_threshold(query_dim, match_threshold)
        except ValueError as exc:
            logger.error(str(exc))
            return []
        
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0.0:
            logger.error("Query embedding has zero norm")
            return []
        query_vec = query_vec / norm_q
        
        try:
            # Fetch all user objects from Firestore
            docs = db.collection("users").document(user_id).collection("personal_objects").get()
        except Exception as e:
            logger.error(f"Failed to fetch objects from Firestore: {e}")
            return []
            
        matches = []
        for doc in docs:
            data = doc.to_dict()
            stored_embedding = data.get("embedding")
            
            if not stored_embedding:
                continue
            
            stored_dim = data.get("embedding_dim", len(stored_embedding))
             
            # Skip incompatible dimensions
            if stored_dim != expected_dim or len(stored_embedding) != expected_dim:
                continue
                
            # Compute Cosine Similarity
            stored_vec = np.array(stored_embedding, dtype=np.float32)
            norm_s = np.linalg.norm(stored_vec)
            if norm_s > 0:
                stored_vec = stored_vec / norm_s
                
            similarity = float(np.dot(query_vec, stored_vec))
            
            if similarity >= effective_threshold:
                matches.append({
                    "object_name": data.get("object_name"),
                    "similarity": similarity,
                    "image_url": data.get("image_url"),
                    "blip_caption": data.get("blip_caption"),
                    "doc_id": doc.id,
                    "model": data.get("model", "Unknown"),
                    "confidence": "high" if similarity >= 0.90 else "medium" if similarity >= 0.80 else "low",
                    "threshold_used": effective_threshold,
                })
                
        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        result = matches[:match_count]
        if result:
            logger.info(f"Found {len(result)} matches above threshold {effective_threshold}")
        else:
            logger.debug(f"No matches found above threshold {effective_threshold}")
        return result
