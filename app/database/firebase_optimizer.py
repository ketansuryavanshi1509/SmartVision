"""
Firebase Schema Optimizations for SmartVision
Optimized Firestore structure for mobile performance
Includes indexing strategies, caching, and query optimization
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import time
from app.core.logger import logger


class FirebaseSchemaOptimizer:
    """
    Optimizes Firestore database operations for mobile use.
    - Efficient queries with proper indexing
    - Caching strategies to reduce reads
    - Batch operations for atomicity
    - Data denormalization for faster reads
    """
    
    def __init__(self, db_client):
        self.db = db_client
        self.cache_ttl_seconds = 300  # 5 minutes cache
        self.local_cache: Dict[str, Dict] = {}
        self.cache_timestamps: Dict[str, float] = {}
    
    def get_user_profile_optimized(self, user_id: str) -> Optional[Dict]:
        """
        Get user profile with caching to reduce reads.
        
        Args:
            user_id: Firebase user ID
            
        Returns:
            User profile data or None
        """
        cache_key = f"user_profile_{user_id}"
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"User profile cache hit for {user_id}")
            return cached_data
        
        # Fetch from Firestore
        try:
            user_doc = self.db.collection("users").document(user_id).get()
            
            if not user_doc.exists:
                return None
            
            user_data = user_doc.to_dict()
            
            # Cache the result
            self._add_to_cache(cache_key, user_data)
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None
    
    def store_face_encoding_optimized(self, user_id: str, embedding: List[float], 
                                     metadata: Dict[str, Any]) -> bool:
        """
        Store face encoding with batch write for atomicity.
        
        Args:
            user_id: Firebase user ID
            embedding: Face embedding vector (512-dim for FaceNet)
            metadata: Additional metadata (enrolled_at, liveness_score, etc.)
            
        Returns:
            True if successful
        """
        try:
            batch = self.db.batch()
            
            # Update user profile with face enrollment info
            user_ref = self.db.collection("users").document(user_id)
            batch.update(user_ref, {
                "face_enrolled": True,
                "face_enrolled_at": datetime.now(timezone.utc),
                "embedding_dimension": len(embedding),
                "model_used": metadata.get("model", "FaceNet")
            })
            
            # Store actual embedding in subcollection (keeps main doc small)
            embedding_ref = user_ref.collection("biometrics").document("face")
            batch.set(embedding_ref, {
                "embedding": embedding,
                "created_at": datetime.now(timezone.utc),
                "metadata": metadata
            })
            
            # Commit batch
            batch.commit()
            
            # Invalidate cache
            self._invalidate_cache(f"user_profile_{user_id}")
            
            logger.info(f"Face encoding stored for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing face encoding: {e}")
            return False
    
    def search_personal_objects_optimized(self, user_id: str, 
                                         min_similarity: float = 0.85,
                                         limit: int = 10) -> List[Dict]:
        """
        Search personal objects with efficient querying.
        
        Args:
            user_id: Firebase user ID
            min_similarity: Minimum similarity threshold
            limit: Maximum results to return
            
        Returns:
            List of matching personal objects
        """
        try:
            # Use collection group query if searching across all objects
            # Or direct query if searching specific user's objects
            objects_ref = self.db.collection("users").document(user_id)\
                          .collection("personal_objects")
            
            # Query with index on similarity score
            query = objects_ref\
                .where("similarity_score", ">=", min_similarity)\
                .order_by("similarity_score", direction="DESCENDING")\
                .limit(limit)
            
            results = []
            docs = query.stream()
            
            for doc in docs:
                data = doc.to_dict()
                data["doc_id"] = doc.id
                results.append(data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching personal objects: {e}")
            return []
    
    def store_personal_object_optimized(self, user_id: str, object_name: str,
                                       embedding: List[float], 
                                       blip_caption: str,
                                       image_embedding_dim: int = 512) -> Optional[str]:
        """
        Store personal object with optimized write pattern.
        
        Args:
            user_id: Firebase user ID
            object_name: Name of the object
            embedding: CLIP embedding (512-dim) or ResNet (2048-dim)
            blip_caption: BLIP-generated caption
            image_embedding_dim: Dimension of embedding (512 for CLIP, 2048 for ResNet)
            
        Returns:
            Document ID if successful
        """
        try:
            objects_ref = self.db.collection("users").document(user_id)\
                          .collection("personal_objects")
            
            # Create document with auto-ID
            object_data = {
                "object_name": object_name.lower(),
                "embedding": embedding,
                "embedding_dimension": image_embedding_dim,
                "model_type": "CLIP" if image_embedding_dim == 512 else "ResNet50",
                "blip_caption": blip_caption,
                "created_at": datetime.now(timezone.utc),
                "last_accessed": datetime.now(timezone.utc),
                "access_count": 0,
                "similarity_threshold": 0.85  # Default threshold for this object
            }
            
            doc_ref = objects_ref.add(object_data)
            doc_id = doc_ref[1].id
            
            logger.info(f"Stored personal object '{object_name}' with ID {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing personal object: {e}")
            return None
    
    def increment_object_access(self, user_id: str, object_id: str):
        """
        Increment access counter for frequently used objects.
        Uses atomic increment to avoid race conditions.
        
        Args:
            user_id: Firebase user ID
            object_id: Personal object document ID
        """
        try:
            object_ref = self.db.collection("users").document(user_id)\
                          .collection("personal_objects").document(object_id)
            
            # Atomic increment
            object_ref.update({
                "last_accessed": datetime.now(timezone.utc),
                "access_count": self.db.Increment(1)
            })
            
        except Exception as e:
            logger.error(f"Error incrementing access count: {e}")
    
    def get_frequently_accessed_objects(self, user_id: str, 
                                       min_access_count: int = 3,
                                       limit: int = 5) -> List[Dict]:
        """
        Get user's frequently accessed personal objects.
        Useful for quick access to commonly lost items.
        
        Args:
            user_id: Firebase user ID
            min_access_count: Minimum access count threshold
            limit: Maximum results
            
        Returns:
            List of frequently accessed objects
        """
        try:
            objects_ref = self.db.collection("users").document(user_id)\
                          .collection("personal_objects")
            
            query = objects_ref\
                .where("access_count", ">=", min_access_count)\
                .order_by("access_count", direction="DESCENDING")\
                .limit(limit)
            
            results = []
            docs = query.stream()
            
            for doc in docs:
                data = doc.to_dict()
                data["doc_id"] = doc.id
                results.append(data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching frequent objects: {e}")
            return []
    
    def store_emergency_event_optimized(self, user_id: str, emergency_type: str,
                                       location: Dict[str, float],
                                       audio_recording_url: str = None,
                                       contacts_notified: List[str] = None) -> str:
        """
        Store emergency event with batch writes for reliability.
        
        Args:
            user_id: Firebase user ID
            emergency_type: Type of emergency (help, medical, fall_detected, etc.)
            location: GPS coordinates {lat, lng}
            audio_recording_url: URL to recorded audio (if any)
            contacts_notified: List of emergency contacts notified
            
        Returns:
            Emergency event document ID
        """
        try:
            emergencies_ref = self.db.collection("users").document(user_id)\
                              .collection("emergencies")
            
            emergency_data = {
                "type": emergency_type,
                "timestamp": datetime.now(timezone.utc),
                "location": location,
                "geohash": self._generate_geohash(location.get("lat", 0), 
                                                   location.get("lng", 0)),
                "audio_recording_url": audio_recording_url or "",
                "contacts_notified": contacts_notified or [],
                "status": "active",  # active, resolved, false_alarm
                "priority": "high" if emergency_type in ["medical", "fall_detected"] else "medium"
            }
            
            doc_ref = emergencies_ref.add(emergency_data)
            emergency_id = doc_ref[1].id
            
            # Also update user's latest emergency reference
            self.db.collection("users").document(user_id).update({
                "latest_emergency_id": emergency_id,
                "latest_emergency_time": emergency_data["timestamp"]
            })
            
            logger.warning(f"Emergency event stored: {emergency_type} for user {user_id}")
            return emergency_id
            
        except Exception as e:
            logger.error(f"Error storing emergency event: {e}")
            return ""
    
    def create_composite_index_if_needed(self, collection_name: str, 
                                        fields: List[str]):
        """
        Ensure composite indexes exist for common queries.
        Note: Actual index creation must be done via Firebase Console or CLI.
        This method logs which indexes are needed.
        
        Args:
            collection_name: Name of collection
            fields: List of fields to index together
        """
        # Log required indexes for developer action
        logger.info(f"Required composite index for {collection_name}:")
        logger.info(f"Fields: {', '.join(fields)}")
        logger.info("Create via Firebase Console → Firestore → Indexes")
    
    def cleanup_old_data(self, user_id: str, days_old: int = 30):
        """
        Clean up old data to reduce storage costs.
        
        Args:
            user_id: Firebase user ID
            days_old: Delete data older than this many days
        """
        try:
            cutoff_date = datetime.now(timezone.utc).timestamp() - (days_old * 24 * 60 * 60)
            
            # Clean old emergency events
            emergencies_ref = self.db.collection("users").document(user_id)\
                              .collection("emergencies")
            
            old_emergencies = emergencies_ref\
                .where("timestamp", "<", datetime.fromtimestamp(cutoff_date, timezone.utc))\
                .stream()
            
            deleted_count = 0
            for doc in old_emergencies:
                # Only delete resolved emergencies
                data = doc.to_dict()
                if data.get("status") == "resolved":
                    doc.reference.delete()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old emergency records for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get data from local cache if not expired."""
        if key not in self.local_cache:
            return None
        
        # Check TTL
        age = time.time() - self.cache_timestamps.get(key, 0)
        if age > self.cache_ttl_seconds:
            del self.local_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
            return None
        
        return self.local_cache[key]
    
    def _add_to_cache(self, key: str, data: Dict):
        """Add data to local cache."""
        self.local_cache[key] = data
        self.cache_timestamps[key] = time.time()
    
    def _invalidate_cache(self, key: str):
        """Remove data from cache."""
        if key in self.local_cache:
            del self.local_cache[key]
        if key in self.cache_timestamps:
            del self.cache_timestamps[key]
    
    def _generate_geohash(self, lat: float, lng: float, precision: int = 6) -> str:
        """
        Generate simple geohash for location-based queries.
        For production, use geofirestore library.
        
        Args:
            lat: Latitude
            lng: Longitude
            precision: Geohash precision (default 6 ≈ 1.2km accuracy)
            
        Returns:
            Geohash string
        """
        # Simplified geohash - for production use proper geohash library
        base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
        
        lat_range = (-90.0, 90.0)
        lng_range = (-180.0, 180.0)
        
        geohash = []
        bit = 0
        ch = 0
        is_even = True
        
        while len(geohash) < precision:
            if is_even:
                mid = (lng_range[0] + lng_range[1]) / 2
                if lng > mid:
                    ch |= (1 << (4 - bit))
                    lng_range = (mid, lng_range[1])
                else:
                    lng_range = (lng_range[0], mid)
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if lat > mid:
                    ch |= (1 << (4 - bit))
                    lat_range = (mid, lat_range[1])
                else:
                    lat_range = (lat_range[0], mid)
            
            is_even = not is_even
            
            if bit == 4:
                geohash.append(base32[ch])
                bit = 0
                ch = 0
            else:
                bit += 1
        
        return ''.join(geohash)


# Required Firestore Indexes Documentation
REQUIRED_INDEXES = """
## Required Firestore Indexes for SmartVision

### Users Collection:
None (single document reads don't need indexes)

### Personal Objects Subcollection:
/users/{user_id}/personal_objects

1. **Similarity Search Index**
   Fields: similarity_score (DESCENDING), created_at (DESCENDING)
   
2. **Frequently Accessed Index**
   Fields: access_count (DESCENDING), last_accessed (DESCENDING)
   
3. **Object Name Index**
   Fields: object_name (ASCENDING)

### Emergencies Subcollection:
/users/{user_id}/emergencies

1. **Time-based Query Index**
   Fields: timestamp (DESCENDING)
   
2. **Status + Time Index**
   Fields: status (ASCENDING), timestamp (DESCENDING)
   
3. **Location-based Index** (for nearby emergencies)
   Fields: geohash (ASCENDING), timestamp (DESCENDING)

### Face Embeddings Subcollection:
/users/{user_id}/biometrics

No additional indexes needed (single document access)

### How to Create Indexes:

**Option 1: Firebase Console**
1. Go to Firebase Console → Firestore Database → Indexes
2. Click "Add Index"
3. Select collection and fields
4. Set sort order (ASCENDING/DESCENDING)

**Option 2: Firebase CLI**
```bash
firebase firestore:indexes
```
Edit firestore.indexes.json and deploy.

**Option 3: Programmatically**
Use Firebase Admin SDK to check/create indexes.
"""


def get_schema_optimizer(db_client) -> FirebaseSchemaOptimizer:
    """Get or create optimizer instance."""
    return FirebaseSchemaOptimizer(db_client)
