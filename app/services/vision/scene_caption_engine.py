"""
Continuous Scene Captioning for SmartVision
Implements "Human Brain" mode - only speaks IMPORTANT changes
Updates every 1.5 seconds with intelligent filtering
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Set
from app.core.logger import logger


class SceneCaptionEngine:
    """
    Continuous scene captioning that thinks like a human brain.
    Only speaks IMPORTANT changes, not every detail.
    """
    
    def __init__(self):
        self.caption_interval = 1.5  # seconds between captions
        self.last_caption_time = 0.0
        self.last_caption_text = ""
        self.scene_memory = SceneMemory()
        self.importance_threshold = 0.6  # Minimum importance to speak
        self.change_detection = ChangeDetector()
        
        # Cooldowns to prevent repetition
        self.spoken_topics_cooldown: Dict[str, float] = {}
        self.topic_cooldown_seconds = 10.0  # Don't repeat same topic for 10s
        
        # Context tracking
        self.consecutive_similar_frames = 0
        self.max_consecutive_similar = 3  # After 3 similar frames, stay silent
        
    def should_generate_caption(self, current_time: float) -> bool:
        """Check if enough time has passed since last caption."""
        return (current_time - self.last_caption_time) >= self.caption_interval
    
    def generate_smart_caption(self, 
                               yolo_detections: List[Dict],
                               blip_caption: str,
                               obstacles: List[Dict],
                               personal_objects: List[Dict],
                               navigation_mode: bool = False) -> Optional[str]:
        """
        Generate intelligent caption using "human brain" logic.
        
        Args:
            yolo_detections: Current YOLO object detections
            blip_caption: BLIP's raw scene description
            obstacles: Detected obstacles with threat levels
            personal_objects: Matched personal objects
            navigation_mode: True if user is navigating
            
        Returns:
            Important caption text or None if nothing important
        """
        current_time = time.time()
        
        # Check timing
        if not self.should_generate_caption(current_time):
            return None
        
        # Analyze what changed
        changes = self.change_detection.detect_changes(
            current_detections=yolo_detections,
            current_caption=blip_caption,
            memory=self.scene_memory
        )
        
        # Update scene memory
        self.scene_memory.update(
            detections=yolo_detections,
            caption=blip_caption,
            timestamp=current_time
        )
        
        # Priority 1: Obstacles (safety first!)
        obstacle_warning = self._process_obstacles(obstacles, navigation_mode)
        if obstacle_warning:
            self._update_caption_time()
            return obstacle_warning
        
        # Priority 2: Personal objects (user is looking for these)
        personal_object_alert = self._process_personal_objects(personal_objects)
        if personal_object_alert:
            self._update_caption_time()
            return personal_object_alert
        
        # Priority 3: Significant scene changes
        scene_change_caption = self._process_scene_changes(changes, blip_caption)
        if scene_change_caption:
            self._update_caption_time()
            return scene_change_caption
        
        # Priority 4: New important objects appeared
        new_object_caption = self._process_new_important_objects(changes.new_objects)
        if new_object_caption:
            self._update_caption_time()
            return new_object_caption
        
        # Nothing important - stay silent (like human brain filters)
        return None
    
    def _process_obstacles(self, obstacles: List[Dict], 
                          navigation_mode: bool) -> Optional[str]:
        """Process obstacle warnings (highest priority)."""
        if not obstacles:
            return None
        
        # Sort by threat level
        high_threats = [o for o in obstacles if o.get('threat_level') == 'high']
        medium_threats = [o for o in obstacles if o.get('threat_level') == 'medium']
        
        # Always warn about high threats
        if high_threats:
            obstacle = high_threats[0]  # Most critical
            class_name = obstacle.get('class_name', 'object')
            distance = obstacle.get('distance_meters', 0.0)
            
            if distance < 0.5:
                warning = f"STOP! {class_name.capitalize()} right in front of you!"
            else:
                warning = f"Warning! {class_name.capitalize()} very close, {distance:.1f} meters"
            
            return warning
        
        # Medium threats only if not recently spoken
        if medium_threats and navigation_mode:
            obstacle = medium_threats[0]
            class_name = obstacle.get('class_name', 'object')
            distance = obstacle.get('distance_meters', 0.0)
            
            topic_key = f"obstacle_{class_name}"
            if not self._is_on_cooldown(topic_key):
                warning = f"Caution: {class_name.capitalize()} {distance:.1f} meters ahead"
                self._add_to_cooldown(topic_key)
                return warning
        
        return None
    
    def _process_personal_objects(self, personal_objects: List[Dict]) -> Optional[str]:
        """Process personal object matches."""
        if not personal_objects:
            return None
        
        # Get highest confidence match
        best_match = max(personal_objects, key=lambda x: x.get('similarity', 0.0))
        similarity = best_match.get('similarity', 0.0)
        object_name = best_match.get('object_name', 'object')
        
        # Only announce if good match
        if similarity >= 0.85:
            topic_key = f"personal_{object_name}"
            
            if not self._is_on_cooldown(topic_key):
                confidence = best_match.get('confidence', 'medium')
                
                if similarity >= 0.92:
                    alert = f"Your {object_name} is right here!"
                elif similarity >= 0.85:
                    alert = f"I can see your {object_name}"
                else:
                    alert = f"That looks like your {object_name}"
                
                self._add_to_cooldown(topic_key)
                return alert
        
        return None
    
    def _process_scene_changes(self, changes, 
                              current_caption: str) -> Optional[str]:
        """Process significant scene changes."""
        if not changes.is_significant():
            return None
        
        # Check if we've been seeing similar frames
        if changes.similarity_to_last > 0.8:
            self.consecutive_similar_frames += 1
            
            if self.consecutive_similar_frames >= self.max_consecutive_similar:
                # Scene hasn't changed much - stay silent
                return None
        else:
            # Scene changed significantly - reset counter
            self.consecutive_similar_frames = 0
        
        # Determine what changed
        change_descriptions = []
        
        if changes.new_people_detected:
            change_descriptions.append("People nearby")
        
        if changes.environment_changed:
            if changes.indoor_outdoor_switch:
                change_descriptions.append(f"Moved {changes.indoor_outdoor_direction}")
            elif changes.lighting_changed:
                change_descriptions.append("Lighting changed significantly")
        
        if changes.crowd_density_changed:
            direction = "more crowded" if changes.crowd_increase else "less crowded"
            change_descriptions.append(f"Area is {direction}")
        
        if change_descriptions:
            caption = "; ".join(change_descriptions)
            
            # Check cooldown
            topic_key = f"scene_{hash(caption) % 10000}"
            if not self._is_on_cooldown(topic_key):
                self._add_to_cooldown(topic_key)
                return caption
        
        return None
    
    def _process_new_important_objects(self, 
                                      new_objects: List[str]) -> Optional[str]:
        """Process newly appeared important objects."""
        if not new_objects:
            return None
        
        # Filter for important objects only
        important_categories = {
            'person', 'people', 'crowd',
            'vehicle', 'car', 'bus', 'bicycle',
            'door', 'exit', 'entrance', 'stairs', 'elevator',
            'traffic light', 'crosswalk'
        }
        
        important_new = [
            obj for obj in new_objects 
            if any(cat in obj.lower() for cat in important_categories)
        ]
        
        if not important_new:
            return None
        
        # Limit to most important one
        primary_object = important_new[0]
        
        topic_key = f"new_{primary_object}"
        if not self._is_on_cooldown(topic_key):
            self._add_to_cooldown(topic_key)
            return f"{primary_object.capitalize()} detected"
        
        return None
    
    def _is_on_cooldown(self, topic: str) -> bool:
        """Check if topic is still on cooldown."""
        if topic not in self.spoken_topics_cooldown:
            return False
        
        elapsed = time.time() - self.spoken_topics_cooldown[topic]
        return elapsed < self.topic_cooldown_seconds
    
    def _add_to_cooldown(self, topic: str):
        """Add topic to cooldown tracker."""
        self.spoken_topics_cooldown[topic] = time.time()
        
        # Clean old topics
        now = time.time()
        expired = [
            t for t, ts in self.spoken_topics_cooldown.items()
            if (now - ts) > self.topic_cooldown_seconds * 2
        ]
        
        for topic in expired:
            del self.spoken_topics_cooldown[topic]
    
    def _update_caption_time(self):
        """Update last caption timestamp."""
        self.last_caption_time = time.time()
    
    def reset(self):
        """Reset all state (when user stops moving or changes context)."""
        self.scene_memory.clear()
        self.spoken_topics_cooldown.clear()
        self.consecutive_similar_frames = 0
        self.last_caption_time = 0.0


class SceneMemory:
    """Maintains short-term memory of recent scenes for change detection."""
    
    def __init__(self, max_frames: int = 10):
        self.max_frames = max_frames
        self.frames: List[Dict] = []
        self.last_caption = ""
        self.last_update_time = 0.0
    
    def update(self, detections: List[Dict], caption: str, timestamp: float):
        """Add current frame to memory."""
        frame_data = {
            'timestamp': timestamp,
            'caption': caption,
            'detections': detections.copy() if detections else [],
            'object_set': self._extract_object_set(detections)
        }
        
        self.frames.append(frame_data)
        self.last_caption = caption
        self.last_update_time = timestamp
        
        # Trim old frames
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]
    
    def get_recent_objects(self, seconds: float = 3.0) -> Set[str]:
        """Get objects seen in recent frames."""
        now = time.time()
        cutoff = now - seconds
        
        objects = set()
        for frame in self.frames:
            if frame['timestamp'] >= cutoff:
                objects.update(frame['object_set'])
        
        return objects
    
    def get_average_similarity(self) -> float:
        """Calculate average similarity of recent frames."""
        if len(self.frames) < 2:
            return 1.0
        
        similarities = []
        for i in range(1, len(self.frames)):
            sim = self._calculate_frame_similarity(
                self.frames[i-1], 
                self.frames[i]
            )
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _extract_object_set(self, detections: List[Dict]) -> Set[str]:
        """Extract set of object class names from detections."""
        if not detections:
            return set()
        
        return {det.get('class_name', '').lower() for det in detections}
    
    def _calculate_frame_similarity(self, frame1: Dict, frame2: Dict) -> float:
        """Calculate similarity between two frames."""
        set1 = frame1.get('object_set', set())
        set2 = frame2.get('object_set', set())
        
        if not set1 and not set2:
            return 1.0
        
        if not set1 or not set2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def clear(self):
        """Clear all memory."""
        self.frames.clear()
        self.last_caption = ""
        self.last_update_time = 0.0


class ChangeDetector:
    """Detects significant changes between frames."""
    
    def __init__(self):
        self.significant_change_threshold = 0.5  # Jaccard similarity
        self.people_classes = {'person', 'people', 'human', 'pedestrian'}
        self.vehicle_classes = {'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'van'}
        self.navigation_classes = {'door', 'exit', 'entrance', 'stairs', 'elevator', 'escalator'}
    
    def detect_changes(self, 
                      current_detections: List[Dict],
                      current_caption: str,
                      memory: SceneMemory) -> 'ChangeInfo':
        """Detect what changed from recent frames."""
        current_objects = self._extract_objects(current_detections)
        recent_objects = memory.get_recent_objects(seconds=3.0)
        
        # Calculate overall change
        similarity = self._calculate_jaccard_similarity(current_objects, recent_objects)
        
        # Detect specific changes
        new_objects = current_objects - recent_objects
        disappeared_objects = recent_objects - current_objects
        
        # Check for people changes
        current_people = current_objects & self.people_classes
        recent_people = recent_objects & self.people_classes
        new_people_detected = bool(current_people and not recent_people)
        
        # Check environment changes
        indoor_outdoor_switch = self._detect_indoor_outdoor_switch(
            current_caption, memory.last_caption
        )
        
        lighting_changed = self._detect_lighting_change(
            current_caption, memory.last_caption
        )
        
        # Crowd density change
        crowd_increase = self._detect_crowd_increase(current_detections, memory)
        crowd_density_changed = crowd_increase or self._detect_crowd_decrease(current_detections, memory)
        
        return ChangeInfo(
            similarity_to_last=similarity,
            new_objects=list(new_objects),
            disappeared_objects=list(disappeared_objects),
            new_people_detected=new_people_detected,
            indoor_outdoor_switch=indoor_outdoor_switch,
            indoor_outdoor_direction=self._get_direction(current_caption, memory.last_caption),
            lighting_changed=lighting_changed,
            crowd_density_changed=crowd_density_changed,
            crowd_increase=crowd_increase,
            environment_changed=indoor_outdoor_switch or lighting_changed or crowd_density_changed
        )
    
    def _extract_objects(self, detections: List[Dict]) -> Set[str]:
        """Extract object class names."""
        if not detections:
            return set()
        return {det.get('class_name', '').lower() for det in detections}
    
    def _calculate_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_indoor_outdoor_switch(self, current: str, previous: str) -> bool:
        """Detect if scene switched between indoor and outdoor."""
        indoor_keywords = {'room', 'wall', 'ceiling', 'floor', 'indoor', 'inside'}
        outdoor_keywords = {'sky', 'street', 'outdoor', 'outside', 'building', 'tree'}
        
        current_lower = current.lower()
        previous_lower = previous.lower()
        
        was_indoor = any(kw in previous_lower for kw in indoor_keywords)
        was_outdoor = any(kw in previous_lower for kw in outdoor_keywords)
        is_indoor = any(kw in current_lower for kw in indoor_keywords)
        is_outdoor = any(kw in current_lower for kw in outdoor_keywords)
        
        return (was_indoor and is_outdoor) or (was_outdoor and is_indoor)
    
    def _detect_lighting_change(self, current: str, previous: str) -> bool:
        """Detect significant lighting changes."""
        lighting_keywords = {
            'bright', 'dark', 'dim', 'shadow', 'sunlight', 'night', 'day'
        }
        
        current_words = set(current.lower().split())
        previous_words = set(previous.lower().split())
        
        current_lighting = current_words & lighting_keywords
        previous_lighting = previous_words & lighting_keywords
        
        return bool(current_lighting != previous_lighting)
    
    def _detect_crowd_increase(self, 
                              current_detections: List[Dict],
                              memory: SceneMemory) -> bool:
        """Detect if crowd size increased significantly."""
        current_people_count = sum(
            1 for det in current_detections 
            if det.get('class_name', '').lower() in self.people_classes
        )
        
        # Get average from recent frames
        if len(memory.frames) < 3:
            return False
        
        recent_avg_people = np.mean([
            sum(1 for det in frame.get('detections', [])
                if det.get('class_name', '').lower() in self.people_classes)
            for frame in memory.frames[-3:]
        ])
        
        return current_people_count > recent_avg_people * 1.5  # 50% increase
    
    def _detect_crowd_decrease(self, 
                              current_detections: List[Dict],
                              memory: SceneMemory) -> bool:
        """Detect if crowd size decreased significantly."""
        current_people_count = sum(
            1 for det in current_detections 
            if det.get('class_name', '').lower() in self.people_classes
        )
        
        if len(memory.frames) < 3:
            return False
        
        recent_avg_people = np.mean([
            sum(1 for det in frame.get('detections', [])
                if det.get('class_name', '').lower() in self.people_classes)
            for frame in memory.frames[-3:]
        ])
        
        return current_people_count < recent_avg_people * 0.5  # 50% decrease
    
    def _get_direction(self, current: str, previous: str) -> str:
        """Determine direction of indoor/outdoor switch."""
        outdoor_keywords = {'sky', 'street', 'outdoor', 'outside', 'building', 'tree'}
        
        current_lower = current.lower()
        
        if any(kw in current_lower for kw in outdoor_keywords):
            return "outside"
        return "inside"


class ChangeInfo:
    """Container for change detection results."""
    
    def __init__(self,
                 similarity_to_last: float,
                 new_objects: List[str],
                 disappeared_objects: List[str],
                 new_people_detected: bool,
                 indoor_outdoor_switch: bool,
                 indoor_outdoor_direction: str,
                 lighting_changed: bool,
                 crowd_density_changed: bool,
                 crowd_increase: bool,
                 environment_changed: bool):
        self.similarity_to_last = similarity_to_last
        self.new_objects = new_objects
        self.disappeared_objects = disappeared_objects
        self.new_people_detected = new_people_detected
        self.indoor_outdoor_switch = indoor_outdoor_switch
        self.indoor_outdoor_direction = indoor_outdoor_direction
        self.lighting_changed = lighting_changed
        self.crowd_density_changed = crowd_density_changed
        self.crowd_increase = crowd_increase
        self.environment_changed = environment_changed
    
    def is_significant(self) -> bool:
        """Check if changes are significant enough to report."""
        return (
            self.similarity_to_last < 0.7 or
            self.new_people_detected or
            self.environment_changed or
            len(self.new_objects) > 2
        )


# Singleton instance
_scene_caption_engine = None


def get_scene_caption_engine() -> SceneCaptionEngine:
    """Get or create singleton SceneCaptionEngine instance."""
    global _scene_caption_engine
    if _scene_caption_engine is None:
        _scene_caption_engine = SceneCaptionEngine()
    return _scene_caption_engine
