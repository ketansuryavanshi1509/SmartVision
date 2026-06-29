import cv2
import numpy as np
import threading
import time
import logging
import warnings
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from app.services.objects.personal_objects_manager import PersonalObjectManager
from app.core.logger import logger

class VisionEngine:
    # DeepFace face recognition removed as per user request
    def get_detected_obstacles(self):
        """Return a list of detected obstacles (e.g., car, bus, truck, bicycle, person)."""
        obstacle_classes = set([
            'car', 'bus', 'truck', 'bicycle', 'person', 'motorcycle'
        ])
        with self.frame_lock:
            dets = self.detections.copy() if hasattr(self, 'detections') else []
        obstacles = [d['class_name'] for d in dets if d['class_name'] in obstacle_classes]
        return list(set(obstacles))
    def get_detected_landmarks(self):
        """Return a list of detected landmark objects (e.g., buildings, traffic lights, stop signs)."""
        # For demo, treat 'traffic light', 'stop sign', 'bench', 'parking meter', 'bus', 'train', 'boat', 'building' as landmarks
        landmark_classes = set([
            'traffic light', 'stop sign', 'bench', 'parking meter', 'bus', 'train', 'boat', 'building'
        ])
        with self.frame_lock:
            dets = self.detections.copy() if hasattr(self, 'detections') else []
        landmarks = [d['class_name'] for d in dets if d['class_name'] in landmark_classes]
        return list(set(landmarks))
    """
    Vision processing engine using YOLOv8 (Ultralytics) with CNN-based personal object recognition.
    - Adds navigation_mode flag: when True, normal auto-announcements are suppressed.
    - Emergency classes are still announced in navigation mode.
    - Uses CNN feature extraction for enhanced personal object recognition.
    """

    def __init__(self, camera_index=0):
        """Initialize the vision engine with CNN feature extraction."""
        # Reduce noisy Hugging Face HTTP logs during model load.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.camera = None
        self.model = None
        self.is_running = False
        self.current_frame = None
        self.detections = []
        self.frame_lock = threading.Lock()
        # BLIP image captioning
        self.caption_model = None
        self.caption_processor = None
        
        # Announcement / voice integration
        self.last_announced_objects = set()
        self.announcement_cooldown = 1.0
        self.last_announcement_time = 0
        self.voice_engine = None
        self.announcement_callback = None
        
        # Duplicate prevention for vision announcements
        self.recent_vision_announcements = set()
        self.vision_announcement_timestamps = {}

        # Track objects detected above confidence threshold continuously
        self.object_detection_start_times = {}  # key: class_name, value: first detection timestamp

        # Motion detection
        self.previous_frame_gray = None
        self.motion_threshold = 5000

        # Navigation mode: when True, only emergency classes will be auto-announced
        self.navigation_mode = False
        # Default emergency/obstacle classes — announced in navigation mode.
        self.emergency_classes = set(['person', 'car', 'motorcycle', 'truck', 'bus', 'bicycle'])

        # Personal object storage with CNN features
        self.personal_objects_mgr = PersonalObjectManager()
        self.personal_objects = {}  # key: object_name, value: image_path
        self.personal_object_features = {}  # key: object_name, value: CNN feature vector
        self.last_personal_search = {}  # key: object_name, value: timestamp of last search
        # Personal object match smoothing/cooldown
        self.personal_match_min_consecutive = 2
        self.personal_match_cooldown_s = 5.0
        self.match_history = {}
        self._personal_match_last_candidate = None
        self._personal_match_last_spoken = {}
        self.last_personal_match_time = 0.0
        self._match_log_last_ts = 0.0

        # Scene caption cooldown
        self.last_scene_time = 0.0
        self.SCENE_COOLDOWN = 3.0
        self.last_caption = None
        self.last_spoken_time = 0.0

        # Initialize CNN feature extractor
        self._init_cnn_feature_extractor()
        
        # Initialize model
        self._setup_model()

    def _suppress_hf_logs(self):
        return

    def _setup_model(self):
        """Initialize YOLOv8 model (yolov8n for faster inference)."""
        try:
            self.model = YOLO('app/models/yolov8n.pt')
            # COCO class names
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

            # BLIP image captioning setup
            try:
                warnings.filterwarnings(
                    "ignore",
                    message=".*position_ids.*",
                    category=UserWarning,
                )
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.caption_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    use_fast=False,
                )
                self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    tie_word_embeddings=False,
                )
                print("BLIP image captioning model loaded successfully")
            except Exception as e:
                print(f"Error loading BLIP model: {e}")
            print("YOLOv8 model initialized successfully")
        except Exception as e:
            print(f"Error initializing YOLOv8 model: {e}")
            self.model = None

    def describe_scene(self):
        """Generate a scene description using BLIP image captioning from the current frame."""
        if self.voice_engine and getattr(self.voice_engine, "state", "idle") != "idle":
            return None
        if self.caption_model is None or self.caption_processor is None:
            return "Scene description model not available."
        with self.frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
        if frame is None:
            return "No frame available for description."
        try:
            from PIL import Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.caption_processor(images=image, return_tensors="pt")
            out = self.caption_model.generate(**inputs)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            caption = self.clean_caption(caption)
            if caption == self.last_caption and (time.time() - self.last_spoken_time) < 10:
                return ""
            self.last_caption = caption
            self.last_spoken_time = time.time()
            return caption
        except Exception as e:
            print(f"Error generating scene description: {e}")
            return "Could not generate scene description."

    def generate_blip_caption(self, frame=None):
        """Generate a BLIP caption for a provided frame (or current frame)."""
        if self.voice_engine and getattr(self.voice_engine, "state", "idle") != "idle":
            return None
        if self.caption_model is None or self.caption_processor is None:
            return None
        if frame is None:
            with self.frame_lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None
        if frame is None:
            return None
        try:
            from PIL import Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.caption_processor(images=image, return_tensors="pt")
            out = self.caption_model.generate(**inputs)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            caption = self.clean_caption(caption)
            if caption == self.last_caption and (time.time() - self.last_spoken_time) < 10:
                return None
            self.last_caption = caption
            self.last_spoken_time = time.time()
            return caption
        except Exception as e:
            print(f"Error generating BLIP caption: {e}")
            return None

    @staticmethod
    def clean_caption(text: str) -> str:
        words = (text or "").split()
        cleaned = []
        for w in words:
            if not cleaned or w != cleaned[-1]:
                cleaned.append(w)
        return " ".join(cleaned)

    def _normalize_embedding(self, embedding):
        """Normalize embedding to unit L2 norm."""
        try:
            vec = np.array(embedding, dtype=np.float32)
            if vec.shape[0] != 2048:
                raise ValueError(f"Embedding dimension mismatch: {vec.shape[0]} (expected 2048)")
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                raise ValueError("Embedding norm is zero")
            return (vec / norm).tolist()
        except Exception as e:
            raise ValueError(f"Embedding normalization failed: {e}") from e

    def _get_normalized_embedding(self, frame):
        """Extract and normalize CNN features from a frame."""
        features = self._extract_features(frame)
        if features is None:
            raise ValueError("Failed to extract embedding")
        embedding = features.squeeze().tolist()
        embedding = [float(x) for x in embedding]
        normalized = self._normalize_embedding(embedding)
        return normalized

    def _get_clip_normalized_embedding(self, frame):
        """Extract and normalize CLIP embedding from a frame."""
        embedding = self._extract_clip_embedding(frame)
        if embedding is None:
            raise ValueError("Failed to extract CLIP embedding")
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != 512:
            raise ValueError(f"CLIP embedding dimension mismatch: {vec.shape[0]} (expected 512)")
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ValueError("CLIP embedding norm is zero")
        return (vec / norm).astype(np.float32).tolist()

    def _get_search_embeddings(self, frame):
        """Build search embeddings in priority order: CLIP first, ResNet fallback."""
        embeddings = []

        if hasattr(self, "clip_matcher") and self.clip_matcher:
            try:
                clip_embedding = self._get_clip_normalized_embedding(frame)
                embeddings.append(("CLIP", clip_embedding))
            except Exception as exc:
                logger.warning("Failed to prepare CLIP search embedding: %s", exc)

        try:
            resnet_embedding = self._get_normalized_embedding(frame)
            embeddings.append(("ResNet50", resnet_embedding))
        except Exception as exc:
            logger.warning("Failed to prepare ResNet search embedding: %s", exc)

        if not embeddings:
            raise ValueError("Failed to extract any search embeddings")

        return embeddings

    def _should_speak_personal_match(self, object_name: str):
        """Apply smoothing and cooldown logic for personal object announcements."""
        now = time.time()
        name = object_name.lower()

        if self._personal_match_last_candidate == name:
            self.match_history[name] = self.match_history.get(name, 0) + 1
        else:
            if self._personal_match_last_candidate:
                self.match_history[self._personal_match_last_candidate] = 0
            self._personal_match_last_candidate = name
            self.match_history[name] = 1

        if self.match_history.get(name, 0) < self.personal_match_min_consecutive:
            return False, "smoothing"

        last_spoken = self._personal_match_last_spoken.get(name, 0.0)
        if now - last_spoken < self.personal_match_cooldown_s:
            return False, "cooldown"

        self._personal_match_last_spoken[name] = now
        self.last_personal_match_time = now
        return True, "ok"

    def reset_personal_match_state(self):
        """Reset personal match smoothing when a frame misses."""
        if self._personal_match_last_candidate:
            self.match_history[self._personal_match_last_candidate] = 0
        self._personal_match_last_candidate = None

    def should_generate_scene_caption(self, now=None) -> bool:
        now = now or time.time()
        if now - self.last_scene_time < self.SCENE_COOLDOWN:
            return False
        if now - self.last_personal_match_time < self.SCENE_COOLDOWN:
            return False
        return True

    def record_scene_caption(self, now=None):
        self.last_scene_time = now or time.time()

    @staticmethod
    def format_personal_match(name: str) -> str:
        name = (name or "").strip().lower()
        if name in ["me", "myself"]:
            return "I can see you"
        return f"I can see your {name}"

    # ---------------- Camera control ----------------
    def start_camera(self, camera_index=0):
        """Start camera capture."""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise Exception(f"Could not open camera {camera_index}")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.is_running = True
            print("Camera started successfully")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def stop_camera(self):
        """Stop camera capture."""
        self.is_running = False
        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None
        print("Camera stopped")

    def capture_frame(self):
        """Capture single frame."""
        if not self.camera or not self.camera.isOpened():
            return None
        ret, frame = self.camera.read()
        if ret:
            with self.frame_lock:
                self.current_frame = frame.copy()
            return frame
        return None

    # ---------------- Motion detection ----------------
    def detect_motion(self, frame):
        """Detect motion by frame differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.previous_frame_gray is None:
            self.previous_frame_gray = gray
            return False
        frame_delta = cv2.absdiff(self.previous_frame_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_area = cv2.countNonZero(thresh)
        self.previous_frame_gray = gray
        return motion_area > self.motion_threshold

    # ---------------- Object detection ----------------
    def detect_objects(self, frame):
        """Detect objects in frame using YOLOv8. Auto-announces objects unless muted by navigation mode."""
        if self.model is None:
            return []

        # If no motion, skip detection (optional optimization).
        # During navigation we must keep evaluating every frame for live obstacle alerts.
        try:
            if (not self.navigation_mode) and (not self.detect_motion(frame)):
                # still keep current_frame updated
                with self.frame_lock:
                    self.current_frame = frame.copy()
                return []
        except Exception:
            # If motion detection fails, continue with detection (safer)
            pass

        try:
            results = self.model(frame, verbose=False)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        if conf > 0.4:
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                            detections.append({
                                'bbox': box,
                                'score': conf,
                                'class_name': class_name,
                                'class_id': class_id
                            })
            with self.frame_lock:
                self.detections = detections

            # Auto-announce according to navigation mode
            self._auto_announce_objects(detections)
            return detections
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []

    def get_personal_objects_list(self, user_id):
        """Get a list of stored personal objects."""
        return self.personal_objects_mgr.get_all_objects(user_id)

    # ---------------- Announcements ----------------
    def _auto_announce_objects(self, detections):
        """Automatically announce detected objects:
           - If navigation_mode == False: announce only high-confidence objects with longer delay
           - If navigation_mode == True: only announce classes in emergency_classes every 5 seconds
           - Only announce if confidence >= 0.85 continuously for 5 seconds (non-navigation)
           - Only announce if confidence >= 0.75 in navigation mode (every 5 seconds)
        """
        # Allow emergency announcements during active navigation.
        # Outside navigation mode, keep the old suppression while voice is in an active dialog state.
        if (
            not self.navigation_mode
            and self.voice_engine
            and getattr(self.voice_engine, "state", "idle") != "idle"
        ):
            return
        if not detections:
            return
        if (not self.voice_engine) and (not self.announcement_callback):
            return

        current_time = time.time()
        
        # Different behavior based on navigation mode
        if self.navigation_mode:
            # In navigation mode: announce emergency/obstacle classes with tighter cadence.
            confidence_threshold = 0.65
            announcement_delay = 3.0
            
            # Only announce emergency objects during navigation
            emergency_detections = [det for det in detections if det['class_name'] in self.emergency_classes and det['score'] >= confidence_threshold]
            
            if emergency_detections and (current_time - self.last_announcement_time >= announcement_delay):
                # Group similar objects
                announced_objects = {}
                for det in emergency_detections:
                    class_name = det['class_name']
                    if class_name not in announced_objects:
                        announced_objects[class_name] = 1
                    else:
                        announced_objects[class_name] += 1
                
                # Build announcement text
                announcement_parts = []
                for obj_name, count in announced_objects.items():
                    if count == 1:
                        announcement_parts.append(f"a {obj_name}")
                    else:
                        announcement_parts.append(f"{count} {obj_name}s")

                if announcement_parts:
                    text = f"Caution: {', '.join(announcement_parts)} nearby"

                    # Check for duplicates
                    if self._is_duplicate_vision_announcement(text):
                        print(f"Duplicate navigation announcement blocked: {text}")
                        return

                    print(f"Navigation announcement: {text}")
                    try:
                        if self.announcement_callback:
                            self.announcement_callback(text)
                        else:
                            # Use emergency priority to interrupt speech immediately
                            try:
                                if self.voice_engine and hasattr(self.voice_engine, "emergency_override"):
                                    self.voice_engine.emergency_override(text)
                                elif self.voice_engine:
                                    self.voice_engine.speak(text, blocking=False, priority="EMERGENCY")
                                else:
                                    import requests
                                    requests.post(
                                        "http://localhost:5000/api/speak",
                                        json={"text": text, "priority": 100, "source": "vision_engine"},
                                        timeout=1,
                                    )
                                # Track the announcement
                                self._track_vision_announcement(text)
                            except Exception:
                                # Fallback to direct voice engine call
                                if self.voice_engine:
                                    self.voice_engine.speak(text, blocking=False, priority="EMERGENCY")
                    except Exception as e:
                        print(f"Error speaking announcement: {e}")

                    self.last_announcement_time = current_time
        else:
            # Outside navigation mode: announce high-confidence objects with longer delay
            confidence_threshold = 0.85  # Higher threshold to reduce chatter
            announcement_delay = 8.0  # Longer delay to reduce frequency

            # Update detection start times for objects above confidence threshold
            current_objects = {}
            for det in detections:
                class_name = det['class_name']
                confidence = det['score']

                if confidence >= confidence_threshold:
                    if class_name not in self.object_detection_start_times:
                        self.object_detection_start_times[class_name] = current_time
                    current_objects[class_name] = confidence
                else:
                    # Remove if confidence drops below threshold
                    if class_name in self.object_detection_start_times:
                        del self.object_detection_start_times[class_name]

            # Remove objects no longer detected
            for obj in list(self.object_detection_start_times.keys()):
                if obj not in current_objects:
                    del self.object_detection_start_times[obj]

            # Find objects detected continuously for announcement_delay seconds
            ready_to_announce = set()
            for obj, start_time in self.object_detection_start_times.items():
                if current_time - start_time >= announcement_delay:
                    ready_to_announce.add(obj)

            # Check cooldown
            if current_time - self.last_announcement_time < self.announcement_cooldown:
                return

            # Announce objects that have been detected continuously
            to_announce = list(ready_to_announce)

            # Remove those already announced recently
            new_ann = [o for o in to_announce if o not in self.last_announced_objects]
            if not new_ann:
                return

            # Build announcement text
            if len(new_ann) == 1:
                text = f"I see a {new_ann[0]}"
            else:
                text = "I see " + ", ".join(new_ann)
            
            # Check for duplicates
            if self._is_duplicate_vision_announcement(text):
                print(f"Duplicate vision announcement blocked: {text}")
                return
            
            print(f"Auto-announcing: {text}")
            try:
                if self.announcement_callback:
                    self.announcement_callback(text)
                else:
                    # Use centralized speech manager for proper queuing
                    try:
                        import requests
                        requests.post('http://localhost:5000/api/speak', 
                                    json={'text': text, 'priority': 60, 'source': 'vision_engine'},
                                    timeout=1)
                        # Track the announcement
                        self._track_vision_announcement(text)
                    except:
                        # Fallback to direct voice engine call
                        self.voice_engine.speak(text, blocking=False)
            except Exception as e:
                print(f"Error speaking announcement: {e}")

            self.last_announced_objects = set(to_announce)
            self.last_announcement_time = current_time

            # Clear tracking for announced objects
            for obj in new_ann:
                if obj in self.object_detection_start_times:
                    del self.object_detection_start_times[obj]

            if not current_objects:
                self.last_announced_objects.clear()
                self.object_detection_start_times.clear()

    def _is_duplicate_vision_announcement(self, text: str) -> bool:
        """Check if a vision announcement is a duplicate"""
        current_time = time.time()
        
        # Check exact duplicate
        if text in self.recent_vision_announcements:
            last_time = self.vision_announcement_timestamps.get(text, 0)
            if current_time - last_time < 3.0:  # 3 second cooldown
                return True
        
        return False
    
    def _track_vision_announcement(self, text: str):
        """Track a vision announcement to prevent duplicates"""
        current_time = time.time()
        self.recent_vision_announcements.add(text)
        self.vision_announcement_timestamps[text] = current_time
        
        # Cleanup old entries
        cleanup_threshold = 6.0
        old_announcements = [text for text, timestamp in self.vision_announcement_timestamps.items()
                           if current_time - timestamp > cleanup_threshold]
        for old_text in old_announcements:
            self.recent_vision_announcements.discard(old_text)
            if old_text in self.vision_announcement_timestamps:
                del self.vision_announcement_timestamps[old_text]
    
    # ---------------- Helper setters ----------------
    def set_voice_engine(self, voice_engine):
        """Set the voice engine for announcements."""
        self.voice_engine = voice_engine

    def set_announcement_callback(self, callback):
        """Set callback to run announcement on GUI/main thread."""
        self.announcement_callback = callback

    def set_navigation_mode(self, enabled: bool):
        """Enable or disable navigation mode. When enabled, most non-emergency announcements are suppressed."""
        self.navigation_mode = bool(enabled)
        if enabled:
            # Reset timer so emergency alerts can fire immediately after navigation starts.
            self.last_announcement_time = 0.0
            print("Navigation mode enabled: normal announcements suppressed except emergencies.")
        else:
            print("Navigation mode disabled: normal announcements resumed.")

    def set_emergency_classes(self, classes):
        """Set emergency classes (iterable of class names)."""
        self.emergency_classes = set(classes)

    # ---------------- Visualization ----------------
    def draw_detections(self, frame):
        """Draw bounding boxes and labels on the frame."""
        if not getattr(self, "detections", None):
            return frame
        for det in self.detections:
            bbox = det['bbox']
            cname = det['class_name']
            score = det['score']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cname}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame

    def get_current_frame_with_detections(self, ar_overlay=False, next_instruction=None):
        """Return the current frame with drawn detections and optional AR overlay."""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        frame = self.draw_detections(frame)
        if ar_overlay:
            frame = self.draw_ar_navigation_overlay(frame, next_instruction)
        return frame

    def draw_ar_navigation_overlay(self, frame, next_instruction=None):
        """Draw AR navigation overlays (arrows, instructions) on the frame."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        # Draw navigation arrow (centered)
        arrow_start = (w // 2, h - 60)
        arrow_end = (w // 2, h // 2)
        cv2.arrowedLine(overlay, arrow_start, arrow_end, (0, 0, 255), 8, tipLength=0.3)
        # Draw next instruction text
        if next_instruction:
            cv2.putText(overlay, next_instruction, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return overlay

    def _init_cnn_feature_extractor(self):
        """Initialize both CNN (ResNet50) and CLIP feature extractors for personal object recognition.
        CLIP is now the primary model; ResNet50 kept for backward compatibility."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            from PIL import Image
            import numpy as np
            
            # Initialize CLIP first (primary model)
            self._init_clip_model()
            
            # Keep ResNet50 for backward compatibility with old embeddings
            weights = models.ResNet50_Weights.DEFAULT
            self.feature_extractor = models.resnet50(weights=weights)
            # Remove the final classification layer (outputs 2048-d embedding)
            self.feature_extractor.fc = nn.Identity()
            self.feature_extractor.eval()
            
            # Image preprocessing transforms for ResNet
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Check if CUDA is available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.feature_extractor = self.feature_extractor.to(self.device)
            
            logger.info(f"CNN feature extractor (ResNet50) initialized on {self.device}")
            logger.info("CLIP ViT-B/32 is primary model for new objects")
            
        except Exception as e:
            logger.error(f"Error initializing feature extractors: {e}")
            self.feature_extractor = None
            self.transform = None
    
    def _init_clip_model(self):
        """Initialize CLIP model globally for object matching."""
        try:
            from app.vision.object_matcher import get_clip_matcher
            self.clip_matcher = get_clip_matcher()
            logger.info("CLIP object matcher initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP: {e}")
            self.clip_matcher = None
    
    def _init_obstacle_detector(self):
        """Initialize MiDaS depth estimation for obstacle detection during navigation."""
        try:
            # Lazy initialization - only load when needed for navigation
            self.obstacle_detector = None
            logger.info("Obstacle detector module ready (lazy load MiDaS)")
        except Exception as e:
            logger.error(f"Failed to initialize obstacle detector: {e}")
            self.obstacle_detector = None
    
    def _init_scene_caption_engine(self):
        """Initialize continuous scene captioning with 'human brain' mode."""
        try:
            from app.services.vision.scene_caption_engine import get_scene_caption_engine
            self.scene_caption_engine = get_scene_caption_engine()
            logger.info("Scene caption engine initialized (human brain mode)")
        except Exception as e:
            logger.error(f"Failed to initialize scene caption engine: {e}")
            self.scene_caption_engine = None

    def _extract_features(self, image):
        """Extract CNN (ResNet50) features from an image. Kept for backward compatibility."""
        if self.feature_extractor is None or self.transform is None:
            return None
            
        try:
            import torch
            from PIL import Image
            import numpy as np
            
            # Convert OpenCV image to PIL
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                # Flatten the features
                features = features.view(features.size(0), -1)
                # Convert to numpy
                features_np = features.cpu().numpy().flatten()
            
            return features_np
            
        except Exception as e:
            logger.error(f"Error extracting ResNet features: {e}")
            return None
    
    def _extract_clip_embedding(self, image):
        """Extract CLIP embedding from an image (primary method for new objects)."""
        if self.clip_matcher is None:
            logger.warning("CLIP matcher not initialized")
            return None
        
        try:
            embedding = self.clip_matcher.extract_embedding(image)
            return embedding
        except Exception as e:
            logger.error(f"Error extracting CLIP embedding: {e}")
            return None
    
    def enable_obstacle_detection(self):
        """Enable MiDaS-based obstacle detection for navigation mode."""
        if self.obstacle_detector is None:
            try:
                from app.services.navigation.obstacle_detector import get_obstacle_detector
                self.obstacle_detector = get_obstacle_detector()
                logger.info("Obstacle detector enabled with MiDaS depth estimation")
            except Exception as e:
                logger.error(f"Failed to enable obstacle detection: {e}")
                return False
        return True
    
    def disable_obstacle_detection(self):
        """Disable obstacle detection to conserve resources."""
        self.obstacle_detector = None
        logger.info("Obstacle detector disabled")
    
    def process_obstacles(self, yolo_detections: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Process obstacles using depth estimation.
        
        Args:
            yolo_detections: List of YOLO object detections
            
        Returns:
            Tuple of (obstacles_list, warning_message)
        """
        if self.obstacle_detector is None or self.current_frame is None:
            return [], ""
        
        try:
            # Estimate depth from current frame
            depth_map = self.obstacle_detector.estimate_depth(self.current_frame)
            
            if depth_map is None:
                return [], ""
            
            # Detect obstacles combining YOLO + depth
            obstacles, warning_message = self.obstacle_detector.detect_obstacles(
                depth_map, 
                yolo_detections
            )
            
            return obstacles, warning_message
            
        except Exception as e:
            logger.error(f"Obstacle processing error: {e}")
            return [], ""
    
    def generate_continuous_caption(self,
                                   yolo_detections: List[Dict],
                                   blip_caption: str,
                                   obstacles: List[Dict] = None,
                                   personal_objects: List[Dict] = None,
                                   navigation_mode: bool = False) -> Optional[str]:
        """
        Generate continuous scene caption with 'human brain' filtering.
        Only speaks IMPORTANT changes, not every detail.
        
        Args:
            yolo_detections: YOLO object detections
            blip_caption: Raw BLIP scene description
            obstacles: Detected obstacles (optional)
            personal_objects: Matched personal objects (optional)
            navigation_mode: True if user is navigating
            
        Returns:
            Important caption text or None if nothing important
        """
        if self.scene_caption_engine is None:
            return None
        
        try:
            caption = self.scene_caption_engine.generate_smart_caption(
                yolo_detections=yolo_detections,
                blip_caption=blip_caption,
                obstacles=obstacles or [],
                personal_objects=personal_objects or [],
                navigation_mode=navigation_mode
            )
            
            return caption
            
        except Exception as e:
            logger.error(f"Scene caption generation error: {e}")
            return None
    
    def reset_scene_captioning(self):
        """Reset scene captioning state (when context changes)."""
        if self.scene_caption_engine:
            self.scene_caption_engine.reset()
            logger.info("Scene captioning state reset")

    def _calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between two feature vectors."""
        if features1 is None or features2 is None:
            return 0.0
            
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([features1], [features2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def store_personal_object(self, supabase, user_id, object_name, image_url):
        """Store a personal object using CLIP embedding (primary) with ResNet fallback."""
        logger = logging.getLogger("smartvision.vision")

        if self.current_frame is None:
            raise ValueError("No current frame available for embedding")

        # Try CLIP first (primary method)
        embedding = None
        model_used = "CLIP"
        
        if hasattr(self, 'clip_matcher') and self.clip_matcher:
            logger.info("Extracting CLIP embedding for object '%s'", object_name)
            embedding = self._extract_clip_embedding(self.current_frame)
        
        # Fallback to ResNet if CLIP fails
        if embedding is None and self.feature_extractor is not None:
            logger.warning("CLIP failed, falling back to ResNet50")
            embedding = self._extract_features(self.current_frame)
            model_used = "ResNet50"
        
        if embedding is None:
            raise ValueError("Failed to extract features with both CLIP and ResNet")

        embedding = np.array(embedding).flatten()
        expected_dim = 512 if model_used == "CLIP" else 2048
        
        if len(embedding) != expected_dim:
            raise ValueError(f"Embedding length mismatch: {len(embedding)} != {expected_dim}")

        norm = float(np.linalg.norm(embedding))
        if norm == 0.0:
            raise ValueError("Embedding norm is zero")

        embedding = embedding / norm
        embedding_list = embedding.astype(float).tolist()

        blip_caption = self.generate_blip_caption(self.current_frame)

        logger.info("Storing personal object '%s' using %s (%d-dim embedding)", 
                   object_name, model_used, len(embedding_list))

        try:
            result = self.personal_objects_mgr.store_object(
                user_id=user_id,
                object_name=object_name,
                image_url=image_url,
                embedding=embedding_list,
                blip_caption=blip_caption
            )
            return result
        except Exception as e:
            logger.error(f"Failed to store personal object: {e}")
            raise ValueError(f"Storage failed: {e}")
    
    def _store_personal_object_local(self, object_name, frame):
        """Fallback method to store personal object locally."""
        import os
        import time
        
        # Create objects directory if it doesn't exist
        os.makedirs('personal_objects', exist_ok=True)
        
        # Save the frame as an image
        image_path = f'personal_objects/{object_name.lower().replace(" ", "_")}_{int(time.time())}.jpg'
        success = cv2.imwrite(image_path, frame)
        
        if success:
            # Extract CNN features for the stored object
            features = self._extract_features(frame)
            if features is not None:
                self.personal_object_features[object_name.lower()] = features
                print(f"Stored personal object '{object_name}' with CNN features at {image_path}")
            else:
                print(f"Stored personal object '{object_name}' at {image_path} (no features extracted)")
            
            self.personal_objects[object_name.lower()] = image_path
            return True
        else:
            print(f"Failed to store personal object '{object_name}' locally")
            return False

    def search_for_personal_object(self, object_name, user_id=None):
        """Search for a named personal object in the current frame."""
        import re
        
        object_name_lower = object_name.lower()

        with self.frame_lock:
            current_frame = self.current_frame.copy() if self.current_frame is not None else None
        
        if current_frame is None:
            return f"Camera not available to search for your {object_name}."

        if user_id:
            try:
                results = self.search_similar_objects(
                    frame=current_frame,
                    match_count=10,
                    match_threshold=None,
                    user_id=user_id,
                )
                named_matches = [
                    match for match in results
                    if (match.get("object_name") or "").lower() == object_name_lower
                ]
                if named_matches:
                    best_match = max(named_matches, key=lambda item: float(item.get("similarity", 0.0)))
                    similarity = float(best_match.get("similarity", 0.0))
                    confidence_text = "high" if similarity >= 0.92 else "medium" if similarity >= 0.80 else "low"
                    return f"Yes, I see your {object_name} with {confidence_text} confidence. Similarity: {similarity:.2f}"
            except Exception as exc:
                logger.warning("Named personal object search failed, falling back to local memory: %s", exc)

        # Backward-compatible local-memory fallback for older flows in the same process.
        if object_name_lower in self.personal_object_features:
            current_embedding = None
            model_used = "CLIP"

            if hasattr(self, 'clip_matcher') and self.clip_matcher:
                current_embedding = self._extract_clip_embedding(current_frame)

            if current_embedding is None and self.feature_extractor is not None:
                logger.warning("CLIP not available, using ResNet for local personal object search")
                current_embedding = self._extract_features(current_frame)
                model_used = "ResNet"

            if current_embedding is not None:
                stored_features = self.personal_object_features[object_name_lower]
                if model_used == "CLIP" and hasattr(self, 'clip_matcher'):
                    similarity = self.clip_matcher.cosine_similarity(
                        np.array(current_embedding),
                        np.array(stored_features)
                    )
                    threshold = 0.85
                else:
                    similarity = self._calculate_similarity(np.array(stored_features), np.array(current_embedding))
                    threshold = 0.70

                if similarity >= threshold:
                    confidence_text = "high" if similarity >= 0.92 else "medium" if similarity >= 0.85 else "low"
                    return f"Yes, I see your {object_name} with {confidence_text} confidence. Similarity: {similarity:.2f}"

        yolo_result = self._search_by_yolo_detection(object_name)
        if "Yes" in yolo_result:
            return yolo_result
        return f"I don't see your {object_name} right now."
    
    def _search_by_yolo_detection(self, object_name):
        """Fallback method using YOLO detection for object search."""
        object_name_lower = object_name.lower()
        
        # Look for the object in current detections
        with self.frame_lock:
            current_detections = self.detections.copy() if hasattr(self, 'detections') else []
        
        # Check if the requested object is in current detections
        found_objects = []
        for det in current_detections:
            class_name = det['class_name'].lower()
            # Check for exact match or partial match in class names
            if object_name_lower in class_name or class_name in object_name_lower:
                found_objects.append(det)
        
        if found_objects:
            count = len(found_objects)
            if count == 1:
                return f"Yes, I see your {object_name} in the frame."
            else:
                return f"Yes, I see {count} {object_name}s in the frame."
        else:
            return f"I don't see your {object_name} right now."

    def search_similar_objects(self, frame=None, access_token=None, match_count=5, match_threshold=None, user_id=None):
        """Search stored personal objects using CLIP first, ResNet fallback for legacy data."""
        if not user_id:
            raise ValueError("User ID is required for similarity search")

        if frame is None:
            with self.frame_lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None

        if frame is None:
            raise ValueError("No frame available for similarity search")

        logger = logging.getLogger("smartvision.vision")
        now = time.time()
        threshold_label = "auto" if match_threshold is None else f"{float(match_threshold):.3f}"

        try:
            candidate_embeddings = self._get_search_embeddings(frame)
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

        combined_results = []
        seen_doc_ids = set()

        for model_name, embedding_list in candidate_embeddings:
            if (now - self._match_log_last_ts) >= 1.0:
                logger.info(
                    "Calling Firestore similarity search | model=%s | len=%d | count=%d | threshold=%s | user=%s",
                    model_name,
                    len(embedding_list),
                    match_count,
                    threshold_label,
                    user_id,
                )
                self._match_log_last_ts = now

            try:
                results = self.personal_objects_mgr.search_similar(
                    user_id=user_id,
                    query_embedding=embedding_list,
                    match_count=match_count,
                    match_threshold=match_threshold
                )
            except Exception as e:
                logger.error("Similarity search failed for %s: %s", model_name, e)
                continue

            for result in results:
                doc_id = result.get("doc_id")
                if doc_id and doc_id in seen_doc_ids:
                    continue
                enriched = dict(result)
                enriched["query_model"] = model_name
                combined_results.append(enriched)
                if doc_id:
                    seen_doc_ids.add(doc_id)

        combined_results.sort(key=lambda item: float(item.get("similarity", 0.0)), reverse=True)
        return combined_results[:match_count]

    def cleanup(self):
        """Cleanup resources."""
        self.stop_camera()
        if hasattr(self, 'model') and self.model:
            try:
                del self.model
            except Exception:
                pass
        print("Vision engine cleaned up")

