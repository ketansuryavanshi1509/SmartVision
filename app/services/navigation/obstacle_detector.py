"""
Obstacle Detection Module for SmartVision
Uses MiDaS for monocular depth estimation and obstacle warning system
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from app.core.logger import logger


class ObstacleDetector:
    """Detect obstacles using depth estimation from MiDaS model."""
    
    def __init__(self):
        self.midas_model = None
        self.midas_transform = None
        self.device = None
        self.depth_threshold = 0.7  # Normalized depth threshold (0-1)
        self.obstacle_warning_distance = 1.5  # meters
        self._initialize_midas()
    
    def _initialize_midas(self):
        """Load MiDaS small model for mobile-friendly depth estimation."""
        try:
            import torch
            
            # Check if CUDA is available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"MiDaS will run on {self.device}")
            
            # Load MiDaS small model (faster for mobile)
            logger.info("Loading MiDaS small depth estimation model...")
            
            # Import timm for MiDaS
            import timm
            
            # Load MiDaS pretrained model
            self.midas_model = torch.hub.load(
                'intel-isl/MiDaS',
                'MiDaS_small',  # Small variant for speed
                pretrained=True
            )
            self.midas_model.to(self.device)
            self.midas_model.eval()
            
            # Load MiDaS transforms
            midas_transforms = torch.hub.load(
                'intel-isl/MiDaS',
                'transforms'
            )
            
            # Use small transform for MiDaS small model
            self.midas_transform = midas_transforms.small_transform
            
            logger.info("MiDaS depth estimation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise
    
    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map from single RGB image.
        
        Args:
            image: RGB image array (BGR from OpenCV converted to RGB)
            
        Returns:
            Depth map (same resolution as input, values 0-1 where higher = closer)
        """
        if self.midas_model is None or self.midas_transform is None:
            logger.error("MiDaS model not initialized")
            return None
        
        try:
            import torch
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Apply transforms
            input_batch = self.midas_transform(image_rgb).to(self.device)
            
            # Predict depth
            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                
                # Interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
            
            # Normalize to 0-1 range
            depth_map = prediction.squeeze().cpu().numpy()
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            
            if depth_max - depth_min > 0:
                depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth_map)
            
            return depth_normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    def detect_obstacles(self, depth_map: np.ndarray, 
                        yolo_detections: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Detect obstacles based on depth map and YOLO detections.
        
        Args:
            depth_map: Normalized depth map from MiDaS
            yolo_detections: List of YOLO object detections
            
        Returns:
            Tuple of (obstacles_list, warning_message)
        """
        if depth_map is None or len(depth_map.shape) != 2:
            return [], ""
        
        obstacles = []
        warnings = []
        
        # Analyze each YOLO detection for distance
        for det in yolo_detections:
            bbox = det.get('bbox')  # [x1, y1, x2, y2]
            class_name = det.get('class_name')
            confidence = det.get('confidence', 0.0)
            
            if bbox is None:
                continue
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get average depth in the object's region
            roi_depth = depth_map[y1:y2, x1:x2]
            avg_depth = float(np.mean(roi_depth))
            
            # Higher depth value = closer object
            # Invert so lower value = closer (more intuitive)
            inverted_depth = 1.0 - avg_depth
            
            # Estimate distance (rough approximation)
            # This is calibrated for typical camera FOV
            estimated_distance_meters = self._depth_to_distance(inverted_depth)
            
            # Determine if this is an obstacle
            is_obstacle = self._is_obstacle(class_name, estimated_distance_meters)
            
            if is_obstacle:
                threat_level = self._assess_threat_level(estimated_distance_meters)
                
                obstacle_info = {
                    'class_name': class_name,
                    'confidence': confidence,
                    'distance_meters': estimated_distance_meters,
                    'threat_level': threat_level,  # 'high', 'medium', 'low'
                    'bbox': bbox,
                    'depth_value': avg_depth
                }
                
                obstacles.append(obstacle_info)
                
                # Generate warning if close
                if estimated_distance_meters < self.obstacle_warning_distance:
                    warning = self._generate_warning(obstacle_info)
                    warnings.append(warning)
        
        # Sort by threat level (high first)
        threat_order = {'high': 0, 'medium': 1, 'low': 2}
        obstacles.sort(key=lambda x: threat_order.get(x['threat_level'], 3))
        
        # Combine warnings
        warning_message = "; ".join(warnings) if warnings else ""
        
        return obstacles, warning_message
    
    def _depth_to_distance(self, inverted_depth: float) -> float:
        """
        Convert normalized depth value to approximate distance in meters.
        
        This is a rough calibration based on typical camera parameters.
        For more accurate measurements, per-camera calibration would be needed.
        
        Args:
            inverted_depth: Normalized depth (0=far, 1=close)
            
        Returns:
            Estimated distance in meters
        """
        # Simple linear mapping (can be improved with calibration)
        # Assuming typical smartphone camera FOV (~60 degrees)
        # Range: 0.5m (very close) to 10m (far)
        
        min_distance = 0.5  # closest detectable
        max_distance = 10.0  # farthest detectable
        
        distance = min_distance + (max_distance - min_distance) * (1.0 - inverted_depth)
        
        # Clamp to valid range
        distance = max(min_distance, min(max_distance, distance))
        
        return round(distance, 2)
    
    def _is_obstacle(self, class_name: str, distance: float) -> bool:
        """
        Determine if detected object should be considered an obstacle.
        
        Args:
            class_name: YOLO class name
            distance: Estimated distance in meters
            
        Returns:
            True if object is an obstacle at this distance
        """
        # Objects that are always obstacles (regardless of distance)
        always_obstacles = {
            'person', 'bicycle', 'motorcycle', 'bus', 'truck',
            'traffic light', 'fire hydrant', 'stop sign', 'car', 'van'
        }
        
        # Objects that are obstacles only when close
        conditional_obstacles = {
            'chair', 'couch', 'bed', 'dining table', 'bench',
            'backpack', 'umbrella', 'suitcase', 'handbag'
        }
        
        class_lower = class_name.lower()
        
        # Check if always an obstacle
        if any(obs in class_lower for obs in always_obstacles):
            return True
        
        # Check if conditional obstacle and within range
        if any(obs in class_lower for obs in conditional_obstacles):
            return distance < 3.0  # Only warn if within 3 meters
        
        # Default: consider as obstacle if very close
        return distance < 2.0
    
    def _assess_threat_level(self, distance: float) -> str:
        """
        Assess threat level based on distance.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Threat level: 'high', 'medium', or 'low'
        """
        if distance < 1.0:
            return 'high'  # Immediate danger
        elif distance < 2.0:
            return 'medium'  # Caution needed
        else:
            return 'low'  # Awareness only
    
    def _generate_warning(self, obstacle: Dict) -> str:
        """
        Generate natural language warning for an obstacle.
        
        Args:
            obstacle: Obstacle information dict
            
        Returns:
            Warning message string
        """
        class_name = obstacle.get('class_name', 'object')
        distance = obstacle.get('distance_meters', 0.0)
        threat_level = obstacle.get('threat_level', 'low')
        
        # Direction (simplified - assumes center of frame)
        direction = "ahead"
        
        if threat_level == 'high':
            if distance < 0.5:
                return f"STOP! {class_name.capitalize()} right in front of you!"
            else:
                return f"Warning! {class_name.capitalize()} very close, {distance:.1f} meters {direction}"
        elif threat_level == 'medium':
            return f"Caution: {class_name.capitalize()} {distance:.1f} meters {direction}"
        else:
            return f"{class_name.capitalize()} detected {distance:.1f} meters away"
    
    def create_depth_visualization(self, depth_map: np.ndarray, 
                                  original_image: np.ndarray,
                                  alpha: float = 0.5) -> np.ndarray:
        """
        Create colorized depth visualization overlay.
        
        Args:
            depth_map: Normalized depth map
            original_image: Original BGR image
            alpha: Transparency factor (0.0-1.0)
            
        Returns:
            Overlay image showing depth (for debugging/visualization)
        """
        if depth_map is None or original_image is None:
            return original_image
        
        # Convert depth to colormap (blue=far, red=close)
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO
        )
        
        # Blend with original image
        overlay = cv2.addWeighted(
            original_image,
            1.0 - alpha,
            depth_colored,
            alpha,
            0
        )
        
        return overlay
    
    def get_navigation_advice(self, obstacles: List[Dict], 
                             free_space_percentage: float) -> str:
        """
        Generate navigation advice based on obstacle positions.
        
        Args:
            obstacles: List of detected obstacles
            free_space_percentage: Percentage of clear path ahead
            
        Returns:
            Natural language navigation advice
        """
        if not obstacles:
            if free_space_percentage > 80:
                return "Path is clear, you can proceed safely"
            else:
                return "Path ahead has some clutter, proceed with caution"
        
        # Find most critical obstacle
        primary_obstacle = obstacles[0]  # Already sorted by threat
        class_name = primary_obstacle.get('class_name', 'object')
        distance = primary_obstacle.get('distance_meters', 0.0)
        threat = primary_obstacle.get('threat_level', 'low')
        
        if threat == 'high':
            if distance < 0.5:
                return f"EMERGENCY STOP! {class_name.capitalize()} directly in your path!"
            else:
                return f"Stop! There's a {class_name} {distance:.1f} meters ahead"
        elif threat == 'medium':
            return f"Slow down. {class_name.capitalize()} {distance:.1f} meters ahead. Consider going around it"
        else:
            return f"Awareness: {class_name.capitalize()} {distance:.1f} meters in your path"
    
    def calculate_free_space(self, depth_map: np.ndarray, 
                            image_width: int) -> Tuple[float, str]:
        """
        Calculate percentage of free space in different regions.
        
        Args:
            depth_map: Normalized depth map
            image_width: Width of the image
            
        Returns:
            Tuple of (free_space_percentage, directional_hint)
        """
        if depth_map is None:
            return 0.0, ""
        
        height, width = depth_map.shape
        
        # Divide into left, center, right sections
        third_width = width // 3
        left_region = depth_map[:, :third_width]
        center_region = depth_map[:, third_width:2*third_width]
        right_region = depth_map[:, 2*third_width:]
        
        # Calculate average depth in each region (inverted: higher = farther)
        left_depth = float(np.mean(left_region))
        center_depth = float(np.mean(center_region))
        right_depth = float(np.mean(right_region))
        
        # Determine which direction has most space
        depths = {'left': left_depth, 'center': center_depth, 'right': right_depth}
        best_direction = max(depths, key=depths.get)
        
        # Calculate overall free space
        avg_depth = float(np.mean(depth_map))
        free_space_percentage = avg_depth * 100
        
        # Generate directional hint
        if best_direction == 'left':
            hint = "More space to your left"
        elif best_direction == 'right':
            hint = "More space to your right"
        else:
            hint = "Path ahead is clear"
        
        return free_space_percentage, hint


# Singleton instance
_obstacle_detector = None


def get_obstacle_detector() -> ObstacleDetector:
    """Get or create singleton ObstacleDetector instance."""
    global _obstacle_detector
    if _obstacle_detector is None:
        _obstacle_detector = ObstacleDetector()
    return _obstacle_detector
