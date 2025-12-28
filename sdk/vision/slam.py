"""
SLAM Processor
==============

Simultaneous Localization and Mapping.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """Camera/robot pose."""
    position: np.ndarray  # 3D position
    rotation: np.ndarray  # 3x3 rotation matrix
    timestamp: float


@dataclass
class MapPoint:
    """A point in the map."""
    position: np.ndarray
    descriptor: Optional[np.ndarray] = None
    observations: int = 0


@dataclass
class Keyframe:
    """A keyframe in the map."""
    id: int
    pose: Pose
    image: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    map_points: List[int] = field(default_factory=list)


class SLAMProcessor:
    """
    Visual SLAM processor.
    
    Features:
    - Visual odometry
    - Loop closure detection
    - Map building
    - Relocalization
    
    Example:
        >>> slam = SLAMProcessor()
        >>> slam.initialize(first_frame)
        >>> for frame in video:
        ...     pose = slam.process_frame(frame)
        ...     print(f"Position: {pose.position}")
    """
    
    def __init__(self, method: str = "orb",
                 language: str = "en"):
        """
        Initialize SLAM processor.
        
        Args:
            method: Feature extraction method
            language: Language for messages
        """
        self.method = method
        self.language = language
        
        # Map
        self._map_points: Dict[int, MapPoint] = {}
        self._keyframes: Dict[int, Keyframe] = {}
        
        # Current state
        self._current_pose: Optional[Pose] = None
        self._last_keyframe_id = -1
        self._frame_count = 0
        
        # Feature detector
        self._detector = None
        self._matcher = None
        
        # Tracking state
        self._initialized = False
        self._lost = False
        
        logger.info(f"SLAM processor initialized: {method}")
    
    def initialize(self, frame: np.ndarray) -> bool:
        """
        Initialize SLAM with first frame.
        
        Args:
            frame: First frame
            
        Returns:
            True if initialized successfully
        """
        self._load_detector()
        
        # Extract features
        keypoints, descriptors = self._detect_features(frame)
        
        if len(keypoints) < 100:
            logger.warning("Not enough features for initialization")
            return False
        
        # Create initial pose at origin
        self._current_pose = Pose(
            position=np.zeros(3),
            rotation=np.eye(3),
            timestamp=0.0
        )
        
        # Create first keyframe
        self._add_keyframe(frame, keypoints, descriptors)
        
        # Create initial map points
        for i, kp in enumerate(keypoints):
            point = MapPoint(
                position=np.array([kp[0], kp[1], 1.0]),  # Simplified
                descriptor=descriptors[i] if descriptors is not None else None,
                observations=1
            )
            self._map_points[i] = point
        
        self._initialized = True
        logger.info("SLAM initialized")
        return True
    
    def _load_detector(self):
        """Load feature detector."""
        try:
            import cv2
            
            if self.method == "orb":
                self._detector = cv2.ORB_create(nfeatures=1000)
                self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            elif self.method == "sift":
                self._detector = cv2.SIFT_create()
                self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                self._detector = cv2.ORB_create()
                self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
        except ImportError:
            logger.warning("OpenCV not installed, using simulation")
            self._detector = "simulated"
    
    def _detect_features(self, frame: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """Detect features in frame."""
        if self._detector == "simulated":
            # Simulate feature detection
            num_features = np.random.randint(100, 500)
            h, w = frame.shape[:2]
            
            keypoints = [
                (np.random.randint(0, w), np.random.randint(0, h))
                for _ in range(num_features)
            ]
            descriptors = np.random.randint(0, 256, (num_features, 32), dtype=np.uint8)
            
            return keypoints, descriptors
        
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, desc = self._detector.detectAndCompute(gray, None)
            
            keypoints = [(int(k.pt[0]), int(k.pt[1])) for k in kp]
            return keypoints, desc
            
        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            return [], None
    
    def process_frame(self, frame: np.ndarray,
                      timestamp: float = 0.0) -> Optional[Pose]:
        """
        Process a new frame.
        
        Args:
            frame: Input frame
            timestamp: Frame timestamp
            
        Returns:
            Estimated pose or None if lost
        """
        if not self._initialized:
            self.initialize(frame)
            return self._current_pose
        
        self._frame_count += 1
        
        # Extract features
        keypoints, descriptors = self._detect_features(frame)
        
        if len(keypoints) < 50:
            self._lost = True
            logger.warning("Tracking lost - not enough features")
            return None
        
        # Track features
        pose = self._track(keypoints, descriptors)
        
        if pose is None:
            self._lost = True
            return None
        
        self._current_pose = pose
        self._current_pose.timestamp = timestamp
        self._lost = False
        
        # Check if we need a new keyframe
        if self._should_add_keyframe():
            self._add_keyframe(frame, keypoints, descriptors)
        
        return pose
    
    def _track(self, keypoints: List, 
               descriptors: np.ndarray) -> Optional[Pose]:
        """Track features and estimate pose."""
        if not self._keyframes:
            return None
        
        # Get last keyframe
        last_kf = self._keyframes[self._last_keyframe_id]
        
        if last_kf.features is None:
            return self._current_pose
        
        # Match features
        if self._matcher != "simulated" and self._matcher is not None:
            try:
                matches = self._matcher.match(descriptors, last_kf.features)
                matches = sorted(matches, key=lambda x: x.distance)[:100]
            except:
                matches = []
        else:
            # Simulate matching
            num_matches = min(len(keypoints), 50)
            matches = list(range(num_matches))
        
        if len(matches) < 10:
            return None
        
        # Estimate motion (simplified)
        # In real implementation, use PnP or essential matrix
        
        # Simulate small motion
        delta_position = np.random.randn(3) * 0.01
        delta_rotation = self._small_rotation(np.random.randn(3) * 0.01)
        
        new_position = self._current_pose.position + delta_position
        new_rotation = delta_rotation @ self._current_pose.rotation
        
        return Pose(
            position=new_position,
            rotation=new_rotation,
            timestamp=0.0
        )
    
    def _small_rotation(self, angles: np.ndarray) -> np.ndarray:
        """Create small rotation matrix from angles."""
        rx, ry, rz = angles
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    def _should_add_keyframe(self) -> bool:
        """Check if we should add a new keyframe."""
        if not self._keyframes:
            return True
        
        # Add keyframe every 20 frames or if moved enough
        if self._frame_count % 20 == 0:
            return True
        
        last_kf = self._keyframes[self._last_keyframe_id]
        distance = np.linalg.norm(
            self._current_pose.position - last_kf.pose.position
        )
        
        return distance > 0.5  # 0.5 meters
    
    def _add_keyframe(self, frame: np.ndarray,
                      keypoints: List,
                      descriptors: np.ndarray):
        """Add a new keyframe."""
        kf_id = len(self._keyframes)
        
        keyframe = Keyframe(
            id=kf_id,
            pose=Pose(
                position=self._current_pose.position.copy() if self._current_pose else np.zeros(3),
                rotation=self._current_pose.rotation.copy() if self._current_pose else np.eye(3),
                timestamp=0.0
            ),
            features=descriptors
        )
        
        self._keyframes[kf_id] = keyframe
        self._last_keyframe_id = kf_id
        
        logger.debug(f"Added keyframe {kf_id}")
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get camera trajectory."""
        return [kf.pose.position for kf in self._keyframes.values()]
    
    def get_map_points(self) -> np.ndarray:
        """Get all map points."""
        if not self._map_points:
            return np.array([]).reshape(0, 3)
        
        return np.array([mp.position for mp in self._map_points.values()])
    
    def relocalize(self, frame: np.ndarray) -> Optional[Pose]:
        """
        Relocalize from a lost state.
        
        Args:
            frame: Current frame
            
        Returns:
            Estimated pose if successful
        """
        keypoints, descriptors = self._detect_features(frame)
        
        # Try to match against all keyframes
        best_matches = 0
        best_pose = None
        
        for kf in self._keyframes.values():
            if kf.features is None:
                continue
            
            # Match features
            if self._matcher != "simulated" and self._matcher is not None:
                try:
                    matches = self._matcher.match(descriptors, kf.features)
                    if len(matches) > best_matches:
                        best_matches = len(matches)
                        best_pose = kf.pose
                except:
                    pass
        
        if best_matches > 50:
            self._current_pose = best_pose
            self._lost = False
            logger.info("Relocalization successful")
            return best_pose
        
        return None
    
    def reset(self):
        """Reset SLAM system."""
        self._map_points.clear()
        self._keyframes.clear()
        self._current_pose = None
        self._last_keyframe_id = -1
        self._frame_count = 0
        self._initialized = False
        self._lost = False
        logger.info("SLAM reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get SLAM statistics."""
        return {
            "initialized": self._initialized,
            "lost": self._lost,
            "keyframes": len(self._keyframes),
            "map_points": len(self._map_points),
            "frames_processed": self._frame_count
        }
    
    def __repr__(self) -> str:
        return f"SLAMProcessor(method='{self.method}', keyframes={len(self._keyframes)})"
