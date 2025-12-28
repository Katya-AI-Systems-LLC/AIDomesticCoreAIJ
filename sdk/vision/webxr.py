"""
WebXR Integration
=================

Integration with WebXR for AR/VR applications.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class XRSessionMode(Enum):
    """XR session modes."""
    INLINE = "inline"
    IMMERSIVE_VR = "immersive-vr"
    IMMERSIVE_AR = "immersive-ar"


class XRReferenceSpace(Enum):
    """XR reference space types."""
    VIEWER = "viewer"
    LOCAL = "local"
    LOCAL_FLOOR = "local-floor"
    BOUNDED_FLOOR = "bounded-floor"
    UNBOUNDED = "unbounded"


@dataclass
class XRPose:
    """XR device pose."""
    position: np.ndarray
    orientation: np.ndarray  # Quaternion
    timestamp: float


@dataclass
class XRHitTestResult:
    """Result from XR hit test."""
    position: np.ndarray
    normal: np.ndarray
    distance: float


@dataclass
class XRAnchor:
    """An XR anchor."""
    anchor_id: str
    pose: XRPose
    metadata: Dict[str, Any]


class WebXRIntegration:
    """
    WebXR integration for AR/VR applications.
    
    Features:
    - Session management
    - Pose tracking
    - Hit testing
    - Anchor management
    - Hand tracking
    
    Example:
        >>> xr = WebXRIntegration()
        >>> session = await xr.request_session(XRSessionMode.IMMERSIVE_AR)
        >>> pose = xr.get_viewer_pose()
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize WebXR integration.
        
        Args:
            language: Language for messages
        """
        self.language = language
        
        # Session state
        self._session_active = False
        self._session_mode: Optional[XRSessionMode] = None
        self._reference_space: Optional[XRReferenceSpace] = None
        
        # Tracking state
        self._viewer_pose: Optional[XRPose] = None
        self._input_sources: List[Dict] = []
        
        # Anchors
        self._anchors: Dict[str, XRAnchor] = {}
        
        # Hit test sources
        self._hit_test_sources: List[str] = []
        
        logger.info("WebXR integration initialized")
    
    async def is_supported(self, mode: XRSessionMode) -> bool:
        """
        Check if XR mode is supported.
        
        Args:
            mode: XR session mode
            
        Returns:
            True if supported
        """
        # In browser, this would check navigator.xr.isSessionSupported
        # Here we simulate support
        return mode in [XRSessionMode.INLINE, XRSessionMode.IMMERSIVE_VR, XRSessionMode.IMMERSIVE_AR]
    
    async def request_session(self, mode: XRSessionMode,
                               features: Optional[List[str]] = None) -> bool:
        """
        Request an XR session.
        
        Args:
            mode: Session mode
            features: Required features
            
        Returns:
            True if session started
        """
        if self._session_active:
            logger.warning("Session already active")
            return False
        
        supported = await self.is_supported(mode)
        if not supported:
            logger.error(f"XR mode not supported: {mode}")
            return False
        
        self._session_mode = mode
        self._session_active = True
        
        # Set default reference space
        if mode == XRSessionMode.IMMERSIVE_AR:
            self._reference_space = XRReferenceSpace.LOCAL_FLOOR
        elif mode == XRSessionMode.IMMERSIVE_VR:
            self._reference_space = XRReferenceSpace.LOCAL
        else:
            self._reference_space = XRReferenceSpace.VIEWER
        
        # Initialize viewer pose
        self._viewer_pose = XRPose(
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),  # Identity quaternion
            timestamp=0.0
        )
        
        logger.info(f"XR session started: {mode.value}")
        return True
    
    async def end_session(self):
        """End the current XR session."""
        if not self._session_active:
            return
        
        self._session_active = False
        self._session_mode = None
        self._viewer_pose = None
        self._input_sources.clear()
        self._anchors.clear()
        
        logger.info("XR session ended")
    
    def get_viewer_pose(self) -> Optional[XRPose]:
        """Get current viewer pose."""
        if not self._session_active:
            return None
        
        # Simulate pose update
        if self._viewer_pose:
            # Add small random motion
            self._viewer_pose.position += np.random.randn(3) * 0.001
            self._viewer_pose.timestamp += 0.016  # ~60 FPS
        
        return self._viewer_pose
    
    def set_reference_space(self, space: XRReferenceSpace) -> bool:
        """
        Set the reference space.
        
        Args:
            space: Reference space type
            
        Returns:
            True if set successfully
        """
        if not self._session_active:
            return False
        
        self._reference_space = space
        logger.info(f"Reference space set: {space.value}")
        return True
    
    async def request_hit_test_source(self, 
                                       origin: np.ndarray,
                                       direction: np.ndarray) -> str:
        """
        Request a hit test source.
        
        Args:
            origin: Ray origin
            direction: Ray direction
            
        Returns:
            Hit test source ID
        """
        source_id = f"hit_test_{len(self._hit_test_sources)}"
        self._hit_test_sources.append(source_id)
        return source_id
    
    def get_hit_test_results(self, source_id: str) -> List[XRHitTestResult]:
        """
        Get hit test results.
        
        Args:
            source_id: Hit test source ID
            
        Returns:
            List of hit test results
        """
        if source_id not in self._hit_test_sources:
            return []
        
        # Simulate hit test results
        num_results = np.random.randint(0, 3)
        results = []
        
        for _ in range(num_results):
            results.append(XRHitTestResult(
                position=np.random.randn(3),
                normal=np.array([0, 1, 0]),
                distance=np.random.uniform(0.5, 5.0)
            ))
        
        return results
    
    async def create_anchor(self, pose: XRPose,
                             metadata: Optional[Dict] = None) -> XRAnchor:
        """
        Create an XR anchor.
        
        Args:
            pose: Anchor pose
            metadata: Additional metadata
            
        Returns:
            Created anchor
        """
        anchor_id = f"anchor_{len(self._anchors)}"
        
        anchor = XRAnchor(
            anchor_id=anchor_id,
            pose=pose,
            metadata=metadata or {}
        )
        
        self._anchors[anchor_id] = anchor
        logger.info(f"Created anchor: {anchor_id}")
        
        return anchor
    
    def get_anchor(self, anchor_id: str) -> Optional[XRAnchor]:
        """Get an anchor by ID."""
        return self._anchors.get(anchor_id)
    
    def get_all_anchors(self) -> List[XRAnchor]:
        """Get all anchors."""
        return list(self._anchors.values())
    
    async def delete_anchor(self, anchor_id: str) -> bool:
        """Delete an anchor."""
        if anchor_id in self._anchors:
            del self._anchors[anchor_id]
            return True
        return False
    
    def get_input_sources(self) -> List[Dict]:
        """Get XR input sources (controllers, hands)."""
        if not self._session_active:
            return []
        
        # Simulate input sources
        return [
            {
                "handedness": "left",
                "target_ray_mode": "tracked-pointer",
                "grip_space": True
            },
            {
                "handedness": "right",
                "target_ray_mode": "tracked-pointer",
                "grip_space": True
            }
        ]
    
    def get_hand_tracking(self, handedness: str) -> Optional[Dict]:
        """
        Get hand tracking data.
        
        Args:
            handedness: "left" or "right"
            
        Returns:
            Hand tracking data
        """
        if not self._session_active:
            return None
        
        # Simulate hand tracking
        joints = [
            "wrist", "thumb-metacarpal", "thumb-phalanx-proximal",
            "thumb-phalanx-distal", "thumb-tip",
            "index-finger-metacarpal", "index-finger-phalanx-proximal",
            "index-finger-phalanx-intermediate", "index-finger-phalanx-distal",
            "index-finger-tip"
            # ... more joints
        ]
        
        return {
            "handedness": handedness,
            "joints": {
                joint: {
                    "position": np.random.randn(3).tolist(),
                    "radius": 0.01
                }
                for joint in joints
            }
        }
    
    def to_json(self) -> str:
        """Export state to JSON for web integration."""
        state = {
            "session_active": self._session_active,
            "session_mode": self._session_mode.value if self._session_mode else None,
            "reference_space": self._reference_space.value if self._reference_space else None,
            "viewer_pose": {
                "position": self._viewer_pose.position.tolist(),
                "orientation": self._viewer_pose.orientation.tolist()
            } if self._viewer_pose else None,
            "anchors": [
                {
                    "id": a.anchor_id,
                    "position": a.pose.position.tolist(),
                    "orientation": a.pose.orientation.tolist()
                }
                for a in self._anchors.values()
            ]
        }
        return json.dumps(state)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get XR statistics."""
        return {
            "session_active": self._session_active,
            "session_mode": self._session_mode.value if self._session_mode else None,
            "anchors": len(self._anchors),
            "hit_test_sources": len(self._hit_test_sources)
        }
    
    def __repr__(self) -> str:
        mode = self._session_mode.value if self._session_mode else "none"
        return f"WebXRIntegration(mode='{mode}', active={self._session_active})"
