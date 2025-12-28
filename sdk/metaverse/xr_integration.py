"""
XR Integration
==============

WebXR, VR, and AR integration for metaverse.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class XRDeviceType(Enum):
    """XR device types."""
    HEADSET_VR = "vr_headset"
    HEADSET_AR = "ar_headset"
    HANDHELD_AR = "handheld_ar"
    CONTROLLER = "controller"
    HAND_TRACKING = "hand"


class XRPlatform(Enum):
    """Supported XR platforms."""
    WEBXR = "webxr"
    OCULUS = "oculus"
    VIVE = "vive"
    PICO = "pico"
    APPLE_VISION = "vision_pro"
    HOLOLENS = "hololens"
    MAGIC_LEAP = "magic_leap"


@dataclass
class XRDevice:
    """XR device."""
    device_id: str
    device_type: XRDeviceType
    platform: XRPlatform
    capabilities: List[str]
    tracking_space: str
    connected: bool = False


@dataclass
class XRPose:
    """XR pose (position + orientation)."""
    position: np.ndarray
    orientation: np.ndarray  # Quaternion
    timestamp: float


@dataclass
class XRInputState:
    """XR controller/hand input state."""
    device_id: str
    pose: XRPose
    buttons: Dict[str, bool]
    axes: Dict[str, float]
    grip_value: float
    trigger_value: float


class XRManager:
    """
    XR device and session manager.
    
    Features:
    - Multi-platform XR support
    - Hand tracking
    - Spatial anchors
    - Pass-through AR
    - Gesture recognition
    - Voice input
    
    Example:
        >>> xr = XRManager()
        >>> await xr.initialize(XRPlatform.WEBXR)
        >>> session = await xr.start_session("immersive-vr")
    """
    
    PLATFORM_CAPABILITIES = {
        XRPlatform.WEBXR: ["hand-tracking", "local-floor", "bounded-floor"],
        XRPlatform.OCULUS: ["hand-tracking", "passthrough", "guardian", "voice"],
        XRPlatform.VIVE: ["lighthouse", "eye-tracking", "lip-tracking"],
        XRPlatform.APPLE_VISION: ["passthrough", "eye-tracking", "hand-tracking", "spatial-audio"],
        XRPlatform.HOLOLENS: ["passthrough", "hand-tracking", "spatial-mapping", "voice"],
        XRPlatform.MAGIC_LEAP: ["passthrough", "hand-tracking", "eye-tracking"]
    }
    
    def __init__(self):
        """Initialize XR manager."""
        self._platform: Optional[XRPlatform] = None
        self._devices: Dict[str, XRDevice] = {}
        self._session_active = False
        self._session_mode: Optional[str] = None
        
        # Tracking state
        self._head_pose: Optional[XRPose] = None
        self._controller_states: Dict[str, XRInputState] = {}
        self._hand_poses: Dict[str, List[XRPose]] = {}  # Per-joint poses
        
        # Spatial anchors
        self._anchors: Dict[str, XRPose] = {}
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info("XR Manager initialized")
    
    async def initialize(self, platform: XRPlatform) -> bool:
        """
        Initialize XR for platform.
        
        Args:
            platform: Target platform
            
        Returns:
            True if successful
        """
        self._platform = platform
        
        # Detect devices
        await self._detect_devices()
        
        logger.info(f"XR initialized for {platform.value}")
        return True
    
    async def _detect_devices(self):
        """Detect available XR devices."""
        # Simulated device detection
        if self._platform in [XRPlatform.OCULUS, XRPlatform.VIVE]:
            # VR headset with controllers
            self._devices["headset"] = XRDevice(
                device_id="headset",
                device_type=XRDeviceType.HEADSET_VR,
                platform=self._platform,
                capabilities=self.PLATFORM_CAPABILITIES.get(self._platform, []),
                tracking_space="local-floor",
                connected=True
            )
            
            self._devices["left_controller"] = XRDevice(
                device_id="left_controller",
                device_type=XRDeviceType.CONTROLLER,
                platform=self._platform,
                capabilities=["haptics", "buttons", "thumbstick"],
                tracking_space="local-floor",
                connected=True
            )
            
            self._devices["right_controller"] = XRDevice(
                device_id="right_controller",
                device_type=XRDeviceType.CONTROLLER,
                platform=self._platform,
                capabilities=["haptics", "buttons", "thumbstick"],
                tracking_space="local-floor",
                connected=True
            )
        
        elif self._platform == XRPlatform.APPLE_VISION:
            # Vision Pro with hand tracking
            self._devices["headset"] = XRDevice(
                device_id="headset",
                device_type=XRDeviceType.HEADSET_AR,
                platform=self._platform,
                capabilities=self.PLATFORM_CAPABILITIES.get(self._platform, []),
                tracking_space="unbounded",
                connected=True
            )
            
            self._devices["left_hand"] = XRDevice(
                device_id="left_hand",
                device_type=XRDeviceType.HAND_TRACKING,
                platform=self._platform,
                capabilities=["gesture", "pinch"],
                tracking_space="unbounded",
                connected=True
            )
            
            self._devices["right_hand"] = XRDevice(
                device_id="right_hand",
                device_type=XRDeviceType.HAND_TRACKING,
                platform=self._platform,
                capabilities=["gesture", "pinch"],
                tracking_space="unbounded",
                connected=True
            )
    
    async def start_session(self, mode: str = "immersive-vr") -> bool:
        """
        Start XR session.
        
        Args:
            mode: Session mode (immersive-vr, immersive-ar, inline)
            
        Returns:
            True if started
        """
        if self._session_active:
            return False
        
        self._session_active = True
        self._session_mode = mode
        
        self._fire_event("session_started", {"mode": mode})
        
        logger.info(f"XR session started: {mode}")
        return True
    
    async def end_session(self):
        """End XR session."""
        if not self._session_active:
            return
        
        self._session_active = False
        self._fire_event("session_ended", {})
        
        logger.info("XR session ended")
    
    def get_head_pose(self) -> Optional[XRPose]:
        """Get current head pose."""
        if not self._session_active:
            return None
        
        # Simulated pose
        return XRPose(
            position=np.array([0.0, 1.6, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=time.time()
        )
    
    def get_controller_state(self, hand: str) -> Optional[XRInputState]:
        """Get controller input state."""
        device_id = f"{hand}_controller"
        
        if device_id not in self._devices:
            return None
        
        # Simulated input state
        return XRInputState(
            device_id=device_id,
            pose=XRPose(
                position=np.array([0.3 if hand == "right" else -0.3, 1.0, -0.3]),
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                timestamp=time.time()
            ),
            buttons={
                "trigger": False,
                "grip": False,
                "a": False,
                "b": False,
                "thumbstick": False
            },
            axes={"thumbstick_x": 0.0, "thumbstick_y": 0.0},
            grip_value=0.0,
            trigger_value=0.0
        )
    
    def get_hand_joints(self, hand: str) -> Optional[List[XRPose]]:
        """Get hand tracking joint poses."""
        # 25 joints per hand (WebXR standard)
        joint_count = 25
        
        base_position = np.array([0.3 if hand == "right" else -0.3, 1.0, -0.3])
        
        joints = []
        for i in range(joint_count):
            offset = np.array([
                (i % 5) * 0.02,
                (i // 5) * 0.03,
                0
            ])
            
            joints.append(XRPose(
                position=base_position + offset,
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                timestamp=time.time()
            ))
        
        return joints
    
    async def create_anchor(self, pose: XRPose,
                            name: str = None) -> str:
        """
        Create spatial anchor.
        
        Args:
            pose: Anchor pose
            name: Optional name
            
        Returns:
            Anchor ID
        """
        anchor_id = name or f"anchor_{len(self._anchors)}"
        self._anchors[anchor_id] = pose
        
        logger.debug(f"Anchor created: {anchor_id}")
        return anchor_id
    
    def get_anchor(self, anchor_id: str) -> Optional[XRPose]:
        """Get anchor pose."""
        return self._anchors.get(anchor_id)
    
    async def delete_anchor(self, anchor_id: str) -> bool:
        """Delete spatial anchor."""
        if anchor_id in self._anchors:
            del self._anchors[anchor_id]
            return True
        return False
    
    def trigger_haptic(self, device_id: str,
                       intensity: float = 0.5,
                       duration: float = 0.1):
        """Trigger haptic feedback."""
        if device_id in self._devices:
            logger.debug(f"Haptic: {device_id} intensity={intensity}")
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def _fire_event(self, event: str, data: Any):
        """Fire event."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def get_devices(self) -> List[XRDevice]:
        """Get connected devices."""
        return list(self._devices.values())
    
    def is_session_active(self) -> bool:
        """Check if session is active."""
        return self._session_active
    
    def get_capabilities(self) -> List[str]:
        """Get platform capabilities."""
        if self._platform:
            return self.PLATFORM_CAPABILITIES.get(self._platform, [])
        return []
    
    def __repr__(self) -> str:
        return f"XRManager(platform={self._platform}, active={self._session_active})"
