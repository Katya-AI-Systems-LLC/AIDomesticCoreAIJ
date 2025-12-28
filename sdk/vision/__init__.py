"""
Vision Module
=============

Computer vision and 3D processing for AIPlatform SDK.

Features:
- Object detection and recognition
- Face recognition and analysis
- Gesture recognition
- Video stream processing
- 3D vision and SLAM
- WebXR integration
"""

from .detector import ObjectDetector
from .face import FaceRecognizer
from .gesture import GestureRecognizer
from .video import VideoStreamProcessor
from .vision3d import Vision3DEngine
from .slam import SLAMProcessor
from .webxr import WebXRIntegration

__all__ = [
    "ObjectDetector",
    "FaceRecognizer",
    "GestureRecognizer",
    "VideoStreamProcessor",
    "Vision3DEngine",
    "SLAMProcessor",
    "WebXRIntegration"
]
