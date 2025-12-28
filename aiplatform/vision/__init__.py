"""
AI & Vision Lab module for AIPlatform SDK

This module provides computer vision and multimodal AI capabilities
including object detection, facial recognition, gesture recognition,
3D computer vision, and integration with WebXR/VisionOS/VR platforms.
"""

from .detector import ObjectDetector, FaceDetector, GestureDetector
from .processor import VideoProcessor, ImageProcessor
from .vision3d import Vision3D
from .multimodal import MultimodalModel
from .webxr import WebXRInterface
from ..exceptions import VisionError

__all__ = [
    'ObjectDetector',
    'FaceDetector', 
    'GestureDetector',
    'VideoProcessor',
    'ImageProcessor',
    'Vision3D',
    'MultimodalModel',
    'WebXRInterface',
    'VisionError'
]

__version__ = '1.0.0'
__author__ = 'REChain Network Solutions & Katya AI Systems'