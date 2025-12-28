"""
Edge AI Module
==============

Edge computing and IoT AI deployment.

Features:
- Model optimization for edge devices
- TensorRT, ONNX, TFLite support
- Federated edge learning
- Real-time inference
- Hardware acceleration
"""

from .runtime import EdgeRuntime
from .optimizer import ModelOptimizer
from .inference import EdgeInference
from .federated_edge import FederatedEdge
from .device_manager import DeviceManager

__all__ = [
    "EdgeRuntime",
    "ModelOptimizer",
    "EdgeInference",
    "FederatedEdge",
    "DeviceManager"
]
