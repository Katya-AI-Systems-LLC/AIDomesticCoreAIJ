"""
Examples module for AIPlatform SDK

This module provides working examples and templates demonstrating
the capabilities of the AIPlatform Quantum Infrastructure Zero SDK.
"""

from .quantum_example import QuantumExample
from .qiz_example import QIZExample
from .federated_example import FederatedExample
from .vision_example import VisionExample
from .genai_example import GenAIExample
from ..exceptions import ExampleError

__all__ = [
    'QuantumExample',
    'QIZExample',
    'FederatedExample',
    'VisionExample',
    'GenAIExample',
    'ExampleError'
]

__version__ = '1.0.0'
__author__ = 'REChain Network Solutions & Katya AI Systems'