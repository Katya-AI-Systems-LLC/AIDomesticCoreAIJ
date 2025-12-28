"""
AIPlatform Quantum Infrastructure Zero SDK

A comprehensive SDK for quantum-enhanced AI development with
Quantum Infrastructure Zero (QIZ) capabilities.

This SDK integrates:
- Quantum computing with Qiskit Runtime
- Quantum Infrastructure Zero networking
- Federated Quantum AI
- Computer vision and multimodal AI
- Generative AI model integration
"""

__version__ = "1.0.0"
__author__ = "REChain Network Solutions & Katya AI Systems"
__license__ = "Enterprise"

# Core module imports
from .core import AIPlatform
from .config import Configuration

# Quantum module imports
from .quantum import (
    QuantumCircuit,
    VQE,
    QAOA,
    Grover,
    Shor,
    Kyber,
    Dilithium
)

# QIZ module imports
from .qiz import (
    QIZNode,
    QuantumSignature,
    PostDNS,
    QMPProtocol
)

# Federated module imports
from .federated import (
    FederatedQuantumTrainer,
    FederatedModel
)

# Vision module imports
from .vision import (
    ObjectDetector,
    FaceDetector,
    GestureDetector
)

# GenAI module imports
from .genai import (
    GenAIModel,
    DiffusionModel,
    KatyaAI
)

# Security module imports
from .security import (
    ZeroTrustModel,
    DIDN
)

__all__ = [
    # Core
    "AIPlatform",
    "Configuration",
    
    # Quantum
    "QuantumCircuit",
    "VQE",
    "QAOA",
    "Grover",
    "Shor",
    "Kyber",
    "Dilithium",
    
    # QIZ
    "QIZNode",
    "QuantumSignature",
    "PostDNS",
    "QMPProtocol",
    
    # Federated
    "FederatedQuantumTrainer",
    "FederatedModel",
    
    # Vision
    "ObjectDetector",
    "FaceDetector",
    "GestureDetector",
    
    # GenAI
    "GenAIModel",
    "DiffusionModel",
    "KatyaAI",
    
    # Security
    "ZeroTrustModel",
    "DIDN"
]