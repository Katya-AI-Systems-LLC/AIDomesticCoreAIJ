"""
Federated Quantum AI module for AIPlatform SDK

This module provides federated learning capabilities with quantum-enhanced algorithms:
- Distributed quantum-classical training
- Secure multi-party computation
- Model marketplace with smart contracts
- Collaborative evolution protocols
"""

from .trainer import FederatedQuantumTrainer
from .model import FederatedModel
from .marketplace import ModelMarketplace

__all__ = [
    "FederatedQuantumTrainer",
    "FederatedModel",
    "ModelMarketplace"
]