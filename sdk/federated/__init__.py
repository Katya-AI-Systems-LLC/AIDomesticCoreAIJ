"""
Federated Quantum AI Module
===========================

Distributed quantum-classical machine learning with:
- Federated learning coordination
- Hybrid quantum-classical training
- Model marketplace with smart contracts
- NFT-based weight management
"""

from .coordinator import FederatedCoordinator
from .node import QuantumFederatedNode
from .trainer import HybridTrainer
from .marketplace import ModelMarketplace
from .nft_weights import NFTWeightManager

__all__ = [
    "FederatedCoordinator",
    "QuantumFederatedNode",
    "HybridTrainer",
    "ModelMarketplace",
    "NFTWeightManager"
]
