"""
Quantum Mesh Protocol (QMP) Module
==================================

Implementation of the Quantum Mesh Protocol for
zero-infrastructure networking with quantum signatures.

Features:
- Quantum signature-based routing
- Mesh network topology
- Self-healing network
- Zero-trust communication
"""

from .protocol import QuantumMeshProtocol
from .node import QMPNode
from .router import QMPRouter
from .signature import QuantumSignature
from .mesh import MeshNetwork

__all__ = [
    "QuantumMeshProtocol",
    "QMPNode",
    "QMPRouter",
    "QuantumSignature",
    "MeshNetwork"
]
