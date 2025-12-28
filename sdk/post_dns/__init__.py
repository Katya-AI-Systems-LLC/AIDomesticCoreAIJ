"""
Post-DNS Module
===============

Implementation of Post-DNS architecture for Quantum Infrastructure Zero.

Features:
- Zero-DNS resolution
- Quantum addressing
- Object signature routing
- Decentralized name resolution
"""

from .resolver import PostDNSResolver
from .registry import ZeroDNSRegistry
from .addressing import QuantumAddressing
from .object_router import ObjectSignatureRouter

__all__ = [
    "PostDNSResolver",
    "ZeroDNSRegistry",
    "QuantumAddressing",
    "ObjectSignatureRouter"
]
