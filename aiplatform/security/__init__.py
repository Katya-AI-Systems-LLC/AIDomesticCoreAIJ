"""
Security module for AIPlatform SDK

This module provides security implementations including
Quantum Safe Cryptography, Zero-Trust model, and DIDN integration.
"""

from .crypto import QuantumSafeCrypto, Kyber, Dilithium
from .zerotrust import ZeroTrustModel, ZeroTrustPolicy
from .dids import DIDN, DIDNResolver
from ..exceptions import SecurityError

__all__ = [
    'QuantumSafeCrypto',
    'Kyber',
    'Dilithium',
    'ZeroTrustModel',
    'ZeroTrustPolicy',
    'DIDN',
    'DIDNResolver',
    'SecurityError'
]

__version__ = '1.0.0'
__author__ = 'REChain Network Solutions & Katya AI Systems'