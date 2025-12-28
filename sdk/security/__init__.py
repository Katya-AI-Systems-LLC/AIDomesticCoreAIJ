"""
Security Module
===============

Quantum-safe cryptography and security for AIPlatform SDK.

Features:
- Post-quantum cryptography (Kyber, Dilithium)
- DIDN (Decentralized Identity)
- Zero-trust security
- Secure key management
"""

from .kyber import KyberKEM
from .dilithium import DilithiumSignature
from .didn import DIDNManager
from .zero_trust import ZeroTrustManager
from .key_manager import SecureKeyManager

__all__ = [
    "KyberKEM",
    "DilithiumSignature",
    "DIDNManager",
    "ZeroTrustManager",
    "SecureKeyManager"
]
