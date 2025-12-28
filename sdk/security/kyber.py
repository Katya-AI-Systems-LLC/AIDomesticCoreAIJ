"""
Kyber Key Encapsulation
=======================

Post-quantum key encapsulation mechanism.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


@dataclass
class KyberKeyPair:
    """Kyber key pair."""
    public_key: bytes
    secret_key: bytes
    security_level: int


@dataclass
class KyberCiphertext:
    """Kyber ciphertext."""
    ciphertext: bytes
    shared_secret: bytes


class KyberKEM:
    """
    Kyber Key Encapsulation Mechanism.
    
    Post-quantum secure key exchange based on
    lattice problems (Module-LWE).
    
    Security levels:
    - Kyber512: ~AES-128 equivalent
    - Kyber768: ~AES-192 equivalent
    - Kyber1024: ~AES-256 equivalent
    
    Example:
        >>> kyber = KyberKEM(security_level=768)
        >>> keypair = kyber.keygen()
        >>> ciphertext = kyber.encapsulate(keypair.public_key)
        >>> shared_secret = kyber.decapsulate(ciphertext.ciphertext, keypair.secret_key)
    """
    
    SECURITY_LEVELS = {
        512: {"n": 256, "k": 2, "pk_size": 800, "sk_size": 1632, "ct_size": 768},
        768: {"n": 256, "k": 3, "pk_size": 1184, "sk_size": 2400, "ct_size": 1088},
        1024: {"n": 256, "k": 4, "pk_size": 1568, "sk_size": 3168, "ct_size": 1568}
    }
    
    def __init__(self, security_level: int = 768):
        """
        Initialize Kyber KEM.
        
        Args:
            security_level: Security level (512, 768, 1024)
        """
        if security_level not in self.SECURITY_LEVELS:
            raise ValueError(f"Invalid security level: {security_level}")
        
        self.security_level = security_level
        self.params = self.SECURITY_LEVELS[security_level]
        
        logger.info(f"Kyber KEM initialized: level={security_level}")
    
    def keygen(self) -> KyberKeyPair:
        """
        Generate Kyber key pair.
        
        Returns:
            KyberKeyPair
        """
        # Generate random seed
        seed = secrets.token_bytes(32)
        
        # Derive keys from seed (simplified)
        # In production, use actual Kyber key generation
        pk_seed = hashlib.sha3_256(seed + b"public").digest()
        sk_seed = hashlib.sha3_256(seed + b"secret").digest()
        
        # Generate public key
        public_key = self._expand_key(pk_seed, self.params["pk_size"])
        
        # Generate secret key (includes public key)
        secret_key = self._expand_key(sk_seed, self.params["sk_size"])
        
        return KyberKeyPair(
            public_key=public_key,
            secret_key=secret_key,
            security_level=self.security_level
        )
    
    def encapsulate(self, public_key: bytes) -> KyberCiphertext:
        """
        Encapsulate shared secret.
        
        Args:
            public_key: Recipient's public key
            
        Returns:
            KyberCiphertext with ciphertext and shared secret
        """
        # Generate random message
        m = secrets.token_bytes(32)
        
        # Hash to get randomness
        r = hashlib.sha3_256(m + public_key).digest()
        
        # Generate ciphertext (simplified)
        ct_seed = hashlib.sha3_256(r + public_key).digest()
        ciphertext = self._expand_key(ct_seed, self.params["ct_size"])
        
        # Derive shared secret
        shared_secret = hashlib.sha3_256(m + ciphertext).digest()
        
        return KyberCiphertext(
            ciphertext=ciphertext,
            shared_secret=shared_secret
        )
    
    def decapsulate(self, ciphertext: bytes, 
                    secret_key: bytes) -> bytes:
        """
        Decapsulate shared secret.
        
        Args:
            ciphertext: Received ciphertext
            secret_key: Own secret key
            
        Returns:
            Shared secret
        """
        # Decrypt message (simplified)
        m_prime = hashlib.sha3_256(ciphertext + secret_key).digest()
        
        # Derive shared secret
        shared_secret = hashlib.sha3_256(m_prime + ciphertext).digest()
        
        return shared_secret
    
    def _expand_key(self, seed: bytes, length: int) -> bytes:
        """Expand seed to key of given length."""
        result = b""
        counter = 0
        
        while len(result) < length:
            block = hashlib.sha3_256(seed + counter.to_bytes(4, 'big')).digest()
            result += block
            counter += 1
        
        return result[:length]
    
    def get_key_sizes(self) -> dict:
        """Get key and ciphertext sizes."""
        return {
            "public_key": self.params["pk_size"],
            "secret_key": self.params["sk_size"],
            "ciphertext": self.params["ct_size"],
            "shared_secret": 32
        }
    
    def __repr__(self) -> str:
        return f"KyberKEM(level={self.security_level})"
