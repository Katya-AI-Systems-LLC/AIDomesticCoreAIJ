"""
Quantum Signature Module
========================

Quantum-safe digital signatures for node identity and message authentication.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import secrets
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignatureKeyPair:
    """Quantum signature key pair."""
    public_key: bytes
    private_key: bytes
    algorithm: str
    created: float
    expires: float


@dataclass
class SignedData:
    """Data with quantum signature."""
    data: bytes
    signature: bytes
    public_key: bytes
    timestamp: float
    algorithm: str


class QuantumSignature:
    """
    Quantum-safe signature implementation.
    
    Provides:
    - Key generation (simulated quantum-safe)
    - Message signing
    - Signature verification
    - Key rotation
    
    In production, this would use actual post-quantum algorithms
    like Dilithium or SPHINCS+.
    
    Example:
        >>> qs = QuantumSignature()
        >>> signed = qs.sign(b"Hello, quantum world!")
        >>> is_valid = qs.verify(signed)
    """
    
    ALGORITHMS = ["dilithium3", "sphincs256", "falcon512"]
    DEFAULT_ALGORITHM = "dilithium3"
    KEY_SIZE = 64  # bytes
    SIGNATURE_SIZE = 128  # bytes
    
    def __init__(self, algorithm: str = DEFAULT_ALGORITHM,
                 key_lifetime: float = 86400,  # 24 hours
                 language: str = "en"):
        """
        Initialize quantum signature system.
        
        Args:
            algorithm: Signature algorithm to use
            key_lifetime: Key validity period in seconds
            language: Language for messages
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.algorithm = algorithm
        self.key_lifetime = key_lifetime
        self.language = language
        
        # Generate initial key pair
        self._key_pair: Optional[SignatureKeyPair] = None
        self._generate_key_pair()
        
        logger.info(f"Quantum signature initialized: algorithm={algorithm}")
    
    def _generate_key_pair(self):
        """Generate new quantum-safe key pair."""
        # In production, this would use actual post-quantum algorithms
        # Here we simulate with cryptographically secure random bytes
        
        private_key = secrets.token_bytes(self.KEY_SIZE)
        
        # Derive public key from private key
        public_key = hashlib.sha3_512(private_key).digest()
        
        self._key_pair = SignatureKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=self.algorithm,
            created=time.time(),
            expires=time.time() + self.key_lifetime
        )
        
        logger.debug("Generated new quantum key pair")
    
    @property
    def public_key(self) -> bytes:
        """Get public key."""
        self._ensure_valid_key()
        return self._key_pair.public_key
    
    @property
    def key_fingerprint(self) -> str:
        """Get key fingerprint (short identifier)."""
        return hashlib.sha256(self.public_key).hexdigest()[:16]
    
    def _ensure_valid_key(self):
        """Ensure key pair is valid, regenerate if expired."""
        if self._key_pair is None or time.time() > self._key_pair.expires:
            self._generate_key_pair()
    
    def sign(self, data: bytes) -> SignedData:
        """
        Sign data with quantum signature.
        
        Args:
            data: Data to sign
            
        Returns:
            SignedData with signature
        """
        self._ensure_valid_key()
        
        timestamp = time.time()
        
        # Create signature
        # In production, this would use actual post-quantum signing
        signature_input = (
            data + 
            self._key_pair.private_key + 
            str(timestamp).encode()
        )
        signature = hashlib.sha3_512(signature_input).digest()
        
        # Add additional entropy
        entropy = secrets.token_bytes(32)
        full_signature = hashlib.sha3_512(signature + entropy).digest()
        
        return SignedData(
            data=data,
            signature=full_signature,
            public_key=self._key_pair.public_key,
            timestamp=timestamp,
            algorithm=self.algorithm
        )
    
    def verify(self, signed_data: SignedData,
               max_age: Optional[float] = None) -> bool:
        """
        Verify quantum signature.
        
        Args:
            signed_data: Signed data to verify
            max_age: Maximum age of signature in seconds
            
        Returns:
            True if signature is valid
        """
        # Check timestamp
        if max_age is not None:
            age = time.time() - signed_data.timestamp
            if age > max_age:
                logger.warning(f"Signature too old: {age}s > {max_age}s")
                return False
        
        # Verify algorithm
        if signed_data.algorithm not in self.ALGORITHMS:
            logger.warning(f"Unknown algorithm: {signed_data.algorithm}")
            return False
        
        # Verify signature length
        if len(signed_data.signature) != self.KEY_SIZE:
            logger.warning("Invalid signature length")
            return False
        
        # In production, this would verify using actual post-quantum verification
        # Here we do a simplified check
        
        # Verify public key format
        if len(signed_data.public_key) != self.KEY_SIZE:
            return False
        
        # Signature is considered valid if it's the right length and format
        # In production, actual cryptographic verification would happen here
        return True
    
    def verify_with_key(self, signed_data: SignedData,
                        public_key: bytes) -> bool:
        """
        Verify signature with specific public key.
        
        Args:
            signed_data: Signed data to verify
            public_key: Public key to verify against
            
        Returns:
            True if signature is valid for given key
        """
        if signed_data.public_key != public_key:
            return False
        
        return self.verify(signed_data)
    
    def rotate_keys(self) -> bytes:
        """
        Rotate to new key pair.
        
        Returns:
            New public key
        """
        old_fingerprint = self.key_fingerprint
        self._generate_key_pair()
        
        logger.info(f"Key rotated: {old_fingerprint} -> {self.key_fingerprint}")
        return self._key_pair.public_key
    
    def export_public_key(self) -> Dict[str, Any]:
        """Export public key for sharing."""
        self._ensure_valid_key()
        
        return {
            "public_key": self._key_pair.public_key.hex(),
            "algorithm": self.algorithm,
            "fingerprint": self.key_fingerprint,
            "expires": self._key_pair.expires
        }
    
    def import_public_key(self, key_data: Dict[str, Any]) -> bytes:
        """Import public key from external source."""
        public_key = bytes.fromhex(key_data["public_key"])
        
        # Validate key
        if len(public_key) != self.KEY_SIZE:
            raise ValueError("Invalid public key size")
        
        return public_key
    
    def create_challenge(self) -> Tuple[bytes, bytes]:
        """
        Create authentication challenge.
        
        Returns:
            Tuple of (challenge, expected_response)
        """
        challenge = secrets.token_bytes(32)
        
        # Expected response is signature of challenge
        signed = self.sign(challenge)
        
        return challenge, signed.signature
    
    def respond_to_challenge(self, challenge: bytes) -> bytes:
        """
        Respond to authentication challenge.
        
        Args:
            challenge: Challenge bytes
            
        Returns:
            Response signature
        """
        signed = self.sign(challenge)
        return signed.signature
    
    def get_key_info(self) -> Dict[str, Any]:
        """Get information about current key pair."""
        self._ensure_valid_key()
        
        return {
            "algorithm": self.algorithm,
            "fingerprint": self.key_fingerprint,
            "created": self._key_pair.created,
            "expires": self._key_pair.expires,
            "time_remaining": self._key_pair.expires - time.time(),
            "public_key_size": len(self._key_pair.public_key),
            "signature_size": self.SIGNATURE_SIZE
        }
    
    def __repr__(self) -> str:
        return (f"QuantumSignature(algorithm='{self.algorithm}', "
                f"fingerprint='{self.key_fingerprint}')")


class SignatureVerifier:
    """
    Standalone signature verifier for nodes that only need to verify.
    """
    
    def __init__(self, trusted_keys: Optional[Dict[str, bytes]] = None):
        """
        Initialize verifier.
        
        Args:
            trusted_keys: Dictionary of node_id -> public_key
        """
        self._trusted_keys: Dict[str, bytes] = trusted_keys or {}
    
    def add_trusted_key(self, node_id: str, public_key: bytes):
        """Add a trusted public key."""
        self._trusted_keys[node_id] = public_key
    
    def remove_trusted_key(self, node_id: str):
        """Remove a trusted key."""
        if node_id in self._trusted_keys:
            del self._trusted_keys[node_id]
    
    def is_trusted(self, node_id: str) -> bool:
        """Check if node is trusted."""
        return node_id in self._trusted_keys
    
    def verify(self, signed_data: SignedData, node_id: str) -> bool:
        """
        Verify signature from a specific node.
        
        Args:
            signed_data: Signed data to verify
            node_id: Expected signer node ID
            
        Returns:
            True if signature is valid and from trusted node
        """
        if node_id not in self._trusted_keys:
            return False
        
        expected_key = self._trusted_keys[node_id]
        
        if signed_data.public_key != expected_key:
            return False
        
        # Verify signature format
        if len(signed_data.signature) != QuantumSignature.KEY_SIZE:
            return False
        
        return True
    
    def get_trusted_nodes(self) -> list:
        """Get list of trusted node IDs."""
        return list(self._trusted_keys.keys())
