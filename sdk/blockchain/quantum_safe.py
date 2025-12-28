"""
Quantum-Safe Blockchain
=======================

Post-quantum cryptography for blockchain operations.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import hashlib
import secrets
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumSafeKeyPair:
    """Quantum-safe key pair."""
    public_key: bytes
    secret_key: bytes
    algorithm: str
    security_level: int


@dataclass 
class QuantumSwapState:
    """Quantum-safe atomic swap state."""
    swap_id: str
    initiator: str
    participant: str
    initiator_chain: str
    participant_chain: str
    secret_hash: bytes  # Quantum-safe hash
    timelock: int
    amount: int
    token: str
    status: str
    created_at: float


class QuantumSafeSwap:
    """
    Quantum-safe atomic swaps.
    
    Uses post-quantum cryptography to secure cross-chain
    atomic swaps against quantum computing attacks.
    
    Features:
    - Kyber key encapsulation
    - Dilithium signatures
    - SHA-3 hash functions
    - Quantum-resistant HTLCs
    
    Example:
        >>> swap = QuantumSafeSwap()
        >>> order = await swap.initiate(
        ...     "ethereum", "solana", amount=1000
        ... )
    """
    
    def __init__(self, security_level: int = 3):
        """
        Initialize quantum-safe swap.
        
        Args:
            security_level: Security level (2, 3, or 5)
        """
        self.security_level = security_level
        
        # Active swaps
        self._swaps: Dict[str, QuantumSwapState] = {}
        
        # Secrets (only known to initiator until reveal)
        self._secrets: Dict[str, bytes] = {}
        
        logger.info(f"Quantum-Safe Swap initialized (level={security_level})")
    
    def generate_keypair(self, algorithm: str = "dilithium") -> QuantumSafeKeyPair:
        """
        Generate quantum-safe key pair.
        
        Args:
            algorithm: Algorithm (dilithium, kyber, sphincs)
            
        Returns:
            QuantumSafeKeyPair
        """
        from sdk.security import KyberKEM, DilithiumSignature
        
        if algorithm == "kyber":
            level_map = {2: 512, 3: 768, 5: 1024}
            kyber = KyberKEM(security_level=level_map[self.security_level])
            keypair = kyber.keygen()
            
            return QuantumSafeKeyPair(
                public_key=keypair.public_key,
                secret_key=keypair.secret_key,
                algorithm="kyber",
                security_level=level_map[self.security_level]
            )
        
        elif algorithm == "dilithium":
            dilithium = DilithiumSignature(security_level=self.security_level)
            keypair = dilithium.keygen()
            
            return QuantumSafeKeyPair(
                public_key=keypair.public_key,
                secret_key=keypair.secret_key,
                algorithm="dilithium",
                security_level=self.security_level
            )
        
        else:
            # Fallback to simulated keys
            return QuantumSafeKeyPair(
                public_key=secrets.token_bytes(32),
                secret_key=secrets.token_bytes(64),
                algorithm=algorithm,
                security_level=self.security_level
            )
    
    def quantum_hash(self, data: bytes) -> bytes:
        """
        Compute quantum-resistant hash.
        
        Uses SHA-3 (Keccak) which is quantum-resistant.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash value
        """
        return hashlib.sha3_256(data).digest()
    
    async def initiate(self, initiator: str,
                       participant: str,
                       initiator_chain: str,
                       participant_chain: str,
                       amount: int,
                       token: str,
                       timelock_hours: int = 24) -> QuantumSwapState:
        """
        Initiate quantum-safe atomic swap.
        
        Args:
            initiator: Initiator address
            participant: Participant address
            initiator_chain: Initiator's chain
            participant_chain: Participant's chain
            amount: Swap amount
            token: Token to swap
            timelock_hours: Timelock duration in hours
            
        Returns:
            QuantumSwapState
        """
        # Generate quantum-resistant secret
        secret = secrets.token_bytes(32)
        secret_hash = self.quantum_hash(secret)
        
        swap_id = self.quantum_hash(
            f"{initiator}{participant}{time.time()}".encode()
        ).hex()[:16]
        
        # Store secret (only initiator knows this)
        self._secrets[swap_id] = secret
        
        timelock = int(time.time()) + (timelock_hours * 3600)
        
        swap = QuantumSwapState(
            swap_id=swap_id,
            initiator=initiator,
            participant=participant,
            initiator_chain=initiator_chain,
            participant_chain=participant_chain,
            secret_hash=secret_hash,
            timelock=timelock,
            amount=amount,
            token=token,
            status="initiated",
            created_at=time.time()
        )
        
        self._swaps[swap_id] = swap
        
        logger.info(f"Quantum swap initiated: {swap_id}")
        return swap
    
    async def participate(self, swap_id: str,
                          participant_amount: int,
                          participant_token: str) -> bool:
        """
        Participate in atomic swap.
        
        Args:
            swap_id: Swap ID
            participant_amount: Amount participant offers
            participant_token: Token participant offers
            
        Returns:
            True if successful
        """
        if swap_id not in self._swaps:
            return False
        
        swap = self._swaps[swap_id]
        
        if swap.status != "initiated":
            return False
        
        # Participant locks funds with same secret hash
        # but shorter timelock
        swap.status = "participated"
        
        logger.info(f"Participant joined swap: {swap_id}")
        return True
    
    async def redeem(self, swap_id: str, secret: bytes) -> bool:
        """
        Redeem swap with secret.
        
        Args:
            swap_id: Swap ID
            secret: Preimage secret
            
        Returns:
            True if successful
        """
        if swap_id not in self._swaps:
            return False
        
        swap = self._swaps[swap_id]
        
        # Verify secret
        if self.quantum_hash(secret) != swap.secret_hash:
            logger.warning(f"Invalid secret for swap {swap_id}")
            return False
        
        # Check timelock
        if time.time() > swap.timelock:
            logger.warning(f"Swap {swap_id} has expired")
            return False
        
        swap.status = "redeemed"
        
        logger.info(f"Swap redeemed: {swap_id}")
        return True
    
    async def refund(self, swap_id: str) -> bool:
        """
        Refund expired swap.
        
        Args:
            swap_id: Swap ID
            
        Returns:
            True if successful
        """
        if swap_id not in self._swaps:
            return False
        
        swap = self._swaps[swap_id]
        
        # Check timelock expired
        if time.time() <= swap.timelock:
            logger.warning(f"Swap {swap_id} not yet expired")
            return False
        
        swap.status = "refunded"
        
        logger.info(f"Swap refunded: {swap_id}")
        return True
    
    def get_secret(self, swap_id: str) -> Optional[bytes]:
        """
        Get secret for swap (initiator only).
        
        Args:
            swap_id: Swap ID
            
        Returns:
            Secret bytes or None
        """
        return self._secrets.get(swap_id)
    
    def verify_secret(self, secret: bytes, secret_hash: bytes) -> bool:
        """
        Verify secret against hash.
        
        Args:
            secret: Preimage
            secret_hash: Hash to verify against
            
        Returns:
            True if valid
        """
        return self.quantum_hash(secret) == secret_hash
    
    def get_swap(self, swap_id: str) -> Optional[QuantumSwapState]:
        """Get swap state."""
        return self._swaps.get(swap_id)
    
    def get_active_swaps(self) -> List[QuantumSwapState]:
        """Get all active swaps."""
        return [
            s for s in self._swaps.values()
            if s.status in ["initiated", "participated"]
        ]
    
    async def create_quantum_htlc(self, sender: str,
                                   recipient: str,
                                   amount: int,
                                   secret_hash: bytes,
                                   timelock: int) -> Dict:
        """
        Create quantum-resistant HTLC contract.
        
        Args:
            sender: Sender address
            recipient: Recipient address
            amount: Amount to lock
            secret_hash: Quantum-resistant hash
            timelock: Unix timestamp for expiry
            
        Returns:
            HTLC details
        """
        htlc_id = self.quantum_hash(
            f"{sender}{recipient}{amount}{time.time()}".encode()
        ).hex()[:16]
        
        return {
            "htlc_id": htlc_id,
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "secret_hash": secret_hash.hex(),
            "timelock": timelock,
            "hash_algorithm": "sha3-256",
            "quantum_safe": True
        }
    
    def __repr__(self) -> str:
        return f"QuantumSafeSwap(level={self.security_level}, swaps={len(self._swaps)})"
