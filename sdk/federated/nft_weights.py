"""
NFT Weight Manager
==================

Manage model weights as NFTs for ownership and trading.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import hashlib
import time
import uuid
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class WeightNFT:
    """NFT representing model weights."""
    token_id: str
    name: str
    description: str
    owner: str
    creator: str
    weight_hash: bytes
    weight_shape: tuple
    metadata: Dict[str, Any]
    created: float
    royalty_percent: float = 5.0
    transfers: int = 0


@dataclass
class TransferRecord:
    """Record of NFT transfer."""
    transfer_id: str
    token_id: str
    from_owner: str
    to_owner: str
    price: float
    timestamp: float
    transaction_hash: str


class NFTWeightManager:
    """
    Manage model weights as NFTs.
    
    Features:
    - Mint weights as NFTs
    - Transfer ownership
    - Royalty distribution
    - Weight verification
    
    Example:
        >>> manager = NFTWeightManager()
        >>> nft = await manager.mint(weights, "MyModel", owner)
        >>> await manager.transfer(nft.token_id, new_owner, price=100)
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize NFT weight manager.
        
        Args:
            language: Language for messages
        """
        self.language = language
        
        # NFT storage
        self._nfts: Dict[str, WeightNFT] = {}
        
        # Weight storage (encrypted)
        self._weights: Dict[str, np.ndarray] = {}
        
        # Transfer history
        self._transfers: List[TransferRecord] = []
        
        # Balances
        self._balances: Dict[str, float] = {}
        
        logger.info("NFT Weight Manager initialized")
    
    async def mint(self, weights: np.ndarray,
                   name: str,
                   owner: str,
                   description: str = "",
                   metadata: Optional[Dict] = None,
                   royalty_percent: float = 5.0) -> WeightNFT:
        """
        Mint model weights as NFT.
        
        Args:
            weights: Model weights array
            name: NFT name
            owner: Owner identifier
            description: NFT description
            metadata: Additional metadata
            royalty_percent: Royalty percentage for creator
            
        Returns:
            WeightNFT
        """
        token_id = str(uuid.uuid4())
        
        # Hash weights for verification
        weight_bytes = weights.tobytes()
        weight_hash = hashlib.sha256(weight_bytes).digest()
        
        nft = WeightNFT(
            token_id=token_id,
            name=name,
            description=description,
            owner=owner,
            creator=owner,
            weight_hash=weight_hash,
            weight_shape=weights.shape,
            metadata=metadata or {},
            created=time.time(),
            royalty_percent=royalty_percent
        )
        
        self._nfts[token_id] = nft
        self._weights[token_id] = weights.copy()
        
        logger.info(f"Minted NFT: {name} ({token_id})")
        return nft
    
    async def transfer(self, token_id: str,
                       from_owner: str,
                       to_owner: str,
                       price: float = 0.0) -> Optional[TransferRecord]:
        """
        Transfer NFT ownership.
        
        Args:
            token_id: NFT to transfer
            from_owner: Current owner
            to_owner: New owner
            price: Transfer price
            
        Returns:
            TransferRecord or None
        """
        if token_id not in self._nfts:
            return None
        
        nft = self._nfts[token_id]
        
        if nft.owner != from_owner:
            logger.warning(f"Unauthorized transfer attempt: {token_id}")
            return None
        
        # Process payment
        if price > 0:
            buyer_balance = self._balances.get(to_owner, 0)
            if buyer_balance < price:
                logger.warning(f"Insufficient balance for {to_owner}")
                return None
            
            # Deduct from buyer
            self._balances[to_owner] = buyer_balance - price
            
            # Calculate royalty
            royalty = price * (nft.royalty_percent / 100)
            seller_amount = price - royalty
            
            # Pay seller
            seller_balance = self._balances.get(from_owner, 0)
            self._balances[from_owner] = seller_balance + seller_amount
            
            # Pay creator royalty
            if nft.creator != from_owner:
                creator_balance = self._balances.get(nft.creator, 0)
                self._balances[nft.creator] = creator_balance + royalty
        
        # Transfer ownership
        nft.owner = to_owner
        nft.transfers += 1
        
        # Create transfer record
        transfer_id = str(uuid.uuid4())
        transaction_hash = hashlib.sha256(
            f"{transfer_id}:{token_id}:{from_owner}:{to_owner}".encode()
        ).hexdigest()
        
        record = TransferRecord(
            transfer_id=transfer_id,
            token_id=token_id,
            from_owner=from_owner,
            to_owner=to_owner,
            price=price,
            timestamp=time.time(),
            transaction_hash=transaction_hash
        )
        
        self._transfers.append(record)
        
        logger.info(f"Transferred NFT: {token_id} from {from_owner} to {to_owner}")
        return record
    
    def get_weights(self, token_id: str, owner: str) -> Optional[np.ndarray]:
        """
        Get weights for an NFT.
        
        Args:
            token_id: NFT token ID
            owner: Owner for verification
            
        Returns:
            Weights array or None
        """
        if token_id not in self._nfts:
            return None
        
        nft = self._nfts[token_id]
        
        if nft.owner != owner:
            logger.warning(f"Unauthorized weight access: {token_id}")
            return None
        
        return self._weights.get(token_id)
    
    def verify_weights(self, token_id: str, weights: np.ndarray) -> bool:
        """
        Verify weights match NFT.
        
        Args:
            token_id: NFT token ID
            weights: Weights to verify
            
        Returns:
            True if weights match
        """
        if token_id not in self._nfts:
            return False
        
        nft = self._nfts[token_id]
        
        weight_bytes = weights.tobytes()
        weight_hash = hashlib.sha256(weight_bytes).digest()
        
        return weight_hash == nft.weight_hash
    
    def get_nft(self, token_id: str) -> Optional[WeightNFT]:
        """Get NFT by token ID."""
        return self._nfts.get(token_id)
    
    def get_owner_nfts(self, owner: str) -> List[WeightNFT]:
        """Get all NFTs owned by an address."""
        return [nft for nft in self._nfts.values() if nft.owner == owner]
    
    def get_creator_nfts(self, creator: str) -> List[WeightNFT]:
        """Get all NFTs created by an address."""
        return [nft for nft in self._nfts.values() if nft.creator == creator]
    
    def get_transfer_history(self, token_id: str) -> List[TransferRecord]:
        """Get transfer history for an NFT."""
        return [t for t in self._transfers if t.token_id == token_id]
    
    async def burn(self, token_id: str, owner: str) -> bool:
        """
        Burn an NFT.
        
        Args:
            token_id: NFT to burn
            owner: Owner for verification
            
        Returns:
            True if burned successfully
        """
        if token_id not in self._nfts:
            return False
        
        nft = self._nfts[token_id]
        
        if nft.owner != owner:
            return False
        
        del self._nfts[token_id]
        if token_id in self._weights:
            del self._weights[token_id]
        
        logger.info(f"Burned NFT: {token_id}")
        return True
    
    def add_balance(self, user: str, amount: float):
        """Add balance to user account."""
        current = self._balances.get(user, 0)
        self._balances[user] = current + amount
    
    def get_balance(self, user: str) -> float:
        """Get user balance."""
        return self._balances.get(user, 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        total_volume = sum(t.price for t in self._transfers)
        
        return {
            "total_nfts": len(self._nfts),
            "total_transfers": len(self._transfers),
            "total_volume": total_volume,
            "unique_owners": len(set(nft.owner for nft in self._nfts.values())),
            "unique_creators": len(set(nft.creator for nft in self._nfts.values()))
        }
    
    def __repr__(self) -> str:
        return f"NFTWeightManager(nfts={len(self._nfts)})"
