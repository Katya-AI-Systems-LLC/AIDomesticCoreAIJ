"""
NFT Manager
===========

Cross-chain NFT operations and metaverse integration.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import json
import logging

logger = logging.getLogger(__name__)


class NFTStandard(Enum):
    """NFT standards."""
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    SPL = "spl"
    METAPLEX = "metaplex"
    CW721 = "cw721"  # CosmWasm


@dataclass
class NFTMetadata:
    """NFT metadata."""
    name: str
    description: str
    image: str
    external_url: Optional[str] = None
    animation_url: Optional[str] = None
    attributes: List[Dict] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)


@dataclass
class NFT:
    """NFT representation."""
    token_id: str
    contract_address: str
    chain: str
    standard: NFTStandard
    owner: str
    metadata: NFTMetadata
    royalty_bps: int = 0
    royalty_recipient: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class NFTCollection:
    """NFT collection."""
    collection_id: str
    name: str
    symbol: str
    contract_address: str
    chain: str
    creator: str
    total_supply: int
    max_supply: Optional[int]
    royalty_bps: int
    metadata_uri: str


class NFTManager:
    """
    Cross-chain NFT manager.
    
    Features:
    - Multi-chain NFT minting
    - Cross-chain NFT bridges
    - Metaverse interoperability
    - Royalty management
    - AI-generated NFT support
    
    Example:
        >>> manager = NFTManager()
        >>> nft = await manager.mint(
        ...     chain="ethereum",
        ...     metadata=NFTMetadata(name="AI Art #1", ...)
        ... )
    """
    
    SUPPORTED_CHAINS = [
        "ethereum", "polygon", "solana", "bsc",
        "avalanche", "arbitrum", "optimism"
    ]
    
    def __init__(self):
        """Initialize NFT manager."""
        # Collections
        self._collections: Dict[str, NFTCollection] = {}
        
        # NFTs by token ID
        self._nfts: Dict[str, NFT] = {}
        
        # Ownership index
        self._ownership: Dict[str, List[str]] = {}
        
        logger.info("NFT Manager initialized")
    
    async def create_collection(self, name: str,
                                 symbol: str,
                                 chain: str,
                                 creator: str,
                                 max_supply: Optional[int] = None,
                                 royalty_bps: int = 500,
                                 metadata_uri: str = "") -> NFTCollection:
        """
        Create NFT collection.
        
        Args:
            name: Collection name
            symbol: Collection symbol
            chain: Target chain
            creator: Creator address
            max_supply: Maximum supply (None for unlimited)
            royalty_bps: Royalty in basis points (500 = 5%)
            metadata_uri: Collection metadata URI
            
        Returns:
            NFTCollection
        """
        collection_id = hashlib.sha256(
            f"{name}{symbol}{chain}{time.time()}".encode()
        ).hexdigest()[:16]
        
        contract_address = "0x" + hashlib.sha256(
            f"nft_{collection_id}".encode()
        ).hexdigest()[:40]
        
        collection = NFTCollection(
            collection_id=collection_id,
            name=name,
            symbol=symbol,
            contract_address=contract_address,
            chain=chain,
            creator=creator,
            total_supply=0,
            max_supply=max_supply,
            royalty_bps=royalty_bps,
            metadata_uri=metadata_uri
        )
        
        self._collections[collection_id] = collection
        
        logger.info(f"Collection created: {name} ({collection_id})")
        return collection
    
    async def mint(self, collection_id: str,
                   recipient: str,
                   metadata: NFTMetadata) -> NFT:
        """
        Mint NFT in collection.
        
        Args:
            collection_id: Collection ID
            recipient: Recipient address
            metadata: NFT metadata
            
        Returns:
            Minted NFT
        """
        if collection_id not in self._collections:
            raise ValueError("Collection not found")
        
        collection = self._collections[collection_id]
        
        # Check max supply
        if collection.max_supply and collection.total_supply >= collection.max_supply:
            raise ValueError("Max supply reached")
        
        token_id = f"{collection_id}_{collection.total_supply + 1}"
        
        nft = NFT(
            token_id=token_id,
            contract_address=collection.contract_address,
            chain=collection.chain,
            standard=NFTStandard.ERC721,
            owner=recipient,
            metadata=metadata,
            royalty_bps=collection.royalty_bps,
            royalty_recipient=collection.creator
        )
        
        self._nfts[token_id] = nft
        collection.total_supply += 1
        
        # Update ownership index
        if recipient not in self._ownership:
            self._ownership[recipient] = []
        self._ownership[recipient].append(token_id)
        
        logger.info(f"NFT minted: {token_id} -> {recipient[:10]}...")
        return nft
    
    async def mint_ai_generated(self, collection_id: str,
                                 recipient: str,
                                 prompt: str,
                                 model: str = "diffusion") -> NFT:
        """
        Mint AI-generated NFT.
        
        Args:
            collection_id: Collection ID
            recipient: Recipient address
            prompt: AI generation prompt
            model: AI model to use
            
        Returns:
            Minted NFT
        """
        # Generate AI art (simulated)
        image_hash = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()
        image_url = f"ipfs://Qm{image_hash[:44]}"
        
        metadata = NFTMetadata(
            name=f"AI Art: {prompt[:30]}...",
            description=f"AI-generated art from prompt: {prompt}",
            image=image_url,
            attributes=[
                {"trait_type": "Generator", "value": model},
                {"trait_type": "Prompt Hash", "value": image_hash[:16]}
            ],
            properties={
                "ai_model": model,
                "prompt": prompt,
                "generation_time": time.time()
            }
        )
        
        return await self.mint(collection_id, recipient, metadata)
    
    async def transfer(self, token_id: str,
                       from_address: str,
                       to_address: str) -> bool:
        """Transfer NFT ownership."""
        if token_id not in self._nfts:
            return False
        
        nft = self._nfts[token_id]
        
        if nft.owner != from_address:
            return False
        
        # Update ownership
        nft.owner = to_address
        
        if from_address in self._ownership:
            self._ownership[from_address].remove(token_id)
        
        if to_address not in self._ownership:
            self._ownership[to_address] = []
        self._ownership[to_address].append(token_id)
        
        logger.info(f"NFT transferred: {token_id} {from_address[:10]}... -> {to_address[:10]}...")
        return True
    
    async def bridge_nft(self, token_id: str,
                          from_chain: str,
                          to_chain: str,
                          recipient: str) -> Dict:
        """
        Bridge NFT to another chain.
        
        Args:
            token_id: NFT token ID
            from_chain: Source chain
            to_chain: Destination chain
            recipient: Recipient on dest chain
            
        Returns:
            Bridge operation details
        """
        if token_id not in self._nfts:
            raise ValueError("NFT not found")
        
        nft = self._nfts[token_id]
        
        if nft.chain != from_chain:
            raise ValueError("NFT not on specified chain")
        
        # Lock on source, mint on destination
        bridge_id = hashlib.sha256(
            f"bridge_{token_id}_{to_chain}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Create wrapped NFT on destination
        wrapped_token_id = f"w{token_id}_{to_chain}"
        
        wrapped_nft = NFT(
            token_id=wrapped_token_id,
            contract_address="0x" + hashlib.sha256(
                f"wrapped_{to_chain}".encode()
            ).hexdigest()[:40],
            chain=to_chain,
            standard=nft.standard,
            owner=recipient,
            metadata=nft.metadata,
            royalty_bps=nft.royalty_bps,
            royalty_recipient=nft.royalty_recipient
        )
        
        self._nfts[wrapped_token_id] = wrapped_nft
        
        logger.info(f"NFT bridged: {token_id} ({from_chain} -> {to_chain})")
        
        return {
            "bridge_id": bridge_id,
            "source_token": token_id,
            "dest_token": wrapped_token_id,
            "from_chain": from_chain,
            "to_chain": to_chain
        }
    
    def get_nft(self, token_id: str) -> Optional[NFT]:
        """Get NFT by token ID."""
        return self._nfts.get(token_id)
    
    def get_owned(self, owner: str) -> List[NFT]:
        """Get NFTs owned by address."""
        token_ids = self._ownership.get(owner, [])
        return [self._nfts[tid] for tid in token_ids if tid in self._nfts]
    
    def get_collection(self, collection_id: str) -> Optional[NFTCollection]:
        """Get collection by ID."""
        return self._collections.get(collection_id)
    
    def get_collection_nfts(self, collection_id: str) -> List[NFT]:
        """Get all NFTs in collection."""
        return [
            nft for nft in self._nfts.values()
            if nft.token_id.startswith(collection_id)
        ]
    
    def generate_metadata_uri(self, metadata: NFTMetadata) -> str:
        """Generate metadata URI (simulated IPFS upload)."""
        metadata_json = json.dumps({
            "name": metadata.name,
            "description": metadata.description,
            "image": metadata.image,
            "external_url": metadata.external_url,
            "animation_url": metadata.animation_url,
            "attributes": metadata.attributes,
            "properties": metadata.properties
        })
        
        hash_value = hashlib.sha256(metadata_json.encode()).hexdigest()
        return f"ipfs://Qm{hash_value[:44]}"
    
    def __repr__(self) -> str:
        return f"NFTManager(collections={len(self._collections)}, nfts={len(self._nfts)})"
