"""
Model Marketplace
=================

Decentralized marketplace for AI models with smart contracts.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class ModelLicense(Enum):
    """Model license types."""
    OPEN = "open"
    COMMERCIAL = "commercial"
    RESEARCH = "research"
    EXCLUSIVE = "exclusive"


class ModelStatus(Enum):
    """Model listing status."""
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


@dataclass
class ModelListing:
    """A model listing in the marketplace."""
    listing_id: str
    model_id: str
    name: str
    description: str
    owner: str
    version: str
    license: ModelLicense
    status: ModelStatus
    price: float
    currency: str
    model_hash: bytes
    metadata: Dict[str, Any]
    created: float
    updated: float
    downloads: int = 0
    rating: float = 0.0
    reviews: int = 0


@dataclass
class PurchaseRecord:
    """Record of a model purchase."""
    purchase_id: str
    listing_id: str
    buyer: str
    price: float
    currency: str
    timestamp: float
    transaction_hash: str
    access_token: str


class ModelMarketplace:
    """
    Decentralized marketplace for AI models.
    
    Features:
    - Model listing and discovery
    - Smart contract-based transactions
    - License management
    - Rating and reviews
    - Version control
    
    Example:
        >>> marketplace = ModelMarketplace()
        >>> listing = await marketplace.list_model(model, metadata)
        >>> purchase = await marketplace.purchase_model(listing.listing_id)
    """
    
    def __init__(self, node_id: Optional[str] = None,
                 language: str = "en"):
        """
        Initialize marketplace.
        
        Args:
            node_id: This node's ID
            language: Language for messages
        """
        self.node_id = node_id or "marketplace"
        self.language = language
        
        # Listings
        self._listings: Dict[str, ModelListing] = {}
        
        # Purchases
        self._purchases: Dict[str, PurchaseRecord] = {}
        
        # User balances (simulated)
        self._balances: Dict[str, float] = {}
        
        # Access tokens
        self._access_tokens: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Model marketplace initialized: {node_id}")
    
    async def list_model(self, model_data: bytes,
                         name: str,
                         description: str,
                         owner: str,
                         version: str = "1.0.0",
                         license_type: ModelLicense = ModelLicense.OPEN,
                         price: float = 0.0,
                         currency: str = "QTOKEN",
                         metadata: Optional[Dict] = None) -> ModelListing:
        """
        List a model in the marketplace.
        
        Args:
            model_data: Serialized model data
            name: Model name
            description: Model description
            owner: Owner identifier
            version: Model version
            license_type: License type
            price: Price in tokens
            currency: Currency type
            metadata: Additional metadata
            
        Returns:
            ModelListing
        """
        listing_id = str(uuid.uuid4())
        model_id = hashlib.sha256(model_data).hexdigest()[:16]
        model_hash = hashlib.sha256(model_data).digest()
        
        listing = ModelListing(
            listing_id=listing_id,
            model_id=model_id,
            name=name,
            description=description,
            owner=owner,
            version=version,
            license=license_type,
            status=ModelStatus.ACTIVE,
            price=price,
            currency=currency,
            model_hash=model_hash,
            metadata=metadata or {},
            created=time.time(),
            updated=time.time()
        )
        
        self._listings[listing_id] = listing
        
        logger.info(f"Listed model: {name} ({listing_id})")
        return listing
    
    async def update_listing(self, listing_id: str,
                              owner: str,
                              **updates) -> Optional[ModelListing]:
        """
        Update a model listing.
        
        Args:
            listing_id: Listing to update
            owner: Owner for verification
            **updates: Fields to update
            
        Returns:
            Updated listing or None
        """
        if listing_id not in self._listings:
            return None
        
        listing = self._listings[listing_id]
        
        if listing.owner != owner:
            logger.warning(f"Unauthorized update attempt: {listing_id}")
            return None
        
        # Update allowed fields
        allowed_fields = ["name", "description", "price", "metadata", "status"]
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(listing, field, value)
        
        listing.updated = time.time()
        
        logger.info(f"Updated listing: {listing_id}")
        return listing
    
    async def purchase_model(self, listing_id: str,
                              buyer: str) -> Optional[PurchaseRecord]:
        """
        Purchase a model.
        
        Args:
            listing_id: Listing to purchase
            buyer: Buyer identifier
            
        Returns:
            PurchaseRecord or None
        """
        if listing_id not in self._listings:
            return None
        
        listing = self._listings[listing_id]
        
        if listing.status != ModelStatus.ACTIVE:
            logger.warning(f"Cannot purchase inactive listing: {listing_id}")
            return None
        
        # Check balance
        buyer_balance = self._balances.get(buyer, 0)
        if buyer_balance < listing.price:
            logger.warning(f"Insufficient balance for {buyer}")
            return None
        
        # Process payment
        if listing.price > 0:
            self._balances[buyer] = buyer_balance - listing.price
            owner_balance = self._balances.get(listing.owner, 0)
            self._balances[listing.owner] = owner_balance + listing.price
        
        # Generate access token
        access_token = hashlib.sha256(
            f"{listing_id}:{buyer}:{time.time()}".encode()
        ).hexdigest()
        
        # Create purchase record
        purchase_id = str(uuid.uuid4())
        transaction_hash = hashlib.sha256(
            f"{purchase_id}:{listing_id}:{buyer}".encode()
        ).hexdigest()
        
        purchase = PurchaseRecord(
            purchase_id=purchase_id,
            listing_id=listing_id,
            buyer=buyer,
            price=listing.price,
            currency=listing.currency,
            timestamp=time.time(),
            transaction_hash=transaction_hash,
            access_token=access_token
        )
        
        self._purchases[purchase_id] = purchase
        
        # Store access token
        self._access_tokens[access_token] = {
            "listing_id": listing_id,
            "buyer": buyer,
            "expires": time.time() + 86400 * 365  # 1 year
        }
        
        # Update listing stats
        listing.downloads += 1
        
        logger.info(f"Model purchased: {listing_id} by {buyer}")
        return purchase
    
    def verify_access(self, access_token: str,
                      listing_id: str) -> bool:
        """
        Verify access to a model.
        
        Args:
            access_token: Access token
            listing_id: Listing ID
            
        Returns:
            True if access is valid
        """
        if access_token not in self._access_tokens:
            return False
        
        token_info = self._access_tokens[access_token]
        
        if token_info["listing_id"] != listing_id:
            return False
        
        if token_info["expires"] < time.time():
            return False
        
        return True
    
    async def search_models(self, query: Optional[str] = None,
                             license_type: Optional[ModelLicense] = None,
                             max_price: Optional[float] = None,
                             min_rating: Optional[float] = None,
                             limit: int = 50) -> List[ModelListing]:
        """
        Search for models.
        
        Args:
            query: Search query
            license_type: Filter by license
            max_price: Maximum price
            min_rating: Minimum rating
            limit: Maximum results
            
        Returns:
            List of matching listings
        """
        results = []
        
        for listing in self._listings.values():
            if listing.status != ModelStatus.ACTIVE:
                continue
            
            # Apply filters
            if query and query.lower() not in listing.name.lower():
                if query.lower() not in listing.description.lower():
                    continue
            
            if license_type and listing.license != license_type:
                continue
            
            if max_price is not None and listing.price > max_price:
                continue
            
            if min_rating is not None and listing.rating < min_rating:
                continue
            
            results.append(listing)
            
            if len(results) >= limit:
                break
        
        # Sort by downloads
        results.sort(key=lambda x: x.downloads, reverse=True)
        
        return results
    
    async def rate_model(self, listing_id: str,
                          buyer: str,
                          rating: float,
                          review: Optional[str] = None) -> bool:
        """
        Rate a purchased model.
        
        Args:
            listing_id: Listing to rate
            buyer: Buyer identifier
            rating: Rating (1-5)
            review: Optional review text
            
        Returns:
            True if rated successfully
        """
        if listing_id not in self._listings:
            return False
        
        # Verify purchase
        has_purchased = any(
            p.listing_id == listing_id and p.buyer == buyer
            for p in self._purchases.values()
        )
        
        if not has_purchased:
            return False
        
        listing = self._listings[listing_id]
        
        # Update rating (simple average)
        total_rating = listing.rating * listing.reviews + rating
        listing.reviews += 1
        listing.rating = total_rating / listing.reviews
        
        logger.info(f"Model rated: {listing_id}, rating={rating}")
        return True
    
    def get_listing(self, listing_id: str) -> Optional[ModelListing]:
        """Get a specific listing."""
        return self._listings.get(listing_id)
    
    def get_user_listings(self, owner: str) -> List[ModelListing]:
        """Get all listings by an owner."""
        return [l for l in self._listings.values() if l.owner == owner]
    
    def get_user_purchases(self, buyer: str) -> List[PurchaseRecord]:
        """Get all purchases by a buyer."""
        return [p for p in self._purchases.values() if p.buyer == buyer]
    
    def add_balance(self, user: str, amount: float):
        """Add balance to user account."""
        current = self._balances.get(user, 0)
        self._balances[user] = current + amount
    
    def get_balance(self, user: str) -> float:
        """Get user balance."""
        return self._balances.get(user, 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        active_listings = sum(
            1 for l in self._listings.values()
            if l.status == ModelStatus.ACTIVE
        )
        
        total_volume = sum(p.price for p in self._purchases.values())
        
        return {
            "total_listings": len(self._listings),
            "active_listings": active_listings,
            "total_purchases": len(self._purchases),
            "total_volume": total_volume,
            "unique_buyers": len(set(p.buyer for p in self._purchases.values())),
            "unique_sellers": len(set(l.owner for l in self._listings.values()))
        }
    
    def __repr__(self) -> str:
        return f"ModelMarketplace(listings={len(self._listings)})"
