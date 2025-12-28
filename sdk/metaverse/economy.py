"""
Metaverse Economy
=================

Decentralized economy for virtual worlds.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """Virtual asset types."""
    LAND = "land"
    BUILDING = "building"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"
    CONSUMABLE = "consumable"
    CURRENCY = "currency"


@dataclass
class VirtualAsset:
    """Virtual world asset."""
    asset_id: str
    asset_type: AssetType
    name: str
    owner: str
    metadata: Dict[str, Any]
    nft_token_id: Optional[str] = None
    nft_contract: Optional[str] = None
    nft_chain: Optional[str] = None
    tradeable: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class Listing:
    """Marketplace listing."""
    listing_id: str
    asset_id: str
    seller: str
    price: int
    currency: str
    listing_type: str  # fixed, auction
    expires_at: float
    created_at: float


@dataclass
class Transaction:
    """Economy transaction."""
    tx_id: str
    tx_type: str
    from_address: str
    to_address: str
    asset_id: Optional[str]
    amount: int
    currency: str
    timestamp: float


class MetaverseEconomy:
    """
    Metaverse economic system.
    
    Features:
    - Virtual asset ownership
    - NFT integration
    - Marketplace
    - Currency exchange
    - Land and real estate
    - Creator royalties
    
    Example:
        >>> economy = MetaverseEconomy()
        >>> asset = economy.create_asset(AssetType.LAND, "Plot #1", owner)
        >>> listing = economy.list_for_sale(asset.asset_id, price=1000)
    """
    
    def __init__(self, native_currency: str = "AIP"):
        """
        Initialize economy.
        
        Args:
            native_currency: Native currency symbol
        """
        self.native_currency = native_currency
        
        # Assets
        self._assets: Dict[str, VirtualAsset] = {}
        
        # Listings
        self._listings: Dict[str, Listing] = {}
        
        # Balances
        self._balances: Dict[str, Dict[str, int]] = {}
        
        # Transactions
        self._transactions: List[Transaction] = []
        
        # Exchange rates
        self._exchange_rates: Dict[str, float] = {
            "AIP": 1.0,
            "ETH": 2000.0,
            "USDC": 1.0
        }
        
        logger.info(f"Metaverse Economy initialized (currency: {native_currency})")
    
    def create_asset(self, asset_type: AssetType,
                     name: str,
                     owner: str,
                     metadata: Dict = None,
                     mint_nft: bool = False,
                     nft_chain: str = "ethereum") -> VirtualAsset:
        """
        Create virtual asset.
        
        Args:
            asset_type: Type of asset
            name: Asset name
            owner: Owner address
            metadata: Asset metadata
            mint_nft: Whether to mint as NFT
            nft_chain: Blockchain for NFT
            
        Returns:
            Created asset
        """
        asset_id = hashlib.sha256(
            f"{asset_type.value}_{name}_{owner}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        asset = VirtualAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            name=name,
            owner=owner,
            metadata=metadata or {}
        )
        
        if mint_nft:
            asset.nft_token_id = asset_id
            asset.nft_contract = f"0x{'a' * 40}"  # Simulated
            asset.nft_chain = nft_chain
        
        self._assets[asset_id] = asset
        
        logger.info(f"Asset created: {name} ({asset_type.value})")
        return asset
    
    def transfer_asset(self, asset_id: str,
                       from_address: str,
                       to_address: str) -> bool:
        """Transfer asset ownership."""
        if asset_id not in self._assets:
            return False
        
        asset = self._assets[asset_id]
        
        if asset.owner != from_address:
            return False
        
        asset.owner = to_address
        
        self._record_transaction(
            "transfer",
            from_address,
            to_address,
            asset_id=asset_id,
            amount=0,
            currency=""
        )
        
        logger.info(f"Asset transferred: {asset_id}")
        return True
    
    def get_balance(self, address: str,
                    currency: str = None) -> Dict[str, int]:
        """Get account balances."""
        if address not in self._balances:
            self._balances[address] = {self.native_currency: 0}
        
        if currency:
            return {currency: self._balances[address].get(currency, 0)}
        
        return self._balances[address].copy()
    
    def mint_currency(self, address: str,
                      amount: int,
                      currency: str = None):
        """Mint currency to address."""
        currency = currency or self.native_currency
        
        if address not in self._balances:
            self._balances[address] = {}
        
        current = self._balances[address].get(currency, 0)
        self._balances[address][currency] = current + amount
        
        self._record_transaction(
            "mint", "system", address,
            amount=amount, currency=currency
        )
    
    def transfer_currency(self, from_address: str,
                          to_address: str,
                          amount: int,
                          currency: str = None) -> bool:
        """Transfer currency between addresses."""
        currency = currency or self.native_currency
        
        from_balance = self.get_balance(from_address, currency).get(currency, 0)
        
        if from_balance < amount:
            return False
        
        if to_address not in self._balances:
            self._balances[to_address] = {}
        
        self._balances[from_address][currency] -= amount
        self._balances[to_address][currency] = \
            self._balances[to_address].get(currency, 0) + amount
        
        self._record_transaction(
            "transfer", from_address, to_address,
            amount=amount, currency=currency
        )
        
        return True
    
    def list_for_sale(self, asset_id: str,
                      price: int,
                      currency: str = None,
                      expires_in: int = 86400,
                      listing_type: str = "fixed") -> Listing:
        """
        List asset for sale.
        
        Args:
            asset_id: Asset to sell
            price: Sale price
            currency: Price currency
            expires_in: Expiration in seconds
            listing_type: fixed or auction
            
        Returns:
            Listing
        """
        if asset_id not in self._assets:
            raise ValueError("Asset not found")
        
        listing_id = hashlib.sha256(
            f"listing_{asset_id}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        asset = self._assets[asset_id]
        
        listing = Listing(
            listing_id=listing_id,
            asset_id=asset_id,
            seller=asset.owner,
            price=price,
            currency=currency or self.native_currency,
            listing_type=listing_type,
            expires_at=time.time() + expires_in,
            created_at=time.time()
        )
        
        self._listings[listing_id] = listing
        
        logger.info(f"Listed for sale: {asset.name} at {price} {listing.currency}")
        return listing
    
    def buy_listing(self, listing_id: str, buyer: str) -> bool:
        """
        Buy listed asset.
        
        Args:
            listing_id: Listing ID
            buyer: Buyer address
            
        Returns:
            True if successful
        """
        if listing_id not in self._listings:
            return False
        
        listing = self._listings[listing_id]
        
        # Check not expired
        if time.time() > listing.expires_at:
            del self._listings[listing_id]
            return False
        
        # Check buyer has funds
        balance = self.get_balance(buyer, listing.currency).get(listing.currency, 0)
        if balance < listing.price:
            return False
        
        # Execute purchase
        self.transfer_currency(
            buyer, listing.seller,
            listing.price, listing.currency
        )
        
        self.transfer_asset(
            listing.asset_id,
            listing.seller, buyer
        )
        
        del self._listings[listing_id]
        
        logger.info(f"Purchase completed: {listing_id}")
        return True
    
    def cancel_listing(self, listing_id: str, seller: str) -> bool:
        """Cancel marketplace listing."""
        if listing_id not in self._listings:
            return False
        
        listing = self._listings[listing_id]
        
        if listing.seller != seller:
            return False
        
        del self._listings[listing_id]
        return True
    
    def exchange_currency(self, from_currency: str,
                          to_currency: str,
                          amount: int,
                          address: str) -> int:
        """
        Exchange between currencies.
        
        Args:
            from_currency: Source currency
            to_currency: Target currency
            amount: Amount to exchange
            address: Account address
            
        Returns:
            Amount received
        """
        balance = self.get_balance(address, from_currency).get(from_currency, 0)
        
        if balance < amount:
            raise ValueError("Insufficient balance")
        
        from_rate = self._exchange_rates.get(from_currency, 1.0)
        to_rate = self._exchange_rates.get(to_currency, 1.0)
        
        received = int(amount * from_rate / to_rate)
        
        self._balances[address][from_currency] -= amount
        self._balances[address][to_currency] = \
            self._balances[address].get(to_currency, 0) + received
        
        return received
    
    def _record_transaction(self, tx_type: str,
                             from_addr: str,
                             to_addr: str,
                             asset_id: str = None,
                             amount: int = 0,
                             currency: str = ""):
        """Record transaction."""
        tx = Transaction(
            tx_id=hashlib.sha256(f"{tx_type}_{time.time()}".encode()).hexdigest()[:16],
            tx_type=tx_type,
            from_address=from_addr,
            to_address=to_addr,
            asset_id=asset_id,
            amount=amount,
            currency=currency,
            timestamp=time.time()
        )
        self._transactions.append(tx)
    
    def get_listings(self, asset_type: AssetType = None) -> List[Listing]:
        """Get active listings."""
        listings = list(self._listings.values())
        
        if asset_type:
            listings = [
                l for l in listings
                if self._assets.get(l.asset_id, VirtualAsset).asset_type == asset_type
            ]
        
        return listings
    
    def get_assets_by_owner(self, owner: str) -> List[VirtualAsset]:
        """Get assets owned by address."""
        return [a for a in self._assets.values() if a.owner == owner]
    
    def __repr__(self) -> str:
        return f"MetaverseEconomy(assets={len(self._assets)}, listings={len(self._listings)})"
