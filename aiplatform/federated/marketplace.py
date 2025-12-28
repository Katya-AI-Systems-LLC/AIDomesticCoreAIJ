"""
Federated Model Marketplace for AIPlatform SDK

This module provides a marketplace for federated models with
support for NFT-based model weights and smart contract integration.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from ..exceptions import FederatedError
from .model import FederatedModel, ModelMetadata

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelStatus(Enum):
    """Model status in marketplace."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"

class TransactionType(Enum):
    """Transaction type in marketplace."""
    LISTING = "listing"
    PURCHASE = "purchase"
    TRANSFER = "transfer"
    LICENSE = "license"

@dataclass
class ModelListing:
    """Model listing in marketplace."""
    listing_id: str
    model_id: str
    seller_id: str
    price: float
    currency: str
    description: str
    tags: List[str]
    created_at: datetime
    status: ModelStatus
    metadata: Dict[str, Any]

@dataclass
class Transaction:
    """Transaction in marketplace."""
    transaction_id: str
    listing_id: str
    buyer_id: str
    seller_id: str
    amount: float
    currency: str
    transaction_type: TransactionType
    timestamp: datetime
    status: str
    metadata: Dict[str, Any]

@dataclass
class ModelNFT:
    """NFT representation of model weights."""
    nft_id: str
    model_id: str
    owner_id: str
    metadata_uri: str
    created_at: datetime
    transaction_history: List[str]

class ModelMarketplace:
    """
    Federated Model Marketplace.
    
    Provides a marketplace for federated models with NFT-based
    model weights and smart contract integration.
    """
    
    def __init__(self, blockchain_provider: Optional[Any] = None):
        """
        Initialize model marketplace.
        
        Args:
            blockchain_provider (Any, optional): Blockchain provider for NFT integration
        """
        self._blockchain_provider = blockchain_provider
        self._listings = {}
        self._transactions = {}
        self._nfts = {}
        self._models = {}
        self._participants = set()
        
        logger.info("Model marketplace initialized")
    
    def list_model(self, model: FederatedModel, seller_id: str, 
                   price: float, currency: str = "USD", 
                   description: str = "", tags: Optional[List[str]] = None) -> str:
        """
        List model in marketplace.
        
        Args:
            model (FederatedModel): Model to list
            seller_id (str): Seller identifier
            price (float): Listing price
            currency (str): Currency for listing
            description (str): Model description
            tags (list): Model tags
            
        Returns:
            str: Listing ID
        """
        try:
            # Validate inputs
            if not model:
                raise ValueError("Model cannot be None")
            
            if price < 0:
                raise ValueError("Price cannot be negative")
            
            # Generate listing ID
            listing_id = self._generate_listing_id()
            
            # Create listing
            listing = ModelListing(
                listing_id=listing_id,
                model_id=model.model_id,
                seller_id=seller_id,
                price=price,
                currency=currency,
                description=description or model.get_metadata().description,
                tags=tags or model.get_metadata().tags,
                created_at=datetime.now(),
                status=ModelStatus.PENDING,
                metadata={
                    "model_type": model.get_metadata().model_type,
                    "version": model.get_metadata().version,
                    "architecture": model.get_metadata().architecture
                }
            )
            
            # Store listing
            self._listings[listing_id] = listing
            self._models[model.model_id] = model
            self._participants.add(seller_id)
            
            # Create NFT for model weights
            nft_id = self._create_model_nft(model, seller_id)
            
            # Update listing status
            listing.status = ModelStatus.ACTIVE
            
            logger.info(f"Model {model.model_id} listed in marketplace as {listing_id}")
            return listing_id
            
        except Exception as e:
            logger.error(f"Failed to list model: {e}")
            raise FederatedError(f"Model listing failed: {e}")
    
    def _generate_listing_id(self) -> str:
        """Generate unique listing ID."""
        import uuid
        return f"listing_{uuid.uuid4().hex[:12]}"
    
    def _create_model_nft(self, model: FederatedModel, owner_id: str) -> str:
        """
        Create NFT for model weights.
        
        Args:
            model (FederatedModel): Model to create NFT for
            owner_id (str): Owner identifier
            
        Returns:
            str: NFT ID
        """
        try:
            # Generate NFT ID
            nft_id = f"nft_{model.model_id}_{datetime.now().timestamp()}"
            
            # Create metadata URI (in a real implementation, this would be IPFS hash)
            metadata_uri = f"ipfs://model_metadata/{model.model_id}"
            
            # Create NFT
            nft = ModelNFT(
                nft_id=nft_id,
                model_id=model.model_id,
                owner_id=owner_id,
                metadata_uri=metadata_uri,
                created_at=datetime.now(),
                transaction_history=[]
            )
            
            # Store NFT
            self._nfts[nft_id] = nft
            
            # Mint NFT on blockchain (simulated)
            if self._blockchain_provider:
                try:
                    # In a real implementation, this would interact with blockchain
                    # self._blockchain_provider.mint_nft(nft_id, owner_id, metadata_uri)
                    pass
                except Exception as e:
                    logger.warning(f"Failed to mint NFT on blockchain: {e}")
            
            logger.debug(f"NFT {nft_id} created for model {model.model_id}")
            return nft_id
            
        except Exception as e:
            logger.error(f"Failed to create model NFT: {e}")
            raise FederatedError(f"NFT creation failed: {e}")
    
    def purchase_model(self, listing_id: str, buyer_id: str, 
                      payment_method: str = "crypto") -> bool:
        """
        Purchase model from marketplace.
        
        Args:
            listing_id (str): Listing identifier
            buyer_id (str): Buyer identifier
            payment_method (str): Payment method
            
        Returns:
            bool: True if purchase successful, False otherwise
        """
        try:
            # Validate listing
            if listing_id not in self._listings:
                raise ValueError(f"Listing {listing_id} not found")
            
            listing = self._listings[listing_id]
            if listing.status != ModelStatus.ACTIVE:
                raise ValueError(f"Listing {listing_id} is not active")
            
            # Create transaction
            transaction_id = self._generate_transaction_id()
            
            transaction = Transaction(
                transaction_id=transaction_id,
                listing_id=listing_id,
                buyer_id=buyer_id,
                seller_id=listing.seller_id,
                amount=listing.price,
                currency=listing.currency,
                transaction_type=TransactionType.PURCHASE,
                timestamp=datetime.now(),
                status="pending",
                metadata={
                    "payment_method": payment_method,
                    "model_id": listing.model_id
                }
            )
            
            # Store transaction
            self._transactions[transaction_id] = transaction
            self._participants.add(buyer_id)
            
            # Process payment (simulated)
            if self._process_payment(transaction, payment_method):
                # Transfer model ownership
                self._transfer_model_ownership(listing.model_id, buyer_id)
                
                # Update transaction status
                transaction.status = "completed"
                
                # Transfer NFT ownership
                self._transfer_nft_ownership(listing.model_id, buyer_id)
                
                # Update listing status
                listing.status = ModelStatus.ARCHIVED
                
                logger.info(f"Model {listing.model_id} purchased by {buyer_id}")
                return True
            else:
                transaction.status = "failed"
                logger.error(f"Payment failed for transaction {transaction_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to purchase model: {e}")
            raise FederatedError(f"Model purchase failed: {e}")
    
    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        import uuid
        return f"tx_{uuid.uuid4().hex[:16]}"
    
    def _process_payment(self, transaction: Transaction, payment_method: str) -> bool:
        """
        Process payment for transaction.
        
        Args:
            transaction (Transaction): Transaction to process
            payment_method (str): Payment method
            
        Returns:
            bool: True if payment successful, False otherwise
        """
        try:
            # In a real implementation, this would interact with payment systems
            # For simulation, we'll assume payment is successful
            logger.debug(f"Processing payment for transaction {transaction.transaction_id} "
                        f"using {payment_method}")
            
            # Simulate payment processing delay
            import time
            time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process payment: {e}")
            return False
    
    def _transfer_model_ownership(self, model_id: str, new_owner_id: str):
        """
        Transfer model ownership.
        
        Args:
            model_id (str): Model identifier
            new_owner_id (str): New owner identifier
        """
        try:
            if model_id in self._models:
                model = self._models[model_id]
                metadata = model.get_metadata()
                metadata_updates = {
                    "owner": new_owner_id,
                    "updated_at": datetime.now()
                }
                model.update_metadata(metadata_updates)
                
                logger.debug(f"Model {model_id} ownership transferred to {new_owner_id}")
            
        except Exception as e:
            logger.error(f"Failed to transfer model ownership: {e}")
    
    def _transfer_nft_ownership(self, model_id: str, new_owner_id: str):
        """
        Transfer NFT ownership.
        
        Args:
            model_id (str): Model identifier
            new_owner_id (str): New owner identifier
        """
        try:
            # Find NFT for model
            nft_to_update = None
            for nft in self._nfts.values():
                if nft.model_id == model_id:
                    nft_to_update = nft
                    break
            
            if nft_to_update:
                # Update ownership
                old_owner = nft_to_update.owner_id
                nft_to_update.owner_id = new_owner_id
                nft_to_update.transaction_history.append(
                    f"transfer_{old_owner}_to_{new_owner_id}_{datetime.now().timestamp()}"
                )
                
                # Update on blockchain (simulated)
                if self._blockchain_provider:
                    try:
                        # In a real implementation, this would interact with blockchain
                        # self._blockchain_provider.transfer_nft(nft_to_update.nft_id, new_owner_id)
                        pass
                    except Exception as e:
                        logger.warning(f"Failed to transfer NFT on blockchain: {e}")
                
                logger.debug(f"NFT {nft_to_update.nft_id} ownership transferred to {new_owner_id}")
            
        except Exception as e:
            logger.error(f"Failed to transfer NFT ownership: {e}")
    
    def get_listing(self, listing_id: str) -> Optional[ModelListing]:
        """
        Get model listing.
        
        Args:
            listing_id (str): Listing identifier
            
        Returns:
            ModelListing: Model listing or None if not found
        """
        return self._listings.get(listing_id)
    
    def get_listings(self, status: Optional[ModelStatus] = None, 
                    tags: Optional[List[str]] = None) -> List[ModelListing]:
        """
        Get model listings.
        
        Args:
            status (ModelStatus, optional): Filter by status
            tags (list, optional): Filter by tags
            
        Returns:
            list: List of model listings
        """
        listings = list(self._listings.values())
        
        # Filter by status
        if status:
            listings = [l for l in listings if l.status == status]
        
        # Filter by tags
        if tags:
            listings = [l for l in listings if any(tag in l.tags for tag in tags)]
        
        return listings
    
    def get_model_nft(self, model_id: str) -> Optional[ModelNFT]:
        """
        Get model NFT.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            ModelNFT: Model NFT or None if not found
        """
        for nft in self._nfts.values():
            if nft.model_id == model_id:
                return nft
        return None
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """
        Get transaction.
        
        Args:
            transaction_id (str): Transaction identifier
            
        Returns:
            Transaction: Transaction or None if not found
        """
        return self._transactions.get(transaction_id)
    
    def get_transactions(self, user_id: str) -> List[Transaction]:
        """
        Get transactions for user.
        
        Args:
            user_id (str): User identifier
            
        Returns:
            list: List of transactions
        """
        return [t for t in self._transactions.values() 
                if t.buyer_id == user_id or t.seller_id == user_id]
    
    def get_participants(self) -> List[str]:
        """
        Get marketplace participants.
        
        Returns:
            list: List of participant IDs
        """
        return list(self._participants)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get marketplace statistics.
        
        Returns:
            dict: Marketplace statistics
        """
        active_listings = [l for l in self._listings.values() 
                         if l.status == ModelStatus.ACTIVE]
        completed_transactions = [t for t in self._transactions.values() 
                                if t.status == "completed"]
        
        total_value = sum(t.amount for t in completed_transactions)
        
        return {
            "total_listings": len(self._listings),
            "active_listings": len(active_listings),
            "total_transactions": len(self._transactions),
            "completed_transactions": len(completed_transactions),
            "total_value": total_value,
            "total_participants": len(self._participants),
            "total_models": len(self._models)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert marketplace to dictionary representation.
        
        Returns:
            dict: Dictionary representation of marketplace
        """
        return {
            "listings": {lid: asdict(listing) for lid, listing in self._listings.items()},
            "transactions": {tid: asdict(tx) for tid, tx in self._transactions.items()},
            "nfts": {nid: asdict(nft) for nid, nft in self._nfts.items()},
            "participants": list(self._participants),
            "statistics": self.get_statistics()
        }
    
    def save_marketplace(self, filepath: str) -> bool:
        """
        Save marketplace to file.
        
        Args:
            filepath (str): Path to save marketplace
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            import pickle
            
            # Create saveable representation
            marketplace_data = {
                "listings": self._listings,
                "transactions": self._transactions,
                "nfts": self._nfts,
                "models": {mid: model.to_dict() for mid, model in self._models.items()},
                "participants": self._participants
            }
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(marketplace_data, f)
            
            logger.info(f"Marketplace saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save marketplace: {e}")
            return False
    
    @classmethod
    def load_marketplace(cls, filepath: str) -> 'ModelMarketplace':
        """
        Load marketplace from file.
        
        Args:
            filepath (str): Path to marketplace file
            
        Returns:
            ModelMarketplace: Loaded marketplace
        """
        try:
            import pickle
            
            # Load from file
            with open(filepath, 'rb') as f:
                marketplace_data = pickle.load(f)
            
            # Create new marketplace
            marketplace = cls()
            marketplace._listings = marketplace_data.get("listings", {})
            marketplace._transactions = marketplace_data.get("transactions", {})
            marketplace._nfts = marketplace_data.get("nfts", {})
            marketplace._participants = set(marketplace_data.get("participants", []))
            
            # Load models (simplified)
            models_data = marketplace_data.get("models", {})
            for model_id, model_dict in models_data.items():
                # In a real implementation, this would reconstruct the model
                # For now, we'll just store the dictionary representation
                pass
            
            logger.info(f"Marketplace loaded from {filepath}")
            return marketplace
            
        except Exception as e:
            logger.error(f"Failed to load marketplace: {e}")
            raise FederatedError(f"Marketplace loading failed: {e}")

# Utility functions for marketplace
def create_marketplace(blockchain_provider: Optional[Any] = None) -> ModelMarketplace:
    """
    Create model marketplace.
    
    Args:
        blockchain_provider (Any, optional): Blockchain provider for NFT integration
        
    Returns:
        ModelMarketplace: Created marketplace
    """
    return ModelMarketplace(blockchain_provider)

def list_federated_model(marketplace: ModelMarketplace, model: FederatedModel, 
                        seller_id: str, price: float, currency: str = "USD",
                        description: str = "", tags: Optional[List[str]] = None) -> str:
    """
    List federated model in marketplace.
    
    Args:
        marketplace (ModelMarketplace): Marketplace instance
        model (FederatedModel): Model to list
        seller_id (str): Seller identifier
        price (float): Listing price
        currency (str): Currency for listing
        description (str): Model description
        tags (list): Model tags
        
    Returns:
        str: Listing ID
    """
    return marketplace.list_model(model, seller_id, price, currency, description, tags)

def purchase_model(marketplace: ModelMarketplace, listing_id: str, 
                  buyer_id: str, payment_method: str = "crypto") -> bool:
    """
    Purchase model from marketplace.
    
    Args:
        marketplace (ModelMarketplace): Marketplace instance
        listing_id (str): Listing identifier
        buyer_id (str): Buyer identifier
        payment_method (str): Payment method
        
    Returns:
        bool: True if purchase successful, False otherwise
    """
    return marketplace.purchase_model(listing_id, buyer_id, payment_method)

# Example usage
def example_marketplace():
    """Example of model marketplace usage."""
    # Create marketplace
    marketplace = ModelMarketplace()
    
    # Create dummy federated model
    class DummyBaseModel:
        def __init__(self):
            pass
    
    dummy_model = DummyBaseModel()
    
    federated_model = FederatedModel(
        base_model=dummy_model,
        federation_config={
            "model_id": "example_model_001",
            "model_type": "neural_network",
            "version": "1.0.0",
            "owner": "example_user",
            "description": "Example federated neural network",
            "tags": ["example", "neural_network", "federated"]
        }
    )
    
    # List model in marketplace
    listing_id = marketplace.list_model(
        model=federated_model,
        seller_id="seller_001",
        price=100.0,
        currency="USD",
        description="Example federated model for demonstration",
        tags=["example", "neural_network", "demo"]
    )
    
    print(f"Model listed with ID: {listing_id}")
    
    # Get listing
    listing = marketplace.get_listing(listing_id)
    print(f"Listing details: {listing}")
    
    # Get marketplace statistics
    stats = marketplace.get_statistics()
    print(f"Marketplace statistics: {stats}")
    
    # Convert to dictionary
    marketplace_dict = marketplace.to_dict()
    print(f"Marketplace dictionary keys: {list(marketplace_dict.keys())}")
    
    return marketplace