"""
Decentralized Identifiers (DIDs) and DIDN module for AIPlatform SDK

This module provides implementation of Decentralized Identifiers (DIDs)
and the DIDN (Decentralized Identity Network) protocol.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import base64

from ..exceptions import SecurityError
from .crypto import Dilithium

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class DIDDocument:
    """DID Document representation."""
    id: str
    context: List[str]
    authentication: List[str]
    assertion_method: List[str]
    key_agreement: List[str]
    capability_invocation: List[str]
    capability_delegation: List[str]
    service: List[Dict[str, Any]]
    created: datetime
    updated: datetime
    proof: Optional[Dict[str, Any]] = None

@dataclass
class DIDNRecord:
    """DIDN (Decentralized Identity Network) record."""
    did: str
    document: DIDDocument
    signature: str
    timestamp: datetime
    version: int
    previous_version: Optional[str] = None

class DIDN:
    """
    Decentralized Identity Network implementation.
    
    Provides decentralized identity management using DID standards
    and quantum-safe cryptography.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DIDN.
        
        Args:
            config (dict, optional): DIDN configuration
        """
        self._config = config or {}
        self._is_initialized = False
        self._dids = {}
        self._resolver = None
        self._signer = None
        
        # Initialize DIDN
        self._initialize_didn()
        
        logger.info("DIDN initialized")
    
    def _initialize_didn(self):
        """Initialize DIDN system."""
        try:
            # In a real implementation, this would initialize the DIDN network
            # For simulation, we'll create placeholder information
            self._didn_info = {
                "network": "didn",
                "version": "1.0.0",
                "status": "initialized",
                "capabilities": ["did_management", "decentralized_identity", "quantum_safe"]
            }
            
            # Initialize crypto signer
            self._signer = Dilithium()
            self._signer.keygen()
            
            self._is_initialized = True
            logger.debug("DIDN initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DIDN: {e}")
            raise SecurityError(f"DIDN initialization failed: {e}")
    
    def create_did(self, method: str = "key", public_key: Optional[bytes] = None) -> str:
        """
        Create new DID.
        
        Args:
            method (str): DID method (default: "key")
            public_key (bytes, optional): Public key for DID
            
        Returns:
            str: Created DID
        """
        try:
            if not self._is_initialized:
                raise SecurityError("DIDN not initialized")
            
            # Generate DID
            if public_key is None:
                # Generate new key pair
                keypair = self._signer.keygen()
                public_key = keypair.public_key
            
            # Create DID using public key
            did = self._generate_did_from_key(method, public_key)
            
            # Create DID document
            did_document = self._create_did_document(did, public_key)
            
            # Sign document
            signature = self._sign_document(did_document, public_key)
            
            # Create DIDN record
            didn_record = DIDNRecord(
                did=did,
                document=did_document,
                signature=signature,
                timestamp=datetime.now(),
                version=1
            )
            
            # Store DID
            self._dids[did] = didn_record
            
            logger.debug(f"DID created: {did}")
            return did
            
        except Exception as e:
            logger.error(f"Failed to create DID: {e}")
            raise SecurityError(f"DID creation failed: {e}")
    
    def _generate_did_from_key(self, method: str, public_key: bytes) -> str:
        """
        Generate DID from public key.
        
        Args:
            method (str): DID method
            public_key (bytes): Public key
            
        Returns:
            str: Generated DID
        """
        # Create multibase encoding of public key
        multibase_key = base64.b32encode(public_key).decode('utf-8').rstrip('=')
        
        # Create DID
        did = f"did:{method}:{multibase_key[:32]}"  # Truncate for readability
        return did
    
    def _create_did_document(self, did: str, public_key: bytes) -> DIDDocument:
        """
        Create DID document.
        
        Args:
            did (str): DID
            public_key (bytes): Public key
            
        Returns:
            DIDDocument: Created DID document
        """
        # Create key identifier
        key_id = f"{did}#keys-1"
        
        return DIDDocument(
            id=did,
            context=["https://www.w3.org/ns/did/v1"],
            authentication=[key_id],
            assertion_method=[key_id],
            key_agreement=[key_id],
            capability_invocation=[key_id],
            capability_delegation=[key_id],
            service=[],
            created=datetime.now(),
            updated=datetime.now()
        )
    
    def _sign_document(self, document: DIDDocument, public_key: bytes) -> str:
        """
        Sign DID document.
        
        Args:
            document (DIDDocument): Document to sign
            public_key (bytes): Public key for signing
            
        Returns:
            str: Digital signature
        """
        try:
            # Convert document to JSON for signing
            doc_dict = {
                "id": document.id,
                "context": document.context,
                "authentication": document.authentication,
                "created": document.created.isoformat(),
                "updated": document.updated.isoformat()
            }
            
            doc_json = json.dumps(doc_dict, sort_keys=True)
            doc_bytes = doc_json.encode('utf-8')
            
            # Sign with Dilithium
            signature = self._signer.sign(doc_bytes)
            return signature.hex()
            
        except Exception as e:
            logger.error(f"Failed to sign document: {e}")
            return "invalid_signature"
    
    def resolve_did(self, did: str) -> Optional[DIDDocument]:
        """
        Resolve DID to DID document.
        
        Args:
            did (str): DID to resolve
            
        Returns:
            DIDDocument: Resolved DID document or None if not found
        """
        try:
            if not self._is_initialized:
                raise SecurityError("DIDN not initialized")
            
            if did not in self._dids:
                logger.warning(f"DID not found: {did}")
                return None
            
            didn_record = self._dids[did]
            return didn_record.document
            
        except Exception as e:
            logger.error(f"Failed to resolve DID: {e}")
            return None
    
    def update_did_document(self, did: str, new_document: DIDDocument) -> bool:
        """
        Update DID document.
        
        Args:
            did (str): DID to update
            new_document (DIDDocument): New DID document
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                raise SecurityError("DIDN not initialized")
            
            if did not in self._dids:
                raise SecurityError(f"DID not found: {did}")
            
            # Get current record
            current_record = self._dids[did]
            
            # Sign new document
            signature = self._sign_document(new_document, 
                                         self._signer.get_keypair().public_key)
            
            # Create new record with incremented version
            new_record = DIDNRecord(
                did=did,
                document=new_document,
                signature=signature,
                timestamp=datetime.now(),
                version=current_record.version + 1,
                previous_version=hashlib.sha256(
                    json.dumps(current_record.__dict__, default=str).encode()
                ).hexdigest()
            )
            
            # Update storage
            self._dids[did] = new_record
            
            logger.debug(f"DID document updated: {did} (v{new_record.version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update DID document: {e}")
            return False
    
    def add_service_endpoint(self, did: str, service: Dict[str, Any]) -> bool:
        """
        Add service endpoint to DID document.
        
        Args:
            did (str): DID
            service (dict): Service endpoint information
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                raise SecurityError("DIDN not initialized")
            
            if did not in self._dids:
                raise SecurityError(f"DID not found: {did}")
            
            # Get current document
            current_record = self._dids[did]
            current_document = current_record.document
            
            # Add service
            new_services = current_document.service + [service]
            
            # Create updated document
            updated_document = DIDDocument(
                id=current_document.id,
                context=current_document.context,
                authentication=current_document.authentication,
                assertion_method=current_document.assertion_method,
                key_agreement=current_document.key_agreement,
                capability_invocation=current_document.capability_invocation,
                capability_delegation=current_document.capability_delegation,
                service=new_services,
                created=current_document.created,
                updated=datetime.now()
            )
            
            # Update document
            return self.update_did_document(did, updated_document)
            
        except Exception as e:
            logger.error(f"Failed to add service endpoint: {e}")
            return False
    
    def verify_did_document(self, did: str, document: DIDDocument) -> bool:
        """
        Verify DID document signature.
        
        Args:
            did (str): DID
            document (DIDDocument): Document to verify
            
        Returns:
            bool: True if verified successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                raise SecurityError("DIDN not initialized")
            
            if did not in self._dids:
                raise SecurityError(f"DID not found: {did}")
            
            # Get stored record
            stored_record = self._dids[did]
            
            # Verify signature
            signature = self._sign_document(document, 
                                         self._signer.get_keypair().public_key)
            
            return signature == stored_record.signature
            
        except Exception as e:
            logger.error(f"Failed to verify DID document: {e}")
            return False
    
    def get_did_record(self, did: str) -> Optional[DIDNRecord]:
        """
        Get DIDN record.
        
        Args:
            did (str): DID
            
        Returns:
            DIDNRecord: DIDN record or None if not found
        """
        return self._dids.get(did)
    
    def list_dids(self) -> List[str]:
        """
        List all DIDs.
        
        Returns:
            list: List of DIDs
        """
        return list(self._dids.keys())
    
    def get_did_info(self, did: str) -> Optional[Dict[str, Any]]:
        """
        Get DID information.
        
        Args:
            did (str): DID
            
        Returns:
            dict: DID information or None if not found
        """
        if did not in self._dids:
            return None
        
        record = self._dids[did]
        return {
            "did": record.did,
            "version": record.version,
            "timestamp": record.timestamp.isoformat(),
            "services": len(record.document.service),
            "has_signature": record.signature is not None
        }
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get DIDN network information.
        
        Returns:
            dict: Network information
        """
        return {
            "initialized": self._is_initialized,
            "did_count": len(self._dids),
            "network": self._didn_info,
            "signer_available": self._signer is not None
        }

class DIDNResolver:
    """
    DIDN Resolver implementation.
    
    Provides resolution of DIDs to DID documents across
    multiple DID methods and networks.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DIDN resolver.
        
        Args:
            config (dict, optional): Resolver configuration
        """
        self._config = config or {}
        self._is_initialized = False
        self._resolvers = {}
        self._cache = {}
        self._cache_ttl = self._config.get("cache_ttl", 3600)  # 1 hour default
        
        # Initialize resolver
        self._initialize_resolver()
        
        logger.info("DIDN resolver initialized")
    
    def _initialize_resolver(self):
        """Initialize resolver system."""
        try:
            # In a real implementation, this would initialize the resolver network
            # For simulation, we'll create placeholder information
            self._resolver_info = {
                "resolver": "didn-resolver",
                "version": "1.0.0",
                "status": "initialized",
                "supported_methods": ["key", "web", "ethr", "didn"]
            }
            
            self._is_initialized = True
            logger.debug("DIDN resolver initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize resolver: {e}")
            raise SecurityError(f"Resolver initialization failed: {e}")
    
    def resolve(self, did: str) -> Optional[DIDDocument]:
        """
        Resolve DID to DID document.
        
        Args:
            did (str): DID to resolve
            
        Returns:
            DIDDocument: Resolved DID document or None if not found
        """
        try:
            if not self._is_initialized:
                raise SecurityError("Resolver not initialized")
            
            # Check cache first
            if did in self._cache:
                cached_entry = self._cache[did]
                if (datetime.now() - cached_entry["timestamp"]).total_seconds() < self._cache_ttl:
                    logger.debug(f"Cache hit for DID: {did}")
                    return cached_entry["document"]
                else:
                    # Remove expired cache entry
                    del self._cache[did]
            
            # Parse DID method
            if not did.startswith("did:"):
                raise SecurityError("Invalid DID format")
            
            method = did.split(":")[1] if len(did.split(":")) > 1 else "unknown"
            
            # Resolve using appropriate method
            document = self._resolve_by_method(did, method)
            
            if document:
                # Cache result
                self._cache[did] = {
                    "document": document,
                    "timestamp": datetime.now()
                }
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to resolve DID: {e}")
            return None
    
    def _resolve_by_method(self, did: str, method: str) -> Optional[DIDDocument]:
        """
        Resolve DID by method.
        
        Args:
            did (str): DID to resolve
            method (str): DID method
            
        Returns:
            DIDDocument: Resolved DID document or None if not found
        """
        # In a real implementation, this would use actual method resolvers
        # For simulation, we'll create a placeholder document for known methods
        
        if method in ["key", "web", "ethr", "didn"]:
            # Create placeholder document
            return DIDDocument(
                id=did,
                context=["https://www.w3.org/ns/did/v1"],
                authentication=[f"{did}#key-1"],
                assertion_method=[f"{did}#key-1"],
                key_agreement=[f"{did}#key-1"],
                capability_invocation=[f"{did}#key-1"],
                capability_delegation=[f"{did}#key-1"],
                service=[],
                created=datetime.now(),
                updated=datetime.now()
            )
        else:
            logger.warning(f"Unsupported DID method: {method}")
            return None
    
    def register_resolver(self, method: str, resolver: Any) -> bool:
        """
        Register custom DID method resolver.
        
        Args:
            method (str): DID method
            resolver (Any): Resolver implementation
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        try:
            if not callable(getattr(resolver, 'resolve', None)):
                raise ValueError("Resolver must have a resolve method")
            
            self._resolvers[method] = resolver
            logger.debug(f"Resolver registered for method: {method}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register resolver: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """
        Clear resolver cache.
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            self._cache.clear()
            logger.debug("Resolver cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            dict: Cache information
        """
        return {
            "cache_size": len(self._cache),
            "cache_ttl": self._cache_ttl,
            "cached_dids": list(self._cache.keys())
        }
    
    def get_resolver_info(self) -> Dict[str, Any]:
        """
        Get resolver information.
        
        Returns:
            dict: Resolver information
        """
        return {
            "initialized": self._is_initialized,
            "supported_methods": self._resolver_info["supported_methods"],
            "registered_resolvers": list(self._resolvers.keys()),
            "resolver_info": self._resolver_info
        }

# Utility functions for DIDs
def create_didn(config: Optional[Dict] = None) -> DIDN:
    """
    Create DIDN.
    
    Args:
        config (dict, optional): DIDN configuration
        
    Returns:
        DIDN: Created DIDN
    """
    return DIDN(config)

def create_didn_resolver(config: Optional[Dict] = None) -> DIDNResolver:
    """
    Create DIDN resolver.
    
    Args:
        config (dict, optional): Resolver configuration
        
    Returns:
        DIDNResolver: Created resolver
    """
    return DIDNResolver(config)

def create_did(didn: DIDN, method: str = "key", public_key: Optional[bytes] = None) -> str:
    """
    Create DID.
    
    Args:
        didn (DIDN): DIDN instance
        method (str): DID method
        public_key (bytes, optional): Public key
        
    Returns:
        str: Created DID
    """
    return didn.create_did(method, public_key)

# Example usage
def example_dids():
    """Example of DIDN usage."""
    # Create DIDN
    didn = create_didn({
        "network": "testnet",
        "version": "1.0"
    })
    
    # Create DID
    did = didn.create_did("key")
    print(f"DID created: {did}")
    
    # Resolve DID
    document = didn.resolve_did(did)
    if document:
        print(f"DID resolved: {document.id}")
        print(f"Authentication keys: {document.authentication}")
    
    # Add service endpoint
    service = {
        "id": f"{did}#ai-service",
        "type": "AIPlatformService",
        "serviceEndpoint": "https://ai-platform.example.com/api"
    }
    
    if didn.add_service_endpoint(did, service):
        print("Service endpoint added successfully")
    
    # Get DID info
    did_info = didn.get_did_info(did)
    print(f"DID info: {did_info}")
    
    # Get network info
    network_info = didn.get_network_info()
    print(f"Network info: {network_info}")
    
    return didn

# Advanced DIDN example
def advanced_didn_example():
    """Advanced example of DIDN usage."""
    # Create DIDN and resolver
    didn = create_didn({
        "network": "mainnet",
        "version": "1.0",
        "security_level": "high"
    })
    
    resolver = create_didn_resolver({
        "cache_ttl": 1800,  # 30 minutes
        "max_cache_size": 1000
    })
    
    # Create multiple DIDs
    dids = []
    for i in range(3):
        did = didn.create_did("key")
        dids.append(did)
        print(f"Created DID {i+1}: {did}")
    
    # Add different service endpoints to each DID
    services = [
        {
            "id": f"{dids[0]}#quantum-service",
            "type": "QuantumComputingService",
            "serviceEndpoint": "https://quantum.example.com/api"
        },
        {
            "id": f"{dids[1]}#ai-service",
            "type": "AIService",
            "serviceEndpoint": "https://ai.example.com/api"
        },
        {
            "id": f"{dids[2]}#storage-service",
            "type": "StorageService",
            "serviceEndpoint": "https://storage.example.com/api"
        }
    ]
    
    # Add services
    for i, service in enumerate(services):
        if didn.add_service_endpoint(dids[i], service):
            print(f"Service added to DID {i+1}")
    
    # Resolve all DIDs
    print("\nResolving DIDs:")
    for did in dids:
        document = resolver.resolve(did)
        if document:
            print(f"Resolved {did}: {len(document.service)} services")
        else:
            print(f"Failed to resolve {did}")
    
    # Verify DID documents
    print("\nVerifying DID documents:")
    for did in dids:
        document = didn.resolve_did(did)
        if document:
            is_valid = didn.verify_did_document(did, document)
            print(f"DID {did} verification: {'valid' if is_valid else 'invalid'}")
    
    # Get detailed information
    print("\nDID Information:")
    for did in dids:
        info = didn.get_did_info(did)
        if info:
            print(f"DID: {info['did']}")
            print(f"  Version: {info['version']}")
            print(f"  Services: {info['services']}")
            print(f"  Timestamp: {info['timestamp']}")
    
    # Resolver cache info
    cache_info = resolver.get_cache_info()
    print(f"\nCache info: {cache_info['cache_size']} entries")
    
    return didn, resolver