"""
Zero-DNS Registry
=================

Decentralized name registry for Post-DNS architecture.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import hashlib
import secrets
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    """Entry in the Zero-DNS registry."""
    name: str
    quantum_signature: bytes
    owner_signature: bytes
    node_endpoints: List[str]
    metadata: Dict[str, Any]
    created: float
    updated: float
    expires: float
    version: int
    verified: bool = True


@dataclass
class RegistrationResult:
    """Result of name registration."""
    success: bool
    name: str
    quantum_signature: Optional[bytes] = None
    error: Optional[str] = None
    expires: float = 0.0


class ZeroDNSRegistry:
    """
    Zero-DNS name registry for Quantum Infrastructure Zero.
    
    Provides:
    - Decentralized name registration
    - Quantum signature verification
    - Name ownership management
    - Expiration and renewal
    
    Example:
        >>> registry = ZeroDNSRegistry()
        >>> result = await registry.register("myservice.qiz", owner_key)
        >>> if result.success:
        ...     print(f"Registered with signature: {result.quantum_signature.hex()}")
    """
    
    DEFAULT_TTL = 86400 * 365  # 1 year
    MIN_NAME_LENGTH = 3
    MAX_NAME_LENGTH = 63
    RESERVED_TLDS = ["qiz", "quantum", "mesh", "zero"]
    
    def __init__(self, node_id: Optional[str] = None,
                 language: str = "en"):
        """
        Initialize Zero-DNS registry.
        
        Args:
            node_id: This node's ID
            language: Language for messages
        """
        self.node_id = node_id or "registry_node"
        self.language = language
        
        # Registry storage
        self._entries: Dict[str, RegistryEntry] = {}
        
        # Owner index
        self._owner_index: Dict[bytes, List[str]] = {}
        
        # Statistics
        self._stats = {
            "registrations": 0,
            "updates": 0,
            "deletions": 0,
            "lookups": 0
        }
        
        logger.info(f"Zero-DNS registry initialized: {node_id}")
    
    async def register(self, name: str,
                       owner_signature: bytes,
                       node_endpoints: Optional[List[str]] = None,
                       metadata: Optional[Dict] = None,
                       ttl: Optional[int] = None) -> RegistrationResult:
        """
        Register a new name.
        
        Args:
            name: Name to register
            owner_signature: Owner's quantum signature
            node_endpoints: List of node endpoints
            metadata: Additional metadata
            ttl: Time-to-live in seconds
            
        Returns:
            RegistrationResult
        """
        # Validate name
        validation_error = self._validate_name(name)
        if validation_error:
            return RegistrationResult(
                success=False,
                name=name,
                error=validation_error
            )
        
        # Normalize name
        name = self._normalize_name(name)
        
        # Check if name is already registered
        if name in self._entries:
            existing = self._entries[name]
            if existing.owner_signature != owner_signature:
                return RegistrationResult(
                    success=False,
                    name=name,
                    error="Name already registered by another owner"
                )
            # Update existing registration
            return await self.update(name, owner_signature, 
                                     node_endpoints, metadata)
        
        # Generate quantum signature for this name
        quantum_signature = self._generate_quantum_signature(name, owner_signature)
        
        # Create entry
        current_time = time.time()
        entry = RegistryEntry(
            name=name,
            quantum_signature=quantum_signature,
            owner_signature=owner_signature,
            node_endpoints=node_endpoints or [],
            metadata=metadata or {},
            created=current_time,
            updated=current_time,
            expires=current_time + (ttl or self.DEFAULT_TTL),
            version=1
        )
        
        # Store entry
        self._entries[name] = entry
        
        # Update owner index
        if owner_signature not in self._owner_index:
            self._owner_index[owner_signature] = []
        self._owner_index[owner_signature].append(name)
        
        self._stats["registrations"] += 1
        logger.info(f"Registered name: {name}")
        
        return RegistrationResult(
            success=True,
            name=name,
            quantum_signature=quantum_signature,
            expires=entry.expires
        )
    
    async def update(self, name: str,
                     owner_signature: bytes,
                     node_endpoints: Optional[List[str]] = None,
                     metadata: Optional[Dict] = None) -> RegistrationResult:
        """
        Update an existing registration.
        
        Args:
            name: Name to update
            owner_signature: Owner's quantum signature
            node_endpoints: New node endpoints
            metadata: New metadata
            
        Returns:
            RegistrationResult
        """
        name = self._normalize_name(name)
        
        if name not in self._entries:
            return RegistrationResult(
                success=False,
                name=name,
                error="Name not registered"
            )
        
        entry = self._entries[name]
        
        # Verify ownership
        if entry.owner_signature != owner_signature:
            return RegistrationResult(
                success=False,
                name=name,
                error="Not authorized to update this name"
            )
        
        # Update entry
        if node_endpoints is not None:
            entry.node_endpoints = node_endpoints
        if metadata is not None:
            entry.metadata.update(metadata)
        
        entry.updated = time.time()
        entry.version += 1
        
        self._stats["updates"] += 1
        logger.info(f"Updated name: {name}")
        
        return RegistrationResult(
            success=True,
            name=name,
            quantum_signature=entry.quantum_signature,
            expires=entry.expires
        )
    
    async def renew(self, name: str,
                    owner_signature: bytes,
                    ttl: Optional[int] = None) -> RegistrationResult:
        """
        Renew a registration.
        
        Args:
            name: Name to renew
            owner_signature: Owner's quantum signature
            ttl: New TTL in seconds
            
        Returns:
            RegistrationResult
        """
        name = self._normalize_name(name)
        
        if name not in self._entries:
            return RegistrationResult(
                success=False,
                name=name,
                error="Name not registered"
            )
        
        entry = self._entries[name]
        
        if entry.owner_signature != owner_signature:
            return RegistrationResult(
                success=False,
                name=name,
                error="Not authorized to renew this name"
            )
        
        entry.expires = time.time() + (ttl or self.DEFAULT_TTL)
        entry.updated = time.time()
        
        logger.info(f"Renewed name: {name}")
        
        return RegistrationResult(
            success=True,
            name=name,
            quantum_signature=entry.quantum_signature,
            expires=entry.expires
        )
    
    async def delete(self, name: str,
                     owner_signature: bytes) -> bool:
        """
        Delete a registration.
        
        Args:
            name: Name to delete
            owner_signature: Owner's quantum signature
            
        Returns:
            True if deleted successfully
        """
        name = self._normalize_name(name)
        
        if name not in self._entries:
            return False
        
        entry = self._entries[name]
        
        if entry.owner_signature != owner_signature:
            return False
        
        # Remove from owner index
        if owner_signature in self._owner_index:
            self._owner_index[owner_signature].remove(name)
        
        del self._entries[name]
        
        self._stats["deletions"] += 1
        logger.info(f"Deleted name: {name}")
        
        return True
    
    async def lookup(self, name: str) -> Optional[RegistryEntry]:
        """
        Look up a name in the registry.
        
        Args:
            name: Name to look up
            
        Returns:
            RegistryEntry if found
        """
        name = self._normalize_name(name)
        self._stats["lookups"] += 1
        
        entry = self._entries.get(name)
        
        if entry and entry.expires < time.time():
            # Entry expired
            return None
        
        return entry
    
    async def lookup_by_signature(self, 
                                   quantum_signature: bytes) -> Optional[RegistryEntry]:
        """Look up by quantum signature."""
        for entry in self._entries.values():
            if entry.quantum_signature == quantum_signature:
                if entry.expires >= time.time():
                    return entry
        return None
    
    def get_owner_names(self, owner_signature: bytes) -> List[str]:
        """Get all names owned by a signature."""
        return self._owner_index.get(owner_signature, []).copy()
    
    def _validate_name(self, name: str) -> Optional[str]:
        """Validate name format."""
        if not name:
            return "Name cannot be empty"
        
        # Extract label (part before TLD)
        parts = name.lower().split(".")
        label = parts[0]
        
        if len(label) < self.MIN_NAME_LENGTH:
            return f"Name must be at least {self.MIN_NAME_LENGTH} characters"
        
        if len(label) > self.MAX_NAME_LENGTH:
            return f"Name cannot exceed {self.MAX_NAME_LENGTH} characters"
        
        # Check for valid characters
        if not all(c.isalnum() or c in "-_" for c in label):
            return "Name can only contain alphanumeric characters, hyphens, and underscores"
        
        if label.startswith("-") or label.endswith("-"):
            return "Name cannot start or end with a hyphen"
        
        return None
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name."""
        name = name.lower().strip()
        if "." not in name:
            name = f"{name}.qiz"
        return name
    
    def _generate_quantum_signature(self, name: str, 
                                     owner_signature: bytes) -> bytes:
        """Generate quantum signature for a name."""
        # Combine name, owner, and entropy
        entropy = secrets.token_bytes(32)
        signature_input = name.encode() + owner_signature + entropy
        
        return hashlib.sha3_256(signature_input).digest()
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        current_time = time.time()
        expired = [
            name for name, entry in self._entries.items()
            if entry.expires < current_time
        ]
        
        for name in expired:
            entry = self._entries[name]
            if entry.owner_signature in self._owner_index:
                self._owner_index[entry.owner_signature].remove(name)
            del self._entries[name]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired entries")
        
        return len(expired)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            **self._stats,
            "total_entries": len(self._entries),
            "unique_owners": len(self._owner_index)
        }
    
    def __repr__(self) -> str:
        return f"ZeroDNSRegistry(entries={len(self._entries)})"
