"""
Quantum Addressing Module
=========================

Quantum-based addressing system for Post-DNS architecture.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import hashlib
import secrets
import struct
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumAddress:
    """A quantum network address."""
    signature: bytes  # 32 bytes
    version: int  # Address version
    type_code: int  # Address type
    checksum: bytes  # 4 bytes
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return (
            struct.pack(">BH", self.version, self.type_code) +
            self.signature +
            self.checksum
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'QuantumAddress':
        """Deserialize from bytes."""
        version, type_code = struct.unpack(">BH", data[:3])
        signature = data[3:35]
        checksum = data[35:39]
        return cls(signature, version, type_code, checksum)
    
    def to_string(self) -> str:
        """Convert to human-readable string."""
        # Format: qaddr1:<base32_encoded>
        import base64
        encoded = base64.b32encode(self.to_bytes()).decode().lower().rstrip('=')
        return f"qaddr{self.version}:{encoded}"
    
    @classmethod
    def from_string(cls, address_str: str) -> 'QuantumAddress':
        """Parse from string."""
        import base64
        
        if not address_str.startswith("qaddr"):
            raise ValueError("Invalid quantum address format")
        
        parts = address_str.split(":")
        if len(parts) != 2:
            raise ValueError("Invalid quantum address format")
        
        # Add padding for base32
        encoded = parts[1].upper()
        padding = (8 - len(encoded) % 8) % 8
        encoded += "=" * padding
        
        data = base64.b32decode(encoded)
        return cls.from_bytes(data)
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __eq__(self, other) -> bool:
        if isinstance(other, QuantumAddress):
            return self.signature == other.signature
        return False
    
    def __hash__(self) -> int:
        return hash(self.signature)


class AddressType:
    """Address type codes."""
    NODE = 0x0001
    SERVICE = 0x0002
    USER = 0x0003
    RESOURCE = 0x0004
    MULTICAST = 0x0010
    ANYCAST = 0x0020


class QuantumAddressing:
    """
    Quantum addressing system for QIZ network.
    
    Provides:
    - Address generation from quantum signatures
    - Address validation
    - Address type management
    - Hierarchical addressing
    
    Example:
        >>> qa = QuantumAddressing()
        >>> addr = qa.generate_address(node_signature, AddressType.NODE)
        >>> print(addr.to_string())
    """
    
    VERSION = 1
    SIGNATURE_SIZE = 32
    CHECKSUM_SIZE = 4
    
    def __init__(self, language: str = "en"):
        """
        Initialize quantum addressing.
        
        Args:
            language: Language for messages
        """
        self.language = language
        
        # Address cache
        self._cache: Dict[bytes, QuantumAddress] = {}
        
        logger.info("Quantum addressing initialized")
    
    def generate_address(self, quantum_signature: bytes,
                         address_type: int = AddressType.NODE) -> QuantumAddress:
        """
        Generate quantum address from signature.
        
        Args:
            quantum_signature: Source quantum signature
            address_type: Type of address to generate
            
        Returns:
            QuantumAddress
        """
        # Derive address signature
        if len(quantum_signature) != self.SIGNATURE_SIZE:
            # Hash to correct size
            quantum_signature = hashlib.sha256(quantum_signature).digest()
        
        # Calculate checksum
        checksum = self._calculate_checksum(quantum_signature, address_type)
        
        address = QuantumAddress(
            signature=quantum_signature,
            version=self.VERSION,
            type_code=address_type,
            checksum=checksum
        )
        
        # Cache
        self._cache[quantum_signature] = address
        
        return address
    
    def generate_random_address(self, 
                                 address_type: int = AddressType.NODE) -> QuantumAddress:
        """Generate a random quantum address."""
        signature = secrets.token_bytes(self.SIGNATURE_SIZE)
        return self.generate_address(signature, address_type)
    
    def generate_derived_address(self, parent: QuantumAddress,
                                  child_id: bytes,
                                  address_type: int = AddressType.RESOURCE) -> QuantumAddress:
        """
        Generate derived address from parent.
        
        Args:
            parent: Parent address
            child_id: Child identifier
            address_type: Type of derived address
            
        Returns:
            Derived QuantumAddress
        """
        # Combine parent signature with child ID
        derived_input = parent.signature + child_id
        derived_signature = hashlib.sha256(derived_input).digest()
        
        return self.generate_address(derived_signature, address_type)
    
    def validate_address(self, address: QuantumAddress) -> Tuple[bool, Optional[str]]:
        """
        Validate a quantum address.
        
        Args:
            address: Address to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check version
        if address.version != self.VERSION:
            return False, f"Unsupported version: {address.version}"
        
        # Check signature size
        if len(address.signature) != self.SIGNATURE_SIZE:
            return False, "Invalid signature size"
        
        # Verify checksum
        expected_checksum = self._calculate_checksum(
            address.signature, address.type_code
        )
        if address.checksum != expected_checksum:
            return False, "Invalid checksum"
        
        return True, None
    
    def _calculate_checksum(self, signature: bytes, type_code: int) -> bytes:
        """Calculate address checksum."""
        checksum_input = (
            struct.pack(">BH", self.VERSION, type_code) +
            signature
        )
        full_hash = hashlib.sha256(checksum_input).digest()
        return full_hash[:self.CHECKSUM_SIZE]
    
    def parse_address(self, address_str: str) -> Optional[QuantumAddress]:
        """
        Parse address from string.
        
        Args:
            address_str: Address string
            
        Returns:
            QuantumAddress if valid
        """
        try:
            address = QuantumAddress.from_string(address_str)
            is_valid, _ = self.validate_address(address)
            return address if is_valid else None
        except Exception as e:
            logger.warning(f"Failed to parse address: {e}")
            return None
    
    def get_address_type_name(self, type_code: int) -> str:
        """Get human-readable name for address type."""
        type_names = {
            AddressType.NODE: "Node",
            AddressType.SERVICE: "Service",
            AddressType.USER: "User",
            AddressType.RESOURCE: "Resource",
            AddressType.MULTICAST: "Multicast",
            AddressType.ANYCAST: "Anycast"
        }
        return type_names.get(type_code, "Unknown")
    
    def is_multicast(self, address: QuantumAddress) -> bool:
        """Check if address is multicast."""
        return address.type_code == AddressType.MULTICAST
    
    def is_anycast(self, address: QuantumAddress) -> bool:
        """Check if address is anycast."""
        return address.type_code == AddressType.ANYCAST
    
    def addresses_match(self, addr1: QuantumAddress, 
                        addr2: QuantumAddress) -> bool:
        """Check if two addresses match."""
        return addr1.signature == addr2.signature
    
    def get_cached_address(self, signature: bytes) -> Optional[QuantumAddress]:
        """Get address from cache."""
        return self._cache.get(signature)
    
    def clear_cache(self):
        """Clear address cache."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"QuantumAddressing(version={self.VERSION}, cached={len(self._cache)})"
