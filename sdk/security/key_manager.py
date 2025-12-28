"""
Secure Key Manager
==================

Secure key storage and management.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import hashlib
import secrets
import time
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class StoredKey:
    """A stored cryptographic key."""
    key_id: str
    key_type: str
    algorithm: str
    encrypted_key: bytes
    created: float
    expires: Optional[float]
    metadata: Dict[str, Any]


class SecureKeyManager:
    """
    Secure key management system.
    
    Features:
    - Key generation
    - Secure storage
    - Key rotation
    - Access control
    - Audit logging
    
    Example:
        >>> km = SecureKeyManager(master_key=b"...")
        >>> key_id = km.generate_key("aes-256")
        >>> key = km.get_key(key_id)
    """
    
    ALGORITHMS = {
        "aes-128": 16,
        "aes-256": 32,
        "chacha20": 32,
        "kyber-768": 32,
        "dilithium-3": 32
    }
    
    def __init__(self, master_key: Optional[bytes] = None,
                 language: str = "en"):
        """
        Initialize key manager.
        
        Args:
            master_key: Master encryption key
            language: Language for messages
        """
        self.language = language
        
        # Generate master key if not provided
        if master_key is None:
            master_key = secrets.token_bytes(32)
        
        self._master_key = master_key
        
        # Key storage
        self._keys: Dict[str, StoredKey] = {}
        
        # Access control
        self._access_control: Dict[str, List[str]] = {}
        
        # Audit log
        self._audit_log: List[Dict] = []
        
        logger.info("Secure Key Manager initialized")
    
    def generate_key(self, algorithm: str = "aes-256",
                     expires_in: Optional[float] = None,
                     metadata: Optional[Dict] = None) -> str:
        """
        Generate a new key.
        
        Args:
            algorithm: Key algorithm
            expires_in: Expiration time in seconds
            metadata: Additional metadata
            
        Returns:
            Key ID
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        key_size = self.ALGORITHMS[algorithm]
        
        # Generate key
        raw_key = secrets.token_bytes(key_size)
        
        # Encrypt with master key
        encrypted_key = self._encrypt_key(raw_key)
        
        # Generate key ID
        key_id = hashlib.sha256(raw_key + secrets.token_bytes(8)).hexdigest()[:16]
        
        # Store key
        current_time = time.time()
        
        stored_key = StoredKey(
            key_id=key_id,
            key_type="symmetric",
            algorithm=algorithm,
            encrypted_key=encrypted_key,
            created=current_time,
            expires=current_time + expires_in if expires_in else None,
            metadata=metadata or {}
        )
        
        self._keys[key_id] = stored_key
        
        # Log
        self._log_action("generate", key_id)
        
        logger.info(f"Generated key: {key_id}")
        return key_id
    
    def import_key(self, raw_key: bytes,
                   algorithm: str,
                   metadata: Optional[Dict] = None) -> str:
        """
        Import an existing key.
        
        Args:
            raw_key: Raw key bytes
            algorithm: Key algorithm
            metadata: Additional metadata
            
        Returns:
            Key ID
        """
        # Encrypt with master key
        encrypted_key = self._encrypt_key(raw_key)
        
        # Generate key ID
        key_id = hashlib.sha256(raw_key + secrets.token_bytes(8)).hexdigest()[:16]
        
        stored_key = StoredKey(
            key_id=key_id,
            key_type="symmetric",
            algorithm=algorithm,
            encrypted_key=encrypted_key,
            created=time.time(),
            expires=None,
            metadata=metadata or {}
        )
        
        self._keys[key_id] = stored_key
        
        self._log_action("import", key_id)
        
        return key_id
    
    def get_key(self, key_id: str,
                accessor: Optional[str] = None) -> Optional[bytes]:
        """
        Retrieve a key.
        
        Args:
            key_id: Key ID
            accessor: Accessor identity for access control
            
        Returns:
            Raw key bytes or None
        """
        if key_id not in self._keys:
            return None
        
        stored_key = self._keys[key_id]
        
        # Check expiration
        if stored_key.expires and stored_key.expires < time.time():
            self._log_action("access_denied_expired", key_id, accessor)
            return None
        
        # Check access control
        if accessor and key_id in self._access_control:
            if accessor not in self._access_control[key_id]:
                self._log_action("access_denied", key_id, accessor)
                return None
        
        # Decrypt key
        raw_key = self._decrypt_key(stored_key.encrypted_key)
        
        self._log_action("access", key_id, accessor)
        
        return raw_key
    
    def rotate_key(self, key_id: str) -> Optional[str]:
        """
        Rotate a key (generate new, keep old for decryption).
        
        Args:
            key_id: Key to rotate
            
        Returns:
            New key ID
        """
        if key_id not in self._keys:
            return None
        
        old_key = self._keys[key_id]
        
        # Generate new key with same algorithm
        new_key_id = self.generate_key(
            algorithm=old_key.algorithm,
            metadata={
                **old_key.metadata,
                "rotated_from": key_id
            }
        )
        
        # Mark old key as rotated
        old_key.metadata["rotated_to"] = new_key_id
        old_key.metadata["rotated_at"] = time.time()
        
        self._log_action("rotate", key_id, new_key_id=new_key_id)
        
        return new_key_id
    
    def delete_key(self, key_id: str) -> bool:
        """
        Delete a key.
        
        Args:
            key_id: Key to delete
            
        Returns:
            True if deleted
        """
        if key_id not in self._keys:
            return False
        
        del self._keys[key_id]
        
        if key_id in self._access_control:
            del self._access_control[key_id]
        
        self._log_action("delete", key_id)
        
        return True
    
    def grant_access(self, key_id: str, accessor: str):
        """Grant access to a key."""
        if key_id not in self._access_control:
            self._access_control[key_id] = []
        
        if accessor not in self._access_control[key_id]:
            self._access_control[key_id].append(accessor)
            self._log_action("grant_access", key_id, accessor)
    
    def revoke_access(self, key_id: str, accessor: str):
        """Revoke access to a key."""
        if key_id in self._access_control:
            if accessor in self._access_control[key_id]:
                self._access_control[key_id].remove(accessor)
                self._log_action("revoke_access", key_id, accessor)
    
    def _encrypt_key(self, raw_key: bytes) -> bytes:
        """Encrypt key with master key."""
        # Simple XOR encryption (in production, use proper encryption)
        encrypted = bytes(
            a ^ b for a, b in zip(
                raw_key,
                (self._master_key * (len(raw_key) // len(self._master_key) + 1))[:len(raw_key)]
            )
        )
        return encrypted
    
    def _decrypt_key(self, encrypted_key: bytes) -> bytes:
        """Decrypt key with master key."""
        # XOR is symmetric
        return self._encrypt_key(encrypted_key)
    
    def _log_action(self, action: str, key_id: str,
                    accessor: Optional[str] = None,
                    **kwargs):
        """Log an action."""
        self._audit_log.append({
            "action": action,
            "key_id": key_id,
            "accessor": accessor,
            "timestamp": time.time(),
            **kwargs
        })
        
        # Keep only recent logs
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]
    
    def get_key_info(self, key_id: str) -> Optional[Dict]:
        """Get key information (without the key itself)."""
        if key_id not in self._keys:
            return None
        
        key = self._keys[key_id]
        
        return {
            "key_id": key.key_id,
            "algorithm": key.algorithm,
            "created": key.created,
            "expires": key.expires,
            "metadata": key.metadata
        }
    
    def list_keys(self) -> List[str]:
        """List all key IDs."""
        return list(self._keys.keys())
    
    def get_audit_log(self, key_id: Optional[str] = None,
                      limit: int = 100) -> List[Dict]:
        """Get audit log entries."""
        logs = self._audit_log
        
        if key_id:
            logs = [l for l in logs if l["key_id"] == key_id]
        
        return logs[-limit:]
    
    def __repr__(self) -> str:
        return f"SecureKeyManager(keys={len(self._keys)})"
