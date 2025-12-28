"""
DIDN - Decentralized Identity
=============================

Decentralized identity management for QIZ.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import hashlib
import secrets
import time
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DIDDocument:
    """DID Document."""
    did: str
    public_keys: List[Dict[str, Any]]
    authentication: List[str]
    services: List[Dict[str, Any]]
    created: float
    updated: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifiableCredential:
    """Verifiable credential."""
    id: str
    issuer: str
    subject: str
    claims: Dict[str, Any]
    proof: bytes
    issued: float
    expires: float


class DIDNManager:
    """
    Decentralized Identity Network Manager.
    
    Features:
    - DID creation and management
    - Verifiable credentials
    - Identity verification
    - Key rotation
    
    Example:
        >>> didn = DIDNManager()
        >>> did_doc = didn.create_did()
        >>> credential = didn.issue_credential(did_doc.did, claims)
    """
    
    DID_METHOD = "didn"
    
    def __init__(self, language: str = "en"):
        """
        Initialize DIDN manager.
        
        Args:
            language: Language for messages
        """
        self.language = language
        
        # DID registry
        self._registry: Dict[str, DIDDocument] = {}
        
        # Credentials
        self._credentials: Dict[str, VerifiableCredential] = {}
        
        # Revocation list
        self._revoked: set = set()
        
        logger.info("DIDN Manager initialized")
    
    def create_did(self, public_key: Optional[bytes] = None,
                   services: Optional[List[Dict]] = None) -> DIDDocument:
        """
        Create a new DID.
        
        Args:
            public_key: Optional public key
            services: Optional service endpoints
            
        Returns:
            DIDDocument
        """
        # Generate key if not provided
        if public_key is None:
            public_key = secrets.token_bytes(32)
        
        # Generate DID
        did_suffix = hashlib.sha256(public_key).hexdigest()[:32]
        did = f"did:{self.DID_METHOD}:{did_suffix}"
        
        # Create document
        current_time = time.time()
        
        doc = DIDDocument(
            did=did,
            public_keys=[{
                "id": f"{did}#key-1",
                "type": "Ed25519VerificationKey2020",
                "controller": did,
                "publicKeyMultibase": public_key.hex()
            }],
            authentication=[f"{did}#key-1"],
            services=services or [],
            created=current_time,
            updated=current_time
        )
        
        # Register
        self._registry[did] = doc
        
        logger.info(f"Created DID: {did}")
        return doc
    
    def resolve_did(self, did: str) -> Optional[DIDDocument]:
        """
        Resolve a DID to its document.
        
        Args:
            did: DID to resolve
            
        Returns:
            DIDDocument if found
        """
        return self._registry.get(did)
    
    def update_did(self, did: str, 
                   updates: Dict[str, Any]) -> Optional[DIDDocument]:
        """
        Update a DID document.
        
        Args:
            did: DID to update
            updates: Fields to update
            
        Returns:
            Updated DIDDocument
        """
        if did not in self._registry:
            return None
        
        doc = self._registry[did]
        
        if "services" in updates:
            doc.services = updates["services"]
        if "metadata" in updates:
            doc.metadata.update(updates["metadata"])
        
        doc.updated = time.time()
        
        return doc
    
    def add_key(self, did: str, public_key: bytes,
                key_type: str = "Ed25519VerificationKey2020") -> bool:
        """
        Add a key to DID document.
        
        Args:
            did: DID to update
            public_key: New public key
            key_type: Key type
            
        Returns:
            True if added
        """
        if did not in self._registry:
            return False
        
        doc = self._registry[did]
        key_id = f"{did}#key-{len(doc.public_keys) + 1}"
        
        doc.public_keys.append({
            "id": key_id,
            "type": key_type,
            "controller": did,
            "publicKeyMultibase": public_key.hex()
        })
        
        doc.updated = time.time()
        return True
    
    def revoke_key(self, did: str, key_id: str) -> bool:
        """
        Revoke a key from DID document.
        
        Args:
            did: DID to update
            key_id: Key ID to revoke
            
        Returns:
            True if revoked
        """
        if did not in self._registry:
            return False
        
        doc = self._registry[did]
        doc.public_keys = [k for k in doc.public_keys if k["id"] != key_id]
        doc.authentication = [a for a in doc.authentication if a != key_id]
        doc.updated = time.time()
        
        return True
    
    def issue_credential(self, issuer_did: str,
                         subject_did: str,
                         claims: Dict[str, Any],
                         expires_in: float = 86400 * 365) -> VerifiableCredential:
        """
        Issue a verifiable credential.
        
        Args:
            issuer_did: Issuer's DID
            subject_did: Subject's DID
            claims: Credential claims
            expires_in: Expiration time in seconds
            
        Returns:
            VerifiableCredential
        """
        credential_id = f"urn:uuid:{secrets.token_hex(16)}"
        current_time = time.time()
        
        # Create proof
        proof_input = json.dumps({
            "issuer": issuer_did,
            "subject": subject_did,
            "claims": claims,
            "issued": current_time
        }).encode()
        
        proof = hashlib.sha3_256(proof_input).digest()
        
        credential = VerifiableCredential(
            id=credential_id,
            issuer=issuer_did,
            subject=subject_did,
            claims=claims,
            proof=proof,
            issued=current_time,
            expires=current_time + expires_in
        )
        
        self._credentials[credential_id] = credential
        
        logger.info(f"Issued credential: {credential_id}")
        return credential
    
    def verify_credential(self, credential: VerifiableCredential) -> bool:
        """
        Verify a credential.
        
        Args:
            credential: Credential to verify
            
        Returns:
            True if valid
        """
        # Check expiration
        if credential.expires < time.time():
            return False
        
        # Check revocation
        if credential.id in self._revoked:
            return False
        
        # Verify issuer exists
        if credential.issuer not in self._registry:
            return False
        
        # Verify proof
        proof_input = json.dumps({
            "issuer": credential.issuer,
            "subject": credential.subject,
            "claims": credential.claims,
            "issued": credential.issued
        }).encode()
        
        expected_proof = hashlib.sha3_256(proof_input).digest()
        
        return credential.proof == expected_proof
    
    def revoke_credential(self, credential_id: str) -> bool:
        """
        Revoke a credential.
        
        Args:
            credential_id: Credential to revoke
            
        Returns:
            True if revoked
        """
        self._revoked.add(credential_id)
        return True
    
    def get_credentials(self, subject_did: str) -> List[VerifiableCredential]:
        """Get all credentials for a subject."""
        return [
            c for c in self._credentials.values()
            if c.subject == subject_did and c.id not in self._revoked
        ]
    
    def export_did_document(self, did: str) -> Optional[str]:
        """Export DID document as JSON."""
        doc = self._registry.get(did)
        if not doc:
            return None
        
        return json.dumps({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": doc.did,
            "verificationMethod": doc.public_keys,
            "authentication": doc.authentication,
            "service": doc.services,
            "created": doc.created,
            "updated": doc.updated
        }, indent=2)
    
    def __repr__(self) -> str:
        return f"DIDNManager(dids={len(self._registry)})"
