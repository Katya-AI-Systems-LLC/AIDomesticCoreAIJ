"""
Security Demo Example
=====================

Demonstrates quantum-safe cryptography and security features.
"""

import asyncio
import time

from sdk.security import (
    KyberKEM,
    DilithiumSignature,
    DIDNManager,
    ZeroTrustManager,
    SecureKeyManager
)
from sdk.security.zero_trust import TrustLevel, SecurityContext, AccessDecision


def kyber_demo():
    """
    Demo: Post-quantum key exchange with Kyber.
    """
    print("=" * 60)
    print("Kyber Key Encapsulation Demo")
    print("=" * 60)
    
    for level in [512, 768, 1024]:
        print(f"\n--- Security Level {level} ---")
        
        kyber = KyberKEM(security_level=level)
        sizes = kyber.get_key_sizes()
        
        # Key generation
        start = time.time()
        keypair = kyber.keygen()
        keygen_time = (time.time() - start) * 1000
        
        print(f"Key generation: {keygen_time:.2f}ms")
        print(f"  Public key: {sizes['public_key']} bytes")
        print(f"  Secret key: {sizes['secret_key']} bytes")
        
        # Encapsulation
        start = time.time()
        ciphertext = kyber.encapsulate(keypair.public_key)
        encap_time = (time.time() - start) * 1000
        
        print(f"Encapsulation: {encap_time:.2f}ms")
        print(f"  Ciphertext: {sizes['ciphertext']} bytes")
        
        # Decapsulation
        start = time.time()
        shared_secret = kyber.decapsulate(ciphertext.ciphertext, keypair.secret_key)
        decap_time = (time.time() - start) * 1000
        
        print(f"Decapsulation: {decap_time:.2f}ms")
        print(f"  Shared secret: {len(shared_secret)} bytes")


def dilithium_demo():
    """
    Demo: Post-quantum digital signatures with Dilithium.
    """
    print("\n" + "=" * 60)
    print("Dilithium Digital Signature Demo")
    print("=" * 60)
    
    for level in [2, 3, 5]:
        print(f"\n--- Security Level {level} ---")
        
        dilithium = DilithiumSignature(security_level=level)
        sizes = dilithium.get_sizes()
        
        # Key generation
        start = time.time()
        keypair = dilithium.keygen()
        keygen_time = (time.time() - start) * 1000
        
        print(f"Key generation: {keygen_time:.2f}ms")
        print(f"  Public key: {sizes['public_key']} bytes")
        print(f"  Secret key: {sizes['secret_key']} bytes")
        
        # Sign message
        message = b"Hello, Quantum World! This is a test message for signing."
        
        start = time.time()
        signature = dilithium.sign(message, keypair.secret_key)
        sign_time = (time.time() - start) * 1000
        
        print(f"Signing: {sign_time:.2f}ms")
        print(f"  Signature: {sizes['signature']} bytes")
        
        # Verify signature
        start = time.time()
        is_valid = dilithium.verify(message, signature, keypair.public_key)
        verify_time = (time.time() - start) * 1000
        
        print(f"Verification: {verify_time:.2f}ms")
        print(f"  Valid: {is_valid}")


def didn_demo():
    """
    Demo: Decentralized Identity (DIDN).
    """
    print("\n" + "=" * 60)
    print("Decentralized Identity (DIDN) Demo")
    print("=" * 60)
    
    didn = DIDNManager()
    
    # Create DIDs
    print("\nCreating DIDs...")
    
    alice = didn.create_did()
    bob = didn.create_did()
    issuer = didn.create_did(services=[
        {"id": "#issuer", "type": "CredentialIssuer", "endpoint": "https://issuer.example"}
    ])
    
    print(f"Alice DID: {alice.did}")
    print(f"Bob DID: {bob.did}")
    print(f"Issuer DID: {issuer.did}")
    
    # Resolve DID
    print("\nResolving Alice's DID...")
    resolved = didn.resolve_did(alice.did)
    print(f"  Public keys: {len(resolved.public_keys)}")
    print(f"  Authentication: {resolved.authentication}")
    
    # Issue credential
    print("\nIssuing credential to Alice...")
    credential = didn.issue_credential(
        issuer.did,
        alice.did,
        {
            "type": "VerifiedEmployee",
            "organization": "AIPlatform",
            "role": "Quantum Developer",
            "clearance": "Level 5"
        },
        expires_in=365 * 24 * 3600  # 1 year
    )
    
    print(f"Credential ID: {credential.id}")
    print(f"Claims: {credential.claims}")
    
    # Verify credential
    print("\nVerifying credential...")
    is_valid = didn.verify_credential(credential)
    print(f"Valid: {is_valid}")
    
    # Revoke and re-verify
    print("\nRevoking credential...")
    didn.revoke_credential(credential.id)
    
    is_valid = didn.verify_credential(credential)
    print(f"Valid after revocation: {is_valid}")
    
    # Export DID document
    print("\nExporting DID document...")
    doc_json = didn.export_did_document(alice.did)
    print(doc_json[:200] + "...")


def zero_trust_demo():
    """
    Demo: Zero-Trust Security.
    """
    print("\n" + "=" * 60)
    print("Zero-Trust Security Demo")
    print("=" * 60)
    
    zt = ZeroTrustManager()
    
    # Register policies
    print("\nRegistering access policies...")
    
    zt.register_policy(
        "api/public",
        required_trust=TrustLevel.LOW,
        rate_limit=1000
    )
    
    zt.register_policy(
        "api/user",
        required_trust=TrustLevel.MEDIUM,
        required_attributes={"authenticated": True},
        rate_limit=100
    )
    
    zt.register_policy(
        "api/admin",
        required_trust=TrustLevel.HIGH,
        required_attributes={"authenticated": True, "admin": True},
        time_restrictions={"allowed_hours": (9, 18)},
        rate_limit=50
    )
    
    zt.register_policy(
        "api/quantum",
        required_trust=TrustLevel.VERIFIED,
        required_attributes={"quantum_clearance": True}
    )
    
    print(f"Registered {zt.get_statistics()['policies']} policies")
    
    # Test access scenarios
    scenarios = [
        ("Anonymous user", TrustLevel.NONE, {}, "api/public"),
        ("Logged in user", TrustLevel.MEDIUM, {"authenticated": True}, "api/user"),
        ("Admin user", TrustLevel.HIGH, {"authenticated": True, "admin": True}, "api/admin"),
        ("Low trust admin", TrustLevel.LOW, {"authenticated": True, "admin": True}, "api/admin"),
        ("Quantum researcher", TrustLevel.VERIFIED, {"quantum_clearance": True}, "api/quantum"),
    ]
    
    print("\nTesting access scenarios:")
    print("-" * 60)
    
    for name, trust, attrs, resource in scenarios:
        context = SecurityContext(
            identity=f"user_{name.lower().replace(' ', '_')}",
            device_id="device_001",
            location="office",
            timestamp=time.time(),
            trust_level=trust,
            attributes=attrs
        )
        
        decision = zt.evaluate_access(context, resource)
        
        status = "✓" if decision == AccessDecision.ALLOW else "✗"
        print(f"{status} {name:20} -> {resource:15} = {decision.value}")
    
    # Session management
    print("\nSession management:")
    
    session_id = zt.create_session("user_alice", "device_laptop")
    print(f"Created session: {session_id[:16]}...")
    
    session = zt.validate_session(session_id)
    print(f"Session valid: {session is not None}")
    
    zt.revoke_session(session_id)
    session = zt.validate_session(session_id)
    print(f"After revocation: {session is not None}")
    
    print(f"\nStatistics: {zt.get_statistics()}")


def key_manager_demo():
    """
    Demo: Secure Key Management.
    """
    print("\n" + "=" * 60)
    print("Secure Key Management Demo")
    print("=" * 60)
    
    km = SecureKeyManager()
    
    # Generate keys
    print("\nGenerating encryption keys...")
    
    aes_key = km.generate_key("aes-256", metadata={"purpose": "data_encryption"})
    print(f"AES-256 key: {aes_key}")
    
    chacha_key = km.generate_key("chacha20", metadata={"purpose": "stream_cipher"})
    print(f"ChaCha20 key: {chacha_key}")
    
    # Retrieve key
    print("\nRetrieving keys...")
    key_data = km.get_key(aes_key)
    print(f"Retrieved key length: {len(key_data)} bytes")
    
    # Key info
    info = km.get_key_info(aes_key)
    print(f"Key info: algorithm={info['algorithm']}, purpose={info['metadata']['purpose']}")
    
    # Access control
    print("\nSetting up access control...")
    km.grant_access(aes_key, "service_a")
    km.grant_access(aes_key, "service_b")
    
    # Key rotation
    print("\nRotating key...")
    new_key_id = km.rotate_key(aes_key)
    print(f"New key ID: {new_key_id}")
    
    old_info = km.get_key_info(aes_key)
    print(f"Old key rotated to: {old_info['metadata'].get('rotated_to')}")
    
    # List keys
    print(f"\nTotal keys: {len(km.list_keys())}")
    
    # Audit log
    print("\nAudit log (last 5 entries):")
    for entry in km.get_audit_log(limit=5):
        print(f"  {entry['action']:15} - {entry['key_id']}")


def main():
    """Run all security demos."""
    print("\n" + "=" * 60)
    print("AIPlatform SDK - Security Demo")
    print("=" * 60)
    
    kyber_demo()
    dilithium_demo()
    didn_demo()
    zero_trust_demo()
    key_manager_demo()
    
    print("\n" + "=" * 60)
    print("All security demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
