"""
Zero Trust Security
===================

Zero-trust security model implementation.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


class AccessDecision(Enum):
    """Access decisions."""
    DENY = "deny"
    ALLOW = "allow"
    CHALLENGE = "challenge"
    ELEVATE = "elevate"


@dataclass
class SecurityContext:
    """Security context for a request."""
    identity: str
    device_id: str
    location: Optional[str]
    timestamp: float
    trust_level: TrustLevel
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPolicy:
    """Access control policy."""
    resource: str
    required_trust: TrustLevel
    required_attributes: Dict[str, Any]
    time_restrictions: Optional[Dict] = None
    rate_limit: Optional[int] = None


class ZeroTrustManager:
    """
    Zero Trust security manager.
    
    Features:
    - Continuous verification
    - Least privilege access
    - Micro-segmentation
    - Risk-based authentication
    
    Example:
        >>> zt = ZeroTrustManager()
        >>> context = SecurityContext(identity="user1", ...)
        >>> decision = zt.evaluate_access(context, "resource1")
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize Zero Trust manager.
        
        Args:
            language: Language for messages
        """
        self.language = language
        
        # Policies
        self._policies: Dict[str, AccessPolicy] = {}
        
        # Identity trust scores
        self._trust_scores: Dict[str, float] = {}
        
        # Session tracking
        self._sessions: Dict[str, Dict] = {}
        
        # Anomaly detection
        self._behavior_baseline: Dict[str, Dict] = {}
        
        # Access log
        self._access_log: List[Dict] = []
        
        logger.info("Zero Trust Manager initialized")
    
    def register_policy(self, resource: str,
                        required_trust: TrustLevel = TrustLevel.MEDIUM,
                        required_attributes: Optional[Dict] = None,
                        time_restrictions: Optional[Dict] = None,
                        rate_limit: Optional[int] = None):
        """
        Register an access policy.
        
        Args:
            resource: Resource identifier
            required_trust: Minimum trust level
            required_attributes: Required identity attributes
            time_restrictions: Time-based restrictions
            rate_limit: Requests per minute limit
        """
        policy = AccessPolicy(
            resource=resource,
            required_trust=required_trust,
            required_attributes=required_attributes or {},
            time_restrictions=time_restrictions,
            rate_limit=rate_limit
        )
        
        self._policies[resource] = policy
        logger.info(f"Registered policy for: {resource}")
    
    def evaluate_access(self, context: SecurityContext,
                        resource: str) -> AccessDecision:
        """
        Evaluate access request.
        
        Args:
            context: Security context
            resource: Requested resource
            
        Returns:
            AccessDecision
        """
        # Log access attempt
        self._log_access(context, resource)
        
        # Get policy
        policy = self._policies.get(resource)
        if not policy:
            # Default deny for unknown resources
            return AccessDecision.DENY
        
        # Check trust level
        if context.trust_level.value < policy.required_trust.value:
            if context.trust_level.value >= TrustLevel.LOW.value:
                return AccessDecision.CHALLENGE
            return AccessDecision.DENY
        
        # Check required attributes
        for attr, value in policy.required_attributes.items():
            if context.attributes.get(attr) != value:
                return AccessDecision.DENY
        
        # Check time restrictions
        if policy.time_restrictions:
            if not self._check_time_restrictions(policy.time_restrictions):
                return AccessDecision.DENY
        
        # Check rate limit
        if policy.rate_limit:
            if not self._check_rate_limit(context.identity, resource, policy.rate_limit):
                return AccessDecision.DENY
        
        # Check for anomalies
        if self._detect_anomaly(context):
            return AccessDecision.CHALLENGE
        
        return AccessDecision.ALLOW
    
    def _check_time_restrictions(self, restrictions: Dict) -> bool:
        """Check time-based restrictions."""
        import datetime
        
        now = datetime.datetime.now()
        
        # Check allowed hours
        if "allowed_hours" in restrictions:
            start, end = restrictions["allowed_hours"]
            if not (start <= now.hour < end):
                return False
        
        # Check allowed days
        if "allowed_days" in restrictions:
            if now.weekday() not in restrictions["allowed_days"]:
                return False
        
        return True
    
    def _check_rate_limit(self, identity: str, resource: str,
                          limit: int) -> bool:
        """Check rate limit."""
        key = f"{identity}:{resource}"
        current_time = time.time()
        
        # Get recent requests
        recent = [
            log for log in self._access_log
            if log["identity"] == identity
            and log["resource"] == resource
            and current_time - log["timestamp"] < 60
        ]
        
        return len(recent) < limit
    
    def _detect_anomaly(self, context: SecurityContext) -> bool:
        """Detect anomalous behavior."""
        identity = context.identity
        
        if identity not in self._behavior_baseline:
            # First access, establish baseline
            self._behavior_baseline[identity] = {
                "locations": [context.location],
                "devices": [context.device_id],
                "times": [context.timestamp]
            }
            return False
        
        baseline = self._behavior_baseline[identity]
        
        # Check for new location
        if context.location and context.location not in baseline["locations"]:
            baseline["locations"].append(context.location)
            return True  # New location is anomalous
        
        # Check for new device
        if context.device_id not in baseline["devices"]:
            baseline["devices"].append(context.device_id)
            return True  # New device is anomalous
        
        return False
    
    def _log_access(self, context: SecurityContext, resource: str):
        """Log access attempt."""
        self._access_log.append({
            "identity": context.identity,
            "resource": resource,
            "timestamp": context.timestamp,
            "device": context.device_id,
            "location": context.location
        })
        
        # Keep only recent logs
        cutoff = time.time() - 3600  # 1 hour
        self._access_log = [
            log for log in self._access_log
            if log["timestamp"] > cutoff
        ]
    
    def update_trust(self, identity: str, delta: float):
        """
        Update trust score for identity.
        
        Args:
            identity: Identity to update
            delta: Trust score change
        """
        current = self._trust_scores.get(identity, 0.5)
        new_score = max(0.0, min(1.0, current + delta))
        self._trust_scores[identity] = new_score
    
    def get_trust_level(self, identity: str) -> TrustLevel:
        """Get trust level for identity."""
        score = self._trust_scores.get(identity, 0.5)
        
        if score >= 0.9:
            return TrustLevel.VERIFIED
        elif score >= 0.7:
            return TrustLevel.HIGH
        elif score >= 0.5:
            return TrustLevel.MEDIUM
        elif score >= 0.3:
            return TrustLevel.LOW
        else:
            return TrustLevel.NONE
    
    def create_session(self, identity: str,
                       device_id: str) -> str:
        """
        Create a new session.
        
        Args:
            identity: User identity
            device_id: Device identifier
            
        Returns:
            Session token
        """
        import secrets
        
        session_id = secrets.token_hex(32)
        
        self._sessions[session_id] = {
            "identity": identity,
            "device_id": device_id,
            "created": time.time(),
            "last_activity": time.time(),
            "verified": False
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate a session."""
        session = self._sessions.get(session_id)
        
        if not session:
            return None
        
        # Check session age (max 24 hours)
        if time.time() - session["created"] > 86400:
            del self._sessions[session_id]
            return None
        
        # Check inactivity (max 1 hour)
        if time.time() - session["last_activity"] > 3600:
            del self._sessions[session_id]
            return None
        
        # Update last activity
        session["last_activity"] = time.time()
        
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            "policies": len(self._policies),
            "active_sessions": len(self._sessions),
            "tracked_identities": len(self._trust_scores),
            "recent_access_attempts": len(self._access_log)
        }
    
    def __repr__(self) -> str:
        return f"ZeroTrustManager(policies={len(self._policies)})"
