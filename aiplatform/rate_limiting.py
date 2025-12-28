"""
Rate Limiting Configuration and Implementation

This module provides comprehensive rate limiting functionality using the token bucket algorithm.
Supports multiple tiers (standard, premium, enterprise), per-user limits, and endpoint-specific rules.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum
import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import hashlib
import hmac

# ============================================================================
# Rate Limiting Tier Definitions
# ============================================================================

class TierType(str, Enum):
    """Available subscription tiers for rate limiting."""
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


@dataclass
class RateLimitTier:
    """Configuration for a specific rate limiting tier."""
    name: TierType
    requests_per_hour: int
    requests_per_minute: int
    requests_per_second: int
    concurrent_requests: int
    burst_capacity: float = 1.5  # Multiplier for token bucket burst
    cost_multiplier: float = 1.0  # Endpoint cost adjustment
    description: str = ""
    features: List[str] = field(default_factory=list)


# Standard Tier Configuration
TIER_STANDARD = RateLimitTier(
    name=TierType.STANDARD,
    requests_per_hour=1000,
    requests_per_minute=30,
    requests_per_second=1,
    concurrent_requests=10,
    burst_capacity=1.5,
    cost_multiplier=1.0,
    description="Standard tier with basic rate limits",
    features=[
        "REST API access",
        "Basic support",
        "Weekly data exports",
        "Standard response times",
    ]
)

# Premium Tier Configuration
TIER_PREMIUM = RateLimitTier(
    name=TierType.PREMIUM,
    requests_per_hour=10000,
    requests_per_minute=300,
    requests_per_second=10,
    concurrent_requests=100,
    burst_capacity=2.0,
    cost_multiplier=0.8,
    description="Premium tier with higher rate limits and priority",
    features=[
        "REST API access",
        "GraphQL API access",
        "Priority support",
        "Daily data exports",
        "Advanced analytics",
        "Webhook support",
        "Priority queue processing",
    ]
)

# Enterprise Tier Configuration
TIER_ENTERPRISE = RateLimitTier(
    name=TierType.ENTERPRISE,
    requests_per_hour=float('inf'),
    requests_per_minute=float('inf'),
    requests_per_second=float('inf'),
    concurrent_requests=1000,
    burst_capacity=float('inf'),
    cost_multiplier=0.5,
    description="Enterprise tier with unlimited requests",
    features=[
        "Unlimited REST API access",
        "Unlimited GraphQL access",
        "24/7 dedicated support",
        "Real-time data exports",
        "Custom integrations",
        "Advanced security features",
        "Custom rate limit policies",
        "SLA guarantees",
    ]
)

# Admin Tier Configuration
TIER_ADMIN = RateLimitTier(
    name=TierType.ADMIN,
    requests_per_hour=float('inf'),
    requests_per_minute=float('inf'),
    requests_per_second=float('inf'),
    concurrent_requests=float('inf'),
    burst_capacity=float('inf'),
    cost_multiplier=0.0,
    description="Admin tier with unlimited access",
    features=[
        "Full unrestricted access",
        "All platform features",
        "System administration",
    ]
)

TIER_MAP = {
    TierType.STANDARD: TIER_STANDARD,
    TierType.PREMIUM: TIER_PREMIUM,
    TierType.ENTERPRISE: TIER_ENTERPRISE,
    TierType.ADMIN: TIER_ADMIN,
}

# ============================================================================
# Endpoint-Specific Rate Limits
# ============================================================================

@dataclass
class EndpointRateLimit:
    """Rate limit configuration for specific endpoints."""
    path: str
    method: str
    cost: float = 1.0  # Relative cost (default=1 token per request)
    tier_overrides: Dict[TierType, int] = field(default_factory=dict)  # Per-tier overrides
    daily_limit: Optional[int] = None  # Optional daily limit
    description: str = ""


# Define endpoint-specific rate limits
ENDPOINT_LIMITS = [
    # Health and Status Endpoints
    EndpointRateLimit(
        path="/health",
        method="GET",
        cost=0.1,
        description="Health check endpoint with minimal cost"
    ),
    EndpointRateLimit(
        path="/status",
        method="GET",
        cost=0.1,
        description="Status endpoint with minimal cost"
    ),
    
    # Optimization Endpoints
    EndpointRateLimit(
        path="/optimize",
        method="POST",
        cost=10.0,
        description="Quantum optimization - high cost due to computation",
        tier_overrides={
            TierType.STANDARD: 10,  # 10 optimization requests/hour
            TierType.PREMIUM: 100,
            TierType.ENTERPRISE: float('inf'),
        }
    ),
    EndpointRateLimit(
        path="/optimize/{job_id}",
        method="GET",
        cost=0.5,
        description="Get optimization result"
    ),
    EndpointRateLimit(
        path="/optimize/{job_id}/cancel",
        method="POST",
        cost=1.0,
        description="Cancel optimization job"
    ),
    
    # Vision Analysis Endpoints
    EndpointRateLimit(
        path="/vision/analyze",
        method="POST",
        cost=5.0,
        description="Single image analysis",
        tier_overrides={
            TierType.STANDARD: 50,  # 50 analyses/hour
            TierType.PREMIUM: 500,
            TierType.ENTERPRISE: float('inf'),
        }
    ),
    EndpointRateLimit(
        path="/vision/batch",
        method="POST",
        cost=20.0,
        description="Batch image analysis",
        tier_overrides={
            TierType.STANDARD: 5,  # 5 batch jobs/hour
            TierType.PREMIUM: 50,
            TierType.ENTERPRISE: float('inf'),
        }
    ),
    EndpointRateLimit(
        path="/vision/batch/{job_id}",
        method="GET",
        cost=0.5,
        description="Get batch analysis results"
    ),
    
    # Federated Learning Endpoints
    EndpointRateLimit(
        path="/federated/train",
        method="POST",
        cost=15.0,
        description="Start federated training",
        tier_overrides={
            TierType.STANDARD: 5,  # 5 training jobs/hour
            TierType.PREMIUM: 20,
            TierType.ENTERPRISE: float('inf'),
        }
    ),
    EndpointRateLimit(
        path="/federated/train/{job_id}",
        method="GET",
        cost=0.5,
        description="Get training status"
    ),
    
    # Inference Endpoints
    EndpointRateLimit(
        path="/infer/predict",
        method="POST",
        cost=2.0,
        description="Single inference prediction",
        tier_overrides={
            TierType.STANDARD: 100,
            TierType.PREMIUM: 1000,
            TierType.ENTERPRISE: float('inf'),
        }
    ),
    EndpointRateLimit(
        path="/infer/batch",
        method="POST",
        cost=10.0,
        description="Batch inference",
        tier_overrides={
            TierType.STANDARD: 20,
            TierType.PREMIUM: 200,
            TierType.ENTERPRISE: float('inf'),
        }
    ),
    
    # Model Management Endpoints
    EndpointRateLimit(
        path="/models",
        method="GET",
        cost=0.5,
        description="List models"
    ),
    EndpointRateLimit(
        path="/models/{model_id}",
        method="GET",
        cost=0.5,
        description="Get model details"
    ),
    
    # Project Management Endpoints
    EndpointRateLimit(
        path="/projects",
        method="GET",
        cost=1.0,
        description="List projects"
    ),
    EndpointRateLimit(
        path="/projects",
        method="POST",
        cost=1.0,
        description="Create project"
    ),
    EndpointRateLimit(
        path="/projects/{project_id}",
        method="GET",
        cost=0.5,
        description="Get project details"
    ),
    EndpointRateLimit(
        path="/projects/{project_id}",
        method="PUT",
        cost=1.0,
        description="Update project"
    ),
    EndpointRateLimit(
        path="/projects/{project_id}",
        method="DELETE",
        cost=1.0,
        description="Delete project"
    ),
    
    # Admin Endpoints
    EndpointRateLimit(
        path="/admin/metrics",
        method="GET",
        cost=1.0,
        description="System metrics (admin only)"
    ),
    EndpointRateLimit(
        path="/admin/usage/{user_id}",
        method="GET",
        cost=0.5,
        description="User usage statistics (admin only)"
    ),
]

# ============================================================================
# Token Bucket Algorithm Implementation
# ============================================================================

class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, capacity: float, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: float = 1.0) -> bool:
        """
        Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were available, False otherwise
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self) -> None:
        """Refill bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def get_tokens(self) -> float:
        """Get current number of tokens."""
        with self.lock:
            self._refill()
            return self.tokens
    
    def reset(self) -> None:
        """Reset bucket to full capacity."""
        with self.lock:
            self.tokens = self.capacity
            self.last_refill = time.time()


# ============================================================================
# Rate Limit Checker
# ============================================================================

@dataclass
class RateLimitInfo:
    """Information about current rate limit status."""
    limit: int
    remaining: int
    reset_at: datetime
    reset_in_seconds: int
    is_limited: bool = False
    retry_after_seconds: Optional[int] = None


class RateLimiter:
    """Main rate limiter implementation."""
    
    def __init__(self):
        """Initialize the rate limiter."""
        self.user_buckets: Dict[str, TokenBucket] = {}
        self.endpoint_buckets: Dict[str, TokenBucket] = {}
        self.user_concurrent: Dict[str, int] = {}
        self.endpoint_concurrent: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def check_rate_limit(
        self,
        user_id: str,
        tier: TierType,
        endpoint_path: str,
        endpoint_method: str,
        cost: float = 1.0
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check if request should be rate limited.
        
        Args:
            user_id: User identifier
            tier: User's subscription tier
            endpoint_path: API endpoint path
            endpoint_method: HTTP method
            cost: Token cost of this request
            
        Returns:
            Tuple of (allowed: bool, rate_limit_info: RateLimitInfo)
        """
        tier_config = TIER_MAP[tier]
        
        # Check if tier has unlimited access
        if tier_config.requests_per_hour == float('inf'):
            reset_time = datetime.utcnow() + timedelta(hours=1)
            return True, RateLimitInfo(
                limit=float('inf'),
                remaining=float('inf'),
                reset_at=reset_time,
                reset_in_seconds=3600,
                is_limited=False
            )
        
        # Get or create user bucket
        user_key = f"user:{user_id}"
        if user_key not in self.user_buckets:
            with self.lock:
                if user_key not in self.user_buckets:
                    refill_rate = tier_config.requests_per_hour / 3600.0
                    capacity = tier_config.requests_per_hour * tier_config.burst_capacity
                    self.user_buckets[user_key] = TokenBucket(capacity, refill_rate)
        
        # Get or create endpoint bucket
        endpoint_key = f"{endpoint_path}:{endpoint_method}"
        if endpoint_key not in self.endpoint_buckets:
            with self.lock:
                if endpoint_key not in self.endpoint_buckets:
                    # Find endpoint configuration
                    endpoint_limit = self._find_endpoint_limit(endpoint_path, endpoint_method)
                    refill_rate = tier_config.requests_per_hour / 3600.0
                    capacity = tier_config.requests_per_hour * tier_config.burst_capacity
                    self.endpoint_buckets[endpoint_key] = TokenBucket(capacity, refill_rate)
        
        # Try to consume tokens
        bucket = self.user_buckets[user_key]
        allowed = bucket.consume(cost)
        
        # Get remaining tokens
        remaining = int(bucket.get_tokens())
        reset_time = datetime.utcnow() + timedelta(hours=1)
        
        if not allowed:
            # Calculate retry-after
            retry_after = int((cost - bucket.get_tokens()) / (tier_config.requests_per_hour / 3600.0))
            return False, RateLimitInfo(
                limit=tier_config.requests_per_hour,
                remaining=max(0, remaining),
                reset_at=reset_time,
                reset_in_seconds=3600,
                is_limited=True,
                retry_after_seconds=max(1, retry_after)
            )
        
        return True, RateLimitInfo(
            limit=tier_config.requests_per_hour,
            remaining=max(0, remaining),
            reset_at=reset_time,
            reset_in_seconds=3600,
            is_limited=False
        )
    
    def check_concurrent_limit(
        self,
        user_id: str,
        tier: TierType
    ) -> bool:
        """Check if user has exceeded concurrent request limit."""
        tier_config = TIER_MAP[tier]
        
        with self.lock:
            current = self.user_concurrent.get(user_id, 0)
            if current >= tier_config.concurrent_requests:
                return False
            self.user_concurrent[user_id] = current + 1
        
        return True
    
    def release_concurrent(self, user_id: str) -> None:
        """Release a concurrent request slot."""
        with self.lock:
            self.user_concurrent[user_id] = max(0, self.user_concurrent.get(user_id, 0) - 1)
    
    def reset_user_limits(self, user_id: str) -> None:
        """Reset all limits for a user (admin function)."""
        user_key = f"user:{user_id}"
        with self.lock:
            if user_key in self.user_buckets:
                self.user_buckets[user_key].reset()
            self.user_concurrent[user_id] = 0
    
    def _find_endpoint_limit(
        self,
        path: str,
        method: str
    ) -> Optional[EndpointRateLimit]:
        """Find endpoint-specific rate limit configuration."""
        for limit in ENDPOINT_LIMITS:
            if limit.path == path and limit.method == method:
                return limit
        return None


# ============================================================================
# Middleware Integration
# ============================================================================

class RateLimitMiddleware:
    """Middleware for enforcing rate limits in web frameworks."""
    
    def __init__(self, rate_limiter: RateLimiter):
        """Initialize middleware with rate limiter instance."""
        self.rate_limiter = rate_limiter
    
    def check_request(
        self,
        user_id: str,
        user_tier: TierType,
        endpoint_path: str,
        endpoint_method: str
    ) -> Tuple[bool, Optional[RateLimitInfo]]:
        """
        Check if request is allowed.
        
        Args:
            user_id: User identifier from token
            user_tier: User's subscription tier
            endpoint_path: Request path
            endpoint_method: HTTP method
            
        Returns:
            Tuple of (allowed: bool, rate_limit_info: RateLimitInfo | None)
        """
        # Check concurrent limit
        if not self.rate_limiter.check_concurrent_limit(user_id, user_tier):
            reset_time = datetime.utcnow() + timedelta(seconds=60)
            return False, RateLimitInfo(
                limit=TIER_MAP[user_tier].concurrent_requests,
                remaining=0,
                reset_at=reset_time,
                reset_in_seconds=60,
                is_limited=True,
                retry_after_seconds=60
            )
        
        # Check rate limit
        allowed, info = self.rate_limiter.check_rate_limit(
            user_id=user_id,
            tier=user_tier,
            endpoint_path=endpoint_path,
            endpoint_method=endpoint_method
        )
        
        return allowed, info
    
    def release_concurrent(self, user_id: str) -> None:
        """Release concurrent request slot after handling."""
        self.rate_limiter.release_concurrent(user_id)


# ============================================================================
# Global Rate Limiter Instance
# ============================================================================

# Create global rate limiter instance
rate_limiter = RateLimiter()


# ============================================================================
# Configuration Summary
# ============================================================================

RATE_LIMIT_CONFIG = {
    "tiers": {
        "standard": {
            "requests_per_hour": 1000,
            "requests_per_minute": 30,
            "concurrent_requests": 10,
        },
        "premium": {
            "requests_per_hour": 10000,
            "requests_per_minute": 300,
            "concurrent_requests": 100,
        },
        "enterprise": {
            "requests_per_hour": float('inf'),
            "requests_per_minute": float('inf'),
            "concurrent_requests": 1000,
        },
    },
    "endpoints": {
        "high_cost": [
            "/optimize",
            "/federated/train",
            "/vision/batch",
        ],
        "medium_cost": [
            "/vision/analyze",
            "/infer/batch",
        ],
        "low_cost": [
            "/health",
            "/status",
            "/models",
            "/projects",
        ]
    },
    "response_headers": {
        "X-RateLimit-Limit": "Total limit for time period",
        "X-RateLimit-Remaining": "Requests remaining",
        "X-RateLimit-Reset": "Unix timestamp when limit resets",
        "X-RateLimit-RetryAfter": "Seconds to wait before retry (when limited)",
    }
}
