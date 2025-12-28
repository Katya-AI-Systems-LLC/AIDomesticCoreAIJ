"""
Object Signature Router
=======================

Route messages based on object quantum signatures.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ObjectDescriptor:
    """Descriptor for a routable object."""
    object_id: str
    quantum_signature: bytes
    object_type: str
    owner: str
    location_hints: List[str]
    metadata: Dict[str, Any]
    created: float
    expires: float


@dataclass
class RouteResult:
    """Result of object routing."""
    object_id: str
    found: bool
    endpoints: List[str]
    latency_ms: float
    hops: int
    verified: bool


class ObjectSignatureRouter:
    """
    Route to objects based on their quantum signatures.
    
    Provides:
    - Object registration and discovery
    - Signature-based routing
    - Location tracking
    - Caching and optimization
    
    Example:
        >>> router = ObjectSignatureRouter()
        >>> router.register_object("my_data", signature, ["node1", "node2"])
        >>> result = await router.route_to_object("my_data")
    """
    
    def __init__(self, node_id: Optional[str] = None,
                 cache_ttl: int = 300,
                 language: str = "en"):
        """
        Initialize object router.
        
        Args:
            node_id: This node's ID
            cache_ttl: Cache TTL in seconds
            language: Language for messages
        """
        self.node_id = node_id or "router"
        self.cache_ttl = cache_ttl
        self.language = language
        
        # Object registry
        self._objects: Dict[str, ObjectDescriptor] = {}
        
        # Signature index
        self._signature_index: Dict[bytes, str] = {}
        
        # Route cache
        self._route_cache: Dict[str, RouteResult] = {}
        
        # Route handlers
        self._handlers: Dict[str, Callable] = {}
        
        logger.info(f"Object router initialized: {node_id}")
    
    def register_object(self, object_id: str,
                        quantum_signature: bytes,
                        location_hints: List[str],
                        object_type: str = "data",
                        owner: Optional[str] = None,
                        metadata: Optional[Dict] = None,
                        ttl: int = 3600) -> bool:
        """
        Register an object for routing.
        
        Args:
            object_id: Unique object identifier
            quantum_signature: Object's quantum signature
            location_hints: Nodes where object can be found
            object_type: Type of object
            owner: Owner identifier
            metadata: Additional metadata
            ttl: Time-to-live in seconds
            
        Returns:
            True if registered successfully
        """
        current_time = time.time()
        
        descriptor = ObjectDescriptor(
            object_id=object_id,
            quantum_signature=quantum_signature,
            object_type=object_type,
            owner=owner or self.node_id,
            location_hints=location_hints,
            metadata=metadata or {},
            created=current_time,
            expires=current_time + ttl
        )
        
        self._objects[object_id] = descriptor
        self._signature_index[quantum_signature] = object_id
        
        # Invalidate cache
        if object_id in self._route_cache:
            del self._route_cache[object_id]
        
        logger.info(f"Registered object: {object_id}")
        return True
    
    def unregister_object(self, object_id: str) -> bool:
        """Unregister an object."""
        if object_id not in self._objects:
            return False
        
        descriptor = self._objects[object_id]
        
        # Remove from indices
        if descriptor.quantum_signature in self._signature_index:
            del self._signature_index[descriptor.quantum_signature]
        
        del self._objects[object_id]
        
        # Clear cache
        if object_id in self._route_cache:
            del self._route_cache[object_id]
        
        logger.info(f"Unregistered object: {object_id}")
        return True
    
    def update_location(self, object_id: str,
                        location_hints: List[str]) -> bool:
        """Update object location hints."""
        if object_id not in self._objects:
            return False
        
        self._objects[object_id].location_hints = location_hints
        
        # Invalidate cache
        if object_id in self._route_cache:
            del self._route_cache[object_id]
        
        return True
    
    async def route_to_object(self, object_id: str,
                               use_cache: bool = True) -> RouteResult:
        """
        Route to an object by ID.
        
        Args:
            object_id: Object to route to
            use_cache: Use cached routes
            
        Returns:
            RouteResult
        """
        start_time = time.time()
        
        # Check cache
        if use_cache and object_id in self._route_cache:
            cached = self._route_cache[object_id]
            # Check if still valid
            if object_id in self._objects:
                return cached
        
        # Look up object
        if object_id not in self._objects:
            return RouteResult(
                object_id=object_id,
                found=False,
                endpoints=[],
                latency_ms=(time.time() - start_time) * 1000,
                hops=0,
                verified=False
            )
        
        descriptor = self._objects[object_id]
        
        # Check expiration
        if descriptor.expires < time.time():
            self.unregister_object(object_id)
            return RouteResult(
                object_id=object_id,
                found=False,
                endpoints=[],
                latency_ms=(time.time() - start_time) * 1000,
                hops=0,
                verified=False
            )
        
        # Build result
        result = RouteResult(
            object_id=object_id,
            found=True,
            endpoints=descriptor.location_hints.copy(),
            latency_ms=(time.time() - start_time) * 1000,
            hops=1,
            verified=True
        )
        
        # Cache result
        self._route_cache[object_id] = result
        
        return result
    
    async def route_by_signature(self, 
                                  quantum_signature: bytes) -> RouteResult:
        """
        Route to object by quantum signature.
        
        Args:
            quantum_signature: Object's quantum signature
            
        Returns:
            RouteResult
        """
        object_id = self._signature_index.get(quantum_signature)
        
        if object_id:
            return await self.route_to_object(object_id)
        
        return RouteResult(
            object_id="unknown",
            found=False,
            endpoints=[],
            latency_ms=0,
            hops=0,
            verified=False
        )
    
    def find_objects_by_type(self, object_type: str) -> List[str]:
        """Find all objects of a given type."""
        return [
            obj_id for obj_id, desc in self._objects.items()
            if desc.object_type == object_type and desc.expires >= time.time()
        ]
    
    def find_objects_by_owner(self, owner: str) -> List[str]:
        """Find all objects owned by a given owner."""
        return [
            obj_id for obj_id, desc in self._objects.items()
            if desc.owner == owner and desc.expires >= time.time()
        ]
    
    def get_object_descriptor(self, object_id: str) -> Optional[ObjectDescriptor]:
        """Get object descriptor."""
        return self._objects.get(object_id)
    
    def register_handler(self, object_type: str, handler: Callable):
        """Register handler for object type."""
        self._handlers[object_type] = handler
    
    async def handle_request(self, object_id: str, 
                              request_data: Any) -> Any:
        """
        Handle request for an object.
        
        Args:
            object_id: Target object
            request_data: Request data
            
        Returns:
            Response from handler
        """
        if object_id not in self._objects:
            return None
        
        descriptor = self._objects[object_id]
        handler = self._handlers.get(descriptor.object_type)
        
        if handler:
            return await handler(descriptor, request_data)
        
        return None
    
    async def cleanup_expired(self) -> int:
        """Remove expired objects."""
        current_time = time.time()
        expired = [
            obj_id for obj_id, desc in self._objects.items()
            if desc.expires < current_time
        ]
        
        for obj_id in expired:
            self.unregister_object(obj_id)
        
        return len(expired)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "total_objects": len(self._objects),
            "cached_routes": len(self._route_cache),
            "handlers": len(self._handlers),
            "node_id": self.node_id
        }
    
    def __repr__(self) -> str:
        return f"ObjectSignatureRouter(objects={len(self._objects)})"
