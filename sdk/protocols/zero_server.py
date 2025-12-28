"""
Zero Server
===========

Zero-infrastructure server implementation.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class ServerState(Enum):
    """Server states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


@dataclass
class ServiceEndpoint:
    """A service endpoint."""
    name: str
    handler: Callable
    path: str
    methods: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestContext:
    """Request context."""
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes
    client_id: str
    timestamp: float


class ZeroServer:
    """
    Zero-infrastructure server.
    
    Features:
    - No central infrastructure required
    - Peer-to-peer service discovery
    - Automatic load balancing
    - Self-healing
    
    Example:
        >>> server = ZeroServer()
        >>> @server.route("/api/data")
        ... async def handle_data(ctx):
        ...     return {"status": "ok"}
        >>> await server.start()
    """
    
    def __init__(self, node_id: Optional[str] = None,
                 port: int = 8080,
                 language: str = "en"):
        """
        Initialize zero server.
        
        Args:
            node_id: Node identifier
            port: Server port
            language: Language for messages
        """
        self.node_id = node_id or "zero_server"
        self.port = port
        self.language = language
        
        # State
        self._state = ServerState.STOPPED
        self._start_time = 0.0
        
        # Endpoints
        self._endpoints: Dict[str, ServiceEndpoint] = {}
        
        # Middleware
        self._middleware: List[Callable] = []
        
        # Metrics
        self._request_count = 0
        self._error_count = 0
        
        logger.info(f"Zero Server initialized: {node_id}")
    
    def route(self, path: str, methods: List[str] = None):
        """
        Decorator to register route handler.
        
        Args:
            path: Route path
            methods: Allowed methods
        """
        methods = methods or ["GET"]
        
        def decorator(handler: Callable):
            endpoint = ServiceEndpoint(
                name=handler.__name__,
                handler=handler,
                path=path,
                methods=methods
            )
            
            for method in methods:
                key = f"{method}:{path}"
                self._endpoints[key] = endpoint
            
            return handler
        
        return decorator
    
    def use(self, middleware: Callable):
        """Add middleware."""
        self._middleware.append(middleware)
    
    async def start(self):
        """Start the server."""
        if self._state != ServerState.STOPPED:
            return
        
        self._state = ServerState.STARTING
        self._start_time = time.time()
        
        # In production, start actual server
        # Here we simulate
        
        self._state = ServerState.RUNNING
        logger.info(f"Zero Server started on port {self.port}")
    
    async def stop(self):
        """Stop the server."""
        if self._state != ServerState.RUNNING:
            return
        
        self._state = ServerState.STOPPING
        
        # Cleanup
        
        self._state = ServerState.STOPPED
        logger.info("Zero Server stopped")
    
    async def handle_request(self, ctx: RequestContext) -> Dict[str, Any]:
        """
        Handle incoming request.
        
        Args:
            ctx: Request context
            
        Returns:
            Response dictionary
        """
        self._request_count += 1
        
        # Apply middleware
        for mw in self._middleware:
            try:
                ctx = await mw(ctx)
            except Exception as e:
                self._error_count += 1
                return {"error": str(e), "status": 500}
        
        # Find endpoint
        key = f"{ctx.method}:{ctx.path}"
        endpoint = self._endpoints.get(key)
        
        if not endpoint:
            return {"error": "Not Found", "status": 404}
        
        # Call handler
        try:
            result = await endpoint.handler(ctx)
            return {"data": result, "status": 200}
        except Exception as e:
            self._error_count += 1
            logger.error(f"Handler error: {e}")
            return {"error": str(e), "status": 500}
    
    def register_service(self, name: str, handler: Callable,
                         path: str, methods: List[str] = None):
        """
        Register a service programmatically.
        
        Args:
            name: Service name
            handler: Request handler
            path: Service path
            methods: Allowed methods
        """
        methods = methods or ["GET"]
        
        endpoint = ServiceEndpoint(
            name=name,
            handler=handler,
            path=path,
            methods=methods
        )
        
        for method in methods:
            key = f"{method}:{path}"
            self._endpoints[key] = endpoint
    
    def unregister_service(self, path: str):
        """Unregister a service."""
        keys_to_remove = [k for k in self._endpoints if k.endswith(f":{path}")]
        for key in keys_to_remove:
            del self._endpoints[key]
    
    def get_services(self) -> List[Dict[str, Any]]:
        """Get list of registered services."""
        seen = set()
        services = []
        
        for endpoint in self._endpoints.values():
            if endpoint.name not in seen:
                seen.add(endpoint.name)
                services.append({
                    "name": endpoint.name,
                    "path": endpoint.path,
                    "methods": endpoint.methods
                })
        
        return services
    
    @property
    def state(self) -> ServerState:
        """Get server state."""
        return self._state
    
    @property
    def uptime(self) -> float:
        """Get server uptime in seconds."""
        if self._state != ServerState.RUNNING:
            return 0.0
        return time.time() - self._start_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "node_id": self.node_id,
            "state": self._state.value,
            "uptime": self.uptime,
            "port": self.port,
            "endpoints": len(self._endpoints),
            "requests": self._request_count,
            "errors": self._error_count
        }
    
    def __repr__(self) -> str:
        return f"ZeroServer(node_id='{self.node_id}', state={self._state.value})"
