"""
QMP Node Implementation
=======================

Network node for Quantum Mesh Protocol.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import hashlib
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Node operational states."""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    ONLINE = "online"
    SYNCING = "syncing"
    ERROR = "error"


class NodeRole(Enum):
    """Node roles in the mesh."""
    STANDARD = "standard"
    RELAY = "relay"
    GATEWAY = "gateway"
    COORDINATOR = "coordinator"


@dataclass
class NodeCapabilities:
    """Node capability flags."""
    quantum_crypto: bool = True
    relay: bool = True
    storage: bool = False
    compute: bool = False
    gateway: bool = False
    max_connections: int = 100
    bandwidth_mbps: float = 100.0


@dataclass
class PeerInfo:
    """Information about a peer node."""
    node_id: str
    quantum_signature: bytes
    address: str
    port: int
    role: NodeRole
    capabilities: NodeCapabilities
    latency_ms: float = 0.0
    last_seen: float = 0.0
    trust_score: float = 1.0


class QMPNode:
    """
    Quantum Mesh Protocol network node.
    
    Represents a node in the QMP mesh network with:
    - Quantum signature identity
    - Peer management
    - Message routing
    - Service discovery
    
    Example:
        >>> node = QMPNode(role=NodeRole.RELAY)
        >>> await node.start()
        >>> await node.connect_to_peer("peer_address:7777")
    """
    
    def __init__(self, node_id: Optional[str] = None,
                 role: NodeRole = NodeRole.STANDARD,
                 capabilities: Optional[NodeCapabilities] = None,
                 port: int = 7777,
                 language: str = "en"):
        """
        Initialize QMP node.
        
        Args:
            node_id: Unique node identifier
            role: Node role in the mesh
            capabilities: Node capabilities
            port: Network port
            language: Language for messages
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.role = role
        self.capabilities = capabilities or NodeCapabilities()
        self.port = port
        self.language = language
        
        # Generate quantum signature
        self._quantum_signature = self._generate_signature()
        
        # Peer management
        self._peers: Dict[str, PeerInfo] = {}
        self._pending_peers: Dict[str, Dict] = {}
        
        # Services
        self._services: Dict[str, Dict[str, Any]] = {}
        
        # State
        self._state = NodeState.OFFLINE
        self._start_time = 0.0
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info(f"QMP Node created: {self.node_id}, role={role.value}")
    
    def _generate_signature(self) -> bytes:
        """Generate quantum signature."""
        import secrets
        
        # Quantum-safe signature generation
        entropy = secrets.token_bytes(64)
        node_data = f"{self.node_id}:{self.role.value}:{time.time()}".encode()
        
        signature = hashlib.sha3_512(entropy + node_data).digest()
        return signature
    
    @property
    def quantum_signature(self) -> bytes:
        """Get node's quantum signature."""
        return self._quantum_signature
    
    @property
    def state(self) -> NodeState:
        """Get current node state."""
        return self._state
    
    async def start(self) -> bool:
        """Start the node."""
        if self._state != NodeState.OFFLINE:
            return False
        
        self._state = NodeState.INITIALIZING
        self._start_time = time.time()
        
        try:
            # Initialize networking
            await self._init_networking()
            
            # Start services
            await self._start_services()
            
            self._state = NodeState.ONLINE
            self._emit_event("node_started", {"node_id": self.node_id})
            
            logger.info(f"Node {self.node_id} started")
            return True
            
        except Exception as e:
            self._state = NodeState.ERROR
            logger.error(f"Failed to start node: {e}")
            return False
    
    async def stop(self):
        """Stop the node."""
        if self._state == NodeState.OFFLINE:
            return
        
        # Disconnect from peers
        for peer_id in list(self._peers.keys()):
            await self.disconnect_peer(peer_id)
        
        # Stop services
        await self._stop_services()
        
        self._state = NodeState.OFFLINE
        self._emit_event("node_stopped", {"node_id": self.node_id})
        
        logger.info(f"Node {self.node_id} stopped")
    
    async def _init_networking(self):
        """Initialize network layer."""
        # In production, this would set up actual network sockets
        pass
    
    async def _start_services(self):
        """Start node services."""
        # Discovery service
        self._services["discovery"] = {
            "enabled": True,
            "interval": 30
        }
        
        # Heartbeat service
        self._services["heartbeat"] = {
            "enabled": True,
            "interval": 10
        }
    
    async def _stop_services(self):
        """Stop node services."""
        self._services.clear()
    
    async def connect_to_peer(self, address: str) -> Optional[str]:
        """
        Connect to a peer node.
        
        Args:
            address: Peer address (host:port)
            
        Returns:
            Peer node ID if successful
        """
        if self._state != NodeState.ONLINE:
            return None
        
        try:
            host, port = address.rsplit(":", 1)
            port = int(port)
            
            # Simulate connection
            peer_id = str(uuid.uuid4())
            
            peer_info = PeerInfo(
                node_id=peer_id,
                quantum_signature=self._generate_signature(),
                address=host,
                port=port,
                role=NodeRole.STANDARD,
                capabilities=NodeCapabilities(),
                last_seen=time.time()
            )
            
            self._peers[peer_id] = peer_info
            self._emit_event("peer_connected", {"peer_id": peer_id})
            
            logger.info(f"Connected to peer: {peer_id}")
            return peer_id
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}: {e}")
            return None
    
    async def disconnect_peer(self, peer_id: str) -> bool:
        """Disconnect from a peer."""
        if peer_id in self._peers:
            del self._peers[peer_id]
            self._emit_event("peer_disconnected", {"peer_id": peer_id})
            logger.info(f"Disconnected from peer: {peer_id}")
            return True
        return False
    
    def add_peer(self, peer_info: PeerInfo) -> bool:
        """Add a peer directly."""
        if peer_info.node_id == self.node_id:
            return False
        
        if len(self._peers) >= self.capabilities.max_connections:
            return False
        
        self._peers[peer_info.node_id] = peer_info
        self._emit_event("peer_added", {"peer_id": peer_info.node_id})
        return True
    
    def get_peer(self, peer_id: str) -> Optional[PeerInfo]:
        """Get peer information."""
        return self._peers.get(peer_id)
    
    def get_peers(self) -> List[PeerInfo]:
        """Get all connected peers."""
        return list(self._peers.values())
    
    def get_peer_count(self) -> int:
        """Get number of connected peers."""
        return len(self._peers)
    
    async def send_to_peer(self, peer_id: str, data: bytes) -> bool:
        """Send data to a specific peer."""
        if peer_id not in self._peers:
            return False
        
        # In production, this would send via network
        self._emit_event("data_sent", {
            "peer_id": peer_id,
            "size": len(data)
        })
        
        return True
    
    async def broadcast(self, data: bytes, exclude: Optional[List[str]] = None):
        """Broadcast data to all peers."""
        exclude = exclude or []
        
        for peer_id in self._peers:
            if peer_id not in exclude:
                await self.send_to_peer(peer_id, data)
    
    def register_service(self, name: str, 
                         handler: Callable,
                         metadata: Optional[Dict] = None):
        """Register a service on this node."""
        self._services[name] = {
            "handler": handler,
            "metadata": metadata or {},
            "registered": time.time()
        }
        
        self._emit_event("service_registered", {"service": name})
        logger.info(f"Service registered: {name}")
    
    def unregister_service(self, name: str) -> bool:
        """Unregister a service."""
        if name in self._services:
            del self._services[name]
            self._emit_event("service_unregistered", {"service": name})
            return True
        return False
    
    def get_services(self) -> List[str]:
        """Get list of registered services."""
        return list(self._services.keys())
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable):
        """Unregister event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].remove(handler)
    
    def _emit_event(self, event: str, data: Dict[str, Any]):
        """Emit event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def update_trust_score(self, peer_id: str, delta: float):
        """Update peer trust score."""
        if peer_id in self._peers:
            peer = self._peers[peer_id]
            peer.trust_score = max(0.0, min(1.0, peer.trust_score + delta))
    
    def get_trusted_peers(self, min_trust: float = 0.5) -> List[PeerInfo]:
        """Get peers with trust score above threshold."""
        return [p for p in self._peers.values() if p.trust_score >= min_trust]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "state": self._state.value,
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "peers": len(self._peers),
            "services": len(self._services),
            "capabilities": {
                "quantum_crypto": self.capabilities.quantum_crypto,
                "relay": self.capabilities.relay,
                "storage": self.capabilities.storage,
                "compute": self.capabilities.compute,
                "gateway": self.capabilities.gateway
            }
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get node information for sharing."""
        return {
            "node_id": self.node_id,
            "quantum_signature": self._quantum_signature.hex(),
            "role": self.role.value,
            "port": self.port,
            "capabilities": {
                "quantum_crypto": self.capabilities.quantum_crypto,
                "relay": self.capabilities.relay,
                "max_connections": self.capabilities.max_connections
            }
        }
    
    def __repr__(self) -> str:
        return (f"QMPNode(id='{self.node_id[:8]}...', "
                f"role={self.role.value}, state={self._state.value})")
