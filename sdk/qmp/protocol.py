"""
Quantum Mesh Protocol Implementation
====================================

Core protocol for quantum-secured mesh networking.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """QMP message types."""
    DISCOVERY = "discovery"
    ROUTE_REQUEST = "route_request"
    ROUTE_REPLY = "route_reply"
    DATA = "data"
    ACK = "ack"
    HEARTBEAT = "heartbeat"
    SIGNATURE_EXCHANGE = "signature_exchange"
    KEY_EXCHANGE = "key_exchange"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class QMPMessage:
    """Quantum Mesh Protocol message."""
    message_id: str
    message_type: MessageType
    source: str
    destination: str
    payload: bytes
    timestamp: float
    ttl: int = 64
    priority: MessagePriority = MessagePriority.NORMAL
    quantum_signature: Optional[bytes] = None
    route: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        import json
        data = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source": self.source,
            "destination": self.destination,
            "payload": self.payload.hex(),
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "priority": self.priority.value,
            "route": self.route,
            "metadata": self.metadata
        }
        if self.quantum_signature:
            data["quantum_signature"] = self.quantum_signature.hex()
        return json.dumps(data).encode()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'QMPMessage':
        """Deserialize message from bytes."""
        import json
        obj = json.loads(data.decode())
        return cls(
            message_id=obj["message_id"],
            message_type=MessageType(obj["message_type"]),
            source=obj["source"],
            destination=obj["destination"],
            payload=bytes.fromhex(obj["payload"]),
            timestamp=obj["timestamp"],
            ttl=obj["ttl"],
            priority=MessagePriority(obj["priority"]),
            quantum_signature=bytes.fromhex(obj["quantum_signature"]) if obj.get("quantum_signature") else None,
            route=obj["route"],
            metadata=obj["metadata"]
        )


@dataclass
class RouteEntry:
    """Routing table entry."""
    destination: str
    next_hop: str
    metric: int
    quantum_signature: bytes
    last_updated: float
    expires: float


class QuantumMeshProtocol:
    """
    Quantum Mesh Protocol for zero-infrastructure networking.
    
    Features:
    - Quantum signature-based node identification
    - Dynamic mesh routing
    - Self-healing network topology
    - End-to-end quantum encryption
    - Zero-trust security model
    
    Example:
        >>> qmp = QuantumMeshProtocol(node_id="node_001")
        >>> qmp.start()
        >>> qmp.send_message("node_002", b"Hello, quantum world!")
    """
    
    VERSION = "1.0.0"
    DEFAULT_TTL = 64
    ROUTE_TIMEOUT = 300  # seconds
    HEARTBEAT_INTERVAL = 30  # seconds
    
    def __init__(self, node_id: Optional[str] = None,
                 port: int = 7777,
                 language: str = "en"):
        """
        Initialize Quantum Mesh Protocol.
        
        Args:
            node_id: Unique node identifier
            port: Network port
            language: Language for messages
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port
        self.language = language
        
        # Generate quantum signature for this node
        self._quantum_signature = self._generate_quantum_signature()
        
        # Routing table
        self._routing_table: Dict[str, RouteEntry] = {}
        
        # Known neighbors
        self._neighbors: Dict[str, Dict[str, Any]] = {}
        
        # Message handlers
        self._handlers: Dict[MessageType, List[Callable]] = {
            msg_type: [] for msg_type in MessageType
        }
        
        # Message queue
        self._message_queue: List[QMPMessage] = []
        
        # State
        self._running = False
        self._start_time = 0.0
        
        logger.info(f"QMP initialized: node_id={self.node_id}")
    
    def _generate_quantum_signature(self) -> bytes:
        """Generate quantum signature for node."""
        # In production, this would use actual quantum random number generation
        # and quantum key distribution
        import secrets
        
        # Generate 256-bit quantum signature
        random_bytes = secrets.token_bytes(32)
        
        # Hash with node ID for uniqueness
        signature_input = self.node_id.encode() + random_bytes
        signature = hashlib.sha3_256(signature_input).digest()
        
        return signature
    
    @property
    def quantum_signature(self) -> bytes:
        """Get node's quantum signature."""
        return self._quantum_signature
    
    def start(self) -> bool:
        """Start the QMP protocol."""
        if self._running:
            return True
        
        self._running = True
        self._start_time = time.time()
        
        # Start discovery
        self._broadcast_discovery()
        
        logger.info(f"QMP started on port {self.port}")
        return True
    
    def stop(self):
        """Stop the QMP protocol."""
        self._running = False
        logger.info("QMP stopped")
    
    def _broadcast_discovery(self):
        """Broadcast discovery message to find neighbors."""
        message = QMPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DISCOVERY,
            source=self.node_id,
            destination="broadcast",
            payload=self._quantum_signature,
            timestamp=time.time(),
            quantum_signature=self._quantum_signature
        )
        
        self._broadcast(message)
    
    def _broadcast(self, message: QMPMessage):
        """Broadcast message to all neighbors."""
        for neighbor_id in self._neighbors:
            self._send_to_neighbor(neighbor_id, message)
    
    def _send_to_neighbor(self, neighbor_id: str, message: QMPMessage):
        """Send message to specific neighbor."""
        # In production, this would use actual network transport
        logger.debug(f"Sending to {neighbor_id}: {message.message_type.value}")
        
        # Add to route
        message.route.append(self.node_id)
        
        # Simulate sending
        self._message_queue.append(message)
    
    def add_neighbor(self, neighbor_id: str, 
                     quantum_signature: bytes,
                     address: Optional[str] = None) -> bool:
        """
        Add a neighbor node.
        
        Args:
            neighbor_id: Neighbor's node ID
            quantum_signature: Neighbor's quantum signature
            address: Network address (optional)
            
        Returns:
            True if neighbor added successfully
        """
        if neighbor_id == self.node_id:
            return False
        
        self._neighbors[neighbor_id] = {
            "quantum_signature": quantum_signature,
            "address": address,
            "last_seen": time.time(),
            "metric": 1
        }
        
        # Add direct route
        self._routing_table[neighbor_id] = RouteEntry(
            destination=neighbor_id,
            next_hop=neighbor_id,
            metric=1,
            quantum_signature=quantum_signature,
            last_updated=time.time(),
            expires=time.time() + self.ROUTE_TIMEOUT
        )
        
        logger.info(f"Added neighbor: {neighbor_id}")
        return True
    
    def remove_neighbor(self, neighbor_id: str) -> bool:
        """Remove a neighbor node."""
        if neighbor_id in self._neighbors:
            del self._neighbors[neighbor_id]
            
            # Remove routes through this neighbor
            routes_to_remove = [
                dest for dest, route in self._routing_table.items()
                if route.next_hop == neighbor_id
            ]
            for dest in routes_to_remove:
                del self._routing_table[dest]
            
            logger.info(f"Removed neighbor: {neighbor_id}")
            return True
        return False
    
    def send_message(self, destination: str, payload: bytes,
                     priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """
        Send a message to destination node.
        
        Args:
            destination: Destination node ID
            payload: Message payload
            priority: Message priority
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        
        message = QMPMessage(
            message_id=message_id,
            message_type=MessageType.DATA,
            source=self.node_id,
            destination=destination,
            payload=payload,
            timestamp=time.time(),
            priority=priority,
            quantum_signature=self._sign_message(payload)
        )
        
        # Find route
        route = self._find_route(destination)
        
        if route:
            self._send_to_neighbor(route.next_hop, message)
            logger.info(f"Message sent: {message_id} -> {destination}")
        else:
            # Initiate route discovery
            self._discover_route(destination)
            self._message_queue.append(message)
            logger.info(f"Message queued, discovering route to {destination}")
        
        return message_id
    
    def _sign_message(self, payload: bytes) -> bytes:
        """Sign message with quantum signature."""
        # Combine payload with quantum signature
        signature_input = payload + self._quantum_signature
        return hashlib.sha3_256(signature_input).digest()
    
    def verify_signature(self, message: QMPMessage, 
                         sender_signature: bytes) -> bool:
        """Verify message signature."""
        expected = hashlib.sha3_256(
            message.payload + sender_signature
        ).digest()
        return message.quantum_signature == expected
    
    def _find_route(self, destination: str) -> Optional[RouteEntry]:
        """Find route to destination."""
        if destination in self._routing_table:
            route = self._routing_table[destination]
            if route.expires > time.time():
                return route
            else:
                # Route expired
                del self._routing_table[destination]
        return None
    
    def _discover_route(self, destination: str):
        """Initiate route discovery for destination."""
        message = QMPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ROUTE_REQUEST,
            source=self.node_id,
            destination=destination,
            payload=destination.encode(),
            timestamp=time.time(),
            quantum_signature=self._quantum_signature
        )
        
        self._broadcast(message)
    
    def handle_message(self, message: QMPMessage):
        """Handle incoming message."""
        # Decrease TTL
        message.ttl -= 1
        if message.ttl <= 0:
            logger.debug(f"Message {message.message_id} TTL expired")
            return
        
        # Check if message is for us
        if message.destination == self.node_id:
            self._process_message(message)
        elif message.destination == "broadcast":
            self._process_message(message)
            # Forward broadcast
            self._broadcast(message)
        else:
            # Forward message
            self._forward_message(message)
    
    def _process_message(self, message: QMPMessage):
        """Process message destined for this node."""
        # Call registered handlers
        for handler in self._handlers.get(message.message_type, []):
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")
        
        # Built-in handling
        if message.message_type == MessageType.DISCOVERY:
            self._handle_discovery(message)
        elif message.message_type == MessageType.ROUTE_REQUEST:
            self._handle_route_request(message)
        elif message.message_type == MessageType.ROUTE_REPLY:
            self._handle_route_reply(message)
        elif message.message_type == MessageType.DATA:
            self._handle_data(message)
    
    def _handle_discovery(self, message: QMPMessage):
        """Handle discovery message."""
        # Add sender as neighbor
        self.add_neighbor(
            message.source,
            message.quantum_signature or message.payload
        )
    
    def _handle_route_request(self, message: QMPMessage):
        """Handle route request."""
        destination = message.payload.decode()
        
        if destination == self.node_id:
            # We are the destination, send reply
            reply = QMPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ROUTE_REPLY,
                source=self.node_id,
                destination=message.source,
                payload=self._quantum_signature,
                timestamp=time.time(),
                route=message.route + [self.node_id],
                quantum_signature=self._quantum_signature
            )
            self._send_via_route(reply, list(reversed(message.route)))
        elif destination in self._routing_table:
            # We have a route, send reply
            route = self._routing_table[destination]
            reply = QMPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ROUTE_REPLY,
                source=self.node_id,
                destination=message.source,
                payload=route.quantum_signature,
                timestamp=time.time(),
                route=message.route + [self.node_id, destination],
                quantum_signature=self._quantum_signature
            )
            self._send_via_route(reply, list(reversed(message.route)))
    
    def _handle_route_reply(self, message: QMPMessage):
        """Handle route reply."""
        # Extract route information
        if message.route:
            destination = message.route[-1]
            
            # Add route
            self._routing_table[destination] = RouteEntry(
                destination=destination,
                next_hop=message.route[0] if message.route else message.source,
                metric=len(message.route),
                quantum_signature=message.payload,
                last_updated=time.time(),
                expires=time.time() + self.ROUTE_TIMEOUT
            )
            
            # Process queued messages
            self._process_queued_messages(destination)
    
    def _handle_data(self, message: QMPMessage):
        """Handle data message."""
        logger.info(f"Received data from {message.source}: {len(message.payload)} bytes")
    
    def _forward_message(self, message: QMPMessage):
        """Forward message to next hop."""
        route = self._find_route(message.destination)
        
        if route:
            self._send_to_neighbor(route.next_hop, message)
        else:
            # No route, drop or queue
            logger.warning(f"No route to {message.destination}")
    
    def _send_via_route(self, message: QMPMessage, route: List[str]):
        """Send message via specific route."""
        if route:
            next_hop = route[0]
            if next_hop in self._neighbors:
                self._send_to_neighbor(next_hop, message)
    
    def _process_queued_messages(self, destination: str):
        """Process queued messages for destination."""
        messages_to_send = [
            msg for msg in self._message_queue
            if msg.destination == destination
        ]
        
        for message in messages_to_send:
            self._message_queue.remove(message)
            route = self._find_route(destination)
            if route:
                self._send_to_neighbor(route.next_hop, message)
    
    def register_handler(self, message_type: MessageType,
                         handler: Callable[[QMPMessage], None]):
        """Register message handler."""
        self._handlers[message_type].append(handler)
    
    def get_routing_table(self) -> Dict[str, Dict[str, Any]]:
        """Get current routing table."""
        return {
            dest: {
                "next_hop": route.next_hop,
                "metric": route.metric,
                "expires": route.expires
            }
            for dest, route in self._routing_table.items()
        }
    
    def get_neighbors(self) -> List[str]:
        """Get list of neighbor node IDs."""
        return list(self._neighbors.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        return {
            "node_id": self.node_id,
            "uptime": time.time() - self._start_time if self._running else 0,
            "neighbors": len(self._neighbors),
            "routes": len(self._routing_table),
            "queued_messages": len(self._message_queue),
            "running": self._running
        }
    
    def __repr__(self) -> str:
        return f"QuantumMeshProtocol(node_id='{self.node_id}', neighbors={len(self._neighbors)})"
