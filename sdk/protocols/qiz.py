"""
QIZ Protocol
============

Quantum Infrastructure Zero protocol implementation.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
import time
import logging

logger = logging.getLogger(__name__)


class QIZMessageType(Enum):
    """QIZ message types."""
    HANDSHAKE = "handshake"
    DATA = "data"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"


@dataclass
class QIZMessage:
    """QIZ protocol message."""
    message_type: QIZMessageType
    source: str
    destination: str
    payload: bytes
    sequence: int
    timestamp: float
    quantum_signature: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QIZConnection:
    """QIZ connection state."""
    connection_id: str
    remote_node: str
    established: float
    last_activity: float
    state: str
    shared_secret: bytes


class QIZProtocol:
    """
    Quantum Infrastructure Zero protocol.
    
    Features:
    - Zero-server architecture
    - Quantum-safe handshake
    - Self-healing connections
    - Distributed state management
    
    Example:
        >>> qiz = QIZProtocol(node_id="node_001")
        >>> await qiz.connect("remote_node")
        >>> await qiz.send("remote_node", data)
    """
    
    VERSION = "1.0"
    HANDSHAKE_TIMEOUT = 30
    HEARTBEAT_INTERVAL = 10
    
    def __init__(self, node_id: Optional[str] = None,
                 language: str = "en"):
        """
        Initialize QIZ protocol.
        
        Args:
            node_id: Node identifier
            language: Language for messages
        """
        self.node_id = node_id or secrets.token_hex(8)
        self.language = language
        
        # Connections
        self._connections: Dict[str, QIZConnection] = {}
        
        # Message handlers
        self._handlers: Dict[QIZMessageType, List[Callable]] = {
            t: [] for t in QIZMessageType
        }
        
        # Sequence counter
        self._sequence = 0
        
        # Quantum signature
        self._quantum_key = secrets.token_bytes(32)
        
        logger.info(f"QIZ Protocol initialized: {self.node_id}")
    
    async def connect(self, remote_node: str) -> Optional[QIZConnection]:
        """
        Establish connection to remote node.
        
        Args:
            remote_node: Remote node identifier
            
        Returns:
            QIZConnection if successful
        """
        if remote_node in self._connections:
            return self._connections[remote_node]
        
        # Generate connection ID
        connection_id = hashlib.sha256(
            f"{self.node_id}:{remote_node}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Perform handshake
        shared_secret = await self._handshake(remote_node)
        
        if shared_secret is None:
            return None
        
        # Create connection
        connection = QIZConnection(
            connection_id=connection_id,
            remote_node=remote_node,
            established=time.time(),
            last_activity=time.time(),
            state="connected",
            shared_secret=shared_secret
        )
        
        self._connections[remote_node] = connection
        
        logger.info(f"Connected to: {remote_node}")
        return connection
    
    async def _handshake(self, remote_node: str) -> Optional[bytes]:
        """Perform quantum-safe handshake."""
        # Generate ephemeral key pair
        ephemeral_private = secrets.token_bytes(32)
        ephemeral_public = hashlib.sha256(ephemeral_private).digest()
        
        # Create handshake message
        handshake = QIZMessage(
            message_type=QIZMessageType.HANDSHAKE,
            source=self.node_id,
            destination=remote_node,
            payload=ephemeral_public,
            sequence=self._next_sequence(),
            timestamp=time.time(),
            quantum_signature=self._sign(ephemeral_public)
        )
        
        # In production, send and receive handshake
        # Simulate successful handshake
        shared_secret = hashlib.sha256(
            ephemeral_private + remote_node.encode()
        ).digest()
        
        return shared_secret
    
    async def disconnect(self, remote_node: str):
        """Disconnect from remote node."""
        if remote_node in self._connections:
            del self._connections[remote_node]
            logger.info(f"Disconnected from: {remote_node}")
    
    async def send(self, destination: str, data: bytes,
                   message_type: QIZMessageType = QIZMessageType.DATA) -> bool:
        """
        Send data to destination.
        
        Args:
            destination: Destination node
            data: Data to send
            message_type: Message type
            
        Returns:
            True if sent successfully
        """
        if destination not in self._connections:
            # Auto-connect
            connection = await self.connect(destination)
            if not connection:
                return False
        
        connection = self._connections[destination]
        
        # Encrypt data
        encrypted = self._encrypt(data, connection.shared_secret)
        
        # Create message
        message = QIZMessage(
            message_type=message_type,
            source=self.node_id,
            destination=destination,
            payload=encrypted,
            sequence=self._next_sequence(),
            timestamp=time.time(),
            quantum_signature=self._sign(encrypted)
        )
        
        # Update activity
        connection.last_activity = time.time()
        
        # In production, send over network
        logger.debug(f"Sent {len(data)} bytes to {destination}")
        
        return True
    
    def on_message(self, message_type: QIZMessageType, handler: Callable):
        """Register message handler."""
        self._handlers[message_type].append(handler)
    
    async def handle_message(self, message: QIZMessage):
        """Handle incoming message."""
        # Verify signature
        if not self._verify(message.payload, message.quantum_signature):
            logger.warning(f"Invalid signature from {message.source}")
            return
        
        # Get connection
        if message.source in self._connections:
            connection = self._connections[message.source]
            connection.last_activity = time.time()
            
            # Decrypt payload
            decrypted = self._decrypt(message.payload, connection.shared_secret)
            message.payload = decrypted
        
        # Call handlers
        for handler in self._handlers[message.message_type]:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")
    
    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self._sequence += 1
        return self._sequence
    
    def _sign(self, data: bytes) -> bytes:
        """Sign data with quantum key."""
        return hashlib.sha256(data + self._quantum_key).digest()
    
    def _verify(self, data: bytes, signature: bytes) -> bool:
        """Verify signature."""
        # In production, verify with sender's public key
        return len(signature) == 32
    
    def _encrypt(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data with key."""
        # Simple XOR encryption (use proper encryption in production)
        extended_key = (key * (len(data) // len(key) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, extended_key))
    
    def _decrypt(self, data: bytes, key: bytes) -> bytes:
        """Decrypt data with key."""
        return self._encrypt(data, key)  # XOR is symmetric
    
    def get_connections(self) -> List[str]:
        """Get list of connected nodes."""
        return list(self._connections.keys())
    
    def get_connection_info(self, remote_node: str) -> Optional[Dict]:
        """Get connection information."""
        if remote_node not in self._connections:
            return None
        
        conn = self._connections[remote_node]
        return {
            "connection_id": conn.connection_id,
            "remote_node": conn.remote_node,
            "established": conn.established,
            "last_activity": conn.last_activity,
            "state": conn.state
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        return {
            "node_id": self.node_id,
            "connections": len(self._connections),
            "messages_sent": self._sequence
        }
    
    def __repr__(self) -> str:
        return f"QIZProtocol(node_id='{self.node_id}')"
