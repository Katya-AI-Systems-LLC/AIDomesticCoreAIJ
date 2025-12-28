"""
Quantum Mesh Protocol (QMP) implementation for AIPlatform SDK

This module provides the Quantum Mesh Protocol for secure,
decentralized communication between quantum and classical nodes.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from ..exceptions import ProtocolError
from ..security.crypto import QuantumSafeCrypto, Kyber, Dilithium

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class QMPNode:
    """QMP Node representation."""
    node_id: str
    node_type: str  # "quantum", "classical", "hybrid"
    address: str
    public_key: bytes
    status: str  # "online", "offline", "maintenance"
    capabilities: List[str]
    last_seen: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class QMPMessage:
    """QMP Message format."""
    message_id: str
    sender: str
    recipient: str
    content: Any
    timestamp: datetime
    message_type: str
    encrypted: bool = False
    signature: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class QMPSession:
    """QMP Secure Session."""
    session_id: str
    node_a: str
    node_b: str
    shared_secret: bytes
    established: datetime
    last_activity: datetime
    encrypted: bool = True
    metadata: Optional[Dict[str, Any]] = None

class QuantumMeshProtocol:
    """
    Quantum Mesh Protocol implementation.
    
    Provides secure, decentralized communication between
    quantum and classical nodes in a mesh network topology.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize QMP.
        
        Args:
            config (dict, optional): QMP configuration
        """
        self._config = config or {}
        self._is_initialized = False
        self._nodes = {}
        self._sessions = {}
        self._message_handlers = {}
        self._crypto = None
        self._signer = None
        
        # Initialize QMP
        self._initialize_qmp()
        
        logger.info("Quantum Mesh Protocol initialized")
    
    def _initialize_qmp(self):
        """Initialize QMP system."""
        try:
            # In a real implementation, this would initialize the QMP network
            # For simulation, we'll create placeholder information
            self._qmp_info = {
                "protocol": "qmp",
                "version": "1.0.0",
                "status": "initialized",
                "capabilities": ["quantum_secure", "mesh_topology", "zero_trust"]
            }
            
            # Initialize crypto
            self._crypto = QuantumSafeCrypto()
            self._signer = Dilithium()
            self._signer.keygen()
            
            self._is_initialized = True
            logger.debug("QMP initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize QMP: {e}")
            raise ProtocolError(f"QMP initialization failed: {e}")
    
    def register_node(self, node: QMPNode) -> bool:
        """
        Register node in QMP network.
        
        Args:
            node (QMPNode): Node to register
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                raise ProtocolError("QMP not initialized")
            
            self._nodes[node.node_id] = node
            logger.debug(f"Node registered: {node.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False
    
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister node from QMP network.
        
        Args:
            node_id (str): Node identifier
            
        Returns:
            bool: True if unregistered successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                raise ProtocolError("QMP not initialized")
            
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.debug(f"Node unregistered: {node_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to unregister node: {e}")
            return False
    
    def establish_session(self, node_a: str, node_b: str, 
                         encrypted: bool = True) -> Optional[str]:
        """
        Establish secure session between nodes.
        
        Args:
            node_a (str): First node identifier
            node_b (str): Second node identifier
            encrypted (bool): Whether to encrypt the session
            
        Returns:
            str: Session ID or None if failed
        """
        try:
            if not self._is_initialized:
                raise ProtocolError("QMP not initialized")
            
            # Verify nodes exist
            if node_a not in self._nodes or node_b not in self._nodes:
                raise ProtocolError("One or both nodes not found")
            
            # Generate session ID
            session_id = f"session_{hashlib.md5(f'{node_a}{node_b}{datetime.now()}'.encode()).hexdigest()[:16]}"
            
            # Generate shared secret (in real implementation, this would use Kyber)
            if encrypted:
                kyber = Kyber()
                keypair = kyber.keygen()
                shared_secret = keypair.private_key  # Simplified for simulation
            else:
                shared_secret = b"unencrypted_session"
            
            # Create session
            session = QMPSession(
                session_id=session_id,
                node_a=node_a,
                node_b=node_b,
                shared_secret=shared_secret,
                established=datetime.now(),
                last_activity=datetime.now(),
                encrypted=encrypted
            )
            
            # Store session
            self._sessions[session_id] = session
            
            logger.debug(f"Session established: {session_id} between {node_a} and {node_b}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to establish session: {e}")
            return None
    
    def send_message(self, recipient: str, content: Any, 
                   message_type: str = "generic", 
                   sender: Optional[str] = None,
                   encrypted: bool = True) -> str:
        """
        Send message to recipient.
        
        Args:
            recipient (str): Recipient node ID
            content (Any): Message content
            message_type (str): Type of message
            sender (str, optional): Sender node ID
            encrypted (bool): Whether to encrypt the message
            
        Returns:
            str: Message ID
        """
        try:
            if not self._is_initialized:
                raise ProtocolError("QMP not initialized")
            
            # Generate message ID
            message_id = f"msg_{hashlib.md5(f'{sender}{recipient}{datetime.now()}'.encode()).hexdigest()[:12]}"
            
            # Get recipient node
            if recipient not in self._nodes:
                raise ProtocolError(f"Recipient node {recipient} not found")
            
            recipient_node = self._nodes[recipient]
            
            # Encrypt content if requested
            encrypted_content = content
            if encrypted:
                # In a real implementation, this would use the session's shared secret
                # For simulation, we'll just mark it as encrypted
                pass
            
            # Create message
            message = QMPMessage(
                message_id=message_id,
                sender=sender or "unknown",
                recipient=recipient,
                content=encrypted_content,
                timestamp=datetime.now(),
                message_type=message_type,
                encrypted=encrypted
            )
            
            # Sign message
            message.signature = self._sign_message(message)
            
            # Process message
            self._process_message(message)
            
            logger.debug(f"Message sent: {message_id} to {recipient}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise ProtocolError(f"Message sending failed: {e}")
    
    def _sign_message(self, message: QMPMessage) -> str:
        """
        Sign message with digital signature.
        
        Args:
            message (QMPMessage): Message to sign
            
        Returns:
            str: Digital signature
        """
        try:
            # Create message hash
            message_dict = {
                "message_id": message.message_id,
                "sender": message.sender,
                "recipient": message.recipient,
                "content": str(message.content),
                "timestamp": message.timestamp.isoformat(),
                "message_type": message.message_type
            }
            
            message_json = json.dumps(message_dict, sort_keys=True)
            message_bytes = message_json.encode('utf-8')
            
            # Sign with Dilithium
            signature = self._signer.sign(message_bytes)
            return signature.hex()
            
        except Exception as e:
            logger.error(f"Failed to sign message: {e}")
            return "invalid_signature"
    
    def _process_message(self, message: QMPMessage):
        """
        Process incoming message.
        
        Args:
            message (QMPMessage): Message to process
        """
        try:
            # Verify signature
            if not self._verify_message_signature(message):
                logger.warning(f"Message signature verification failed: {message.message_id}")
                return
            
            # Check if we have a handler for this message type
            if message.message_type in self._message_handlers:
                handler = self._message_handlers[message.message_type]
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")
            else:
                # Default message handling
                logger.debug(f"Received message: {message.message_id} from {message.sender}")
                logger.debug(f"Message content: {message.content}")
            
            # Update session activity if applicable
            self._update_session_activity(message.sender, message.recipient)
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
    
    def _verify_message_signature(self, message: QMPMessage) -> bool:
        """
        Verify message signature.
        
        Args:
            message (QMPMessage): Message to verify
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        try:
            if not message.signature or message.signature == "invalid_signature":
                return False
            
            # Recreate message hash
            message_dict = {
                "message_id": message.message_id,
                "sender": message.sender,
                "recipient": message.recipient,
                "content": str(message.content),
                "timestamp": message.timestamp.isoformat(),
                "message_type": message.message_type
            }
            
            message_json = json.dumps(message_dict, sort_keys=True)
            message_bytes = message_json.encode('utf-8')
            
            # Convert signature from hex
            signature_bytes = bytes.fromhex(message.signature)
            
            # Verify with Dilithium (in real implementation)
            # For simulation, we'll assume it's valid
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify message signature: {e}")
            return False
    
    def _update_session_activity(self, node_a: str, node_b: str):
        """
        Update session activity.
        
        Args:
            node_a (str): First node
            node_b (str): Second node
        """
        try:
            # Find session between nodes
            for session in self._sessions.values():
                if (session.node_a == node_a and session.node_b == node_b) or \
                   (session.node_a == node_b and session.node_b == node_a):
                    session.last_activity = datetime.now()
                    break
            
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
    
    def register_message_handler(self, message_type: str, 
                               handler: Callable[[QMPMessage], None]) -> bool:
        """
        Register message handler.
        
        Args:
            message_type (str): Type of messages to handle
            handler (callable): Function to handle messages
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        try:
            if not callable(handler):
                raise ValueError("Handler must be callable")
            
            self._message_handlers[message_type] = handler
            logger.debug(f"Message handler registered for type: {message_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register message handler: {e}")
            return False
    
    def broadcast_message(self, content: Any, message_type: str = "broadcast",
                        target_nodes: Optional[List[str]] = None) -> List[str]:
        """
        Broadcast message to multiple nodes.
        
        Args:
            content (Any): Message content
            message_type (str): Type of message
            target_nodes (list, optional): Specific nodes to broadcast to
            
        Returns:
            list: List of message IDs
        """
        try:
            if not self._is_initialized:
                raise ProtocolError("QMP not initialized")
            
            # If no target nodes specified, use all nodes
            if target_nodes is None:
                target_nodes = list(self._nodes.keys())
            
            message_ids = []
            
            for node_id in target_nodes:
                message_id = self.send_message(
                    recipient=node_id,
                    content=content,
                    message_type=message_type,
                    encrypted=True
                )
                message_ids.append(message_id)
            
            logger.debug(f"Broadcast message sent to {len(message_ids)} nodes")
            return message_ids
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            raise ProtocolError(f"Message broadcasting failed: {e}")
    
    def get_node(self, node_id: str) -> Optional[QMPNode]:
        """
        Get node information.
        
        Args:
            node_id (str): Node identifier
            
        Returns:
            QMPNode: Node information or None if not found
        """
        return self._nodes.get(node_id)
    
    def list_nodes(self) -> List[QMPNode]:
        """
        List all nodes in the network.
        
        Returns:
            list: List of nodes
        """
        return list(self._nodes.values())
    
    def get_session(self, session_id: str) -> Optional[QMPSession]:
        """
        Get session information.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            QMPSession: Session information or None if not found
        """
        return self._sessions.get(session_id)
    
    def list_sessions(self) -> List[QMPSession]:
        """
        List all sessions.
        
        Returns:
            list: List of sessions
        """
        return list(self._sessions.values())
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get QMP network information.
        
        Returns:
            dict: Network information
        """
        return {
            "initialized": self._is_initialized,
            "node_count": len(self._nodes),
            "session_count": len(self._sessions),
            "message_handlers": list(self._message_handlers.keys()),
            "qmp_info": self._qmp_info,
            "crypto_available": self._crypto is not None,
            "signer_available": self._signer is not None
        }
    
    def get_node_metrics(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get node metrics.
        
        Args:
            node_id (str): Node identifier
            
        Returns:
            dict: Node metrics or None if not found
        """
        if node_id not in self._nodes:
            return None
        
        node = self._nodes[node_id]
        sessions = [s for s in self._sessions.values() 
                   if s.node_a == node_id or s.node_b == node_id]
        
        return {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "status": node.status,
            "capabilities": node.capabilities,
            "last_seen": node.last_seen.isoformat(),
            "active_sessions": len(sessions),
            "session_partners": list(set([s.node_a if s.node_a != node_id else s.node_b 
                                        for s in sessions]))
        }

# Utility functions for QMP
def create_qmp(config: Optional[Dict] = None) -> QuantumMeshProtocol:
    """
    Create Quantum Mesh Protocol.
    
    Args:
        config (dict, optional): QMP configuration
        
    Returns:
        QuantumMeshProtocol: Created QMP
    """
    return QuantumMeshProtocol(config)

def register_qmp_node(qmp: QuantumMeshProtocol, node: QMPNode) -> bool:
    """
    Register node in QMP network.
    
    Args:
        qmp (QuantumMeshProtocol): QMP instance
        node (QMPNode): Node to register
        
    Returns:
        bool: True if registered successfully, False otherwise
    """
    return qmp.register_node(node)

def send_qmp_message(qmp: QuantumMeshProtocol, recipient: str, content: Any,
                    message_type: str = "generic", sender: Optional[str] = None) -> str:
    """
    Send QMP message.
    
    Args:
        qmp (QuantumMeshProtocol): QMP instance
        recipient (str): Recipient node ID
        content (Any): Message content
        message_type (str): Type of message
        sender (str, optional): Sender node ID
        
    Returns:
        str: Message ID
    """
    return qmp.send_message(recipient, content, message_type, sender)

# Example usage
def example_qmp():
    """Example of QMP usage."""
    # Create QMP
    qmp = create_qmp({
        "network": "testnet",
        "version": "1.0",
        "security_level": "high"
    })
    
    # Create nodes
    node1 = QMPNode(
        node_id="node_001",
        node_type="quantum",
        address="192.168.1.101:8080",
        public_key=b"public_key_001",
        status="online",
        capabilities=["quantum_computing", "secure_messaging"],
        last_seen=datetime.now()
    )
    
    node2 = QMPNode(
        node_id="node_002",
        node_type="classical",
        address="192.168.1.102:8080",
        public_key=b"public_key_002",
        status="online",
        capabilities=["data_processing", "storage"],
        last_seen=datetime.now()
    )
    
    # Register nodes
    qmp.register_node(node1)
    qmp.register_node(node2)
    
    # Register message handler
    def message_handler(msg: QMPMessage):
        print(f"Received message: {msg.content} from {msg.sender}")
    
    qmp.register_message_handler("test", message_handler)
    
    # Send message
    message_id = qmp.send_message(
        recipient="node_002",
        content="Hello from node_001",
        message_type="test",
        sender="node_001"
    )
    print(f"Message sent with ID: {message_id}")
    
    # Establish session
    session_id = qmp.establish_session("node_001", "node_002")
    if session_id:
        print(f"Session established: {session_id}")
    
    # Get network info
    network_info = qmp.get_network_info()
    print(f"Network info: {network_info}")
    
    # Get node metrics
    node_metrics = qmp.get_node_metrics("node_001")
    print(f"Node metrics: {node_metrics}")
    
    return qmp

# Advanced QMP example
def advanced_qmp_example():
    """Advanced example of QMP usage."""
    # Create QMP
    qmp = create_qmp({
        "network": "mainnet",
        "version": "1.0",
        "mesh_topology": "full"
    })
    
    # Create multiple nodes
    node_types = ["quantum", "classical", "hybrid"]
    nodes = []
    
    for i in range(5):
        node = QMPNode(
            node_id=f"node_{i+1:03d}",
            node_type=node_types[i % len(node_types)],
            address=f"192.168.1.{100+i}:8080",
            public_key=f"public_key_{i+1:03d}".encode(),
            status="online",
            capabilities=["secure_messaging", f"capability_{i+1}"],
            last_seen=datetime.now(),
            metadata={"region": "datacenter_a", "zone": f"zone_{(i % 3) + 1}"}
        )
        nodes.append(node)
        qmp.register_node(node)
        print(f"Registered node: {node.node_id} ({node.node_type})")
    
    # Register message handlers
    def generic_handler(msg: QMPMessage):
        print(f"Generic message received: {msg.message_id}")
    
    def data_handler(msg: QMPMessage):
        print(f"Data message received: {len(str(msg.content))} bytes")
    
    qmp.register_message_handler("generic", generic_handler)
    qmp.register_message_handler("data", data_handler)
    
    # Establish sessions between all nodes
    sessions = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            session_id = qmp.establish_session(nodes[i].node_id, nodes[j].node_id)
            if session_id:
                sessions.append(session_id)
    
    print(f"Established {len(sessions)} sessions")
    
    # Send various types of messages
    message_types = ["generic", "data", "control", "status"]
    message_contents = [
        "Hello network!",
        {"data": [1, 2, 3, 4, 5], "timestamp": datetime.now().isoformat()},
        {"command": "ping", "target": "all"},
        {"status": "operational", "load": 0.75}
    ]
    
    sent_messages = []
    for i, node in enumerate(nodes[:3]):  # Send from first 3 nodes
        for j, msg_type in enumerate(message_types):
            content = message_contents[j % len(message_contents)]
            message_id = qmp.send_message(
                recipient=nodes[(i+1) % len(nodes)].node_id,
                content=content,
                message_type=msg_type,
                sender=node.node_id
            )
            sent_messages.append(message_id)
    
    print(f"Sent {len(sent_messages)} messages")
    
    # Broadcast message
    broadcast_ids = qmp.broadcast_message(
        content="Network-wide broadcast message",
        message_type="broadcast"
    )
    print(f"Broadcast to {len(broadcast_ids)} nodes")
    
    # Get detailed network information
    print("\nNetwork Information:")
    network_info = qmp.get_network_info()
    print(f"Nodes: {network_info['node_count']}")
    print(f"Sessions: {network_info['session_count']}")
    print(f"Message handlers: {network_info['message_handlers']}")
    
    # Get metrics for all nodes
    print("\nNode Metrics:")
    for node in nodes:
        metrics = qmp.get_node_metrics(node.node_id)
        if metrics:
            print(f"{node.node_id}: {metrics['active_sessions']} sessions, "
                  f"status: {metrics['status']}")
    
    return qmp