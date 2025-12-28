"""
Quantum Mesh Protocol (QMP) implementation for AIPlatform Quantum Infrastructure Zero SDK

This module provides the Quantum Mesh Protocol for secure, quantum-enhanced
communication between QIZ nodes in the network.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..exceptions import QIZRoutingError, QIZSecurityError
from .signature import QuantumSignature
from .node import QIZNode

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class QMPMessageType(Enum):
    """QMP message types."""
    HELLO = "hello"
    DATA = "data"
    ACK = "ack"
    NACK = "nack"
    DISCOVERY = "discovery"
    ROUTING = "routing"
    SECURITY = "security"
    ERROR = "error"

class QMPMessagePriority(Enum):
    """QMP message priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class QMPMessage:
    """QMP message structure."""
    message_type: QMPMessageType
    sender: str
    recipient: str
    payload: Dict[str, Any]
    timestamp: datetime
    priority: QMPMessagePriority
    signature: str
    sequence_number: int
    ttl: int = 300  # 5 minutes default TTL

class QMPProtocol:
    """
    Quantum Mesh Protocol implementation.
    
    Provides secure, quantum-enhanced communication between QIZ nodes
    with entanglement-based synchronization and quantum key distribution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize QMP protocol.
        
        Args:
            config (dict, optional): QMP configuration
        """
        self._config = config or {}
        self._nodes = {}  # node_id -> node_info
        self._connections = {}  # connection_id -> connection_info
        self._message_queue = asyncio.Queue()
        self._message_handlers = {}
        self._security_enabled = self._config.get("security_enabled", True)
        self._encryption_enabled = self._config.get("encryption_enabled", True)
        self._sequence_counter = 0
        self._is_running = False
        self._routing_table = {}
        self._entanglement_map = {}
        
        # Initialize protocol components
        self._initialize_protocol()
    
    def _initialize_protocol(self):
        """Initialize protocol components."""
        # Register default message handlers
        self.register_message_handler(QMPMessageType.HELLO, self._handle_hello)
        self.register_message_handler(QMPMessageType.DATA, self._handle_data)
        self.register_message_handler(QMPMessageType.ACK, self._handle_ack)
        self.register_message_handler(QMPMessageType.NACK, self._handle_nack)
        self.register_message_handler(QMPMessageType.DISCOVERY, self._handle_discovery)
        self.register_message_handler(QMPMessageType.ROUTING, self._handle_routing)
        self.register_message_handler(QMPMessageType.SECURITY, self._handle_security)
        self.register_message_handler(QMPMessageType.ERROR, self._handle_error)
    
    async def start(self) -> bool:
        """
        Start QMP protocol.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            logger.info("Starting Quantum Mesh Protocol")
            
            self._is_running = True
            
            # Start message processing loop
            self._message_processor_task = asyncio.create_task(
                self._process_message_queue()
            )
            
            logger.info("Quantum Mesh Protocol started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start QMP protocol: {e}")
            raise QIZRoutingError(f"Failed to start QMP: {e}")
    
    async def stop(self) -> bool:
        """
        Stop QMP protocol.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            logger.info("Stopping Quantum Mesh Protocol")
            
            self._is_running = False
            
            # Cancel message processor task
            if hasattr(self, '_message_processor_task'):
                self._message_processor_task.cancel()
                try:
                    await self._message_processor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Quantum Mesh Protocol stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop QMP protocol: {e}")
            raise QIZRoutingError(f"Failed to stop QMP: {e}")
    
    async def connect_node(self, node: QIZNode) -> bool:
        """
        Connect node to QMP network.
        
        Args:
            node (QIZNode): Node to connect
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            if not self._is_running:
                raise QIZRoutingError("QMP protocol not running")
            
            node_id = node.node_id
            quantum_signature = node.quantum_signature
            
            # Create connection info
            connection_info = {
                "node_id": node_id,
                "signature": quantum_signature,
                "connected_at": datetime.now(),
                "status": "connected",
                "capabilities": node.get_node_info().get("capabilities", []),
                "entanglement_key": self._generate_entanglement_key(node_id)
            }
            
            # Store connection
            self._connections[node_id] = connection_info
            self._nodes[node_id] = node
            
            # Update routing table
            self._update_routing_table(node_id, quantum_signature)
            
            # Send hello message
            await self._send_hello(node_id)
            
            logger.info(f"Node {node_id} connected to QMP network")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect node {node.node_id}: {e}")
            raise QIZRoutingError(f"Failed to connect node: {e}")
    
    async def disconnect_node(self, node_id: str) -> bool:
        """
        Disconnect node from QMP network.
        
        Args:
            node_id (str): ID of node to disconnect
            
        Returns:
            bool: True if disconnected successfully, False otherwise
        """
        try:
            if not self._is_running:
                raise QIZRoutingError("QMP protocol not running")
            
            if node_id not in self._connections:
                return False
            
            # Update connection status
            self._connections[node_id]["status"] = "disconnected"
            self._connections[node_id]["disconnected_at"] = datetime.now()
            
            # Remove from routing table
            if node_id in self._routing_table:
                del self._routing_table[node_id]
            
            # Remove entanglement key
            if node_id in self._entanglement_map:
                del self._entanglement_map[node_id]
            
            logger.info(f"Node {node_id} disconnected from QMP network")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect node {node_id}: {e}")
            raise QIZRoutingError(f"Failed to disconnect node: {e}")
    
    async def send_message(self, recipient: str, payload: Dict[str, Any], 
                          message_type: QMPMessageType = QMPMessageType.DATA,
                          priority: QMPMessagePriority = QMPMessagePriority.NORMAL,
                          ttl: int = 300) -> bool:
        """
        Send message to recipient.
        
        Args:
            recipient (str): Recipient node ID
            payload (dict): Message payload
            message_type (QMPMessageType): Type of message
            priority (QMPMessagePriority): Message priority
            ttl (int): Time to live in seconds
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            if not self._is_running:
                raise QIZRoutingError("QMP protocol not running")
            
            # Create message
            self._sequence_counter += 1
            message = QMPMessage(
                message_type=message_type,
                sender="qmp_protocol",  # System message
                recipient=recipient,
                payload=payload,
                timestamp=datetime.now(),
                priority=priority,
                signature=self._generate_message_signature(payload),
                sequence_number=self._sequence_counter,
                ttl=ttl
            )
            
            # Add to message queue
            await self._message_queue.put(message)
            
            logger.debug(f"Message queued for {recipient}: {message_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {recipient}: {e}")
            raise QIZRoutingError(f"Failed to send message: {e}")
    
    async def broadcast_message(self, payload: Dict[str, Any],
                               message_type: QMPMessageType = QMPMessageType.DATA,
                               priority: QMPMessagePriority = QMPMessagePriority.NORMAL) -> bool:
        """
        Broadcast message to all connected nodes.
        
        Args:
            payload (dict): Message payload
            message_type (QMPMessageType): Type of message
            priority (QMPMessagePriority): Message priority
            
        Returns:
            bool: True if message broadcast successfully, False otherwise
        """
        try:
            if not self._is_running:
                raise QIZRoutingError("QMP protocol not running")
            
            # Send message to all connected nodes
            success_count = 0
            for node_id in self._connections:
                if self._connections[node_id]["status"] == "connected":
                    try:
                        await self.send_message(
                            recipient=node_id,
                            payload=payload,
                            message_type=message_type,
                            priority=priority
                        )
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to send broadcast message to {node_id}: {e}")
            
            logger.info(f"Broadcast message sent to {success_count} nodes")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            raise QIZRoutingError(f"Failed to broadcast message: {e}")
    
    def register_message_handler(self, message_type: QMPMessageType, 
                                 handler: Callable[[QMPMessage], Any]) -> bool:
        """
        Register message handler.
        
        Args:
            message_type (QMPMessageType): Type of messages to handle
            handler (callable): Handler function
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        try:
            self._message_handlers[message_type] = handler
            logger.debug(f"Registered handler for {message_type.value} messages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register handler for {message_type.value}: {e}")
            return False
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get QMP network information.
        
        Returns:
            dict: Network information
        """
        return {
            "nodes_connected": len([c for c in self._connections.values() if c["status"] == "connected"]),
            "total_nodes": len(self._connections),
            "routing_entries": len(self._routing_table),
            "entangled_pairs": len(self._entanglement_map),
            "message_queue_size": self._message_queue.qsize(),
            "is_running": self._is_running
        }
    
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about specific node.
        
        Args:
            node_id (str): Node ID
            
        Returns:
            dict: Node information, or None if not found
        """
        return self._connections.get(node_id)
    
    def list_connected_nodes(self) -> List[str]:
        """
        List all connected nodes.
        
        Returns:
            list: List of connected node IDs
        """
        return [node_id for node_id, conn in self._connections.items() 
                if conn["status"] == "connected"]
    
    async def _process_message_queue(self):
        """Process message queue."""
        while self._is_running:
            try:
                # Get message from queue
                message = await self._message_queue.get()
                
                # Check TTL
                if self._is_message_expired(message):
                    logger.debug(f"Message expired: {message.sequence_number}")
                    continue
                
                # Process message
                await self._handle_message(message)
                
                # Mark task as done
                self._message_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Message processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _handle_message(self, message: QMPMessage):
        """Handle incoming message."""
        try:
            # Verify message signature
            if self._security_enabled and not self._verify_message_signature(message):
                logger.warning(f"Invalid signature for message {message.sequence_number}")
                await self._send_error(message.sender, "Invalid signature")
                return
            
            # Get message handler
            handler = self._message_handlers.get(message.message_type)
            if not handler:
                logger.warning(f"No handler for message type: {message.message_type.value}")
                await self._send_error(message.sender, f"No handler for {message.message_type.value}")
                return
            
            # Handle message
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error in message handler for {message.message_type.value}: {e}")
                await self._send_error(message.sender, f"Handler error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_hello(self, message: QMPMessage):
        """Handle HELLO message."""
        logger.debug(f"Received HELLO from {message.sender}")
        
        # Send ACK
        await self._send_ack(message.sender, {"response": "hello_ack"})
    
    async def _handle_data(self, message: QMPMessage):
        """Handle DATA message."""
        logger.debug(f"Received DATA from {message.sender}")
        
        # Process data (in a real implementation, this would do something with the data)
        data_type = message.payload.get("type", "unknown")
        logger.info(f"Processing {data_type} data from {message.sender}")
        
        # Send ACK
        await self._send_ack(message.sender, {"response": "data_received"})
    
    async def _handle_ack(self, message: QMPMessage):
        """Handle ACK message."""
        logger.debug(f"Received ACK from {message.sender}")
        # In a real implementation, this would handle acknowledgments
    
    async def _handle_nack(self, message: QMPMessage):
        """Handle NACK message."""
        logger.warning(f"Received NACK from {message.sender}: {message.payload}")
        # In a real implementation, this would handle negative acknowledgments
    
    async def _handle_discovery(self, message: QMPMessage):
        """Handle DISCOVERY message."""
        logger.debug(f"Received DISCOVERY from {message.sender}")
        
        # Send network information
        discovery_response = {
            "nodes": self.list_connected_nodes(),
            "protocol_version": "1.0",
            "capabilities": ["qmp", "quantum_routing", "entanglement"]
        }
        
        await self.send_message(
            recipient=message.sender,
            payload=discovery_response,
            message_type=QMPMessageType.DISCOVERY,
            priority=QMPMessagePriority.LOW
        )
    
    async def _handle_routing(self, message: QMPMessage):
        """Handle ROUTING message."""
        logger.debug(f"Received ROUTING from {message.sender}")
        # In a real implementation, this would handle routing updates
    
    async def _handle_security(self, message: QMPMessage):
        """Handle SECURITY message."""
        logger.debug(f"Received SECURITY from {message.sender}")
        # In a real implementation, this would handle security updates
    
    async def _handle_error(self, message: QMPMessage):
        """Handle ERROR message."""
        error_msg = message.payload.get("error", "Unknown error")
        logger.error(f"Received ERROR from {message.sender}: {error_msg}")
    
    def _is_message_expired(self, message: QMPMessage) -> bool:
        """Check if message is expired."""
        if message.ttl <= 0:
            return False
        expiration_time = message.timestamp.timestamp() + message.ttl
        return datetime.now().timestamp() > expiration_time
    
    def _generate_message_signature(self, payload: Dict[str, Any]) -> str:
        """Generate signature for message payload."""
        try:
            return QuantumSignature.generate_signature(payload)
        except Exception as e:
            logger.error(f"Failed to generate message signature: {e}")
            return "invalid_signature"
    
    def _verify_message_signature(self, message: QMPMessage) -> bool:
        """Verify message signature."""
        try:
            # In a real implementation, this would verify the quantum signature
            # For simulation, we'll do a basic check
            return message.signature != "invalid_signature" and len(message.signature) > 10
        except Exception as e:
            logger.error(f"Failed to verify message signature: {e}")
            return False
    
    def _generate_entanglement_key(self, node_id: str) -> str:
        """Generate entanglement key for node."""
        # In a real implementation, this would use quantum entanglement
        # For simulation, we'll generate a pseudo-quantum key
        import secrets
        return f"ent_{secrets.token_hex(16)}_{node_id[:8]}"
    
    def _update_routing_table(self, node_id: str, quantum_signature: str):
        """Update routing table with node information."""
        self._routing_table[node_id] = {
            "signature": quantum_signature,
            "last_updated": datetime.now(),
            "routes": []  # In a real implementation, this would contain routing information
        }
    
    async def _send_hello(self, recipient: str) -> bool:
        """Send HELLO message to node."""
        hello_payload = {
            "protocol_version": "1.0",
            "capabilities": ["qmp", "quantum_routing", "entanglement"],
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.send_message(
            recipient=recipient,
            payload=hello_payload,
            message_type=QMPMessageType.HELLO,
            priority=QMPMessagePriority.HIGH
        )
    
    async def _send_ack(self, recipient: str, payload: Dict[str, Any]) -> bool:
        """Send ACK message to node."""
        return await self.send_message(
            recipient=recipient,
            payload=payload,
            message_type=QMPMessageType.ACK,
            priority=QMPMessagePriority.NORMAL
        )
    
    async def _send_error(self, recipient: str, error_message: str) -> bool:
        """Send ERROR message to node."""
        error_payload = {
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.send_message(
            recipient=recipient,
            payload=error_payload,
            message_type=QMPMessageType.ERROR,
            priority=QMPMessagePriority.HIGH
        )
    
    @property
    def is_running(self) -> bool:
        """Check if QMP protocol is running."""
        return self._is_running

# Utility functions for QMP operations
async def create_qmp_protocol(config: Optional[Dict] = None) -> QMPProtocol:
    """
    Create and initialize QMP protocol.
    
    Args:
        config (dict, optional): QMP configuration
        
    Returns:
        QMPProtocol: Initialized QMP protocol instance
    """
    qmp = QMPProtocol(config)
    await qmp.start()
    return qmp

async def connect_node_to_qmp(qmp: QMPProtocol, node: QIZNode) -> bool:
    """
    Connect node to QMP network.
    
    Args:
        qmp (QMPProtocol): QMP protocol instance
        node (QIZNode): Node to connect
        
    Returns:
        bool: True if connected successfully, False otherwise
    """
    return await qmp.connect_node(node)

async def send_qmp_message(qmp: QMPProtocol, recipient: str, payload: Dict[str, Any],
                          message_type: QMPMessageType = QMPMessageType.DATA) -> bool:
    """
    Send message via QMP protocol.
    
    Args:
        qmp (QMPProtocol): QMP protocol instance
        recipient (str): Recipient node ID
        payload (dict): Message payload
        message_type (QMPMessageType): Type of message
        
    Returns:
        bool: True if message sent successfully, False otherwise
    """
    return await qmp.send_message(recipient, payload, message_type)

class QMPNetwork:
    """QMP network manager for multiple protocol instances."""
    
    def __init__(self):
        self._protocols = {}
        self._default_config = {}
    
    async def create_network(self, name: str, config: Optional[Dict] = None) -> QMPProtocol:
        """
        Create QMP network.
        
        Args:
            name (str): Network name
            config (dict, optional): Network configuration
            
        Returns:
            QMPProtocol: Created protocol instance
        """
        if name in self._protocols:
            raise QIZRoutingError(f"QMP network {name} already exists")
        
        network_config = {**self._default_config, **(config or {})}
        qmp = QMPProtocol(network_config)
        await qmp.start()
        
        self._protocols[name] = qmp
        return qmp
    
    def get_network(self, name: str) -> Optional[QMPProtocol]:
        """
        Get QMP network.
        
        Args:
            name (str): Network name
            
        Returns:
            QMPProtocol: Network instance, or None if not found
        """
        return self._protocols.get(name)
    
    async def destroy_network(self, name: str) -> bool:
        """
        Destroy QMP network.
        
        Args:
            name (str): Network name
            
        Returns:
            bool: True if destroyed successfully, False otherwise
        """
        if name not in self._protocols:
            return False
        
        protocol = self._protocols[name]
        await protocol.stop()
        del self._protocols[name]
        return True
    
    def list_networks(self) -> List[str]:
        """
        List all QMP networks.
        
        Returns:
            list: List of network names
        """
        return list(self._protocols.keys())

# Global QMP network manager
_global_qmp_network = QMPNetwork()

def get_qmp_network() -> QMPNetwork:
    """
    Get global QMP network manager.
    
    Returns:
        QMPNetwork: Global QMP network manager instance
    """
    return _global_qmp_network

# Example usage
async def example_qmp_usage():
    """Example of QMP protocol usage."""
    # Create QMP protocol
    qmp = QMPProtocol()
    await qmp.start()
    
    # Send a test message
    test_payload = {
        "message": "Hello QMP Network",
        "timestamp": datetime.now().isoformat(),
        "test_data": {"value": 42, "status": "active"}
    }
    
    # Send broadcast message
    await qmp.broadcast_message(
        payload=test_payload,
        message_type=QMPMessageType.DATA,
        priority=QMPMessagePriority.NORMAL
    )
    
    # Get network info
    network_info = qmp.get_network_info()
    print(f"Network info: {network_info}")
    
    await qmp.stop()
    return True