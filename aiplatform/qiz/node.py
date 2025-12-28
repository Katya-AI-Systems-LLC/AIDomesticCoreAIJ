"""
QIZ Node implementation for AIPlatform Quantum Infrastructure Zero SDK

This module provides the core node functionality for the Quantum Infrastructure Zero,
including zero-server architecture, quantum signature generation, and QMP protocol integration.
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..exceptions import QIZNodeError
from .signature import QuantumSignature

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class NodeInfo:
    """Information about a QIZ node."""
    node_id: str
    capabilities: List[str]
    status: str
    last_seen: datetime
    quantum_signature: str
    metadata: Dict[str, Any]

class QIZNode:
    """
    Quantum Infrastructure Zero Node implementation.
    
    Represents a node in the QIZ network with zero-server architecture,
    quantum signature-based identification, and QMP protocol support.
    """
    
    def __init__(self, node_id: str, config: Optional[Dict] = None):
        """
        Initialize QIZ node.
        
        Args:
            node_id (str): Unique node identifier
            config (dict, optional): Node configuration
        """
        self._node_id = node_id
        self._config = config or {}
        self._status = "initialized"
        self._capabilities = self._config.get("capabilities", ["quantum", "ai"])
        self._connections = {}
        self._services = {}
        self._quantum_signature = None
        self._is_running = False
        self._discovery_enabled = self._config.get("discovery_enabled", True)
        
        # Initialize quantum signature
        self._generate_quantum_signature()
        
        logger.info(f"QIZ Node {self._node_id} initialized")
    
    def _generate_quantum_signature(self):
        """Generate quantum signature for this node."""
        try:
            # Create signature based on node configuration
            signature_data = {
                "node_id": self._node_id,
                "capabilities": self._capabilities,
                "timestamp": datetime.now().isoformat(),
                "config_hash": self._hash_config()
            }
            
            # Generate quantum signature
            self._quantum_signature = QuantumSignature.generate_signature(
                signature_data,
                method="sha256"  # In real implementation, this would use quantum randomness
            )
            
            logger.debug(f"Generated quantum signature for node {self._node_id}")
            
        except Exception as e:
            raise QIZNodeError(f"Failed to generate quantum signature: {e}")
    
    def _hash_config(self) -> str:
        """Generate hash of node configuration."""
        config_str = json.dumps(self._config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    async def start(self) -> bool:
        """
        Start the QIZ node.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            logger.info(f"Starting QIZ Node {self._node_id}")
            
            # Verify quantum signature
            if not self._quantum_signature:
                self._generate_quantum_signature()
            
            # Start services
            await self._start_services()
            
            # Enable discovery if configured
            if self._discovery_enabled:
                await self._start_discovery()
            
            self._is_running = True
            self._status = "running"
            
            logger.info(f"QIZ Node {self._node_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start QIZ Node {self._node_id}: {e}")
            self._status = "error"
            raise QIZNodeError(f"Failed to start node: {e}")
    
    async def _start_services(self):
        """Start node services."""
        # In a real implementation, this would start various services
        # based on node capabilities
        logger.debug(f"Starting services for node {self._node_id}")
        
        # Simulate service startup
        for capability in self._capabilities:
            service_name = f"{capability}_service"
            self._services[service_name] = {
                "status": "running",
                "started_at": datetime.now()
            }
            logger.debug(f"Started {service_name} for node {self._node_id}")
    
    async def _start_discovery(self):
        """Start node discovery service."""
        # In a real implementation, this would start the discovery protocol
        logger.debug(f"Starting discovery for node {self._node_id}")
    
    async def stop(self) -> bool:
        """
        Stop the QIZ node.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            logger.info(f"Stopping QIZ Node {self._node_id}")
            
            # Stop services
            await self._stop_services()
            
            # Stop discovery
            if self._discovery_enabled:
                await self._stop_discovery()
            
            self._is_running = False
            self._status = "stopped"
            
            logger.info(f"QIZ Node {self._node_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop QIZ Node {self._node_id}: {e}")
            raise QIZNodeError(f"Failed to stop node: {e}")
    
    async def _stop_services(self):
        """Stop node services."""
        logger.debug(f"Stopping services for node {self._node_id}")
        
        # Simulate service shutdown
        for service_name in list(self._services.keys()):
            self._services[service_name]["status"] = "stopped"
            self._services[service_name]["stopped_at"] = datetime.now()
            logger.debug(f"Stopped {service_name} for node {self._node_id}")
    
    async def _stop_discovery(self):
        """Stop node discovery service."""
        logger.debug(f"Stopping discovery for node {self._node_id}")
    
    async def connect(self, node_signature: str, address: Optional[str] = None) -> bool:
        """
        Connect to another QIZ node.
        
        Args:
            node_signature (str): Quantum signature of target node
            address (str, optional): Network address of target node
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            logger.info(f"Connecting to node with signature {node_signature}")
            
            # In a real implementation, this would:
            # 1. Verify the quantum signature
            # 2. Establish QMP connection
            # 3. Perform security handshake
            # 4. Register connection
            
            # For simulation, we'll just store the connection
            connection_info = {
                "signature": node_signature,
                "address": address,
                "connected_at": datetime.now(),
                "status": "connected"
            }
            
            # Use first 8 characters of signature as connection ID
            connection_id = node_signature[:8]
            self._connections[connection_id] = connection_info
            
            logger.info(f"Connected to node {node_signature}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to node {node_signature}: {e}")
            raise QIZNodeError(f"Connection failed: {e}")
    
    async def disconnect(self, node_signature: str) -> bool:
        """
        Disconnect from a QIZ node.
        
        Args:
            node_signature (str): Quantum signature of target node
            
        Returns:
            bool: True if disconnected successfully, False otherwise
        """
        try:
            logger.info(f"Disconnecting from node with signature {node_signature}")
            
            # Find connection by signature
            connection_id = None
            for conn_id, conn_info in self._connections.items():
                if conn_info.get("signature") == node_signature:
                    connection_id = conn_id
                    break
            
            if connection_id and connection_id in self._connections:
                self._connections[connection_id]["status"] = "disconnected"
                self._connections[connection_id]["disconnected_at"] = datetime.now()
                logger.info(f"Disconnected from node {node_signature}")
                return True
            else:
                logger.warning(f"No active connection to node {node_signature}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to disconnect from node {node_signature}: {e}")
            raise QIZNodeError(f"Disconnection failed: {e}")
    
    async def broadcast(self, message: Dict[str, Any]) -> bool:
        """
        Broadcast message to all connected nodes.
        
        Args:
            message (dict): Message to broadcast
            
        Returns:
            bool: True if broadcast successful, False otherwise
        """
        try:
            logger.debug(f"Broadcasting message to {len(self._connections)} nodes")
            
            # Add node information to message
            message["sender"] = self._node_id
            message["timestamp"] = datetime.now().isoformat()
            message["signature"] = self._quantum_signature
            
            # In a real implementation, this would send the message
            # to all connected nodes via QMP protocol
            
            logger.debug(f"Broadcast message sent from node {self._node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to broadcast message from node {self._node_id}: {e}")
            raise QIZNodeError(f"Broadcast failed: {e}")
    
    async def send_direct(self, target_node: str, message: Dict[str, Any]) -> bool:
        """
        Send direct message to specific node.
        
        Args:
            target_node (str): Target node signature
            message (dict): Message to send
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            logger.debug(f"Sending direct message to node {target_node}")
            
            # Add node information to message
            message["sender"] = self._node_id
            message["timestamp"] = datetime.now().isoformat()
            message["signature"] = self._quantum_signature
            
            # In a real implementation, this would send the message
            # directly to the target node via QMP protocol
            
            logger.debug(f"Direct message sent to node {target_node}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send direct message to node {target_node}: {e}")
            raise QIZNodeError(f"Direct message failed: {e}")
    
    def get_node_info(self) -> Dict[str, Any]:
        """
        Get information about this node.
        
        Returns:
            dict: Node information
        """
        return {
            "node_id": self._node_id,
            "status": self._status,
            "capabilities": self._capabilities,
            "quantum_signature": self._quantum_signature,
            "is_running": self._is_running,
            "connections": len(self._connections),
            "services": list(self._services.keys())
        }
    
    def get_connections(self) -> Dict[str, Any]:
        """
        Get information about node connections.
        
        Returns:
            dict: Connection information
        """
        return self._connections
    
    def get_services(self) -> Dict[str, Any]:
        """
        Get information about node services.
        
        Returns:
            dict: Service information
        """
        return self._services
    
    @property
    def node_id(self) -> str:
        """Get node ID."""
        return self._node_id
    
    @property
    def status(self) -> str:
        """Get node status."""
        return self._status
    
    @property
    def is_running(self) -> bool:
        """Check if node is running."""
        return self._is_running
    
    @property
    def quantum_signature(self) -> Optional[str]:
        """Get node quantum signature."""
        return self._quantum_signature

# Utility functions for node management
async def create_node(node_id: str, config: Optional[Dict] = None) -> QIZNode:
    """
    Create and initialize a QIZ node.
    
    Args:
        node_id (str): Unique node identifier
        config (dict, optional): Node configuration
        
    Returns:
        QIZNode: Initialized node instance
    """
    node = QIZNode(node_id, config)
    return node

async def start_node(node: QIZNode) -> bool:
    """
    Start a QIZ node.
    
    Args:
        node (QIZNode): Node to start
        
    Returns:
        bool: True if started successfully, False otherwise
    """
    return await node.start()

async def stop_node(node: QIZNode) -> bool:
    """
    Stop a QIZ node.
    
    Args:
        node (QIZNode): Node to stop
        
    Returns:
        bool: True if stopped successfully, False otherwise
    """
    return await node.stop()

class NodeRegistry:
    """Registry for managing multiple QIZ nodes."""
    
    def __init__(self):
        self._nodes = {}
        self._node_lock = asyncio.Lock()
    
    async def register_node(self, node: QIZNode) -> bool:
        """
        Register a node in the registry.
        
        Args:
            node (QIZNode): Node to register
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        async with self._node_lock:
            if node.node_id in self._nodes:
                logger.warning(f"Node {node.node_id} already registered")
                return False
            
            self._nodes[node.node_id] = node
            logger.info(f"Registered node {node.node_id}")
            return True
    
    async def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the registry.
        
        Args:
            node_id (str): ID of node to unregister
            
        Returns:
            bool: True if unregistered successfully, False otherwise
        """
        async with self._node_lock:
            if node_id not in self._nodes:
                logger.warning(f"Node {node_id} not found in registry")
                return False
            
            del self._nodes[node_id]
            logger.info(f"Unregistered node {node_id}")
            return True
    
    def get_node(self, node_id: str) -> Optional[QIZNode]:
        """
        Get a registered node.
        
        Args:
            node_id (str): ID of node to retrieve
            
        Returns:
            QIZNode: Node instance, or None if not found
        """
        return self._nodes.get(node_id)
    
    def list_nodes(self) -> List[str]:
        """
        List all registered nodes.
        
        Returns:
            list: List of node IDs
        """
        return list(self._nodes.keys())
    
    async def start_all_nodes(self) -> Dict[str, bool]:
        """
        Start all registered nodes.
        
        Returns:
            dict: Results for each node
        """
        results = {}
        for node_id, node in self._nodes.items():
            try:
                result = await node.start()
                results[node_id] = result
            except Exception as e:
                logger.error(f"Failed to start node {node_id}: {e}")
                results[node_id] = False
        return results
    
    async def stop_all_nodes(self) -> Dict[str, bool]:
        """
        Stop all registered nodes.
        
        Returns:
            dict: Results for each node
        """
        results = {}
        for node_id, node in self._nodes.items():
            try:
                result = await node.stop()
                results[node_id] = result
            except Exception as e:
                logger.error(f"Failed to stop node {node_id}: {e}")
                results[node_id] = False
        return results

# Global node registry
_global_node_registry = NodeRegistry()

def get_node_registry() -> NodeRegistry:
    """
    Get global node registry.
    
    Returns:
        NodeRegistry: Global node registry instance
    """
    return _global_node_registry