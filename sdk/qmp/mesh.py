"""
Mesh Network Module
===================

Self-healing mesh network implementation for QMP.
"""

from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class MeshState(Enum):
    """Mesh network states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HEALING = "healing"
    PARTITIONED = "partitioned"


@dataclass
class MeshNode:
    """Node in the mesh network."""
    node_id: str
    address: str
    port: int
    quantum_signature: bytes
    connected: bool = False
    last_heartbeat: float = 0.0
    latency_ms: float = 0.0
    hops: int = 0


@dataclass
class MeshTopology:
    """Mesh network topology snapshot."""
    nodes: Dict[str, MeshNode]
    edges: List[tuple]
    timestamp: float
    version: int


class MeshNetwork:
    """
    Self-healing mesh network for Quantum Mesh Protocol.
    
    Features:
    - Automatic peer discovery
    - Self-healing on node failure
    - Topology optimization
    - Partition detection and recovery
    
    Example:
        >>> mesh = MeshNetwork(node_id="mesh_001")
        >>> await mesh.start()
        >>> await mesh.join_network("bootstrap_node:7777")
    """
    
    HEARTBEAT_INTERVAL = 10  # seconds
    HEARTBEAT_TIMEOUT = 30  # seconds
    HEALING_INTERVAL = 60  # seconds
    MIN_CONNECTIONS = 3
    MAX_CONNECTIONS = 20
    
    def __init__(self, node_id: Optional[str] = None,
                 port: int = 7777,
                 language: str = "en"):
        """
        Initialize mesh network.
        
        Args:
            node_id: This node's ID
            port: Network port
            language: Language for messages
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port
        self.language = language
        
        # Network state
        self._state = MeshState.DISCONNECTED
        self._nodes: Dict[str, MeshNode] = {}
        self._edges: Set[tuple] = set()
        
        # Bootstrap nodes
        self._bootstrap_nodes: List[str] = []
        
        # Topology version
        self._topology_version = 0
        
        # Event handlers
        self._handlers: Dict[str, List[Callable]] = {}
        
        # Tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._healing_task: Optional[asyncio.Task] = None
        
        logger.info(f"Mesh network initialized: {self.node_id}")
    
    @property
    def state(self) -> MeshState:
        """Get current mesh state."""
        return self._state
    
    @property
    def node_count(self) -> int:
        """Get number of nodes in mesh."""
        return len(self._nodes)
    
    @property
    def connected_count(self) -> int:
        """Get number of connected nodes."""
        return sum(1 for n in self._nodes.values() if n.connected)
    
    async def start(self) -> bool:
        """Start the mesh network."""
        if self._state != MeshState.DISCONNECTED:
            return False
        
        self._state = MeshState.CONNECTING
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._healing_task = asyncio.create_task(self._healing_loop())
        
        self._state = MeshState.CONNECTED
        self._emit("mesh_started", {"node_id": self.node_id})
        
        logger.info(f"Mesh network started: {self.node_id}")
        return True
    
    async def stop(self):
        """Stop the mesh network."""
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._healing_task:
            self._healing_task.cancel()
        
        # Disconnect from all nodes
        for node_id in list(self._nodes.keys()):
            await self.disconnect_node(node_id)
        
        self._state = MeshState.DISCONNECTED
        self._emit("mesh_stopped", {"node_id": self.node_id})
        
        logger.info(f"Mesh network stopped: {self.node_id}")
    
    async def join_network(self, bootstrap_address: str) -> bool:
        """
        Join mesh network via bootstrap node.
        
        Args:
            bootstrap_address: Bootstrap node address (host:port)
            
        Returns:
            True if joined successfully
        """
        self._bootstrap_nodes.append(bootstrap_address)
        
        try:
            # Connect to bootstrap node
            node = await self._connect_to_address(bootstrap_address)
            
            if node:
                # Request peer list
                peers = await self._request_peers(node.node_id)
                
                # Connect to peers
                for peer_address in peers[:self.MAX_CONNECTIONS]:
                    await self._connect_to_address(peer_address)
                
                self._emit("network_joined", {"bootstrap": bootstrap_address})
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to join network: {e}")
            return False
    
    async def _connect_to_address(self, address: str) -> Optional[MeshNode]:
        """Connect to node at address."""
        try:
            host, port = address.rsplit(":", 1)
            port = int(port)
            
            # Simulate connection
            node_id = str(uuid.uuid4())
            
            node = MeshNode(
                node_id=node_id,
                address=host,
                port=port,
                quantum_signature=b"simulated_signature",
                connected=True,
                last_heartbeat=time.time()
            )
            
            self._nodes[node_id] = node
            self._edges.add((self.node_id, node_id))
            self._topology_version += 1
            
            self._emit("node_connected", {"node_id": node_id})
            logger.info(f"Connected to node: {node_id}")
            
            return node
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}: {e}")
            return None
    
    async def disconnect_node(self, node_id: str) -> bool:
        """Disconnect from a node."""
        if node_id not in self._nodes:
            return False
        
        node = self._nodes[node_id]
        node.connected = False
        
        # Remove edges
        self._edges = {e for e in self._edges if node_id not in e}
        
        del self._nodes[node_id]
        self._topology_version += 1
        
        self._emit("node_disconnected", {"node_id": node_id})
        logger.info(f"Disconnected from node: {node_id}")
        
        return True
    
    async def _request_peers(self, node_id: str) -> List[str]:
        """Request peer list from a node."""
        # In production, this would send actual network request
        # Simulate returning some peer addresses
        return [
            f"192.168.1.{i}:7777" 
            for i in range(2, min(10, self.MAX_CONNECTIONS))
        ]
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to connected nodes."""
        while True:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                
                current_time = time.time()
                
                for node_id, node in list(self._nodes.items()):
                    if not node.connected:
                        continue
                    
                    # Check for timeout
                    if current_time - node.last_heartbeat > self.HEARTBEAT_TIMEOUT:
                        logger.warning(f"Node {node_id} heartbeat timeout")
                        await self.disconnect_node(node_id)
                        continue
                    
                    # Send heartbeat (simulated)
                    await self._send_heartbeat(node_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _send_heartbeat(self, node_id: str):
        """Send heartbeat to node."""
        if node_id in self._nodes:
            self._nodes[node_id].last_heartbeat = time.time()
    
    async def _healing_loop(self):
        """Periodic network healing."""
        while True:
            try:
                await asyncio.sleep(self.HEALING_INTERVAL)
                
                # Check if we need more connections
                if self.connected_count < self.MIN_CONNECTIONS:
                    self._state = MeshState.HEALING
                    await self._heal_network()
                    self._state = MeshState.CONNECTED
                
                # Optimize topology
                await self._optimize_topology()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Healing error: {e}")
    
    async def _heal_network(self):
        """Attempt to heal network by finding new connections."""
        logger.info("Healing network...")
        
        # Try bootstrap nodes
        for bootstrap in self._bootstrap_nodes:
            if self.connected_count >= self.MIN_CONNECTIONS:
                break
            
            await self._connect_to_address(bootstrap)
        
        # Request more peers from connected nodes
        for node_id in list(self._nodes.keys()):
            if self.connected_count >= self.MIN_CONNECTIONS:
                break
            
            peers = await self._request_peers(node_id)
            for peer in peers:
                if self.connected_count >= self.MAX_CONNECTIONS:
                    break
                await self._connect_to_address(peer)
        
        self._emit("network_healed", {"connections": self.connected_count})
    
    async def _optimize_topology(self):
        """Optimize network topology."""
        # Remove duplicate edges
        # Balance connections
        # Prefer lower latency paths
        pass
    
    def add_node(self, node: MeshNode):
        """Add a node to the mesh."""
        if node.node_id == self.node_id:
            return
        
        self._nodes[node.node_id] = node
        self._topology_version += 1
        
        self._emit("node_added", {"node_id": node.node_id})
    
    def remove_node(self, node_id: str):
        """Remove a node from the mesh."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._edges = {e for e in self._edges if node_id not in e}
            self._topology_version += 1
            
            self._emit("node_removed", {"node_id": node_id})
    
    def add_edge(self, node1: str, node2: str):
        """Add an edge between nodes."""
        if node1 != node2:
            self._edges.add((min(node1, node2), max(node1, node2)))
            self._topology_version += 1
    
    def remove_edge(self, node1: str, node2: str):
        """Remove an edge between nodes."""
        edge = (min(node1, node2), max(node1, node2))
        if edge in self._edges:
            self._edges.remove(edge)
            self._topology_version += 1
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node."""
        neighbors = []
        for edge in self._edges:
            if edge[0] == node_id:
                neighbors.append(edge[1])
            elif edge[1] == node_id:
                neighbors.append(edge[0])
        return neighbors
    
    def get_topology(self) -> MeshTopology:
        """Get current mesh topology."""
        return MeshTopology(
            nodes=self._nodes.copy(),
            edges=list(self._edges),
            timestamp=time.time(),
            version=self._topology_version
        )
    
    def is_connected(self, node1: str, node2: str) -> bool:
        """Check if two nodes are connected (directly or indirectly)."""
        if node1 == node2:
            return True
        
        # BFS to find path
        visited = set()
        queue = [node1]
        
        while queue:
            current = queue.pop(0)
            if current == node2:
                return True
            
            if current in visited:
                continue
            visited.add(current)
            
            queue.extend(self.get_neighbors(current))
        
        return False
    
    def detect_partitions(self) -> List[Set[str]]:
        """Detect network partitions."""
        if not self._nodes:
            return []
        
        partitions = []
        unvisited = set(self._nodes.keys())
        unvisited.add(self.node_id)
        
        while unvisited:
            # Start new partition
            start = next(iter(unvisited))
            partition = set()
            queue = [start]
            
            while queue:
                current = queue.pop(0)
                if current in partition:
                    continue
                
                partition.add(current)
                unvisited.discard(current)
                
                for neighbor in self.get_neighbors(current):
                    if neighbor not in partition:
                        queue.append(neighbor)
            
            partitions.append(partition)
        
        if len(partitions) > 1:
            self._state = MeshState.PARTITIONED
            self._emit("partition_detected", {"partitions": len(partitions)})
        
        return partitions
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable):
        """Unregister event handler."""
        if event in self._handlers:
            self._handlers[event].remove(handler)
    
    def _emit(self, event: str, data: Dict[str, Any]):
        """Emit event to handlers."""
        for handler in self._handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics."""
        return {
            "node_id": self.node_id,
            "state": self._state.value,
            "total_nodes": len(self._nodes),
            "connected_nodes": self.connected_count,
            "edges": len(self._edges),
            "topology_version": self._topology_version,
            "bootstrap_nodes": len(self._bootstrap_nodes)
        }
    
    def __repr__(self) -> str:
        return (f"MeshNetwork(node_id='{self.node_id[:8]}...', "
                f"nodes={len(self._nodes)}, state={self._state.value})")
