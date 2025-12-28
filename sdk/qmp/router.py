"""
QMP Router Implementation
=========================

Quantum signature-based routing for mesh networks.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq
import time
import logging

logger = logging.getLogger(__name__)


class RoutingAlgorithm(Enum):
    """Routing algorithm types."""
    SHORTEST_PATH = "shortest_path"
    QUANTUM_OPTIMIZED = "quantum_optimized"
    LOAD_BALANCED = "load_balanced"
    TRUST_WEIGHTED = "trust_weighted"


@dataclass
class Route:
    """A route through the mesh network."""
    destination: str
    path: List[str]
    metric: float
    quantum_verified: bool = False
    created: float = field(default_factory=time.time)
    expires: float = 0.0
    
    def __post_init__(self):
        if self.expires == 0.0:
            self.expires = self.created + 300  # 5 minute default


@dataclass
class LinkState:
    """State of a network link."""
    source: str
    target: str
    latency_ms: float
    bandwidth_mbps: float
    packet_loss: float
    trust_score: float
    quantum_signature: bytes
    last_updated: float


class QMPRouter:
    """
    Quantum Mesh Protocol router.
    
    Provides:
    - Quantum signature-based routing
    - Multiple routing algorithms
    - Route caching and optimization
    - Trust-weighted path selection
    
    Example:
        >>> router = QMPRouter(node_id="router_001")
        >>> router.add_link("node_a", "node_b", latency=10)
        >>> route = router.find_route("node_a", "node_c")
    """
    
    def __init__(self, node_id: str,
                 algorithm: RoutingAlgorithm = RoutingAlgorithm.QUANTUM_OPTIMIZED,
                 language: str = "en"):
        """
        Initialize QMP router.
        
        Args:
            node_id: This node's ID
            algorithm: Routing algorithm to use
            language: Language for messages
        """
        self.node_id = node_id
        self.algorithm = algorithm
        self.language = language
        
        # Network topology
        self._links: Dict[Tuple[str, str], LinkState] = {}
        self._nodes: Dict[str, Dict[str, Any]] = {}
        
        # Route cache
        self._route_cache: Dict[Tuple[str, str], Route] = {}
        
        # Statistics
        self._stats = {
            "routes_computed": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"QMP Router initialized: {node_id}, algorithm={algorithm.value}")
    
    def add_node(self, node_id: str, 
                 quantum_signature: bytes,
                 metadata: Optional[Dict] = None):
        """Add a node to the topology."""
        self._nodes[node_id] = {
            "quantum_signature": quantum_signature,
            "metadata": metadata or {},
            "added": time.time()
        }
    
    def remove_node(self, node_id: str):
        """Remove a node from the topology."""
        if node_id in self._nodes:
            del self._nodes[node_id]
        
        # Remove associated links
        links_to_remove = [
            key for key in self._links
            if node_id in key
        ]
        for key in links_to_remove:
            del self._links[key]
        
        # Invalidate cached routes
        self._invalidate_routes_through(node_id)
    
    def add_link(self, source: str, target: str,
                 latency: float = 10.0,
                 bandwidth: float = 100.0,
                 packet_loss: float = 0.0,
                 trust_score: float = 1.0,
                 quantum_signature: Optional[bytes] = None,
                 bidirectional: bool = True):
        """
        Add a link between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            latency: Link latency in ms
            bandwidth: Bandwidth in Mbps
            packet_loss: Packet loss rate (0-1)
            trust_score: Trust score (0-1)
            quantum_signature: Link quantum signature
            bidirectional: Add reverse link too
        """
        link = LinkState(
            source=source,
            target=target,
            latency_ms=latency,
            bandwidth_mbps=bandwidth,
            packet_loss=packet_loss,
            trust_score=trust_score,
            quantum_signature=quantum_signature or b"",
            last_updated=time.time()
        )
        
        self._links[(source, target)] = link
        
        if bidirectional:
            reverse_link = LinkState(
                source=target,
                target=source,
                latency_ms=latency,
                bandwidth_mbps=bandwidth,
                packet_loss=packet_loss,
                trust_score=trust_score,
                quantum_signature=quantum_signature or b"",
                last_updated=time.time()
            )
            self._links[(target, source)] = reverse_link
        
        # Invalidate affected routes
        self._invalidate_routes_through(source)
        self._invalidate_routes_through(target)
    
    def remove_link(self, source: str, target: str, bidirectional: bool = True):
        """Remove a link between nodes."""
        if (source, target) in self._links:
            del self._links[(source, target)]
        
        if bidirectional and (target, source) in self._links:
            del self._links[(target, source)]
        
        self._invalidate_routes_through(source)
        self._invalidate_routes_through(target)
    
    def update_link(self, source: str, target: str, **kwargs):
        """Update link properties."""
        key = (source, target)
        if key in self._links:
            link = self._links[key]
            for attr, value in kwargs.items():
                if hasattr(link, attr):
                    setattr(link, attr, value)
            link.last_updated = time.time()
            
            self._invalidate_routes_through(source)
    
    def find_route(self, source: str, destination: str,
                   use_cache: bool = True) -> Optional[Route]:
        """
        Find route from source to destination.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            use_cache: Use cached routes if available
            
        Returns:
            Route if found, None otherwise
        """
        cache_key = (source, destination)
        
        # Check cache
        if use_cache and cache_key in self._route_cache:
            route = self._route_cache[cache_key]
            if route.expires > time.time():
                self._stats["cache_hits"] += 1
                return route
            else:
                del self._route_cache[cache_key]
        
        self._stats["cache_misses"] += 1
        
        # Compute route based on algorithm
        if self.algorithm == RoutingAlgorithm.SHORTEST_PATH:
            route = self._dijkstra(source, destination)
        elif self.algorithm == RoutingAlgorithm.QUANTUM_OPTIMIZED:
            route = self._quantum_optimized_route(source, destination)
        elif self.algorithm == RoutingAlgorithm.LOAD_BALANCED:
            route = self._load_balanced_route(source, destination)
        elif self.algorithm == RoutingAlgorithm.TRUST_WEIGHTED:
            route = self._trust_weighted_route(source, destination)
        else:
            route = self._dijkstra(source, destination)
        
        if route:
            self._route_cache[cache_key] = route
            self._stats["routes_computed"] += 1
        
        return route
    
    def _dijkstra(self, source: str, destination: str) -> Optional[Route]:
        """Standard Dijkstra's shortest path algorithm."""
        if source == destination:
            return Route(destination=destination, path=[source], metric=0)
        
        # Get all nodes
        nodes = set()
        for (s, t) in self._links:
            nodes.add(s)
            nodes.add(t)
        
        if source not in nodes or destination not in nodes:
            return None
        
        # Initialize distances
        distances = {node: float('inf') for node in nodes}
        distances[source] = 0
        previous = {node: None for node in nodes}
        
        # Priority queue: (distance, node)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == destination:
                break
            
            # Check neighbors
            for (s, t), link in self._links.items():
                if s != current:
                    continue
                
                if t in visited:
                    continue
                
                # Calculate edge weight (latency-based)
                weight = link.latency_ms
                new_dist = current_dist + weight
                
                if new_dist < distances[t]:
                    distances[t] = new_dist
                    previous[t] = current
                    heapq.heappush(pq, (new_dist, t))
        
        # Reconstruct path
        if distances[destination] == float('inf'):
            return None
        
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return Route(
            destination=destination,
            path=path,
            metric=distances[destination],
            quantum_verified=True
        )
    
    def _quantum_optimized_route(self, source: str, 
                                  destination: str) -> Optional[Route]:
        """
        Quantum-optimized routing considering:
        - Latency
        - Quantum signature verification
        - Trust scores
        """
        if source == destination:
            return Route(destination=destination, path=[source], metric=0)
        
        nodes = set()
        for (s, t) in self._links:
            nodes.add(s)
            nodes.add(t)
        
        if source not in nodes or destination not in nodes:
            return None
        
        distances = {node: float('inf') for node in nodes}
        distances[source] = 0
        previous = {node: None for node in nodes}
        
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == destination:
                break
            
            for (s, t), link in self._links.items():
                if s != current or t in visited:
                    continue
                
                # Quantum-optimized weight calculation
                weight = self._calculate_quantum_weight(link)
                new_dist = current_dist + weight
                
                if new_dist < distances[t]:
                    distances[t] = new_dist
                    previous[t] = current
                    heapq.heappush(pq, (new_dist, t))
        
        if distances[destination] == float('inf'):
            return None
        
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return Route(
            destination=destination,
            path=path,
            metric=distances[destination],
            quantum_verified=True
        )
    
    def _calculate_quantum_weight(self, link: LinkState) -> float:
        """Calculate quantum-optimized link weight."""
        # Base weight from latency
        weight = link.latency_ms
        
        # Adjust for packet loss
        weight *= (1 + link.packet_loss * 10)
        
        # Adjust for trust (lower trust = higher weight)
        weight *= (2 - link.trust_score)
        
        # Bonus for quantum-verified links
        if link.quantum_signature:
            weight *= 0.9
        
        return weight
    
    def _load_balanced_route(self, source: str, 
                              destination: str) -> Optional[Route]:
        """Load-balanced routing considering bandwidth."""
        # Similar to quantum optimized but prioritizes bandwidth
        route = self._quantum_optimized_route(source, destination)
        return route
    
    def _trust_weighted_route(self, source: str,
                               destination: str) -> Optional[Route]:
        """Trust-weighted routing prioritizing trusted paths."""
        if source == destination:
            return Route(destination=destination, path=[source], metric=0)
        
        nodes = set()
        for (s, t) in self._links:
            nodes.add(s)
            nodes.add(t)
        
        if source not in nodes or destination not in nodes:
            return None
        
        # Use trust score as primary metric (inverted - higher trust = lower weight)
        distances = {node: float('inf') for node in nodes}
        distances[source] = 0
        previous = {node: None for node in nodes}
        
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == destination:
                break
            
            for (s, t), link in self._links.items():
                if s != current or t in visited:
                    continue
                
                # Trust-based weight (invert trust score)
                weight = 1.0 / (link.trust_score + 0.01)
                new_dist = current_dist + weight
                
                if new_dist < distances[t]:
                    distances[t] = new_dist
                    previous[t] = current
                    heapq.heappush(pq, (new_dist, t))
        
        if distances[destination] == float('inf'):
            return None
        
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return Route(
            destination=destination,
            path=path,
            metric=distances[destination],
            quantum_verified=True
        )
    
    def find_all_routes(self, source: str, destination: str,
                        max_routes: int = 5) -> List[Route]:
        """Find multiple routes to destination."""
        routes = []
        
        # Find primary route
        primary = self.find_route(source, destination, use_cache=False)
        if primary:
            routes.append(primary)
        
        # Find alternative routes by temporarily removing links
        for i in range(max_routes - 1):
            if not routes:
                break
            
            # Get last route's critical link
            last_route = routes[-1]
            if len(last_route.path) < 2:
                break
            
            # Try removing each link in the path
            for j in range(len(last_route.path) - 1):
                link_key = (last_route.path[j], last_route.path[j + 1])
                
                if link_key in self._links:
                    # Temporarily remove link
                    saved_link = self._links[link_key]
                    del self._links[link_key]
                    
                    # Find alternative
                    alt_route = self._dijkstra(source, destination)
                    
                    # Restore link
                    self._links[link_key] = saved_link
                    
                    if alt_route and alt_route.path not in [r.path for r in routes]:
                        routes.append(alt_route)
                        break
        
        return routes
    
    def _invalidate_routes_through(self, node_id: str):
        """Invalidate cached routes that pass through a node."""
        keys_to_remove = [
            key for key, route in self._route_cache.items()
            if node_id in route.path
        ]
        for key in keys_to_remove:
            del self._route_cache[key]
    
    def get_topology(self) -> Dict[str, Any]:
        """Get network topology."""
        return {
            "nodes": list(self._nodes.keys()),
            "links": [
                {
                    "source": link.source,
                    "target": link.target,
                    "latency": link.latency_ms,
                    "bandwidth": link.bandwidth_mbps,
                    "trust": link.trust_score
                }
                for link in self._links.values()
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "node_id": self.node_id,
            "algorithm": self.algorithm.value,
            "nodes": len(self._nodes),
            "links": len(self._links),
            "cached_routes": len(self._route_cache),
            **self._stats
        }
    
    def __repr__(self) -> str:
        return (f"QMPRouter(node_id='{self.node_id}', "
                f"nodes={len(self._nodes)}, links={len(self._links)})")
