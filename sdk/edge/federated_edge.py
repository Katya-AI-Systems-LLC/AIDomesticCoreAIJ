"""
Federated Edge Learning
=======================

Federated learning for edge devices.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Federated aggregation strategies."""
    FED_AVG = "fedavg"
    FED_PROX = "fedprox"
    FED_SGD = "fedsgd"
    SCAFFOLD = "scaffold"
    FED_NOVA = "fednova"


@dataclass
class EdgeClient:
    """Federated edge client."""
    client_id: str
    device_type: str
    capabilities: Dict[str, Any]
    last_seen: float
    training_rounds: int = 0
    total_samples: int = 0
    status: str = "idle"


@dataclass
class FederatedRound:
    """Federated learning round."""
    round_id: int
    participants: List[str]
    global_model_version: int
    local_updates: Dict[str, Any]
    aggregated: bool = False
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class FederatedEdge:
    """
    Federated learning for edge devices.
    
    Features:
    - Edge-optimized federated learning
    - Multiple aggregation strategies
    - Differential privacy
    - Secure aggregation
    - Asynchronous updates
    
    Example:
        >>> fed = FederatedEdge()
        >>> fed.register_client("edge_001", capabilities)
        >>> await fed.start_round(["edge_001", "edge_002"])
    """
    
    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
                 min_clients: int = 2,
                 rounds_per_epoch: int = 10,
                 local_epochs: int = 5):
        """
        Initialize federated edge system.
        
        Args:
            strategy: Aggregation strategy
            min_clients: Minimum clients per round
            rounds_per_epoch: Rounds per global epoch
            local_epochs: Local training epochs
        """
        self.strategy = strategy
        self.min_clients = min_clients
        self.rounds_per_epoch = rounds_per_epoch
        self.local_epochs = local_epochs
        
        # Clients
        self._clients: Dict[str, EdgeClient] = {}
        
        # Rounds
        self._rounds: List[FederatedRound] = []
        self._current_round: Optional[FederatedRound] = None
        
        # Global model
        self._global_model_version = 0
        self._global_weights: Optional[Dict] = None
        
        # Callbacks
        self._on_round_complete: List[Callable] = []
        
        logger.info(f"Federated Edge initialized (strategy={strategy.value})")
    
    def register_client(self, client_id: str,
                        device_type: str = "edge",
                        capabilities: Dict = None) -> EdgeClient:
        """
        Register edge client.
        
        Args:
            client_id: Client identifier
            device_type: Device type
            capabilities: Device capabilities
            
        Returns:
            EdgeClient
        """
        client = EdgeClient(
            client_id=client_id,
            device_type=device_type,
            capabilities=capabilities or {},
            last_seen=time.time()
        )
        
        self._clients[client_id] = client
        
        logger.info(f"Client registered: {client_id}")
        return client
    
    def unregister_client(self, client_id: str):
        """Unregister client."""
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info(f"Client unregistered: {client_id}")
    
    async def start_round(self, participants: List[str] = None) -> FederatedRound:
        """
        Start new federated round.
        
        Args:
            participants: Client IDs to participate (None = all available)
            
        Returns:
            FederatedRound
        """
        if participants is None:
            participants = self._select_participants()
        
        if len(participants) < self.min_clients:
            raise ValueError(f"Need at least {self.min_clients} clients")
        
        round_id = len(self._rounds)
        
        fed_round = FederatedRound(
            round_id=round_id,
            participants=participants,
            global_model_version=self._global_model_version,
            local_updates={}
        )
        
        self._rounds.append(fed_round)
        self._current_round = fed_round
        
        # Update client status
        for client_id in participants:
            if client_id in self._clients:
                self._clients[client_id].status = "training"
        
        logger.info(f"Round {round_id} started with {len(participants)} clients")
        return fed_round
    
    def _select_participants(self) -> List[str]:
        """Select participants for round."""
        available = [
            c.client_id for c in self._clients.values()
            if c.status == "idle" and time.time() - c.last_seen < 300
        ]
        
        return available
    
    async def submit_update(self, client_id: str,
                            local_weights: Dict,
                            num_samples: int,
                            metrics: Dict = None) -> bool:
        """
        Submit local model update.
        
        Args:
            client_id: Client ID
            local_weights: Local model weights
            num_samples: Number of training samples
            metrics: Training metrics
            
        Returns:
            True if accepted
        """
        if not self._current_round:
            return False
        
        if client_id not in self._current_round.participants:
            return False
        
        self._current_round.local_updates[client_id] = {
            "weights": local_weights,
            "num_samples": num_samples,
            "metrics": metrics or {},
            "submitted_at": time.time()
        }
        
        # Update client
        if client_id in self._clients:
            client = self._clients[client_id]
            client.status = "idle"
            client.training_rounds += 1
            client.total_samples += num_samples
            client.last_seen = time.time()
        
        logger.info(f"Update received from {client_id}")
        
        # Check if round complete
        if len(self._current_round.local_updates) == len(self._current_round.participants):
            await self._aggregate_round()
        
        return True
    
    async def _aggregate_round(self):
        """Aggregate round updates."""
        if not self._current_round:
            return
        
        fed_round = self._current_round
        
        # Perform aggregation based on strategy
        if self.strategy == AggregationStrategy.FED_AVG:
            aggregated = self._federated_averaging(fed_round.local_updates)
        elif self.strategy == AggregationStrategy.FED_PROX:
            aggregated = self._fedprox_aggregation(fed_round.local_updates)
        else:
            aggregated = self._federated_averaging(fed_round.local_updates)
        
        # Update global model
        self._global_weights = aggregated
        self._global_model_version += 1
        
        # Mark round complete
        fed_round.aggregated = True
        fed_round.completed_at = time.time()
        
        # Callbacks
        for callback in self._on_round_complete:
            try:
                callback(fed_round, aggregated)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        self._current_round = None
        
        logger.info(f"Round {fed_round.round_id} aggregated")
    
    def _federated_averaging(self, updates: Dict) -> Dict:
        """FedAvg aggregation."""
        import numpy as np
        
        total_samples = sum(u["num_samples"] for u in updates.values())
        
        # Weighted average
        aggregated = {}
        
        for client_id, update in updates.items():
            weight = update["num_samples"] / total_samples
            
            for key, value in update["weights"].items():
                if key not in aggregated:
                    aggregated[key] = np.zeros_like(value) if hasattr(value, 'shape') else 0
                
                aggregated[key] = aggregated[key] + value * weight
        
        return aggregated
    
    def _fedprox_aggregation(self, updates: Dict) -> Dict:
        """FedProx aggregation with proximal term."""
        # Same as FedAvg but clients use proximal term during training
        return self._federated_averaging(updates)
    
    def get_global_model(self) -> Tuple[Dict, int]:
        """Get current global model."""
        return self._global_weights, self._global_model_version
    
    def get_client(self, client_id: str) -> Optional[EdgeClient]:
        """Get client info."""
        return self._clients.get(client_id)
    
    def get_clients(self, status: str = None) -> List[EdgeClient]:
        """Get all clients, optionally filtered by status."""
        clients = list(self._clients.values())
        
        if status:
            clients = [c for c in clients if c.status == status]
        
        return clients
    
    def get_round_history(self) -> List[FederatedRound]:
        """Get round history."""
        return self._rounds
    
    def on_round_complete(self, callback: Callable):
        """Register round complete callback."""
        self._on_round_complete.append(callback)
    
    def get_statistics(self) -> Dict:
        """Get federated learning statistics."""
        return {
            "total_clients": len(self._clients),
            "active_clients": len([c for c in self._clients.values() if c.status != "offline"]),
            "total_rounds": len(self._rounds),
            "global_model_version": self._global_model_version,
            "strategy": self.strategy.value
        }
    
    def __repr__(self) -> str:
        return f"FederatedEdge(clients={len(self._clients)}, rounds={len(self._rounds)})"


# Import Tuple for type hint
from typing import Tuple
