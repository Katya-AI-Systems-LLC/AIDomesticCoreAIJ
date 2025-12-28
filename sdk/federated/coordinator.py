"""
Federated Learning Coordinator
==============================

Coordinates distributed quantum-classical training across nodes.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import uuid
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Model aggregation strategies."""
    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal"
    SCAFFOLD = "scaffold"
    QUANTUM_SECURE = "quantum_secure"


class TrainingPhase(Enum):
    """Training phases."""
    IDLE = "idle"
    DISTRIBUTING = "distributing"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    VALIDATING = "validating"


@dataclass
class TrainingRound:
    """A single training round."""
    round_id: str
    round_number: int
    phase: TrainingPhase
    participants: List[str]
    start_time: float
    end_time: Optional[float] = None
    global_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ParticipantUpdate:
    """Update from a training participant."""
    node_id: str
    round_id: str
    weights: np.ndarray
    num_samples: int
    local_loss: float
    metrics: Dict[str, float]
    quantum_signature: bytes
    timestamp: float


class FederatedCoordinator:
    """
    Coordinator for federated quantum-classical learning.
    
    Features:
    - Distributed training coordination
    - Multiple aggregation strategies
    - Quantum-secure aggregation
    - Participant management
    - Model versioning
    
    Example:
        >>> coordinator = FederatedCoordinator()
        >>> await coordinator.start_training(model, num_rounds=10)
        >>> final_model = coordinator.get_global_model()
    """
    
    def __init__(self, 
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.QUANTUM_SECURE,
                 min_participants: int = 2,
                 participation_threshold: float = 0.8,
                 language: str = "en"):
        """
        Initialize federated coordinator.
        
        Args:
            aggregation_strategy: Strategy for aggregating updates
            min_participants: Minimum participants per round
            participation_threshold: Required participation rate
            language: Language for messages
        """
        self.aggregation_strategy = aggregation_strategy
        self.min_participants = min_participants
        self.participation_threshold = participation_threshold
        self.language = language
        
        # Participants
        self._participants: Dict[str, Dict[str, Any]] = {}
        
        # Training state
        self._current_round: Optional[TrainingRound] = None
        self._round_history: List[TrainingRound] = []
        self._global_weights: Optional[np.ndarray] = None
        self._model_version = 0
        
        # Pending updates
        self._pending_updates: Dict[str, ParticipantUpdate] = {}
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {}
        
        logger.info(f"Federated coordinator initialized: strategy={aggregation_strategy.value}")
    
    def register_participant(self, node_id: str,
                             capabilities: Optional[Dict] = None,
                             quantum_signature: Optional[bytes] = None) -> bool:
        """
        Register a training participant.
        
        Args:
            node_id: Participant node ID
            capabilities: Node capabilities
            quantum_signature: Node's quantum signature
            
        Returns:
            True if registered successfully
        """
        if node_id in self._participants:
            return False
        
        self._participants[node_id] = {
            "capabilities": capabilities or {},
            "quantum_signature": quantum_signature,
            "registered": time.time(),
            "rounds_participated": 0,
            "total_samples": 0,
            "last_seen": time.time()
        }
        
        self._emit("participant_registered", {"node_id": node_id})
        logger.info(f"Registered participant: {node_id}")
        return True
    
    def unregister_participant(self, node_id: str) -> bool:
        """Unregister a participant."""
        if node_id in self._participants:
            del self._participants[node_id]
            self._emit("participant_unregistered", {"node_id": node_id})
            return True
        return False
    
    def get_participants(self) -> List[str]:
        """Get list of registered participants."""
        return list(self._participants.keys())
    
    def get_active_participants(self) -> List[str]:
        """Get participants active in current round."""
        if self._current_round:
            return self._current_round.participants.copy()
        return []
    
    async def start_training(self, initial_weights: np.ndarray,
                              num_rounds: int = 10,
                              epochs_per_round: int = 1) -> Dict[str, Any]:
        """
        Start federated training.
        
        Args:
            initial_weights: Initial model weights
            num_rounds: Number of training rounds
            epochs_per_round: Local epochs per round
            
        Returns:
            Training results
        """
        if len(self._participants) < self.min_participants:
            raise ValueError(f"Need at least {self.min_participants} participants")
        
        self._global_weights = initial_weights.copy()
        self._model_version = 0
        
        training_start = time.time()
        
        for round_num in range(num_rounds):
            logger.info(f"Starting round {round_num + 1}/{num_rounds}")
            
            # Execute training round
            round_result = await self._execute_round(round_num, epochs_per_round)
            
            if round_result:
                self._round_history.append(round_result)
                self._emit("round_completed", {
                    "round": round_num,
                    "loss": round_result.global_loss
                })
        
        training_time = time.time() - training_start
        
        return {
            "rounds_completed": len(self._round_history),
            "final_loss": self._round_history[-1].global_loss if self._round_history else None,
            "training_time": training_time,
            "model_version": self._model_version,
            "participants": len(self._participants)
        }
    
    async def _execute_round(self, round_num: int,
                              epochs: int) -> Optional[TrainingRound]:
        """Execute a single training round."""
        round_id = str(uuid.uuid4())
        
        # Select participants
        participants = self._select_participants()
        
        if len(participants) < self.min_participants:
            logger.warning("Not enough participants for round")
            return None
        
        # Create round
        training_round = TrainingRound(
            round_id=round_id,
            round_number=round_num,
            phase=TrainingPhase.DISTRIBUTING,
            participants=participants,
            start_time=time.time()
        )
        self._current_round = training_round
        
        # Distribute model
        await self._distribute_model(participants)
        
        # Wait for training
        training_round.phase = TrainingPhase.TRAINING
        await self._wait_for_updates(participants, timeout=300)
        
        # Aggregate updates
        training_round.phase = TrainingPhase.AGGREGATING
        aggregation_result = await self._aggregate_updates()
        
        if aggregation_result:
            self._global_weights = aggregation_result["weights"]
            training_round.global_loss = aggregation_result["loss"]
            training_round.metrics = aggregation_result["metrics"]
            self._model_version += 1
        
        # Validate
        training_round.phase = TrainingPhase.VALIDATING
        # Validation logic here
        
        training_round.end_time = time.time()
        training_round.phase = TrainingPhase.IDLE
        
        # Clear pending updates
        self._pending_updates.clear()
        
        return training_round
    
    def _select_participants(self) -> List[str]:
        """Select participants for training round."""
        # Select all active participants
        active = [
            node_id for node_id, info in self._participants.items()
            if time.time() - info["last_seen"] < 300  # Active in last 5 minutes
        ]
        
        # Apply participation threshold
        required = int(len(self._participants) * self.participation_threshold)
        required = max(required, self.min_participants)
        
        if len(active) >= required:
            return active[:required]
        
        return active
    
    async def _distribute_model(self, participants: List[str]):
        """Distribute global model to participants."""
        for node_id in participants:
            # In production, this would send model over network
            self._emit("model_distributed", {
                "node_id": node_id,
                "version": self._model_version
            })
    
    async def _wait_for_updates(self, participants: List[str],
                                 timeout: float):
        """Wait for updates from participants."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if we have enough updates
            received = len(self._pending_updates)
            required = int(len(participants) * self.participation_threshold)
            
            if received >= required:
                return
            
            await asyncio.sleep(1)
        
        logger.warning(f"Timeout waiting for updates. Received {len(self._pending_updates)}/{len(participants)}")
    
    def submit_update(self, update: ParticipantUpdate) -> bool:
        """
        Submit training update from participant.
        
        Args:
            update: Participant's training update
            
        Returns:
            True if accepted
        """
        if not self._current_round:
            return False
        
        if update.node_id not in self._current_round.participants:
            return False
        
        if update.round_id != self._current_round.round_id:
            return False
        
        self._pending_updates[update.node_id] = update
        
        # Update participant stats
        if update.node_id in self._participants:
            self._participants[update.node_id]["rounds_participated"] += 1
            self._participants[update.node_id]["total_samples"] += update.num_samples
            self._participants[update.node_id]["last_seen"] = time.time()
        
        self._emit("update_received", {"node_id": update.node_id})
        return True
    
    async def _aggregate_updates(self) -> Optional[Dict[str, Any]]:
        """Aggregate participant updates."""
        if not self._pending_updates:
            return None
        
        if self.aggregation_strategy == AggregationStrategy.FEDAVG:
            return self._federated_averaging()
        elif self.aggregation_strategy == AggregationStrategy.QUANTUM_SECURE:
            return await self._quantum_secure_aggregation()
        else:
            return self._federated_averaging()
    
    def _federated_averaging(self) -> Dict[str, Any]:
        """Standard federated averaging."""
        total_samples = sum(u.num_samples for u in self._pending_updates.values())
        
        if total_samples == 0:
            return None
        
        # Weighted average of weights
        aggregated_weights = None
        total_loss = 0.0
        
        for update in self._pending_updates.values():
            weight = update.num_samples / total_samples
            
            if aggregated_weights is None:
                aggregated_weights = update.weights * weight
            else:
                aggregated_weights += update.weights * weight
            
            total_loss += update.local_loss * weight
        
        return {
            "weights": aggregated_weights,
            "loss": total_loss,
            "metrics": {"participants": len(self._pending_updates)}
        }
    
    async def _quantum_secure_aggregation(self) -> Dict[str, Any]:
        """Quantum-secure aggregation with verification."""
        # Verify quantum signatures
        verified_updates = {}
        
        for node_id, update in self._pending_updates.items():
            # In production, verify quantum signature
            if self._verify_signature(update):
                verified_updates[node_id] = update
        
        if not verified_updates:
            return None
        
        # Perform secure aggregation
        total_samples = sum(u.num_samples for u in verified_updates.values())
        aggregated_weights = None
        total_loss = 0.0
        
        for update in verified_updates.values():
            weight = update.num_samples / total_samples
            
            if aggregated_weights is None:
                aggregated_weights = update.weights * weight
            else:
                aggregated_weights += update.weights * weight
            
            total_loss += update.local_loss * weight
        
        return {
            "weights": aggregated_weights,
            "loss": total_loss,
            "metrics": {
                "participants": len(verified_updates),
                "verified": True
            }
        }
    
    def _verify_signature(self, update: ParticipantUpdate) -> bool:
        """Verify participant's quantum signature."""
        if update.node_id not in self._participants:
            return False
        
        expected_sig = self._participants[update.node_id].get("quantum_signature")
        
        if expected_sig and update.quantum_signature != expected_sig:
            return False
        
        return True
    
    def get_global_model(self) -> Optional[np.ndarray]:
        """Get current global model weights."""
        return self._global_weights.copy() if self._global_weights is not None else None
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return [
            {
                "round": r.round_number,
                "loss": r.global_loss,
                "participants": len(r.participants),
                "duration": (r.end_time - r.start_time) if r.end_time else None
            }
            for r in self._round_history
        ]
    
    def on(self, event: str, callback: Callable):
        """Register event callback."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _emit(self, event: str, data: Dict[str, Any]):
        """Emit event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "participants": len(self._participants),
            "rounds_completed": len(self._round_history),
            "model_version": self._model_version,
            "aggregation_strategy": self.aggregation_strategy.value,
            "current_phase": self._current_round.phase.value if self._current_round else "idle"
        }
    
    def __repr__(self) -> str:
        return (f"FederatedCoordinator(participants={len(self._participants)}, "
                f"rounds={len(self._round_history)})")
