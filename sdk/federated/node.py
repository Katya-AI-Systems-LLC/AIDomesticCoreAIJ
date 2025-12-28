"""
Quantum Federated Node
======================

Participant node for federated quantum-classical learning.
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class LocalTrainingConfig:
    """Configuration for local training."""
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    quantum_layers: int = 2
    use_quantum: bool = True


class QuantumFederatedNode:
    """
    Federated learning participant with quantum capabilities.
    
    Features:
    - Local model training
    - Quantum circuit integration
    - Secure update submission
    - Privacy preservation
    
    Example:
        >>> node = QuantumFederatedNode(node_id="node_001")
        >>> await node.connect_to_coordinator("coordinator_address")
        >>> await node.participate_in_training()
    """
    
    def __init__(self, node_id: str,
                 quantum_signature: Optional[bytes] = None,
                 config: Optional[LocalTrainingConfig] = None,
                 language: str = "en"):
        """
        Initialize federated node.
        
        Args:
            node_id: Unique node identifier
            quantum_signature: Node's quantum signature
            config: Training configuration
            language: Language for messages
        """
        self.node_id = node_id
        self.quantum_signature = quantum_signature or self._generate_signature()
        self.config = config or LocalTrainingConfig()
        self.language = language
        
        # Model state
        self._local_weights: Optional[np.ndarray] = None
        self._global_weights: Optional[np.ndarray] = None
        
        # Training data
        self._train_data: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        
        # Coordinator connection
        self._coordinator_address: Optional[str] = None
        self._connected = False
        
        # Training state
        self._current_round: Optional[str] = None
        self._training_history: list = []
        
        logger.info(f"Quantum federated node initialized: {node_id}")
    
    def _generate_signature(self) -> bytes:
        """Generate quantum signature."""
        import secrets
        import hashlib
        
        entropy = secrets.token_bytes(32)
        return hashlib.sha256(self.node_id.encode() + entropy).digest()
    
    def set_training_data(self, data: np.ndarray, labels: np.ndarray):
        """
        Set local training data.
        
        Args:
            data: Training features
            labels: Training labels
        """
        self._train_data = data
        self._train_labels = labels
        logger.info(f"Set training data: {len(data)} samples")
    
    async def connect_to_coordinator(self, address: str) -> bool:
        """
        Connect to federated coordinator.
        
        Args:
            address: Coordinator address
            
        Returns:
            True if connected successfully
        """
        self._coordinator_address = address
        
        # In production, establish actual connection
        # Simulate connection
        self._connected = True
        
        logger.info(f"Connected to coordinator: {address}")
        return True
    
    async def disconnect(self):
        """Disconnect from coordinator."""
        self._connected = False
        self._coordinator_address = None
        logger.info("Disconnected from coordinator")
    
    def receive_global_model(self, weights: np.ndarray, round_id: str):
        """
        Receive global model from coordinator.
        
        Args:
            weights: Global model weights
            round_id: Current round ID
        """
        self._global_weights = weights.copy()
        self._local_weights = weights.copy()
        self._current_round = round_id
        
        logger.info(f"Received global model for round {round_id}")
    
    async def train_local(self) -> Dict[str, Any]:
        """
        Perform local training.
        
        Returns:
            Training results
        """
        if self._local_weights is None:
            raise ValueError("No model weights to train")
        
        if self._train_data is None:
            raise ValueError("No training data")
        
        start_time = time.time()
        
        # Perform local training
        if self.config.use_quantum:
            result = await self._quantum_train()
        else:
            result = self._classical_train()
        
        training_time = time.time() - start_time
        
        self._training_history.append({
            "round": self._current_round,
            "loss": result["loss"],
            "time": training_time
        })
        
        logger.info(f"Local training completed: loss={result['loss']:.4f}")
        
        return {
            **result,
            "training_time": training_time,
            "num_samples": len(self._train_data)
        }
    
    async def _quantum_train(self) -> Dict[str, Any]:
        """Perform quantum-enhanced training."""
        # Simulate quantum training
        # In production, this would use actual quantum circuits
        
        num_samples = len(self._train_data)
        
        for epoch in range(self.config.epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_data = self._train_data[batch_indices]
                batch_labels = self._train_labels[batch_indices]
                
                # Quantum forward pass (simulated)
                predictions = self._quantum_forward(batch_data)
                
                # Compute loss
                loss = self._compute_loss(predictions, batch_labels)
                epoch_loss += loss
                num_batches += 1
                
                # Quantum gradient computation (simulated)
                gradients = self._quantum_gradients(batch_data, batch_labels)
                
                # Update weights
                self._local_weights -= self.config.learning_rate * gradients
            
            avg_loss = epoch_loss / num_batches
            logger.debug(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
        
        return {
            "loss": avg_loss,
            "weights": self._local_weights,
            "quantum_enhanced": True
        }
    
    def _classical_train(self) -> Dict[str, Any]:
        """Perform classical training."""
        num_samples = len(self._train_data)
        
        for epoch in range(self.config.epochs):
            indices = np.random.permutation(num_samples)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_data = self._train_data[batch_indices]
                batch_labels = self._train_labels[batch_indices]
                
                # Forward pass
                predictions = self._forward(batch_data)
                
                # Compute loss
                loss = self._compute_loss(predictions, batch_labels)
                epoch_loss += loss
                num_batches += 1
                
                # Compute gradients
                gradients = self._compute_gradients(batch_data, batch_labels)
                
                # Update weights
                self._local_weights -= self.config.learning_rate * gradients
            
            avg_loss = epoch_loss / num_batches
        
        return {
            "loss": avg_loss,
            "weights": self._local_weights,
            "quantum_enhanced": False
        }
    
    def _quantum_forward(self, data: np.ndarray) -> np.ndarray:
        """Quantum forward pass (simulated)."""
        # Simulate quantum layer output
        return np.tanh(data @ self._local_weights.reshape(-1, 1)).flatten()
    
    def _forward(self, data: np.ndarray) -> np.ndarray:
        """Classical forward pass."""
        return np.tanh(data @ self._local_weights.reshape(-1, 1)).flatten()
    
    def _compute_loss(self, predictions: np.ndarray, 
                      labels: np.ndarray) -> float:
        """Compute loss."""
        return float(np.mean((predictions - labels) ** 2))
    
    def _quantum_gradients(self, data: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
        """Compute quantum gradients (simulated)."""
        # Parameter shift rule simulation
        epsilon = 0.01
        gradients = np.zeros_like(self._local_weights)
        
        for i in range(len(self._local_weights)):
            # Shift up
            weights_plus = self._local_weights.copy()
            weights_plus[i] += epsilon
            
            # Shift down
            weights_minus = self._local_weights.copy()
            weights_minus[i] -= epsilon
            
            # Compute gradient
            loss_plus = self._compute_loss(
                np.tanh(data @ weights_plus.reshape(-1, 1)).flatten(),
                labels
            )
            loss_minus = self._compute_loss(
                np.tanh(data @ weights_minus.reshape(-1, 1)).flatten(),
                labels
            )
            
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients
    
    def _compute_gradients(self, data: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
        """Compute classical gradients."""
        predictions = self._forward(data)
        error = predictions - labels
        
        # Gradient of tanh
        grad_tanh = 1 - predictions ** 2
        
        # Gradient w.r.t. weights
        gradients = np.mean(
            data * (error * grad_tanh).reshape(-1, 1),
            axis=0
        )
        
        return gradients
    
    def get_update(self) -> Dict[str, Any]:
        """
        Get update to send to coordinator.
        
        Returns:
            Update dictionary
        """
        return {
            "node_id": self.node_id,
            "round_id": self._current_round,
            "weights": self._local_weights,
            "num_samples": len(self._train_data) if self._train_data is not None else 0,
            "quantum_signature": self.quantum_signature,
            "timestamp": time.time()
        }
    
    def get_training_history(self) -> list:
        """Get local training history."""
        return self._training_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            "node_id": self.node_id,
            "connected": self._connected,
            "coordinator": self._coordinator_address,
            "training_rounds": len(self._training_history),
            "data_samples": len(self._train_data) if self._train_data is not None else 0,
            "quantum_enabled": self.config.use_quantum
        }
    
    def __repr__(self) -> str:
        return f"QuantumFederatedNode(id='{self.node_id}', connected={self._connected})"
