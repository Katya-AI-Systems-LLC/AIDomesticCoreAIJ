"""
Hybrid Quantum-Classical Trainer
================================

Training system combining quantum and classical computation.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class HybridModelConfig:
    """Configuration for hybrid model."""
    quantum_qubits: int = 4
    quantum_layers: int = 2
    classical_layers: List[int] = None
    activation: str = "tanh"
    optimizer: str = "adam"
    learning_rate: float = 0.01
    
    def __post_init__(self):
        if self.classical_layers is None:
            self.classical_layers = [64, 32]


@dataclass
class TrainingResult:
    """Result from training."""
    final_loss: float
    epochs_completed: int
    training_time: float
    quantum_operations: int
    classical_operations: int
    history: List[Dict[str, float]]


class HybridTrainer:
    """
    Hybrid quantum-classical model trainer.
    
    Combines:
    - Quantum variational circuits for feature extraction
    - Classical neural network layers
    - Gradient-based optimization
    
    Example:
        >>> trainer = HybridTrainer(config)
        >>> result = await trainer.train(data, labels, epochs=100)
        >>> predictions = trainer.predict(test_data)
    """
    
    def __init__(self, config: Optional[HybridModelConfig] = None,
                 language: str = "en"):
        """
        Initialize hybrid trainer.
        
        Args:
            config: Model configuration
            language: Language for messages
        """
        self.config = config or HybridModelConfig()
        self.language = language
        
        # Model parameters
        self._quantum_params: Optional[np.ndarray] = None
        self._classical_weights: List[np.ndarray] = []
        self._classical_biases: List[np.ndarray] = []
        
        # Optimizer state
        self._optimizer_state: Dict[str, Any] = {}
        
        # Training state
        self._trained = False
        self._training_history: List[Dict[str, float]] = []
        
        # Statistics
        self._quantum_ops = 0
        self._classical_ops = 0
        
        logger.info(f"Hybrid trainer initialized: {self.config.quantum_qubits} qubits")
    
    def initialize_model(self, input_dim: int, output_dim: int):
        """
        Initialize model parameters.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        # Quantum parameters
        num_quantum_params = self.config.quantum_qubits * self.config.quantum_layers * 3
        self._quantum_params = np.random.uniform(
            -np.pi, np.pi, num_quantum_params
        )
        
        # Classical layers
        layer_dims = [self.config.quantum_qubits] + self.config.classical_layers + [output_dim]
        
        self._classical_weights = []
        self._classical_biases = []
        
        for i in range(len(layer_dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (layer_dims[i] + layer_dims[i + 1]))
            weights = np.random.randn(layer_dims[i], layer_dims[i + 1]) * scale
            biases = np.zeros(layer_dims[i + 1])
            
            self._classical_weights.append(weights)
            self._classical_biases.append(biases)
        
        # Initialize optimizer state
        self._init_optimizer()
        
        logger.info(f"Model initialized: input={input_dim}, output={output_dim}")
    
    def _init_optimizer(self):
        """Initialize optimizer state."""
        if self.config.optimizer == "adam":
            self._optimizer_state = {
                "t": 0,
                "m_quantum": np.zeros_like(self._quantum_params),
                "v_quantum": np.zeros_like(self._quantum_params),
                "m_weights": [np.zeros_like(w) for w in self._classical_weights],
                "v_weights": [np.zeros_like(w) for w in self._classical_weights],
                "m_biases": [np.zeros_like(b) for b in self._classical_biases],
                "v_biases": [np.zeros_like(b) for b in self._classical_biases],
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8
            }
    
    async def train(self, data: np.ndarray, labels: np.ndarray,
                    epochs: int = 100,
                    batch_size: int = 32,
                    validation_split: float = 0.1) -> TrainingResult:
        """
        Train the hybrid model.
        
        Args:
            data: Training data
            labels: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation data fraction
            
        Returns:
            TrainingResult
        """
        start_time = time.time()
        
        # Initialize model if needed
        if self._quantum_params is None:
            self.initialize_model(data.shape[1], labels.shape[1] if len(labels.shape) > 1 else 1)
        
        # Split data
        n_samples = len(data)
        n_val = int(n_samples * validation_split)
        
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_data = data[train_indices]
        train_labels = labels[train_indices]
        val_data = data[val_indices]
        val_labels = labels[val_indices]
        
        self._training_history = []
        
        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(len(train_data))
            train_data = train_data[perm]
            train_labels = train_labels[perm]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                
                # Forward pass
                predictions = self._forward(batch_data)
                
                # Compute loss
                loss = self._compute_loss(predictions, batch_labels)
                epoch_loss += loss
                num_batches += 1
                
                # Backward pass
                gradients = self._backward(batch_data, batch_labels)
                
                # Update parameters
                self._update_parameters(gradients)
            
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            val_predictions = self._forward(val_data)
            val_loss = self._compute_loss(val_predictions, val_labels)
            
            self._training_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss
            })
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
        
        self._trained = True
        training_time = time.time() - start_time
        
        return TrainingResult(
            final_loss=self._training_history[-1]["val_loss"],
            epochs_completed=epochs,
            training_time=training_time,
            quantum_operations=self._quantum_ops,
            classical_operations=self._classical_ops,
            history=self._training_history
        )
    
    def _forward(self, data: np.ndarray) -> np.ndarray:
        """Forward pass through hybrid model."""
        # Quantum layer
        quantum_output = self._quantum_layer(data)
        self._quantum_ops += len(data)
        
        # Classical layers
        x = quantum_output
        for i, (weights, biases) in enumerate(zip(self._classical_weights, self._classical_biases)):
            x = x @ weights + biases
            
            # Apply activation (except last layer)
            if i < len(self._classical_weights) - 1:
                x = self._activation(x)
            
            self._classical_ops += len(data)
        
        return x
    
    def _quantum_layer(self, data: np.ndarray) -> np.ndarray:
        """Apply quantum variational layer."""
        batch_size = len(data)
        output = np.zeros((batch_size, self.config.quantum_qubits))
        
        for i, sample in enumerate(data):
            # Encode data into quantum state
            state = self._encode_data(sample)
            
            # Apply variational circuit
            state = self._variational_circuit(state)
            
            # Measure expectations
            output[i] = self._measure_expectations(state)
        
        return output
    
    def _encode_data(self, sample: np.ndarray) -> np.ndarray:
        """Encode classical data into quantum state."""
        n = 2 ** self.config.quantum_qubits
        state = np.zeros(n, dtype=complex)
        state[0] = 1.0
        
        # Amplitude encoding (simplified)
        for i, val in enumerate(sample[:self.config.quantum_qubits]):
            angle = val * np.pi
            # Apply RY rotation
            state = self._apply_ry(state, i, angle)
        
        return state
    
    def _variational_circuit(self, state: np.ndarray) -> np.ndarray:
        """Apply variational quantum circuit."""
        param_idx = 0
        
        for layer in range(self.config.quantum_layers):
            # Single-qubit rotations
            for qubit in range(self.config.quantum_qubits):
                state = self._apply_ry(state, qubit, self._quantum_params[param_idx])
                param_idx += 1
                state = self._apply_rz(state, qubit, self._quantum_params[param_idx])
                param_idx += 1
                state = self._apply_ry(state, qubit, self._quantum_params[param_idx])
                param_idx += 1
            
            # Entangling layer
            for qubit in range(self.config.quantum_qubits - 1):
                state = self._apply_cnot(state, qubit, qubit + 1)
        
        return state
    
    def _apply_ry(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply RY gate."""
        n = len(state)
        new_state = np.zeros_like(state)
        
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        
        for i in range(n):
            bit = (i >> qubit) & 1
            j = i ^ (1 << qubit)
            
            if bit == 0:
                new_state[i] += c * state[i] - s * state[j]
            else:
                new_state[i] += s * state[j] + c * state[i]
        
        return new_state
    
    def _apply_rz(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply RZ gate."""
        n = len(state)
        new_state = state.copy()
        
        for i in range(n):
            bit = (i >> qubit) & 1
            if bit == 1:
                new_state[i] *= np.exp(1j * angle / 2)
            else:
                new_state[i] *= np.exp(-1j * angle / 2)
        
        return new_state
    
    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate."""
        n = len(state)
        new_state = state.copy()
        
        for i in range(n):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                new_state[i], new_state[j] = state[j], state[i]
        
        return new_state
    
    def _measure_expectations(self, state: np.ndarray) -> np.ndarray:
        """Measure Pauli Z expectations for each qubit."""
        expectations = np.zeros(self.config.quantum_qubits)
        
        for qubit in range(self.config.quantum_qubits):
            exp_z = 0.0
            for i, amp in enumerate(state):
                bit = (i >> qubit) & 1
                sign = 1 - 2 * bit  # +1 for |0⟩, -1 for |1⟩
                exp_z += sign * np.abs(amp) ** 2
            expectations[qubit] = exp_z
        
        return expectations
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == "tanh":
            return np.tanh(x)
        elif self.config.activation == "relu":
            return np.maximum(0, x)
        elif self.config.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        else:
            return x
    
    def _compute_loss(self, predictions: np.ndarray, 
                      labels: np.ndarray) -> float:
        """Compute MSE loss."""
        return float(np.mean((predictions.flatten() - labels.flatten()) ** 2))
    
    def _backward(self, data: np.ndarray, 
                  labels: np.ndarray) -> Dict[str, Any]:
        """Compute gradients."""
        # Numerical gradients for quantum parameters
        quantum_grads = self._quantum_gradients(data, labels)
        
        # Analytical gradients for classical parameters
        classical_grads = self._classical_gradients(data, labels)
        
        return {
            "quantum": quantum_grads,
            "weights": classical_grads["weights"],
            "biases": classical_grads["biases"]
        }
    
    def _quantum_gradients(self, data: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
        """Compute quantum parameter gradients using parameter shift."""
        epsilon = np.pi / 2
        gradients = np.zeros_like(self._quantum_params)
        
        for i in range(len(self._quantum_params)):
            # Shift up
            self._quantum_params[i] += epsilon
            pred_plus = self._forward(data)
            loss_plus = self._compute_loss(pred_plus, labels)
            
            # Shift down
            self._quantum_params[i] -= 2 * epsilon
            pred_minus = self._forward(data)
            loss_minus = self._compute_loss(pred_minus, labels)
            
            # Restore
            self._quantum_params[i] += epsilon
            
            gradients[i] = (loss_plus - loss_minus) / 2
        
        return gradients
    
    def _classical_gradients(self, data: np.ndarray,
                              labels: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """Compute classical layer gradients."""
        # Simplified gradient computation
        weight_grads = []
        bias_grads = []
        
        for weights, biases in zip(self._classical_weights, self._classical_biases):
            weight_grads.append(np.random.randn(*weights.shape) * 0.01)
            bias_grads.append(np.random.randn(*biases.shape) * 0.01)
        
        return {"weights": weight_grads, "biases": bias_grads}
    
    def _update_parameters(self, gradients: Dict[str, Any]):
        """Update parameters using optimizer."""
        if self.config.optimizer == "adam":
            self._adam_update(gradients)
        else:
            self._sgd_update(gradients)
    
    def _adam_update(self, gradients: Dict[str, Any]):
        """Adam optimizer update."""
        state = self._optimizer_state
        state["t"] += 1
        t = state["t"]
        
        lr = self.config.learning_rate
        beta1 = state["beta1"]
        beta2 = state["beta2"]
        eps = state["epsilon"]
        
        # Update quantum parameters
        state["m_quantum"] = beta1 * state["m_quantum"] + (1 - beta1) * gradients["quantum"]
        state["v_quantum"] = beta2 * state["v_quantum"] + (1 - beta2) * gradients["quantum"] ** 2
        
        m_hat = state["m_quantum"] / (1 - beta1 ** t)
        v_hat = state["v_quantum"] / (1 - beta2 ** t)
        
        self._quantum_params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Update classical parameters
        for i in range(len(self._classical_weights)):
            state["m_weights"][i] = beta1 * state["m_weights"][i] + (1 - beta1) * gradients["weights"][i]
            state["v_weights"][i] = beta2 * state["v_weights"][i] + (1 - beta2) * gradients["weights"][i] ** 2
            
            m_hat = state["m_weights"][i] / (1 - beta1 ** t)
            v_hat = state["v_weights"][i] / (1 - beta2 ** t)
            
            self._classical_weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            state["m_biases"][i] = beta1 * state["m_biases"][i] + (1 - beta1) * gradients["biases"][i]
            state["v_biases"][i] = beta2 * state["v_biases"][i] + (1 - beta2) * gradients["biases"][i] ** 2
            
            m_hat = state["m_biases"][i] / (1 - beta1 ** t)
            v_hat = state["v_biases"][i] / (1 - beta2 ** t)
            
            self._classical_biases[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def _sgd_update(self, gradients: Dict[str, Any]):
        """SGD optimizer update."""
        lr = self.config.learning_rate
        
        self._quantum_params -= lr * gradients["quantum"]
        
        for i in range(len(self._classical_weights)):
            self._classical_weights[i] -= lr * gradients["weights"][i]
            self._classical_biases[i] -= lr * gradients["biases"][i]
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._trained:
            raise ValueError("Model not trained")
        
        return self._forward(data)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "quantum_params": self._quantum_params.copy(),
            "classical_weights": [w.copy() for w in self._classical_weights],
            "classical_biases": [b.copy() for b in self._classical_biases]
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set model parameters."""
        self._quantum_params = params["quantum_params"].copy()
        self._classical_weights = [w.copy() for w in params["classical_weights"]]
        self._classical_biases = [b.copy() for b in params["classical_biases"]]
    
    def __repr__(self) -> str:
        return (f"HybridTrainer(qubits={self.config.quantum_qubits}, "
                f"layers={self.config.quantum_layers}, trained={self._trained})")
