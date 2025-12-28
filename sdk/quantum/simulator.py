"""
Quantum Simulator Module
========================

High-performance quantum circuit simulator with support for:
- State vector simulation
- Density matrix simulation
- Noise modeling
- GPU acceleration (when available)
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result from quantum simulation."""
    state_vector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    counts: Dict[str, int] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)
    expectation_values: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    memory_used_mb: float = 0.0


@dataclass
class NoiseModel:
    """Quantum noise model configuration."""
    depolarizing_rate: float = 0.0
    amplitude_damping_rate: float = 0.0
    phase_damping_rate: float = 0.0
    readout_error_rate: float = 0.0
    gate_error_rates: Dict[str, float] = field(default_factory=dict)


class QuantumSimulator:
    """
    High-performance quantum circuit simulator.
    
    Supports:
    - State vector simulation for pure states
    - Density matrix simulation for mixed states
    - Configurable noise models
    - Measurement sampling
    - Expectation value computation
    
    Example:
        >>> simulator = QuantumSimulator(num_qubits=4)
        >>> circuit = QuantumCircuitBuilder(4).h(0).cx(0, 1).measure_all()
        >>> result = simulator.run(circuit, shots=1000)
        >>> print(result.counts)
    """
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    def __init__(self, num_qubits: int,
                 simulation_method: str = "statevector",
                 noise_model: Optional[NoiseModel] = None,
                 seed: Optional[int] = None,
                 language: str = "en"):
        """
        Initialize quantum simulator.
        
        Args:
            num_qubits: Number of qubits to simulate
            simulation_method: 'statevector' or 'density_matrix'
            noise_model: Optional noise model
            seed: Random seed for reproducibility
            language: Language for messages
        """
        self.num_qubits = num_qubits
        self.simulation_method = simulation_method
        self.noise_model = noise_model
        self.language = language
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state
        self._state = None
        self._density_matrix = None
        self.reset()
        
        logger.info(f"Quantum simulator initialized: {num_qubits} qubits, "
                   f"method={simulation_method}")
    
    def reset(self):
        """Reset simulator to |0...0⟩ state."""
        n = 2 ** self.num_qubits
        
        if self.simulation_method == "statevector":
            self._state = np.zeros(n, dtype=complex)
            self._state[0] = 1.0
        else:
            self._density_matrix = np.zeros((n, n), dtype=complex)
            self._density_matrix[0, 0] = 1.0
    
    def run(self, circuit: Any, shots: int = 1024,
            compute_state: bool = True) -> SimulationResult:
        """
        Run quantum circuit simulation.
        
        Args:
            circuit: Quantum circuit to simulate
            shots: Number of measurement shots
            compute_state: Whether to return final state
            
        Returns:
            SimulationResult with simulation results
        """
        import time
        start_time = time.time()
        
        self.reset()
        
        # Get gates from circuit
        if hasattr(circuit, 'gates'):
            gates = circuit.gates
        elif hasattr(circuit, 'data'):
            # Qiskit circuit
            gates = self._extract_qiskit_gates(circuit)
        else:
            gates = []
        
        # Apply gates
        measurement_qubits = []
        for gate in gates:
            if gate.name == "measure":
                measurement_qubits.append(gate.qubits[0])
            else:
                self._apply_gate(gate)
        
        # Apply noise if configured
        if self.noise_model:
            self._apply_noise()
        
        # Compute measurements
        if measurement_qubits or shots > 0:
            counts, probabilities = self._measure(shots, measurement_qubits)
        else:
            counts = {}
            probabilities = {}
        
        execution_time = time.time() - start_time
        
        result = SimulationResult(
            counts=counts,
            probabilities=probabilities,
            execution_time=execution_time,
            memory_used_mb=self._estimate_memory()
        )
        
        if compute_state:
            if self.simulation_method == "statevector":
                result.state_vector = self._state.copy()
            else:
                result.density_matrix = self._density_matrix.copy()
        
        return result
    
    def _apply_gate(self, gate: Any):
        """Apply quantum gate to state."""
        name = gate.name
        qubits = gate.qubits
        params = gate.parameters if hasattr(gate, 'parameters') else []
        
        if name == "h":
            self._apply_single_qubit_gate(self.H, qubits[0])
        elif name == "x":
            self._apply_single_qubit_gate(self.X, qubits[0])
        elif name == "y":
            self._apply_single_qubit_gate(self.Y, qubits[0])
        elif name == "z":
            self._apply_single_qubit_gate(self.Z, qubits[0])
        elif name == "s":
            self._apply_single_qubit_gate(self.S, qubits[0])
        elif name == "t":
            self._apply_single_qubit_gate(self.T, qubits[0])
        elif name == "sdg":
            self._apply_single_qubit_gate(self.S.conj().T, qubits[0])
        elif name == "tdg":
            self._apply_single_qubit_gate(self.T.conj().T, qubits[0])
        elif name == "rx":
            self._apply_single_qubit_gate(self._rx_matrix(params[0]), qubits[0])
        elif name == "ry":
            self._apply_single_qubit_gate(self._ry_matrix(params[0]), qubits[0])
        elif name == "rz":
            self._apply_single_qubit_gate(self._rz_matrix(params[0]), qubits[0])
        elif name == "u":
            self._apply_single_qubit_gate(
                self._u_matrix(params[0], params[1], params[2]), qubits[0])
        elif name == "cx":
            self._apply_cnot(qubits[0], qubits[1])
        elif name == "cy":
            self._apply_controlled_gate(self.Y, qubits[0], qubits[1])
        elif name == "cz":
            self._apply_controlled_gate(self.Z, qubits[0], qubits[1])
        elif name == "swap":
            self._apply_swap(qubits[0], qubits[1])
        elif name == "ccx":
            self._apply_toffoli(qubits[0], qubits[1], qubits[2])
        elif name == "reset":
            self._apply_reset(qubits[0])
        elif name == "barrier":
            pass  # No-op for simulation
    
    def _apply_single_qubit_gate(self, matrix: np.ndarray, qubit: int):
        """Apply single-qubit gate to state."""
        if self.simulation_method == "statevector":
            self._apply_single_qubit_statevector(matrix, qubit)
        else:
            self._apply_single_qubit_density(matrix, qubit)
    
    def _apply_single_qubit_statevector(self, matrix: np.ndarray, qubit: int):
        """Apply single-qubit gate to state vector."""
        n = 2 ** self.num_qubits
        new_state = np.zeros(n, dtype=complex)
        
        for i in range(n):
            bit = (i >> qubit) & 1
            j = i ^ (1 << qubit)  # Flip qubit
            
            if bit == 0:
                new_state[i] += matrix[0, 0] * self._state[i]
                new_state[i] += matrix[0, 1] * self._state[j]
            else:
                new_state[i] += matrix[1, 0] * self._state[j]
                new_state[i] += matrix[1, 1] * self._state[i]
        
        self._state = new_state
    
    def _apply_single_qubit_density(self, matrix: np.ndarray, qubit: int):
        """Apply single-qubit gate to density matrix."""
        # Build full operator
        full_matrix = self._build_full_operator(matrix, qubit)
        
        # Apply: ρ' = U ρ U†
        self._density_matrix = full_matrix @ self._density_matrix @ full_matrix.conj().T
    
    def _build_full_operator(self, matrix: np.ndarray, qubit: int) -> np.ndarray:
        """Build full n-qubit operator from single-qubit gate."""
        result = np.array([[1]], dtype=complex)
        
        for i in range(self.num_qubits):
            if i == qubit:
                result = np.kron(result, matrix)
            else:
                result = np.kron(result, self.I)
        
        return result
    
    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        if self.simulation_method == "statevector":
            n = 2 ** self.num_qubits
            new_state = self._state.copy()
            
            for i in range(n):
                control_bit = (i >> control) & 1
                if control_bit == 1:
                    j = i ^ (1 << target)
                    new_state[i], new_state[j] = self._state[j], self._state[i]
            
            self._state = new_state
        else:
            # Build CNOT matrix
            cnot = self._build_cnot_matrix(control, target)
            self._density_matrix = cnot @ self._density_matrix @ cnot.conj().T
    
    def _build_cnot_matrix(self, control: int, target: int) -> np.ndarray:
        """Build full CNOT matrix."""
        n = 2 ** self.num_qubits
        cnot = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            control_bit = (i >> control) & 1
            if control_bit == 0:
                cnot[i, i] = 1
            else:
                j = i ^ (1 << target)
                cnot[i, j] = 1
        
        return cnot
    
    def _apply_controlled_gate(self, matrix: np.ndarray, 
                                control: int, target: int):
        """Apply controlled single-qubit gate."""
        if self.simulation_method == "statevector":
            n = 2 ** self.num_qubits
            new_state = self._state.copy()
            
            for i in range(n):
                control_bit = (i >> control) & 1
                if control_bit == 1:
                    target_bit = (i >> target) & 1
                    j = i ^ (1 << target)
                    
                    if target_bit == 0:
                        new_state[i] = matrix[0, 0] * self._state[i] + matrix[0, 1] * self._state[j]
                        new_state[j] = matrix[1, 0] * self._state[i] + matrix[1, 1] * self._state[j]
            
            self._state = new_state
    
    def _apply_swap(self, qubit1: int, qubit2: int):
        """Apply SWAP gate."""
        if self.simulation_method == "statevector":
            n = 2 ** self.num_qubits
            new_state = self._state.copy()
            
            for i in range(n):
                bit1 = (i >> qubit1) & 1
                bit2 = (i >> qubit2) & 1
                
                if bit1 != bit2:
                    j = i ^ (1 << qubit1) ^ (1 << qubit2)
                    new_state[i], new_state[j] = self._state[j], self._state[i]
            
            self._state = new_state
    
    def _apply_toffoli(self, control1: int, control2: int, target: int):
        """Apply Toffoli (CCX) gate."""
        if self.simulation_method == "statevector":
            n = 2 ** self.num_qubits
            new_state = self._state.copy()
            
            for i in range(n):
                c1 = (i >> control1) & 1
                c2 = (i >> control2) & 1
                
                if c1 == 1 and c2 == 1:
                    j = i ^ (1 << target)
                    new_state[i], new_state[j] = self._state[j], self._state[i]
            
            self._state = new_state
    
    def _apply_reset(self, qubit: int):
        """Reset qubit to |0⟩."""
        # Measure and conditionally flip
        prob_1 = self._get_qubit_probability(qubit, 1)
        
        if np.random.random() < prob_1:
            # Qubit is |1⟩, apply X to reset
            self._apply_single_qubit_gate(self.X, qubit)
    
    def _get_qubit_probability(self, qubit: int, value: int) -> float:
        """Get probability of qubit being in given state."""
        if self.simulation_method == "statevector":
            prob = 0.0
            n = 2 ** self.num_qubits
            
            for i in range(n):
                if ((i >> qubit) & 1) == value:
                    prob += np.abs(self._state[i]) ** 2
            
            return prob
        else:
            # Trace out other qubits
            prob = 0.0
            n = 2 ** self.num_qubits
            
            for i in range(n):
                if ((i >> qubit) & 1) == value:
                    prob += np.real(self._density_matrix[i, i])
            
            return prob
    
    def _rx_matrix(self, theta: float) -> np.ndarray:
        """RX rotation matrix."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    
    def _ry_matrix(self, theta: float) -> np.ndarray:
        """RY rotation matrix."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    def _rz_matrix(self, phi: float) -> np.ndarray:
        """RZ rotation matrix."""
        return np.array([
            [np.exp(-1j * phi / 2), 0],
            [0, np.exp(1j * phi / 2)]
        ], dtype=complex)
    
    def _u_matrix(self, theta: float, phi: float, lam: float) -> np.ndarray:
        """General U gate matrix."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
        ], dtype=complex)
    
    def _measure(self, shots: int, 
                 qubits: Optional[List[int]] = None) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Perform measurement sampling."""
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        # Calculate probabilities
        n = 2 ** self.num_qubits
        
        if self.simulation_method == "statevector":
            probabilities = np.abs(self._state) ** 2
        else:
            probabilities = np.real(np.diag(self._density_matrix))
        
        # Apply readout error if configured
        if self.noise_model and self.noise_model.readout_error_rate > 0:
            probabilities = self._apply_readout_error(probabilities)
        
        # Sample measurements
        indices = np.random.choice(n, size=shots, p=probabilities)
        
        counts = {}
        for idx in indices:
            # Extract measured qubits
            bitstring = ''.join(str((idx >> q) & 1) for q in reversed(qubits))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        # Calculate probabilities
        prob_dict = {k: v / shots for k, v in counts.items()}
        
        return counts, prob_dict
    
    def _apply_readout_error(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply readout error to probabilities."""
        error_rate = self.noise_model.readout_error_rate
        n = len(probabilities)
        
        # Simple bit-flip error model
        new_probs = np.zeros(n)
        for i in range(n):
            for j in range(n):
                # Probability of measuring j when state is i
                hamming = bin(i ^ j).count('1')
                flip_prob = (error_rate ** hamming) * ((1 - error_rate) ** (self.num_qubits - hamming))
                new_probs[j] += probabilities[i] * flip_prob
        
        return new_probs / new_probs.sum()
    
    def _apply_noise(self):
        """Apply noise model to state."""
        if not self.noise_model:
            return
        
        # Convert to density matrix if needed
        if self.simulation_method == "statevector" and self.noise_model.depolarizing_rate > 0:
            n = 2 ** self.num_qubits
            self._density_matrix = np.outer(self._state, self._state.conj())
            self.simulation_method = "density_matrix"
        
        # Apply depolarizing noise
        if self.noise_model.depolarizing_rate > 0:
            self._apply_depolarizing_noise()
        
        # Apply amplitude damping
        if self.noise_model.amplitude_damping_rate > 0:
            self._apply_amplitude_damping()
    
    def _apply_depolarizing_noise(self):
        """Apply depolarizing noise channel."""
        p = self.noise_model.depolarizing_rate
        n = 2 ** self.num_qubits
        
        # ρ' = (1-p)ρ + p*I/n
        identity = np.eye(n, dtype=complex) / n
        self._density_matrix = (1 - p) * self._density_matrix + p * identity
    
    def _apply_amplitude_damping(self):
        """Apply amplitude damping noise."""
        gamma = self.noise_model.amplitude_damping_rate
        
        for qubit in range(self.num_qubits):
            # Kraus operators for amplitude damping
            K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
            K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
            
            # Apply to each qubit
            self._apply_kraus_channel([K0, K1], qubit)
    
    def _apply_kraus_channel(self, kraus_ops: List[np.ndarray], qubit: int):
        """Apply Kraus channel to density matrix."""
        n = 2 ** self.num_qubits
        new_density = np.zeros((n, n), dtype=complex)
        
        for K in kraus_ops:
            full_K = self._build_full_operator(K, qubit)
            new_density += full_K @ self._density_matrix @ full_K.conj().T
        
        self._density_matrix = new_density
    
    def _extract_qiskit_gates(self, circuit: Any) -> List[Any]:
        """Extract gates from Qiskit circuit."""
        from .circuit_builder import QuantumGate
        
        gates = []
        for instruction in circuit.data:
            name = instruction.operation.name
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            params = list(instruction.operation.params)
            classical_bits = [circuit.find_bit(c).index for c in instruction.clbits]
            
            gates.append(QuantumGate(name, qubits, params, classical_bits))
        
        return gates
    
    def _estimate_memory(self) -> float:
        """Estimate memory usage in MB."""
        n = 2 ** self.num_qubits
        
        if self.simulation_method == "statevector":
            # Complex128: 16 bytes per element
            return n * 16 / (1024 * 1024)
        else:
            # Density matrix: n x n complex
            return n * n * 16 / (1024 * 1024)
    
    def get_state_vector(self) -> Optional[np.ndarray]:
        """Get current state vector."""
        return self._state.copy() if self._state is not None else None
    
    def get_density_matrix(self) -> Optional[np.ndarray]:
        """Get current density matrix."""
        if self._density_matrix is not None:
            return self._density_matrix.copy()
        elif self._state is not None:
            return np.outer(self._state, self._state.conj())
        return None
    
    def compute_expectation(self, observable: np.ndarray) -> float:
        """Compute expectation value of observable."""
        if self.simulation_method == "statevector":
            return np.real(np.conj(self._state) @ observable @ self._state)
        else:
            return np.real(np.trace(observable @ self._density_matrix))
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities for all basis states."""
        n = 2 ** self.num_qubits
        
        if self.simulation_method == "statevector":
            probs = np.abs(self._state) ** 2
        else:
            probs = np.real(np.diag(self._density_matrix))
        
        return {format(i, f'0{self.num_qubits}b'): float(probs[i]) 
                for i in range(n) if probs[i] > 1e-10}
    
    def __repr__(self) -> str:
        return (f"QuantumSimulator(num_qubits={self.num_qubits}, "
                f"method='{self.simulation_method}')")
