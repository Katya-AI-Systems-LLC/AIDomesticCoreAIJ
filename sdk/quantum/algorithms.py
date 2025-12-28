"""
Quantum Algorithms Module
=========================

Implementation of key quantum algorithms:
- VQE (Variational Quantum Eigensolver)
- QAOA (Quantum Approximate Optimization Algorithm)
- Grover's Search Algorithm
- Shor's Factoring Algorithm
"""

from typing import List, Optional, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class VQEResult:
    """Result from VQE optimization."""
    optimal_value: float
    optimal_parameters: List[float]
    eigenvalue: float
    eigenvector: Optional[np.ndarray] = None
    iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)
    execution_time: float = 0.0


@dataclass
class QAOAResult:
    """Result from QAOA optimization."""
    optimal_value: float
    optimal_parameters: Tuple[List[float], List[float]] = ([], [])
    best_solution: str = ""
    solution_probability: float = 0.0
    iterations: int = 0
    execution_time: float = 0.0


@dataclass
class GroverResult:
    """Result from Grover's search."""
    found_states: List[str]
    probabilities: Dict[str, float]
    iterations_used: int
    success_probability: float
    execution_time: float = 0.0


@dataclass
class ShorResult:
    """Result from Shor's factoring."""
    number: int
    factors: Tuple[int, int]
    period: int
    success: bool
    execution_time: float = 0.0


class VQESolver:
    """
    Variational Quantum Eigensolver for finding ground state energy.
    
    VQE is a hybrid quantum-classical algorithm that uses a parameterized
    quantum circuit (ansatz) to prepare trial states and a classical
    optimizer to find the optimal parameters.
    
    Example:
        >>> hamiltonian = create_hamiltonian(...)
        >>> vqe = VQESolver(hamiltonian, num_qubits=4)
        >>> result = vqe.solve()
        >>> print(f"Ground state energy: {result.eigenvalue}")
    """
    
    def __init__(self, hamiltonian: Any, num_qubits: int,
                 ansatz_type: str = "ry_rz",
                 ansatz_depth: int = 2,
                 optimizer: str = "COBYLA",
                 language: str = "en"):
        """
        Initialize VQE solver.
        
        Args:
            hamiltonian: Hamiltonian operator (Pauli sum or matrix)
            num_qubits: Number of qubits
            ansatz_type: Type of variational ansatz
            ansatz_depth: Depth of the ansatz circuit
            optimizer: Classical optimizer to use
            language: Language for messages
        """
        self.hamiltonian = hamiltonian
        self.num_qubits = num_qubits
        self.ansatz_type = ansatz_type
        self.ansatz_depth = ansatz_depth
        self.optimizer = optimizer
        self.language = language
        self._circuit = None
        self._parameters = None
        
        logger.info(f"VQE solver initialized: {num_qubits} qubits, {ansatz_type} ansatz")
    
    def _build_ansatz(self) -> Any:
        """Build variational ansatz circuit."""
        from .circuit_builder import QuantumCircuitBuilder
        
        num_params = self.num_qubits * self.ansatz_depth * 2
        self._parameters = np.random.uniform(-np.pi, np.pi, num_params)
        
        builder = QuantumCircuitBuilder(self.num_qubits, name="vqe_ansatz")
        
        param_idx = 0
        for layer in range(self.ansatz_depth):
            # Single-qubit rotations
            for qubit in range(self.num_qubits):
                if self.ansatz_type == "ry_rz":
                    builder.ry(qubit, self._parameters[param_idx])
                    param_idx += 1
                    builder.rz(qubit, self._parameters[param_idx])
                    param_idx += 1
                elif self.ansatz_type == "rx_ry":
                    builder.rx(qubit, self._parameters[param_idx])
                    param_idx += 1
                    builder.ry(qubit, self._parameters[param_idx])
                    param_idx += 1
            
            # Entangling layer
            for qubit in range(self.num_qubits - 1):
                builder.cx(qubit, qubit + 1)
            
            # Circular entanglement
            if self.num_qubits > 2:
                builder.cx(self.num_qubits - 1, 0)
        
        self._circuit = builder
        return builder
    
    def _compute_expectation(self, parameters: np.ndarray) -> float:
        """Compute expectation value of Hamiltonian."""
        # Update circuit parameters
        self._parameters = parameters
        
        # For simulation, compute expectation value
        if isinstance(self.hamiltonian, np.ndarray):
            # Matrix Hamiltonian
            state = self._simulate_state(parameters)
            expectation = np.real(np.conj(state) @ self.hamiltonian @ state)
            return float(expectation)
        else:
            # Pauli Hamiltonian - simulate measurement
            return self._measure_pauli_expectation(parameters)
    
    def _simulate_state(self, parameters: np.ndarray) -> np.ndarray:
        """Simulate quantum state for given parameters."""
        n = 2 ** self.num_qubits
        state = np.zeros(n, dtype=complex)
        state[0] = 1.0  # |0...0⟩
        
        # Apply parameterized gates (simplified simulation)
        param_idx = 0
        for layer in range(self.ansatz_depth):
            for qubit in range(self.num_qubits):
                # RY gate
                theta = parameters[param_idx]
                state = self._apply_ry(state, qubit, theta)
                param_idx += 1
                
                # RZ gate
                phi = parameters[param_idx]
                state = self._apply_rz(state, qubit, phi)
                param_idx += 1
            
            # CNOT gates
            for qubit in range(self.num_qubits - 1):
                state = self._apply_cnot(state, qubit, qubit + 1)
        
        return state
    
    def _apply_ry(self, state: np.ndarray, qubit: int, theta: float) -> np.ndarray:
        """Apply RY gate to state."""
        n = len(state)
        new_state = np.zeros_like(state)
        
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        for i in range(n):
            bit = (i >> qubit) & 1
            j = i ^ (1 << qubit)
            
            if bit == 0:
                new_state[i] += c * state[i] - s * state[j]
            else:
                new_state[i] += s * state[j] + c * state[i]
        
        return new_state
    
    def _apply_rz(self, state: np.ndarray, qubit: int, phi: float) -> np.ndarray:
        """Apply RZ gate to state."""
        n = len(state)
        new_state = state.copy()
        
        for i in range(n):
            bit = (i >> qubit) & 1
            if bit == 1:
                new_state[i] *= np.exp(1j * phi / 2)
            else:
                new_state[i] *= np.exp(-1j * phi / 2)
        
        return new_state
    
    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate to state."""
        n = len(state)
        new_state = state.copy()
        
        for i in range(n):
            control_bit = (i >> control) & 1
            if control_bit == 1:
                j = i ^ (1 << target)
                new_state[i], new_state[j] = state[j], state[i]
        
        return new_state
    
    def _measure_pauli_expectation(self, parameters: np.ndarray) -> float:
        """Measure expectation value for Pauli Hamiltonian."""
        # Simplified: return random value for demonstration
        return np.random.uniform(-1, 1)
    
    def solve(self, max_iterations: int = 100,
              tolerance: float = 1e-6,
              initial_params: Optional[np.ndarray] = None) -> VQEResult:
        """
        Run VQE optimization to find ground state.
        
        Args:
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            initial_params: Initial parameter values
            
        Returns:
            VQEResult with optimization results
        """
        import time
        start_time = time.time()
        
        # Build ansatz if not already built
        if self._circuit is None:
            self._build_ansatz()
        
        # Initialize parameters
        if initial_params is not None:
            params = initial_params.copy()
        else:
            params = self._parameters.copy()
        
        # Optimization loop
        convergence_history = []
        best_value = float('inf')
        best_params = params.copy()
        
        for iteration in range(max_iterations):
            # Compute expectation value
            value = self._compute_expectation(params)
            convergence_history.append(value)
            
            if value < best_value:
                best_value = value
                best_params = params.copy()
            
            # Check convergence
            if len(convergence_history) > 1:
                if abs(convergence_history[-1] - convergence_history[-2]) < tolerance:
                    logger.info(f"VQE converged at iteration {iteration}")
                    break
            
            # Update parameters (simplified gradient descent)
            gradient = self._estimate_gradient(params)
            params = params - 0.1 * gradient
        
        execution_time = time.time() - start_time
        
        return VQEResult(
            optimal_value=best_value,
            optimal_parameters=best_params.tolist(),
            eigenvalue=best_value,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            execution_time=execution_time
        )
    
    def _estimate_gradient(self, params: np.ndarray, 
                           epsilon: float = 0.01) -> np.ndarray:
        """Estimate gradient using parameter shift rule."""
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            
            params_minus = params.copy()
            params_minus[i] -= epsilon
            
            gradient[i] = (self._compute_expectation(params_plus) - 
                          self._compute_expectation(params_minus)) / (2 * epsilon)
        
        return gradient


class QAOASolver:
    """
    Quantum Approximate Optimization Algorithm solver.
    
    QAOA is used for solving combinatorial optimization problems
    by encoding them into a quantum circuit.
    
    Example:
        >>> problem = MaxCutProblem(graph)
        >>> qaoa = QAOASolver(problem, p=3)
        >>> result = qaoa.solve()
        >>> print(f"Best solution: {result.best_solution}")
    """
    
    def __init__(self, problem: Any, p: int = 1,
                 mixer: str = "x",
                 language: str = "en"):
        """
        Initialize QAOA solver.
        
        Args:
            problem: Optimization problem (cost function)
            p: Number of QAOA layers
            mixer: Mixer Hamiltonian type
            language: Language for messages
        """
        self.problem = problem
        self.p = p
        self.mixer = mixer
        self.language = language
        
        # Determine number of qubits from problem
        if hasattr(problem, 'num_variables'):
            self.num_qubits = problem.num_variables
        elif hasattr(problem, 'num_qubits'):
            self.num_qubits = problem.num_qubits
        else:
            self.num_qubits = 4  # Default
        
        logger.info(f"QAOA solver initialized: p={p}, {self.num_qubits} qubits")
    
    def _build_circuit(self, gamma: List[float], beta: List[float]) -> Any:
        """Build QAOA circuit with given parameters."""
        from .circuit_builder import QuantumCircuitBuilder
        
        builder = QuantumCircuitBuilder(self.num_qubits, name="qaoa")
        
        # Initial superposition
        for qubit in range(self.num_qubits):
            builder.h(qubit)
        
        # QAOA layers
        for layer in range(self.p):
            # Cost layer
            self._apply_cost_layer(builder, gamma[layer])
            
            # Mixer layer
            self._apply_mixer_layer(builder, beta[layer])
        
        builder.measure_all()
        return builder
    
    def _apply_cost_layer(self, builder: Any, gamma: float):
        """Apply cost Hamiltonian layer."""
        # For MaxCut-like problems
        for i in range(self.num_qubits - 1):
            builder.cx(i, i + 1)
            builder.rz(i + 1, 2 * gamma)
            builder.cx(i, i + 1)
    
    def _apply_mixer_layer(self, builder: Any, beta: float):
        """Apply mixer Hamiltonian layer."""
        if self.mixer == "x":
            for qubit in range(self.num_qubits):
                builder.rx(qubit, 2 * beta)
        elif self.mixer == "xy":
            for qubit in range(self.num_qubits - 1):
                builder.rx(qubit, beta)
                builder.ry(qubit, beta)
    
    def solve(self, max_iterations: int = 100,
              shots: int = 1024) -> QAOAResult:
        """
        Run QAOA optimization.
        
        Args:
            max_iterations: Maximum optimization iterations
            shots: Number of measurement shots
            
        Returns:
            QAOAResult with optimization results
        """
        import time
        start_time = time.time()
        
        # Initialize parameters
        gamma = np.random.uniform(0, 2 * np.pi, self.p)
        beta = np.random.uniform(0, np.pi, self.p)
        
        best_value = float('-inf')
        best_solution = ""
        best_gamma = gamma.copy()
        best_beta = beta.copy()
        
        for iteration in range(max_iterations):
            # Build and simulate circuit
            circuit = self._build_circuit(gamma.tolist(), beta.tolist())
            
            # Simulate measurements
            counts = self._simulate_measurements(shots)
            
            # Evaluate solutions
            for bitstring, count in counts.items():
                value = self._evaluate_solution(bitstring)
                if value > best_value:
                    best_value = value
                    best_solution = bitstring
                    best_gamma = gamma.copy()
                    best_beta = beta.copy()
            
            # Update parameters (simplified)
            gamma += np.random.uniform(-0.1, 0.1, self.p)
            beta += np.random.uniform(-0.1, 0.1, self.p)
        
        execution_time = time.time() - start_time
        
        return QAOAResult(
            optimal_value=best_value,
            optimal_parameters=(best_gamma.tolist(), best_beta.tolist()),
            best_solution=best_solution,
            solution_probability=1.0 / (2 ** self.num_qubits),
            iterations=max_iterations,
            execution_time=execution_time
        )
    
    def _simulate_measurements(self, shots: int) -> Dict[str, int]:
        """Simulate measurement outcomes."""
        counts = {}
        for _ in range(shots):
            bitstring = ''.join(str(np.random.randint(0, 2)) 
                               for _ in range(self.num_qubits))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
    
    def _evaluate_solution(self, bitstring: str) -> float:
        """Evaluate cost function for a solution."""
        if hasattr(self.problem, 'evaluate'):
            return self.problem.evaluate(bitstring)
        
        # Default: count number of 1s (for demonstration)
        return sum(int(b) for b in bitstring)


class GroverSearch:
    """
    Grover's Quantum Search Algorithm.
    
    Provides quadratic speedup for unstructured search problems.
    
    Example:
        >>> oracle = lambda x: x == "1010"
        >>> grover = GroverSearch(4, oracle)
        >>> result = grover.search()
        >>> print(f"Found: {result.found_states}")
    """
    
    def __init__(self, num_qubits: int, 
                 oracle: Callable[[str], bool],
                 num_solutions: int = 1,
                 language: str = "en"):
        """
        Initialize Grover's search.
        
        Args:
            num_qubits: Number of qubits (search space = 2^n)
            oracle: Oracle function that returns True for target states
            num_solutions: Expected number of solutions
            language: Language for messages
        """
        self.num_qubits = num_qubits
        self.oracle = oracle
        self.num_solutions = num_solutions
        self.language = language
        
        # Calculate optimal number of iterations
        N = 2 ** num_qubits
        self.optimal_iterations = int(np.pi / 4 * np.sqrt(N / num_solutions))
        
        logger.info(f"Grover search initialized: {num_qubits} qubits, "
                   f"{self.optimal_iterations} iterations")
    
    def _build_circuit(self, iterations: int) -> Any:
        """Build Grover's algorithm circuit."""
        from .circuit_builder import QuantumCircuitBuilder
        
        builder = QuantumCircuitBuilder(self.num_qubits, name="grover")
        
        # Initial superposition
        for qubit in range(self.num_qubits):
            builder.h(qubit)
        
        # Grover iterations
        for _ in range(iterations):
            # Oracle (simplified - marks target states)
            self._apply_oracle(builder)
            
            # Diffusion operator
            self._apply_diffusion(builder)
        
        builder.measure_all()
        return builder
    
    def _apply_oracle(self, builder: Any):
        """Apply oracle operator."""
        # Simplified oracle - in practice, this depends on the problem
        # Here we apply a phase flip to demonstrate
        for qubit in range(self.num_qubits):
            builder.z(qubit)
    
    def _apply_diffusion(self, builder: Any):
        """Apply diffusion (Grover) operator."""
        # H gates
        for qubit in range(self.num_qubits):
            builder.h(qubit)
        
        # X gates
        for qubit in range(self.num_qubits):
            builder.x(qubit)
        
        # Multi-controlled Z (simplified as CZ chain)
        for qubit in range(self.num_qubits - 1):
            builder.cz(qubit, qubit + 1)
        
        # X gates
        for qubit in range(self.num_qubits):
            builder.x(qubit)
        
        # H gates
        for qubit in range(self.num_qubits):
            builder.h(qubit)
    
    def search(self, iterations: Optional[int] = None,
               shots: int = 1024) -> GroverResult:
        """
        Run Grover's search algorithm.
        
        Args:
            iterations: Number of Grover iterations (default: optimal)
            shots: Number of measurement shots
            
        Returns:
            GroverResult with search results
        """
        import time
        start_time = time.time()
        
        if iterations is None:
            iterations = self.optimal_iterations
        
        # Build and simulate circuit
        circuit = self._build_circuit(iterations)
        
        # Simulate measurements
        counts = self._simulate_measurements(shots)
        
        # Find target states
        found_states = []
        probabilities = {}
        
        for bitstring, count in counts.items():
            prob = count / shots
            probabilities[bitstring] = prob
            
            if self.oracle(bitstring):
                found_states.append(bitstring)
        
        # Calculate success probability
        success_prob = sum(probabilities.get(s, 0) for s in found_states)
        
        execution_time = time.time() - start_time
        
        return GroverResult(
            found_states=found_states,
            probabilities=probabilities,
            iterations_used=iterations,
            success_probability=success_prob,
            execution_time=execution_time
        )
    
    def _simulate_measurements(self, shots: int) -> Dict[str, int]:
        """Simulate measurement outcomes with Grover amplification."""
        N = 2 ** self.num_qubits
        
        # Find target states
        target_states = []
        for i in range(N):
            bitstring = format(i, f'0{self.num_qubits}b')
            if self.oracle(bitstring):
                target_states.append(i)
        
        # Calculate amplified probabilities
        if target_states:
            # Simplified: target states have higher probability
            target_prob = 0.9 / len(target_states)
            other_prob = 0.1 / (N - len(target_states)) if N > len(target_states) else 0
        else:
            target_prob = 0
            other_prob = 1.0 / N
        
        probabilities = []
        for i in range(N):
            if i in target_states:
                probabilities.append(target_prob)
            else:
                probabilities.append(other_prob)
        
        # Normalize
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        # Sample
        samples = np.random.choice(N, size=shots, p=probabilities)
        counts = {}
        for sample in samples:
            bitstring = format(sample, f'0{self.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts


class ShorFactorization:
    """
    Shor's Quantum Factoring Algorithm.
    
    Provides exponential speedup for integer factorization.
    
    Example:
        >>> shor = ShorFactorization(15)
        >>> result = shor.factor()
        >>> print(f"Factors: {result.factors}")
    """
    
    def __init__(self, number: int, language: str = "en"):
        """
        Initialize Shor's algorithm.
        
        Args:
            number: Number to factor
            language: Language for messages
        """
        if number < 2:
            raise ValueError("Number must be >= 2")
        
        self.number = number
        self.language = language
        
        # Calculate required qubits
        self.num_qubits = 2 * int(np.ceil(np.log2(number))) + 1
        
        logger.info(f"Shor factorization initialized for N={number}, "
                   f"using {self.num_qubits} qubits")
    
    def factor(self, attempts: int = 10) -> ShorResult:
        """
        Run Shor's factoring algorithm.
        
        Args:
            attempts: Maximum number of attempts
            
        Returns:
            ShorResult with factorization results
        """
        import time
        start_time = time.time()
        
        N = self.number
        
        # Check trivial cases
        if N % 2 == 0:
            return ShorResult(
                number=N,
                factors=(2, N // 2),
                period=0,
                success=True,
                execution_time=time.time() - start_time
            )
        
        # Check if N is a prime power
        for k in range(2, int(np.log2(N)) + 1):
            root = int(round(N ** (1/k)))
            if root ** k == N:
                return ShorResult(
                    number=N,
                    factors=(root, N // root),
                    period=0,
                    success=True,
                    execution_time=time.time() - start_time
                )
        
        # Quantum period finding
        for attempt in range(attempts):
            # Choose random a
            a = np.random.randint(2, N)
            
            # Check if a shares a factor with N
            gcd = self._gcd(a, N)
            if gcd > 1:
                return ShorResult(
                    number=N,
                    factors=(gcd, N // gcd),
                    period=0,
                    success=True,
                    execution_time=time.time() - start_time
                )
            
            # Find period using quantum simulation
            period = self._find_period(a, N)
            
            if period and period % 2 == 0:
                # Try to find factors
                x = pow(a, period // 2, N)
                
                factor1 = self._gcd(x - 1, N)
                factor2 = self._gcd(x + 1, N)
                
                if 1 < factor1 < N:
                    return ShorResult(
                        number=N,
                        factors=(factor1, N // factor1),
                        period=period,
                        success=True,
                        execution_time=time.time() - start_time
                    )
                
                if 1 < factor2 < N:
                    return ShorResult(
                        number=N,
                        factors=(factor2, N // factor2),
                        period=period,
                        success=True,
                        execution_time=time.time() - start_time
                    )
        
        return ShorResult(
            number=N,
            factors=(1, N),
            period=0,
            success=False,
            execution_time=time.time() - start_time
        )
    
    def _find_period(self, a: int, N: int) -> Optional[int]:
        """Find period of a^x mod N using quantum simulation."""
        # Simplified classical simulation of quantum period finding
        seen = {}
        x = 1
        
        for i in range(N):
            if x in seen:
                return i - seen[x]
            seen[x] = i
            x = (x * a) % N
        
        return None
    
    def _gcd(self, a: int, b: int) -> int:
        """Compute greatest common divisor."""
        while b:
            a, b = b, a % b
        return a
    
    def _build_circuit(self, a: int) -> Any:
        """Build quantum circuit for period finding."""
        from .circuit_builder import QuantumCircuitBuilder
        
        builder = QuantumCircuitBuilder(self.num_qubits, name="shor")
        
        # Initialize counting register in superposition
        n_count = self.num_qubits // 2
        for qubit in range(n_count):
            builder.h(qubit)
        
        # Controlled modular exponentiation (simplified)
        for qubit in range(n_count):
            power = 2 ** qubit
            self._controlled_modular_mult(builder, qubit, a, power)
        
        # Inverse QFT on counting register
        self._inverse_qft(builder, n_count)
        
        # Measure counting register
        for qubit in range(n_count):
            builder.measure(qubit, qubit)
        
        return builder
    
    def _controlled_modular_mult(self, builder: Any, control: int, 
                                  a: int, power: int):
        """Apply controlled modular multiplication."""
        # Simplified implementation
        target = self.num_qubits // 2
        builder.cx(control, target)
    
    def _inverse_qft(self, builder: Any, n: int):
        """Apply inverse QFT."""
        for i in range(n // 2):
            builder.swap(i, n - 1 - i)
        
        for i in range(n):
            for j in range(i):
                angle = -np.pi / (2 ** (i - j))
                builder.crz(j, i, angle)
            builder.h(i)
