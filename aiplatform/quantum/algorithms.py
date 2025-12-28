"""
Quantum algorithms module for AIPlatform Quantum Infrastructure Zero SDK

This module provides implementations of key quantum algorithms including:
- Variational Quantum Eigensolver (VQE)
- Quantum Approximate Optimization Algorithm (QAOA)
- Grover's search algorithm
- Shor's factoring algorithm
"""

from typing import Union, List, Optional, Callable
import numpy as np
from scipy.optimize import minimize

try:
    from qiskit.opflow import PauliSumOp
    from qiskit.algorithms import VQE as QiskitVQE
    from qiskit.algorithms.optimizers import Optimizer, SPSA, COBYLA
    from qiskit.circuit import Parameter, QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    PauliSumOp = object
    QiskitVQE = object
    Optimizer = object
    SPSA = object
    COBYLA = object
    Parameter = object
    QuantumCircuit = object

from aiplatform.exceptions import QuantumAlgorithmError
from aiplatform.quantum.circuit import QuantumCircuit as AIQuantumCircuit

class VQE:
    """
    Variational Quantum Eigensolver implementation.
    
    Finds the ground state energy of a Hamiltonian using variational methods.
    """
    
    def __init__(self, hamiltonian, ansatz=None, optimizer=None):
        """
        Initialize VQE.
        
        Args:
            hamiltonian: Hamiltonian operator
            ansatz: Ansatz circuit (trial wavefunction)
            optimizer: Classical optimizer for parameter optimization
        """
        if not QISKIT_AVAILABLE:
            raise QuantumAlgorithmError(
                "Qiskit not available. Please install qiskit to use VQE."
            )
        
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz or self._default_ansatz()
        self._optimizer = optimizer or COBYLA(maxiter=1000)
        self._optimal_parameters = None
        self._ground_state_energy = None
        self._evaluation_count = 0
    
    def _default_ansatz(self):
        """Create default ansatz circuit."""
        # Simple hardware-efficient ansatz
        num_qubits = self._hamiltonian.num_qubits if hasattr(self._hamiltonian, 'num_qubits') else 2
        ansatz = QuantumCircuit(num_qubits)
        
        # Initial layer of Hadamard gates
        for i in range(num_qubits):
            ansatz.h(i)
        
        # Parameterized layers
        params = [Parameter(f'theta_{i}') for i in range(num_qubits * 2)]
        param_idx = 0
        
        for layer in range(2):
            # Rotation gates
            for i in range(num_qubits):
                ansatz.ry(params[param_idx], i)
                param_idx += 1
            
            # Entangling gates
            for i in range(num_qubits - 1):
                ansatz.cx(i, i + 1)
        
        return ansatz
    
    def optimize(self, initial_params: Optional[List[float]] = None, 
                max_iterations: int = 1000) -> float:
        """
        Optimize to find ground state energy.
        
        Args:
            initial_params (list, optional): Initial parameters
            max_iterations (int): Maximum optimization iterations
            
        Returns:
            float: Ground state energy
        """
        if not QISKIT_AVAILABLE:
            raise QuantumAlgorithmError("Qiskit not available.")
        
        try:
            # Set up Qiskit VQE
            qiskit_vqe = QiskitVQE(
                ansatz=self._ansatz,
                optimizer=self._optimizer,
                max_evals_grouped=1
            )
            
            # Run VQE
            result = qiskit_vqe.compute_minimum_eigenvalue(
                operator=self._hamiltonian
            )
            
            # Store results
            self._optimal_parameters = result.optimal_point
            self._ground_state_energy = result.eigenvalue.real
            
            return float(self._ground_state_energy)
            
        except Exception as e:
            raise QuantumAlgorithmError(f"VQE optimization failed: {e}")
    
    def get_ground_state(self) -> Optional[float]:
        """
        Get computed ground state energy.
        
        Returns:
            float: Ground state energy, or None if not computed
        """
        return self._ground_state_energy
    
    def get_optimal_parameters(self) -> Optional[np.ndarray]:
        """
        Get optimal parameters.
        
        Returns:
            numpy.ndarray: Optimal parameters, or None if not computed
        """
        return self._optimal_parameters
    
    @property
    def evaluation_count(self) -> int:
        """Get number of evaluations performed."""
        return self._evaluation_count

class QAOA:
    """
    Quantum Approximate Optimization Algorithm implementation.
    
    Solves combinatorial optimization problems using quantum-classical hybrid approach.
    """
    
    def __init__(self, problem, p: int, mixer=None):
        """
        Initialize QAOA.
        
        Args:
            problem: Optimization problem (cost Hamiltonian)
            p (int): Number of QAOA layers
            mixer: Mixer Hamiltonian
        """
        if not QISKIT_AVAILABLE:
            raise QuantumAlgorithmError(
                "Qiskit not available. Please install qiskit to use QAOA."
            )
        
        self._problem = problem
        self._p = p
        self._mixer = mixer
        self._optimal_parameters = None
        self._optimal_solution = None
        self._solution_value = None
    
    def optimize(self, initial_params: Optional[List[float]] = None) -> List[float]:
        """
        Optimize QAOA parameters.
        
        Args:
            initial_params (list, optional): Initial parameters [betas, gammas]
            
        Returns:
            list: Optimal solution bitstring
        """
        if not QISKIT_AVAILABLE:
            raise QuantumAlgorithmError("Qiskit not available.")
        
        try:
            from qiskit.algorithms import QAOA as QiskitQAOA
            
            # Set up Qiskit QAOA
            qaoa = QiskitQAOA(
                reps=self._p,
                optimizer=COBYLA(maxiter=1000),
                measurement=False
            )
            
            # Run QAOA
            result = qaoa.compute_minimum_eigenvalue(
                operator=self._problem
            )
            
            # Extract solution
            if hasattr(result, 'eigenstate') and result.eigenstate is not None:
                # Get most probable bitstring
                if hasattr(result.eigenstate, 'to_dict'):
                    eigenstate_dict = result.eigenstate.to_dict()
                    if eigenstate_dict:
                        # Get bitstring with highest probability
                        optimal_bitstring = max(eigenstate_dict.items(), 
                                              key=lambda x: abs(x[1])**2)
                        self._optimal_solution = [int(bit) for bit in optimal_bitstring[0]]
                        self._solution_value = result.eigenvalue.real
            
            self._optimal_parameters = result.optimal_point
            
            return self._optimal_solution or []
            
        except Exception as e:
            raise QuantumAlgorithmError(f"QAOA optimization failed: {e}")
    
    def get_solution(self) -> Optional[List[int]]:
        """
        Get optimal solution.
        
        Returns:
            list: Optimal solution bitstring, or None if not computed
        """
        return self._optimal_solution
    
    def get_solution_value(self) -> Optional[float]:
        """
        Get solution value.
        
        Returns:
            float: Solution value, or None if not computed
        """
        return self._solution_value

class Grover:
    """
    Grover's search algorithm implementation.
    
    Provides quadratic speedup for unstructured search problems.
    """
    
    def __init__(self, oracle, num_qubits: int):
        """
        Initialize Grover's algorithm.
        
        Args:
            oracle: Oracle function that marks solutions
            num_qubits (int): Number of qubits
        """
        if not QISKIT_AVAILABLE:
            raise QuantumAlgorithmError(
                "Qiskit not available. Please install qiskit to use Grover."
            )
        
        self._oracle = oracle
        self._num_qubits = num_qubits
        self._solution = None
        self._iterations = None
    
    def search(self, iterations: Optional[int] = None) -> List[int]:
        """
        Perform Grover search.
        
        Args:
            iterations (int, optional): Number of Grover iterations
            
        Returns:
            list: Solution bitstring
        """
        if not QISKIT_AVAILABLE:
            raise QuantumAlgorithmError("Qiskit not available.")
        
        try:
            from qiskit.algorithms import Grover as QiskitGrover
            from qiskit.algorithms.amplitude_amplifiers import GroverOperator
            
            # Estimate optimal iterations if not provided
            if iterations is None:
                # For a single solution in N items, optimal iterations ≈ π/4 * √N
                N = 2 ** self._num_qubits
                iterations = int(np.pi / 4 * np.sqrt(N))
                self._iterations = iterations
            
            # Create Grover operator
            grover_op = GroverOperator(self._oracle)
            
            # Set up Qiskit Grover
            grover = QiskitGrover(
                iterations=iterations,
                grover_operator=grover_op
            )
            
            # Run Grover search
            result = grover.amplify()
            
            # Extract solution
            if hasattr(result, 'top_measurement') and result.top_measurement:
                self._solution = [int(bit) for bit in result.top_measurement]
            
            return self._solution or []
            
        except Exception as e:
            raise QuantumAlgorithmError(f"Grover search failed: {e}")
    
    def get_solution(self) -> Optional[List[int]]:
        """
        Get search solution.
        
        Returns:
            list: Solution bitstring, or None if not computed
        """
        return self._solution

class Shor:
    """
    Shor's factoring algorithm implementation.
    
    Finds prime factors of composite numbers with exponential speedup.
    """
    
    def __init__(self, N: int):
        """
        Initialize Shor's algorithm.
        
        Args:
            N (int): Number to factorize
        """
        if not QISKIT_AVAILABLE:
            raise QuantumAlgorithmError(
                "Qiskit not available. Please install qiskit to use Shor."
            )
        
        if N <= 1:
            raise QuantumAlgorithmError("N must be greater than 1")
        
        if self._is_prime(N):
            raise QuantumAlgorithmError(f"{N} is prime and cannot be factorized")
        
        self._N = N
        self._factors = None
    
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def factorize(self) -> List[int]:
        """
        Factorize the number using Shor's algorithm.
        
        Returns:
            list: Prime factors
        """
        if not QISKIT_AVAILABLE:
            raise QuantumAlgorithmError("Qiskit not available.")
        
        try:
            from qiskit.algorithms import Shor as QiskitShor
            
            # Set up Qiskit Shor
            shor = QiskitShor()
            
            # Run factorization
            result = shor.factor(self._N)
            
            # Extract factors
            if hasattr(result, 'factors') and result.factors:
                self._factors = list(result.factors)
            
            return self._factors or []
            
        except Exception as e:
            # Fallback to classical factorization for small numbers
            if self._N < 1000:
                return self._classical_factorize()
            raise QuantumAlgorithmError(f"Shor's algorithm failed: {e}")
    
    def _classical_factorize(self) -> List[int]:
        """Classical factorization fallback."""
        factors = []
        n = self._N
        
        # Check for factor 2
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        
        # Check for odd factors
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 2
        
        # If n is still greater than 1, then it's prime
        if n > 1:
            factors.append(n)
        
        return factors
    
    def get_factors(self) -> Optional[List[int]]:
        """
        Get computed factors.
        
        Returns:
            list: Prime factors, or None if not computed
        """
        return self._factors

# Utility functions for common problems
def max_cut_problem(graph_edges: List[tuple], num_nodes: int) -> PauliSumOp:
    """
    Create Max-Cut problem Hamiltonian.
    
    Args:
        graph_edges (list): List of (node1, node2) tuples
        num_nodes (int): Number of nodes
        
    Returns:
        PauliSumOp: Max-Cut Hamiltonian
    """
    if not QISKIT_AVAILABLE:
        raise QuantumAlgorithmError("Qiskit not available.")
    
    from qiskit.opflow import PauliSumOp, I, Z
    
    # Create Hamiltonian for Max-Cut
    hamiltonian = PauliSumOp.from_list([("I" * num_nodes, 0.0)])
    
    for (i, j) in graph_edges:
        # Add term: 0.5 * (I - Z_i * Z_j)
        pauli_string = ["I"] * num_nodes
        pauli_string[i] = "Z"
        pauli_string[j] = "Z"
        hamiltonian -= 0.5 * PauliSumOp.from_list([("".join(pauli_string), 1.0)])
        hamiltonian += 0.5 * PauliSumOp.from_list([("I" * num_nodes, 1.0)])
    
    return hamiltonian

def ising_model_problem(j_matrix: np.ndarray, h_vector: np.ndarray) -> PauliSumOp:
    """
    Create Ising model problem Hamiltonian.
    
    Args:
        j_matrix (numpy.ndarray): Coupling matrix
        h_vector (numpy.ndarray): Magnetic field vector
        
    Returns:
        PauliSumOp: Ising Hamiltonian
    """
    if not QISKIT_AVAILABLE:
        raise QuantumAlgorithmError("Qiskit not available.")
    
    from qiskit.opflow import PauliSumOp, I, Z
    
    num_qubits = len(h_vector)
    hamiltonian = PauliSumOp.from_list([("I" * num_qubits, 0.0)])
    
    # Add magnetic field terms
    for i in range(num_qubits):
        if abs(h_vector[i]) > 1e-10:
            pauli_string = ["I"] * num_qubits
            pauli_string[i] = "Z"
            hamiltonian -= h_vector[i] * PauliSumOp.from_list([("".join(pauli_string), 1.0)])
    
    # Add coupling terms
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if abs(j_matrix[i, j]) > 1e-10:
                pauli_string = ["I"] * num_qubits
                pauli_string[i] = "Z"
                pauli_string[j] = "Z"
                hamiltonian -= j_matrix[i, j] * PauliSumOp.from_list([("".join(pauli_string), 1.0)])
    
    return hamiltonian