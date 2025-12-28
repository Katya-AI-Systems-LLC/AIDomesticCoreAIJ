"""
Quantum circuit module for AIPlatform Quantum Infrastructure Zero SDK

This module provides quantum circuit building and manipulation capabilities
with integration to Qiskit for execution on quantum backends.
"""

from typing import Union, List, Optional
import numpy as np

try:
    from qiskit import QuantumCircuit as QiskitCircuit
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QiskitCircuit = object
    Parameter = object

from aiplatform.exceptions import QuantumCircuitError

class QuantumCircuit:
    """
    Quantum circuit representation and manipulation.
    
    This class provides a high-level interface for building and manipulating
    quantum circuits with integration to Qiskit for execution.
    """
    
    def __init__(self, num_qubits: int, num_clbits: int = 0):
        """
        Initialize quantum circuit.
        
        Args:
            num_qubits (int): Number of qubits
            num_clbits (int): Number of classical bits
        """
        if not QISKIT_AVAILABLE:
            raise QuantumCircuitError(
                "Qiskit not available. Please install qiskit to use quantum circuits."
            )
        
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
        self._circuit = QiskitCircuit(num_qubits, num_clbits)
        self._parameters = {}
    
    def h(self, qubit: int) -> 'QuantumCircuit':
        """
        Apply Hadamard gate to qubit.
        
        Args:
            qubit (int): Qubit index
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.h(qubit)
        return self
    
    def x(self, qubit: int) -> 'QuantumCircuit':
        """
        Apply Pauli-X gate to qubit.
        
        Args:
            qubit (int): Qubit index
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.x(qubit)
        return self
    
    def y(self, qubit: int) -> 'QuantumCircuit':
        """
        Apply Pauli-Y gate to qubit.
        
        Args:
            qubit (int): Qubit index
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.y(qubit)
        return self
    
    def z(self, qubit: int) -> 'QuantumCircuit':
        """
        Apply Pauli-Z gate to qubit.
        
        Args:
            qubit (int): Qubit index
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.z(qubit)
        return self
    
    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """
        Apply CNOT gate.
        
        Args:
            control (int): Control qubit index
            target (int): Target qubit index
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.cx(control, target)
        return self
    
    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """
        Apply CZ gate.
        
        Args:
            control (int): Control qubit index
            target (int): Target qubit index
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.cz(control, target)
        return self
    
    def ccx(self, control1: int, control2: int, target: int) -> 'QuantumCircuit':
        """
        Apply Toffoli gate (CCX).
        
        Args:
            control1 (int): First control qubit index
            control2 (int): Second control qubit index
            target (int): Target qubit index
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.ccx(control1, control2, target)
        return self
    
    def rx(self, qubit: int, theta: Union[float, str]) -> 'QuantumCircuit':
        """
        Apply RX rotation gate.
        
        Args:
            qubit (int): Qubit index
            theta (float or str): Rotation angle or parameter name
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        if isinstance(theta, str):
            if theta not in self._parameters:
                self._parameters[theta] = Parameter(theta)
            theta = self._parameters[theta]
        
        self._circuit.rx(theta, qubit)
        return self
    
    def ry(self, qubit: int, theta: Union[float, str]) -> 'QuantumCircuit':
        """
        Apply RY rotation gate.
        
        Args:
            qubit (int): Qubit index
            theta (float or str): Rotation angle or parameter name
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        if isinstance(theta, str):
            if theta not in self._parameters:
                self._parameters[theta] = Parameter(theta)
            theta = self._parameters[theta]
        
        self._circuit.ry(theta, qubit)
        return self
    
    def rz(self, qubit: int, theta: Union[float, str]) -> 'QuantumCircuit':
        """
        Apply RZ rotation gate.
        
        Args:
            qubit (int): Qubit index
            theta (float or str): Rotation angle or parameter name
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        if isinstance(theta, str):
            if theta not in self._parameters:
                self._parameters[theta] = Parameter(theta)
            theta = self._parameters[theta]
        
        self._circuit.rz(theta, qubit)
        return self
    
    def measure(self, qubit: int, clbit: int) -> 'QuantumCircuit':
        """
        Measure qubit to classical bit.
        
        Args:
            qubit (int): Qubit index
            clbit (int): Classical bit index
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.measure(qubit, clbit)
        return self
    
    def measure_all(self) -> 'QuantumCircuit':
        """
        Measure all qubits to classical bits.
        
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.measure_all()
        return self
    
    def append(self, gate, qubits: List[int]) -> 'QuantumCircuit':
        """
        Append gate to circuit.
        
        Args:
            gate: Quantum gate to append
            qubits (list): Qubit indices
            
        Returns:
            QuantumCircuit: Self for chaining
        """
        self._circuit.append(gate, qubits)
        return self
    
    def draw(self, output: str = "text") -> str:
        """
        Draw circuit diagram.
        
        Args:
            output (str): Output format ("text", "mpl", "latex")
            
        Returns:
            str: Circuit diagram
        """
        return self._circuit.draw(output).__str__()
    
    def to_qiskit(self) -> QiskitCircuit:
        """
        Get Qiskit circuit representation.
        
        Returns:
            qiskit.QuantumCircuit: Qiskit circuit
        """
        return self._circuit
    
    @property
    def num_qubits(self) -> int:
        """Get number of qubits."""
        return self._num_qubits
    
    @property
    def num_clbits(self) -> int:
        """Get number of classical bits."""
        return self._num_clbits
    
    @property
    def depth(self) -> int:
        """Get circuit depth."""
        return self._circuit.depth()
    
    def copy(self) -> 'QuantumCircuit':
        """
        Create copy of circuit.
        
        Returns:
            QuantumCircuit: Copy of circuit
        """
        new_circuit = QuantumCircuit(self._num_qubits, self._num_clbits)
        new_circuit._circuit = self._circuit.copy()
        new_circuit._parameters = self._parameters.copy()
        return new_circuit
    
    def inverse(self) -> 'QuantumCircuit':
        """
        Get inverse of circuit.
        
        Returns:
            QuantumCircuit: Inverse circuit
        """
        new_circuit = QuantumCircuit(self._num_qubits, self._num_clbits)
        new_circuit._circuit = self._circuit.inverse()
        return new_circuit
    
    def __len__(self) -> int:
        """Get circuit depth."""
        return self.depth()
    
    def __str__(self) -> str:
        """Get string representation."""
        return self.draw("text")
    
    def __repr__(self) -> str:
        """Get detailed representation."""
        return f"QuantumCircuit(num_qubits={self._num_qubits}, depth={self.depth()})"

# Utility functions
def bell_state(qubit1: int = 0, qubit2: int = 1, num_qubits: int = 2) -> QuantumCircuit:
    """
    Create Bell state circuit.
    
    Args:
        qubit1 (int): First qubit index
        qubit2 (int): Second qubit index
        num_qubits (int): Total number of qubits
        
    Returns:
        QuantumCircuit: Bell state circuit
    """
    circuit = QuantumCircuit(num_qubits)
    circuit.h(qubit1).cx(qubit1, qubit2)
    return circuit

def ghz_state(num_qubits: int) -> QuantumCircuit:
    """
    Create GHZ state circuit.
    
    Args:
        num_qubits (int): Number of qubits
        
    Returns:
        QuantumCircuit: GHZ state circuit
    """
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cx(0, i)
    return circuit

def quantum_fourier_transform(num_qubits: int) -> QuantumCircuit:
    """
    Create Quantum Fourier Transform circuit.
    
    Args:
        num_qubits (int): Number of qubits
        
    Returns:
        QuantumCircuit: QFT circuit
    """
    from qiskit.circuit.library import QFT
    qft = QFT(num_qubits)
    circuit = QuantumCircuit(num_qubits)
    circuit.append(qft, range(num_qubits))
    return circuit