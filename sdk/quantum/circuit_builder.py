"""
Quantum Circuit Builder Module
==============================

Provides high-level interface for building quantum circuits
with IBM Qiskit integration.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumGate:
    """Represents a quantum gate operation."""
    name: str
    qubits: List[int]
    parameters: List[float] = field(default_factory=list)
    classical_bits: List[int] = field(default_factory=list)


@dataclass
class CircuitMetrics:
    """Metrics for quantum circuit analysis."""
    depth: int
    gate_count: int
    qubit_count: int
    classical_bit_count: int
    two_qubit_gate_count: int
    single_qubit_gate_count: int


class QuantumCircuitBuilder:
    """
    High-level quantum circuit builder with IBM Qiskit integration.
    
    Supports:
    - Standard quantum gates (H, X, Y, Z, CNOT, etc.)
    - Parameterized gates (RX, RY, RZ, etc.)
    - Measurement operations
    - Circuit optimization
    - Export to Qiskit format
    
    Example:
        >>> builder = QuantumCircuitBuilder(4)
        >>> builder.h(0).cx(0, 1).measure_all()
        >>> circuit = builder.build()
    """
    
    def __init__(self, num_qubits: int, num_classical_bits: Optional[int] = None,
                 name: str = "quantum_circuit", language: str = "en"):
        """
        Initialize quantum circuit builder.
        
        Args:
            num_qubits: Number of qubits in the circuit
            num_classical_bits: Number of classical bits (defaults to num_qubits)
            name: Circuit name
            language: Language for messages (en, ru, zh, ar)
        """
        self.num_qubits = num_qubits
        self.num_classical_bits = num_classical_bits or num_qubits
        self.name = name
        self.language = language
        self.gates: List[QuantumGate] = []
        self._qiskit_circuit = None
        
        logger.info(f"Created quantum circuit '{name}' with {num_qubits} qubits")
    
    # Single-qubit gates
    def h(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Apply Hadamard gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("h", [qubit]))
        return self
    
    def x(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Apply Pauli-X (NOT) gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("x", [qubit]))
        return self
    
    def y(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Apply Pauli-Y gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("y", [qubit]))
        return self
    
    def z(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Apply Pauli-Z gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("z", [qubit]))
        return self
    
    def s(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Apply S (phase) gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("s", [qubit]))
        return self
    
    def t(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Apply T gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("t", [qubit]))
        return self
    
    def sdg(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Apply S-dagger gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("sdg", [qubit]))
        return self
    
    def tdg(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Apply T-dagger gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("tdg", [qubit]))
        return self
    
    # Rotation gates
    def rx(self, qubit: int, theta: float) -> 'QuantumCircuitBuilder':
        """Apply rotation around X-axis."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("rx", [qubit], [theta]))
        return self
    
    def ry(self, qubit: int, theta: float) -> 'QuantumCircuitBuilder':
        """Apply rotation around Y-axis."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("ry", [qubit], [theta]))
        return self
    
    def rz(self, qubit: int, theta: float) -> 'QuantumCircuitBuilder':
        """Apply rotation around Z-axis."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("rz", [qubit], [theta]))
        return self
    
    def u(self, qubit: int, theta: float, phi: float, lam: float) -> 'QuantumCircuitBuilder':
        """Apply general unitary gate U(θ, φ, λ)."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("u", [qubit], [theta, phi, lam]))
        return self
    
    # Two-qubit gates
    def cx(self, control: int, target: int) -> 'QuantumCircuitBuilder':
        """Apply CNOT (controlled-X) gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        self.gates.append(QuantumGate("cx", [control, target]))
        return self
    
    def cy(self, control: int, target: int) -> 'QuantumCircuitBuilder':
        """Apply controlled-Y gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.gates.append(QuantumGate("cy", [control, target]))
        return self
    
    def cz(self, control: int, target: int) -> 'QuantumCircuitBuilder':
        """Apply controlled-Z gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.gates.append(QuantumGate("cz", [control, target]))
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuitBuilder':
        """Apply SWAP gate."""
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.gates.append(QuantumGate("swap", [qubit1, qubit2]))
        return self
    
    def crx(self, control: int, target: int, theta: float) -> 'QuantumCircuitBuilder':
        """Apply controlled rotation around X-axis."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.gates.append(QuantumGate("crx", [control, target], [theta]))
        return self
    
    def cry(self, control: int, target: int, theta: float) -> 'QuantumCircuitBuilder':
        """Apply controlled rotation around Y-axis."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.gates.append(QuantumGate("cry", [control, target], [theta]))
        return self
    
    def crz(self, control: int, target: int, theta: float) -> 'QuantumCircuitBuilder':
        """Apply controlled rotation around Z-axis."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.gates.append(QuantumGate("crz", [control, target], [theta]))
        return self
    
    # Three-qubit gates
    def ccx(self, control1: int, control2: int, target: int) -> 'QuantumCircuitBuilder':
        """Apply Toffoli (CCX) gate."""
        self._validate_qubit(control1)
        self._validate_qubit(control2)
        self._validate_qubit(target)
        self.gates.append(QuantumGate("ccx", [control1, control2, target]))
        return self
    
    def cswap(self, control: int, target1: int, target2: int) -> 'QuantumCircuitBuilder':
        """Apply Fredkin (CSWAP) gate."""
        self._validate_qubit(control)
        self._validate_qubit(target1)
        self._validate_qubit(target2)
        self.gates.append(QuantumGate("cswap", [control, target1, target2]))
        return self
    
    # Measurement
    def measure(self, qubit: int, classical_bit: int) -> 'QuantumCircuitBuilder':
        """Measure a qubit into a classical bit."""
        self._validate_qubit(qubit)
        self._validate_classical_bit(classical_bit)
        self.gates.append(QuantumGate("measure", [qubit], classical_bits=[classical_bit]))
        return self
    
    def measure_all(self) -> 'QuantumCircuitBuilder':
        """Measure all qubits."""
        for i in range(self.num_qubits):
            self.measure(i, i)
        return self
    
    # Barrier
    def barrier(self, qubits: Optional[List[int]] = None) -> 'QuantumCircuitBuilder':
        """Add barrier for visualization and optimization."""
        if qubits is None:
            qubits = list(range(self.num_qubits))
        for q in qubits:
            self._validate_qubit(q)
        self.gates.append(QuantumGate("barrier", qubits))
        return self
    
    # Reset
    def reset(self, qubit: int) -> 'QuantumCircuitBuilder':
        """Reset qubit to |0⟩ state."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("reset", [qubit]))
        return self
    
    # Circuit operations
    def inverse(self) -> 'QuantumCircuitBuilder':
        """Return inverse of the circuit."""
        new_builder = QuantumCircuitBuilder(
            self.num_qubits, 
            self.num_classical_bits,
            f"{self.name}_inverse",
            self.language
        )
        
        # Reverse gates and apply inverse
        for gate in reversed(self.gates):
            if gate.name in ["h", "x", "y", "z", "cx", "cy", "cz", "swap", "ccx", "cswap"]:
                new_builder.gates.append(gate)
            elif gate.name in ["s", "t"]:
                new_builder.gates.append(QuantumGate(f"{gate.name}dg", gate.qubits))
            elif gate.name in ["sdg", "tdg"]:
                new_builder.gates.append(QuantumGate(gate.name[:-2], gate.qubits))
            elif gate.name in ["rx", "ry", "rz"]:
                new_builder.gates.append(QuantumGate(gate.name, gate.qubits, [-p for p in gate.parameters]))
            elif gate.name == "u":
                theta, phi, lam = gate.parameters
                new_builder.gates.append(QuantumGate("u", gate.qubits, [-theta, -lam, -phi]))
        
        return new_builder
    
    def compose(self, other: 'QuantumCircuitBuilder') -> 'QuantumCircuitBuilder':
        """Compose with another circuit."""
        if other.num_qubits != self.num_qubits:
            raise ValueError("Circuits must have the same number of qubits")
        
        new_builder = QuantumCircuitBuilder(
            self.num_qubits,
            max(self.num_classical_bits, other.num_classical_bits),
            f"{self.name}_{other.name}",
            self.language
        )
        new_builder.gates = self.gates.copy() + other.gates.copy()
        return new_builder
    
    # Build and export
    def build(self) -> Any:
        """Build and return Qiskit QuantumCircuit."""
        try:
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
            
            qr = QuantumRegister(self.num_qubits, 'q')
            cr = ClassicalRegister(self.num_classical_bits, 'c')
            circuit = QuantumCircuit(qr, cr, name=self.name)
            
            for gate in self.gates:
                self._apply_gate_to_qiskit(circuit, gate, qr, cr)
            
            self._qiskit_circuit = circuit
            return circuit
            
        except ImportError:
            logger.warning("Qiskit not installed, returning gate list")
            return self.gates
    
    def _apply_gate_to_qiskit(self, circuit, gate: QuantumGate, qr, cr):
        """Apply gate to Qiskit circuit."""
        name = gate.name
        qubits = [qr[q] for q in gate.qubits]
        params = gate.parameters
        
        if name == "h":
            circuit.h(qubits[0])
        elif name == "x":
            circuit.x(qubits[0])
        elif name == "y":
            circuit.y(qubits[0])
        elif name == "z":
            circuit.z(qubits[0])
        elif name == "s":
            circuit.s(qubits[0])
        elif name == "t":
            circuit.t(qubits[0])
        elif name == "sdg":
            circuit.sdg(qubits[0])
        elif name == "tdg":
            circuit.tdg(qubits[0])
        elif name == "rx":
            circuit.rx(params[0], qubits[0])
        elif name == "ry":
            circuit.ry(params[0], qubits[0])
        elif name == "rz":
            circuit.rz(params[0], qubits[0])
        elif name == "u":
            circuit.u(params[0], params[1], params[2], qubits[0])
        elif name == "cx":
            circuit.cx(qubits[0], qubits[1])
        elif name == "cy":
            circuit.cy(qubits[0], qubits[1])
        elif name == "cz":
            circuit.cz(qubits[0], qubits[1])
        elif name == "swap":
            circuit.swap(qubits[0], qubits[1])
        elif name == "crx":
            circuit.crx(params[0], qubits[0], qubits[1])
        elif name == "cry":
            circuit.cry(params[0], qubits[0], qubits[1])
        elif name == "crz":
            circuit.crz(params[0], qubits[0], qubits[1])
        elif name == "ccx":
            circuit.ccx(qubits[0], qubits[1], qubits[2])
        elif name == "cswap":
            circuit.cswap(qubits[0], qubits[1], qubits[2])
        elif name == "measure":
            circuit.measure(qr[gate.qubits[0]], cr[gate.classical_bits[0]])
        elif name == "barrier":
            circuit.barrier(*qubits)
        elif name == "reset":
            circuit.reset(qubits[0])
    
    def to_qasm(self) -> str:
        """Export circuit to OpenQASM 2.0 format."""
        circuit = self.build()
        if hasattr(circuit, 'qasm'):
            return circuit.qasm()
        
        # Manual QASM generation if Qiskit not available
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self.num_qubits}];",
            f"creg c[{self.num_classical_bits}];"
        ]
        
        for gate in self.gates:
            lines.append(self._gate_to_qasm(gate))
        
        return "\n".join(lines)
    
    def _gate_to_qasm(self, gate: QuantumGate) -> str:
        """Convert gate to QASM instruction."""
        name = gate.name
        qubits = ",".join([f"q[{q}]" for q in gate.qubits])
        
        if gate.parameters:
            params = ",".join([str(p) for p in gate.parameters])
            return f"{name}({params}) {qubits};"
        elif name == "measure":
            return f"measure q[{gate.qubits[0]}] -> c[{gate.classical_bits[0]}];"
        else:
            return f"{name} {qubits};"
    
    def get_metrics(self) -> CircuitMetrics:
        """Get circuit metrics."""
        single_qubit = 0
        two_qubit = 0
        
        for gate in self.gates:
            if gate.name in ["measure", "barrier", "reset"]:
                continue
            if len(gate.qubits) == 1:
                single_qubit += 1
            else:
                two_qubit += 1
        
        return CircuitMetrics(
            depth=self._calculate_depth(),
            gate_count=len([g for g in self.gates if g.name not in ["barrier"]]),
            qubit_count=self.num_qubits,
            classical_bit_count=self.num_classical_bits,
            two_qubit_gate_count=two_qubit,
            single_qubit_gate_count=single_qubit
        )
    
    def _calculate_depth(self) -> int:
        """Calculate circuit depth."""
        qubit_depths = [0] * self.num_qubits
        
        for gate in self.gates:
            if gate.name in ["barrier"]:
                continue
            
            max_depth = max(qubit_depths[q] for q in gate.qubits)
            for q in gate.qubits:
                qubit_depths[q] = max_depth + 1
        
        return max(qubit_depths) if qubit_depths else 0
    
    def _validate_qubit(self, qubit: int):
        """Validate qubit index."""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits})")
    
    def _validate_classical_bit(self, bit: int):
        """Validate classical bit index."""
        if not 0 <= bit < self.num_classical_bits:
            raise ValueError(f"Classical bit index {bit} out of range [0, {self.num_classical_bits})")
    
    def __repr__(self) -> str:
        metrics = self.get_metrics()
        return (f"QuantumCircuitBuilder(name='{self.name}', "
                f"qubits={self.num_qubits}, depth={metrics.depth}, "
                f"gates={metrics.gate_count})")


# Convenience functions
def create_bell_state() -> QuantumCircuitBuilder:
    """Create Bell state circuit."""
    return QuantumCircuitBuilder(2, name="bell_state").h(0).cx(0, 1).measure_all()


def create_ghz_state(n: int) -> QuantumCircuitBuilder:
    """Create GHZ state circuit for n qubits."""
    builder = QuantumCircuitBuilder(n, name=f"ghz_{n}")
    builder.h(0)
    for i in range(n - 1):
        builder.cx(i, i + 1)
    return builder.measure_all()


def create_qft(n: int) -> QuantumCircuitBuilder:
    """Create Quantum Fourier Transform circuit."""
    builder = QuantumCircuitBuilder(n, name=f"qft_{n}")
    
    for i in range(n):
        builder.h(i)
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            builder.crz(j, i, angle)
    
    # Swap qubits
    for i in range(n // 2):
        builder.swap(i, n - 1 - i)
    
    return builder
