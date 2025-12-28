"""
Quantum Module Tests
====================

Tests for quantum computing functionality.
"""

import pytest
import numpy as np
import asyncio

from sdk.quantum import (
    QuantumCircuitBuilder,
    QuantumSimulator,
    VQESolver,
    QAOASolver,
    GroverSearch
)


class TestQuantumCircuitBuilder:
    """Tests for QuantumCircuitBuilder."""
    
    def test_initialization(self):
        """Test circuit initialization."""
        circuit = QuantumCircuitBuilder(num_qubits=4)
        assert circuit.num_qubits == 4
        assert circuit.num_classical_bits == 0
    
    def test_single_qubit_gates(self):
        """Test single qubit gates."""
        circuit = QuantumCircuitBuilder(num_qubits=2)
        
        circuit.h(0)
        circuit.x(1)
        circuit.y(0)
        circuit.z(1)
        circuit.s(0)
        circuit.t(1)
        
        assert circuit.depth > 0
    
    def test_rotation_gates(self):
        """Test rotation gates."""
        circuit = QuantumCircuitBuilder(num_qubits=2)
        
        circuit.rx(0, np.pi / 4)
        circuit.ry(1, np.pi / 2)
        circuit.rz(0, np.pi)
        
        assert circuit.depth > 0
    
    def test_two_qubit_gates(self):
        """Test two-qubit gates."""
        circuit = QuantumCircuitBuilder(num_qubits=3)
        
        circuit.cx(0, 1)
        circuit.cz(1, 2)
        circuit.swap(0, 2)
        
        assert circuit.depth > 0
    
    def test_measurement(self):
        """Test measurement operations."""
        circuit = QuantumCircuitBuilder(num_qubits=2)
        
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        assert circuit.num_classical_bits == 2
    
    def test_bell_state(self):
        """Test Bell state creation."""
        circuit = QuantumCircuitBuilder(num_qubits=2)
        circuit.bell_state(0, 1)
        
        assert circuit.depth >= 2
    
    def test_ghz_state(self):
        """Test GHZ state creation."""
        circuit = QuantumCircuitBuilder(num_qubits=4)
        circuit.ghz_state([0, 1, 2, 3])
        
        assert circuit.depth >= 4


class TestQuantumSimulator:
    """Tests for QuantumSimulator."""
    
    def test_initialization(self):
        """Test simulator initialization."""
        sim = QuantumSimulator(num_qubits=3)
        assert sim.num_qubits == 3
    
    def test_hadamard_gate(self):
        """Test Hadamard gate simulation."""
        sim = QuantumSimulator(num_qubits=1)
        sim.h(0)
        
        state = sim.get_statevector()
        # |+⟩ state should have equal amplitudes
        assert np.allclose(np.abs(state[0]), np.abs(state[1]), atol=0.01)
    
    def test_cnot_gate(self):
        """Test CNOT gate simulation."""
        sim = QuantumSimulator(num_qubits=2)
        sim.x(0)  # |10⟩
        sim.cx(0, 1)  # |11⟩
        
        state = sim.get_statevector()
        assert np.abs(state[3]) > 0.99  # |11⟩ = index 3
    
    def test_measurement(self):
        """Test measurement."""
        sim = QuantumSimulator(num_qubits=2)
        sim.x(0)
        
        results = sim.measure(shots=100)
        
        assert len(results) > 0
        assert "10" in results or "01" in results  # Depending on bit ordering


class TestVQESolver:
    """Tests for VQE solver."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test VQE initialization."""
        vqe = VQESolver(num_qubits=4, ansatz="ry_rz")
        assert vqe.num_qubits == 4
    
    @pytest.mark.asyncio
    async def test_simple_hamiltonian(self):
        """Test VQE with simple Hamiltonian."""
        vqe = VQESolver(num_qubits=2)
        
        hamiltonian = {
            "ZZ": [(0, 1, 1.0)],
            "X": [(0, 0.5), (1, 0.5)]
        }
        
        result = await vqe.solve(hamiltonian, max_iterations=10)
        
        assert result.energy is not None
        assert result.iterations > 0


class TestQAOASolver:
    """Tests for QAOA solver."""
    
    @pytest.mark.asyncio
    async def test_maxcut(self):
        """Test QAOA for MaxCut."""
        qaoa = QAOASolver(num_qubits=4, p_layers=1)
        
        edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 0, 1.0)
        ]
        
        result = await qaoa.solve_maxcut(edges, num_iterations=5)
        
        assert result.solution is not None
        assert len(result.solution) == 4


class TestGroverSearch:
    """Tests for Grover's search."""
    
    @pytest.mark.asyncio
    async def test_search(self):
        """Test Grover's search."""
        grover = GroverSearch(num_qubits=3)
        
        target = "101"
        result = await grover.search(target)
        
        assert result.found
        assert result.solution == target


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
