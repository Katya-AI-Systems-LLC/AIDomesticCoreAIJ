"""
Quantum-AI Hybrid Example
=========================

Demonstrates combining quantum computing with AI for optimization.
"""

import asyncio
import numpy as np

# SDK imports
from sdk.quantum import QuantumCircuitBuilder, VQESolver, QAOASolver
from sdk.federated import HybridTrainer, FederatedCoordinator
from sdk.genai import UnifiedGenAI


async def quantum_optimization_example():
    """
    Example: Using VQE for molecular simulation.
    """
    print("=" * 60)
    print("Quantum Optimization with VQE")
    print("=" * 60)
    
    # Create VQE solver
    vqe = VQESolver(num_qubits=4, ansatz="ry_rz")
    
    # Define a simple Hamiltonian (H2 molecule approximation)
    hamiltonian = {
        "ZZ": [(0, 1, 0.5), (1, 2, 0.3)],
        "X": [(0, 0.2), (1, 0.2), (2, 0.2)],
        "Z": [(0, -0.5), (1, -0.3)]
    }
    
    # Run VQE
    print("\nRunning VQE optimization...")
    result = await vqe.solve(hamiltonian, max_iterations=50)
    
    print(f"Ground state energy: {result.energy:.6f}")
    print(f"Optimal parameters: {result.optimal_params[:5]}...")
    print(f"Iterations: {result.iterations}")
    print(f"Converged: {result.converged}")


async def qaoa_combinatorial_example():
    """
    Example: Using QAOA for combinatorial optimization.
    """
    print("\n" + "=" * 60)
    print("Combinatorial Optimization with QAOA")
    print("=" * 60)
    
    # Create QAOA solver
    qaoa = QAOASolver(num_qubits=5, p_layers=2)
    
    # Define MaxCut problem
    graph_edges = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 0, 1.0),
        (0, 2, 0.5)
    ]
    
    print("\nSolving MaxCut problem...")
    result = await qaoa.solve_maxcut(graph_edges, num_iterations=30)
    
    print(f"Best cut value: {result.objective_value:.4f}")
    print(f"Solution bitstring: {result.solution}")
    print(f"Iterations: {result.iterations}")


async def hybrid_quantum_ml_example():
    """
    Example: Hybrid quantum-classical machine learning.
    """
    print("\n" + "=" * 60)
    print("Hybrid Quantum-Classical ML Training")
    print("=" * 60)
    
    # Create hybrid trainer
    from sdk.federated.trainer import HybridModelConfig
    
    config = HybridModelConfig(
        quantum_qubits=4,
        quantum_layers=2,
        classical_layers=[32, 16],
        learning_rate=0.01
    )
    
    trainer = HybridTrainer(config)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 4).astype(np.float32)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.float32)
    
    print("\nTraining hybrid quantum-classical model...")
    result = await trainer.train(X_train, y_train, epochs=20, batch_size=16)
    
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Training time: {result.training_time:.2f}s")
    print(f"Quantum operations: {result.quantum_operations}")
    print(f"Classical operations: {result.classical_operations}")


async def federated_quantum_example():
    """
    Example: Federated learning with quantum nodes.
    """
    print("\n" + "=" * 60)
    print("Federated Quantum Learning")
    print("=" * 60)
    
    from sdk.federated import FederatedCoordinator, QuantumFederatedNode
    
    # Create coordinator
    coordinator = FederatedCoordinator(min_participants=2)
    
    # Create participant nodes
    nodes = []
    for i in range(3):
        node = QuantumFederatedNode(node_id=f"node_{i}")
        
        # Generate local data
        np.random.seed(i)
        data = np.random.randn(50, 10).astype(np.float32)
        labels = np.random.randn(50).astype(np.float32)
        node.set_training_data(data, labels)
        
        coordinator.register_participant(
            node.node_id,
            quantum_signature=node.quantum_signature
        )
        nodes.append(node)
    
    print(f"\nRegistered {len(nodes)} federated nodes")
    
    # Initialize global model
    initial_weights = np.random.randn(10).astype(np.float32)
    
    print("Starting federated training...")
    # In production, this would run full training
    # Here we demonstrate the setup
    
    print(f"Coordinator stats: {coordinator.get_statistics()}")


async def quantum_genai_integration():
    """
    Example: Quantum-enhanced GenAI.
    """
    print("\n" + "=" * 60)
    print("Quantum-Enhanced GenAI")
    print("=" * 60)
    
    # Create quantum circuit for feature extraction
    circuit = QuantumCircuitBuilder(num_qubits=4)
    
    # Build variational circuit
    for i in range(4):
        circuit.ry(i, np.pi / 4)
    
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    
    for i in range(4):
        circuit.rz(i, np.pi / 3)
    
    circuit.measure_all()
    
    print(f"Created quantum feature circuit with {circuit.num_qubits} qubits")
    print(f"Circuit depth: {circuit.depth}")
    
    # Use GenAI for interpretation
    genai = UnifiedGenAI()
    
    print("\nQuantum-AI integration ready for:")
    print("- Quantum feature extraction")
    print("- AI-driven circuit optimization")
    print("- Hybrid inference pipelines")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AIPlatform SDK - Quantum-AI Hybrid Examples")
    print("=" * 60)
    
    await quantum_optimization_example()
    await qaoa_combinatorial_example()
    await hybrid_quantum_ml_example()
    await federated_quantum_example()
    await quantum_genai_integration()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
