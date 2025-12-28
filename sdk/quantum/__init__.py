"""
Quantum Computing Module for AIPlatform SDK
============================================

Multi-backend quantum computing with support for:
- IBM Qiskit Runtime (Nighthawk, Heron, Eagle)
- AWS Braket (IonQ, Rigetti, IQM, QuEra)
- Google Quantum AI (Cirq, Sycamore)
- VQE, QAOA, Grover, Shor algorithms
- Quantum Safe Cryptography
- Hybrid quantum-classical computing
"""

from .circuit_builder import QuantumCircuitBuilder
from .qiskit_runtime import QiskitRuntimeClient
from .algorithms import VQESolver, QAOASolver, GroverSearch, ShorFactorization
from .simulator import QuantumSimulator
from .backends import IBMQuantumBackend, NighthawkBackend, HeronBackend
from .aws_braket import AWSBraketClient, BraketDevice
from .google_quantum import GoogleQuantumClient, GoogleProcessor

__all__ = [
    "QuantumCircuitBuilder",
    "QiskitRuntimeClient",
    "VQESolver",
    "QAOASolver",
    "GroverSearch",
    "ShorFactorization",
    "QuantumSimulator",
    "IBMQuantumBackend",
    "NighthawkBackend",
    "HeronBackend",
    # AWS Braket
    "AWSBraketClient",
    "BraketDevice",
    # Google Quantum
    "GoogleQuantumClient",
    "GoogleProcessor"
]
