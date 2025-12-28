"""
Quantum Computing Example for AIPlatform SDK

This example demonstrates quantum computing capabilities with multilingual support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiplatform.quantum import (
    create_quantum_circuit,
    create_vqe_solver,
    create_qaoa_solver,
    create_grover_search,
    create_shor_algorithm,
    create_quantum_safe_crypto
)
import numpy as np


def quantum_circuit_example(language='en'):
    """Demonstrate quantum circuit creation and execution."""
    print(f"=== {translate('quantum_circuit_example', language) or 'Quantum Circuit Example'} ===")
    
    # Create a 3-qubit circuit
    circuit = create_quantum_circuit(3, language=language)
    
    # Add some gates
    circuit.h(0)  # Hadamard on qubit 0
    circuit.cx(0, 1)  # CNOT from qubit 0 to 1
    circuit.x(2)  # Pauli-X on qubit 2
    circuit.measure_all()  # Measure all qubits
    
    # Draw the circuit
    print(circuit.draw())
    
    # Execute the circuit
    results = circuit.execute(backend='simulator', shots=1000)
    print(f"Results: {results}")
    print()


def vqe_example(language='en'):
    """Demonstrate VQE solver."""
    print(f"=== {translate('vqe_example', language) or 'VQE Example'} ===")
    
    # Create a simple 2x2 Hamiltonian matrix
    hamiltonian = np.array([[1.0, 0.5], [0.5, 2.0]])
    
    # Create VQE solver
    vqe = create_vqe_solver(hamiltonian, language=language)
    
    # Solve for ground state energy
    results = vqe.solve()
    print(f"VQE Results: {results}")
    print()


def qaoa_example(language='en'):
    """Demonstrate QAOA solver."""
    print(f"=== {translate('qaoa_example', language) or 'QAOA Example'} ===")
    
    # Create a simple graph problem (max cut)
    problem_graph = [(0, 1), (1, 2), (2, 0)]  # Triangle graph
    
    # Create QAOA solver
    qaoa = create_qaoa_solver(problem_graph, max_depth=2, language=language)
    
    # Optimize the problem
    results = qaoa.optimize()
    print(f"QAOA Results: {results}")
    print()


def grover_example(language='en'):
    """Demonstrate Grover's search."""
    print(f"=== {translate('grover_example', language) or 'Grover\\'s Search Example'} ===")
    
    # Simple oracle function (search for state '101')
    def oracle(state):
        return state == '101'
    
    # Create Grover search for 3 qubits
    grover = create_grover_search(oracle, num_qubits=3, language=language)
    
    # Search for solution
    results = grover.search()
    print(f"Grover Results: {results}")
    print()


def shor_example(language='en'):
    """Demonstrate Shor's algorithm."""
    print(f"=== {translate('shor_example', language) or 'Shor\\'s Algorithm Example'} ===")
    
    # Factor the number 15
    shor = create_shor_algorithm(15, language=language)
    
    # Factor the number
    factors = shor.factor()
    print(f"Factors of 15: {factors}")
    print()


def quantum_crypto_example(language='en'):
    """Demonstrate quantum-safe cryptography."""
    print(f"=== {translate('quantum_crypto_example', language) or 'Quantum-Safe Crypto Example'} ===")
    
    # Create quantum-safe crypto
    crypto = create_quantum_safe_crypto(language=language)
    
    # Encrypt some data
    data = b"Hello, Quantum World!"
    encrypted = crypto.encrypt(data, algorithm='kyber')
    print(f"Encrypted: {encrypted}")
    
    # Decrypt the data
    decrypted = crypto.decrypt(encrypted, algorithm='kyber')
    print(f"Decrypted: {decrypted}")
    print()


def translate(key, language):
    """Simple translation function for example titles."""
    translations = {
        'quantum_circuit_example': {
            'ru': 'Пример квантовой схемы',
            'zh': '量子电路示例',
            'ar': 'مثال الدائرة الكمومية'
        },
        'vqe_example': {
            'ru': 'Пример VQE',
            'zh': 'VQE示例',
            'ar': 'مثال VQE'
        },
        'qaoa_example': {
            'ru': 'Пример QAOA',
            'zh': 'QAOA示例',
            'ar': 'مثال QAOA'
        },
        'grover_example': {
            'ru': 'Пример поиска Гровера',
            'zh': 'Grover搜索示例',
            'ar': 'مثال بحث جروفر'
        },
        'shor_example': {
            'ru': 'Пример алгоритма Шора',
            'zh': 'Shor算法示例',
            'ar': 'مثال خوارزمية شور'
        },
        'quantum_crypto_example': {
            'ru': 'Пример квантовой криптографии',
            'zh': '量子密码学示例',
            'ar': 'مثال التشفير الكمومي'
        }
    }
    
    if key in translations and language in translations[key]:
        return translations[key][language]
    return None


def main():
    """Run all quantum examples."""
    languages = ['en', 'ru', 'zh', 'ar']
    
    for language in languages:
        print(f"\n{'='*50}")
        print(f"QUANTUM EXAMPLES - {language.upper()}")
        print(f"{'='*50}\n")
        
        try:
            quantum_circuit_example(language)
            vqe_example(language)
            qaoa_example(language)
            grover_example(language)
            shor_example(language)
            quantum_crypto_example(language)
        except Exception as e:
            print(f"Error in {language} examples: {e}")


if __name__ == "__main__":
    main()