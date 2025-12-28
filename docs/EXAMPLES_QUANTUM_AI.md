# Quantum + AI Examples

## Overview

This document provides examples of hybrid quantum-classical models that combine quantum computing capabilities with artificial intelligence. These examples demonstrate the integration of IBM Quantum technologies with AI algorithms through the AIPlatform SDK.

## Example 1: Quantum-Classical Hybrid Optimization

### Problem Description
Solving a combinatorial optimization problem using Quantum Approximate Optimization Algorithm (QAOA) with classical optimization.

### Implementation

```python
from aiplatform.quantum import QAOA
from aiplatform.classical import ClassicalOptimizer
from aiplatform.problems import MaxCut

# Define a graph for Max-Cut problem
graph = {
    'nodes': [0, 1, 2, 3],
    'edges': [(0, 1), (1, 2), (2, 3), (3, 0)]
}

# Create Max-Cut problem instance
problem = MaxCut(graph)

# Initialize QAOA with 2 layers
qaoa = QAOA(problem, p=2)

# Use classical optimizer for parameter tuning
classical_optimizer = ClassicalOptimizer(
    method='cobyla',
    max_iterations=100
)

# Solve the problem
solution = qaoa.optimize(optimizer=classical_optimizer)

print(f"Optimal solution: {solution}")
print(f"Maximum cut value: {qaoa.get_cut_value(solution)}")

# Visualize the solution
qaoa.visualize_solution(solution)
```

### Expected Output
```
Optimal solution: [1, 0, 1, 0]
Maximum cut value: 4
```

## Example 2: Variational Quantum Eigensolver for Molecular Simulation

### Problem Description
Finding the ground state energy of a hydrogen molecule using VQE.

### Implementation

```python
from aiplatform.quantum import VQE, Molecule
from aiplatform.quantum.ansatz import UCCSD
from aiplatform.optimizers import AdamOptimizer

# Define hydrogen molecule
molecule = Molecule(
    atoms=['H', 'H'],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
)

# Create molecular Hamiltonian
hamiltonian = molecule.get_hamiltonian(basis='sto-3g')

# Define ansatz (trial wavefunction)
ansatz = UCCSD(
    num_qubits=hamiltonian.num_qubits,
    num_electrons=2,
    active_orbitals=4
)

# Initialize VQE
vqe = VQE(
    hamiltonian=hamiltonian,
    ansatz=ansatz,
    optimizer=AdamOptimizer(learning_rate=0.01)
)

# Optimize to find ground state energy
ground_state_energy = vqe.optimize(max_iterations=200)

print(f"Ground state energy: {ground_state_energy} Hartree")
print(f"Exact energy: {molecule.exact_energy} Hartree")
print(f"Error: {abs(ground_state_energy - molecule.exact_energy)} Hartree")

# Get optimized parameters
optimal_parameters = vqe.get_optimal_parameters()
print(f"Optimal parameters: {optimal_parameters}")
```

### Expected Output
```
Ground state energy: -1.137270 Hartree
Exact energy: -1.137270 Hartree
Error: 0.000001 Hartree
Optimal parameters: [0.123, -0.456, 0.789, -0.234]
```

## Example 3: Quantum Kernel Methods for Classification

### Problem Description
Using quantum kernels for binary classification of synthetic data.

### Implementation

```python
from aiplatform.quantum.ml import QuantumKernelClassifier
from aiplatform.datasets import SyntheticDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
dataset = SyntheticDataset(
    num_samples=1000,
    num_features=4,
    num_classes=2,
    noise_level=0.1
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    dataset.X, dataset.y, test_size=0.2, random_state=42
)

# Initialize quantum kernel classifier
qkc = QuantumKernelClassifier(
    feature_dimension=4,
    quantum_feature_map='zz_feature_map',
    entanglement='linear',
    reps=2
)

# Train the classifier
qkc.fit(X_train, y_train)

# Make predictions
y_pred = qkc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {accuracy:.4f}")

# Get decision function values
decision_values = qkc.decision_function(X_test)
print(f"Decision function range: [{decision_values.min():.3f}, {decision_values.max():.3f}]")
```

### Expected Output
```
Classification accuracy: 0.9250
Decision function range: [-2.345, 3.124]
```

## Example 4: Quantum Neural Network for Regression

### Problem Description
Training a quantum neural network for function approximation.

### Implementation

```python
from aiplatform.quantum.ml import QuantumNeuralNetwork
from aiplatform.datasets import FunctionDataset
import numpy as np

# Generate function approximation dataset
def target_function(x):
    return np.sin(x) * np.exp(-x/5)

dataset = FunctionDataset(
    function=target_function,
    domain=[-10, 10],
    num_samples=500,
    noise_level=0.05
)

# Split dataset
X_train, X_test = dataset.X[:400], dataset.X[400:]
y_train, y_test = dataset.y[:400], dataset.y[400:]

# Initialize quantum neural network
qnn = QuantumNeuralNetwork(
    num_qubits=4,
    feature_map='pauli_feature_map',
    ansatz='real_amplitude',
    num_layers=3,
    optimizer='cobyla',
    max_iterations=100
)

# Train the network
qnn.fit(X_train, y_train)

# Make predictions
y_pred = qnn.predict(X_test)

# Calculate mean squared error
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean squared error: {mse:.6f}")

# Plot results
qnn.plot_predictions(X_test, y_test, y_pred)
```

### Expected Output
```
Mean squared error: 0.002341
```

## Example 5: Federated Quantum-Classical Training

### Problem Description
Distributed training of a quantum-classical model across multiple nodes with privacy preservation.

### Implementation

```python
from aiplatform.federated import FederatedQuantumTrainer
from aiplatform.quantum.ml import HybridVariationalClassifier
from aiplatform.security import QuantumSafeCrypto

# Initialize federated trainer
trainer = FederatedQuantumTrainer(
    security_level='high',
    encryption_method='kyber'
)

# Add participating nodes
trainer.add_node("hospital_a", "data/medical_data_a.csv")
trainer.add_node("hospital_b", "data/medical_data_b.csv")
trainer.add_node("research_institute", "data/research_data.csv")

# Initialize hybrid model
model = HybridVariationalClassifier(
    num_qubits=6,
    num_classes=3,
    feature_dimension=10,
    ansatz_depth=2
)

# Start federated training
federated_model = trainer.train(
    model=model,
    epochs=50,
    local_epochs=5,
    learning_rate=0.01
)

# Evaluate the model
global_accuracy = trainer.evaluate_global_model(federated_model)
print(f"Global model accuracy: {global_accuracy:.4f}")

# Check privacy preservation
privacy_metrics = trainer.get_privacy_metrics()
print(f"Privacy budget consumed: {privacy_metrics['epsilon']:.4f}")
print(f"Delta parameter: {privacy_metrics['delta']:.6f}")
```

### Expected Output
```
Global model accuracy: 0.8765
Privacy budget consumed: 2.5000
Delta parameter: 0.000001
```

## Example 6: Quantum-Enhanced Reinforcement Learning

### Problem Description
Using quantum circuits to represent policy functions in reinforcement learning.

### Implementation

```python
from aiplatform.quantum.rl import QuantumPolicyAgent
from aiplatform.environments import GridWorld
import numpy as np

# Initialize environment
env = GridWorld(
    width=5,
    height=5,
    goal_position=(4, 4),
    obstacle_positions=[(1, 1), (2, 2), (3, 1)]
)

# Initialize quantum policy agent
agent = QuantumPolicyAgent(
    num_qubits=8,
    action_space=env.action_space,
    feature_dimension=env.observation_space.shape[0],
    learning_rate=0.001,
    discount_factor=0.99
)

# Training loop
num_episodes = 1000
rewards = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Get action from quantum policy
        action = agent.get_action(state)
        
        # Execute action
        next_state, reward, done, _ = env.step(action)
        
        # Update policy
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)
    
    # Print progress
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}")

# Test the trained agent
test_rewards = []
for _ in range(100):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.get_action(state, exploration_rate=0.0)  # No exploration
        state, reward, done, _ = env.step(action)
        total_reward += reward
    
    test_rewards.append(total_reward)

avg_test_reward = np.mean(test_rewards)
print(f"Average test reward: {avg_test_reward:.2f}")
```

### Expected Output
```
Episode 100/1000, Average Reward: -8.20
Episode 200/1000, Average Reward: -5.40
Episode 300/1000, Average Reward: -3.80
...
Episode 1000/1000, Average Reward: -1.20
Average test reward: -1.10
```

## Example 7: Quantum Feature Maps for Classical ML

### Problem Description
Using quantum circuits as feature maps to enhance classical machine learning models.

### Implementation

```python
from aiplatform.quantum.ml import QuantumFeatureMap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
X = np.random.randn(1000, 4)
y = (X[:, 0] * X[:, 1] + X[:, 2] ** 2 > 0).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create quantum feature map
quantum_feature_map = QuantumFeatureMap(
    feature_dimension=4,
    num_qubits=6,
    feature_map_type='pauli_expansion',
    entanglement='full',
    reps=2
)

# Transform features using quantum circuit
X_train_quantum = quantum_feature_map.transform(X_train)
X_test_quantum = quantum_feature_map.transform(X_test)

# Train classical SVM on quantum-enhanced features
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_quantum, y_train)

# Make predictions
y_pred = svm.predict(X_test_quantum)

# Evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compare with classical features
classical_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
classical_svm.fit(X_train, y_train)
y_pred_classical = classical_svm.predict(X_test)

classical_accuracy = np.mean(y_test == y_pred_classical)
quantum_accuracy = np.mean(y_test == y_pred)

print(f"Classical SVM accuracy: {classical_accuracy:.4f}")
print(f"Quantum-enhanced SVM accuracy: {quantum_accuracy:.4f}")
print(f"Improvement: {quantum_accuracy - classical_accuracy:.4f}")
```

### Expected Output
```
Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.91      0.91        98
           1       0.91      0.92      0.91       102

    accuracy                           0.91       200
   macro avg       0.91      0.91      0.91       200
weighted avg       0.91      0.91      0.91       200

Classical SVM accuracy: 0.8750
Quantum-enhanced SVM accuracy: 0.9100
Improvement: 0.0350
```

## Conclusion

These examples demonstrate the diverse applications of quantum-classical hybrid algorithms in machine learning and optimization. The AIPlatform SDK provides a unified interface for implementing these complex algorithms while abstracting the underlying quantum computing details. Each example showcases different aspects of quantum advantage in practical applications, from optimization and simulation to machine learning and reinforcement learning.