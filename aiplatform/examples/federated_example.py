"""
Federated Quantum AI Example for AIPlatform SDK

This example demonstrates federated learning capabilities with multilingual support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiplatform.federated import (
    create_federated_node,
    create_federated_coordinator,
    create_hybrid_model,
    create_model_marketplace,
    create_collaborative_evolution
)
import numpy as np


def federated_learning_example(language='en'):
    """Demonstrate federated learning."""
    print(f"=== {translate('federated_learning_example', language) or 'Federated Learning Example'} ===")
    
    # Create federated coordinator
    coordinator = create_federated_coordinator(language=language)
    
    # Create federated nodes
    node1 = create_federated_node("node_1", "model_1", language=language)
    node2 = create_federated_node("node_2", "model_2", language=language)
    node3 = create_federated_node("node_3", "model_3", language=language)
    
    # Register nodes
    coordinator.register_node(node1)
    coordinator.register_node(node2)
    coordinator.register_node(node3)
    
    # Run federated rounds
    for i in range(3):
        print(f"\n--- {translate('federated_round', language) or 'Federated Round'} {i+1} ---")
        try:
            result = coordinator.run_federated_round()
            print(f"Round result: {result}")
        except Exception as e:
            print(f"Error in round {i+1}: {e}")
    
    # Get final global weights
    global_weights = coordinator.get_global_weights()
    print(f"\nFinal global weights signature: {global_weights.get_signature()}")
    print()


def hybrid_model_example(language='en'):
    """Demonstrate hybrid quantum-classical model."""
    print(f"=== {translate('hybrid_model_example', language) or 'Hybrid Model Example'} ===")
    
    # Create hybrid model components
    quantum_component = {"type": "qiskit_circuit", "qubits": 4}
    classical_component = {"type": "tensorflow_model", "layers": 3}
    
    # Create hybrid model
    hybrid_model = create_hybrid_model(quantum_component, classical_component, language=language)
    
    # Forward pass
    input_data = {"features": np.random.random(10).tolist()}
    result = hybrid_model.forward(input_data)
    
    print(f"Hybrid model result: {result}")
    print()


def model_marketplace_example(language='en'):
    """Demonstrate model marketplace."""
    print(f"=== {translate('model_marketplace_example', language) or 'Model Marketplace Example'} ===")
    
    # Create model marketplace
    marketplace = create_model_marketplace(language=language)
    
    # Deploy smart contract
    contract_code = """
    contract ModelTrading {
        function buyModel(string modelId) public {
            // Implementation for buying model
        }
        
        function sellModel(string modelId, uint price) public {
            // Implementation for selling model
        }
    }
    """
    
    contract_address = marketplace.deploy_smart_contract("ModelTrading", contract_code)
    print(f"Deployed contract at: {contract_address}")
    
    # Mint NFT weights (simulated)
    from aiplatform.federated import ModelWeights
    weights = ModelWeights({"layer_1": [0.1, 0.2, 0.3], "layer_2": [0.4, 0.5]}, language=language)
    
    nft_token = marketplace.mint_nft_weights("resnet50_model", weights)
    print(f"Minted NFT weights with token: {nft_token}")
    
    # List models
    models = marketplace.list_models()
    print(f"Available models: {models}")
    
    # Get model info
    model_info = marketplace.get_model_info(nft_token)
    print(f"Model info: {model_info}")
    print()


def collaborative_evolution_example(language='en'):
    """Demonstrate collaborative evolution."""
    print(f"=== {translate('collaborative_evolution_example', language) or 'Collaborative Evolution Example'} ===")
    
    # Create collaborative evolution
    evolution = create_collaborative_evolution(language=language)
    
    # Add individuals
    for i in range(5):
        genome = {
            "learning_rate": float(np.random.random()),
            "hidden_layers": int(np.random.randint(1, 10)),
            "activation": np.random.choice(["relu", "sigmoid", "tanh"])
        }
        evolution.add_individual(f"individual_{i}", genome)
    
    print(f"Added {len(evolution.population)} individuals")
    
    # Evaluate fitness
    def fitness_function(genome):
        # Simple fitness function based on learning rate and layers
        return genome["learning_rate"] * genome["hidden_layers"]
    
    fitness_scores = []
    for individual_id in evolution.population.keys():
        fitness = evolution.evaluate_fitness(individual_id, fitness_function)
        fitness_scores.append(fitness)
        print(f"{individual_id} fitness: {fitness}")
    
    # Evolve generation
    evolution_result = evolution.evolve_generation()
    print(f"\nEvolution result: {evolution_result}")
    print()


def translate(key, language):
    """Simple translation function for example titles."""
    translations = {
        'federated_learning_example': {
            'ru': 'Пример федеративного обучения',
            'zh': '联邦学习示例',
            'ar': 'مثال التعلم الفيدرالي'
        },
        'federated_round': {
            'ru': 'Федеративный раунд',
            'zh': '联邦轮次',
            'ar': 'جولة فيدرالية'
        },
        'hybrid_model_example': {
            'ru': 'Пример гибридной модели',
            'zh': '混合模型示例',
            'ar': 'مثال النموذج الهجين'
        },
        'model_marketplace_example': {
            'ru': 'Пример рынка моделей',
            'zh': '模型市场示例',
            'ar': 'مثال سوق النماذج'
        },
        'collaborative_evolution_example': {
            'ru': 'Пример совместной эволюции',
            'zh': '协作进化示例',
            'ar': 'مثال التطور التعاوني'
        }
    }
    
    if key in translations and language in translations[key]:
        return translations[key][language]
    return None


def main():
    """Run all federated examples."""
    languages = ['en', 'ru', 'zh', 'ar']
    
    for language in languages:
        print(f"\n{'='*50}")
        print(f"FEDERATED EXAMPLES - {language.upper()}")
        print(f"{'='*50}\n")
        
        try:
            federated_learning_example(language)
            hybrid_model_example(language)
            model_marketplace_example(language)
            collaborative_evolution_example(language)
        except Exception as e:
            print(f"Error in {language} examples: {e}")


if __name__ == "__main__":
    main()