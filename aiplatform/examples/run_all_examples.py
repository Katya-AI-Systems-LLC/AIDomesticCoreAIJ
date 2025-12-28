"""
Run All Examples Script for AIPlatform SDK

This script provides a convenient way to run all examples in the AIPlatform SDK
with various configuration options.
"""

import sys
import os
import argparse
import time
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_quantum_example(language: str = 'en', epochs: int = 2) -> bool:
    """Run quantum-classical hybrid AI example."""
    try:
        print(f"Running Quantum-Classical Hybrid AI Example ({language})...")
        from aiplatform.examples.quantum_ai_hybrid_example import QuantumClassicalHybridAI
        
        hybrid_ai = QuantumClassicalHybridAI(language=language)
        node_ids = hybrid_ai.setup_hybrid_training(num_nodes=2)
        result = hybrid_ai.train_hybrid_model(node_ids, epochs=epochs)
        report = hybrid_ai.generate_training_report(result)
        
        print(f"Quantum example completed successfully")
        print(f"Overall accuracy: {result.overall_accuracy:.2f}")
        print(f"Processing time: {result.processing_time:.3f} seconds")
        print()
        return True
        
    except Exception as e:
        print(f"Error running quantum example: {e}")
        print()
        return False

def run_vision_example(language: str = 'en') -> bool:
    """Run vision demo example."""
    try:
        print(f"Running Vision Demo Example ({language})...")
        from aiplatform.examples.vision_demo import VisionDemo
        
        vision_demo = VisionDemo(language=language)
        results = vision_demo.run_cross_platform_demo()
        
        print(f"Vision example completed successfully")
        print(f"Total demos run: {len(results)}")
        print()
        return True
        
    except Exception as e:
        print(f"Error running vision example: {e}")
        print()
        return False

def run_multimodal_example(language: str = 'en') -> bool:
    """Run multimodal AI example."""
    try:
        print(f"Running Multimodal AI Example ({language})...")
        from aiplatform.examples.multimodal_ai_example import MultimodalAIDemo
        
        multimodal_demo = MultimodalAIDemo(language=language)
        result = multimodal_demo.run_integrated_multimodal_analysis()
        
        print(f"Multimodal example completed successfully")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing time: {result.processing_time:.3f} seconds")
        print()
        return True
        
    except Exception as e:
        print(f"Error running multimodal example: {e}")
        print()
        return False

def run_all_examples_in_language(language: str = 'en', epochs: int = 2) -> Dict[str, bool]:
    """
    Run all examples in a specific language.
    
    Args:
        language (str): Language to run examples in
        epochs (int): Number of epochs for quantum training
        
    Returns:
        dict: Results for each example
    """
    print(f"{'='*60}")
    print(f"RUNNING ALL EXAMPLES IN {language.upper()}")
    print(f"{'='*60}")
    print()
    
    start_time = time.time()
    
    results = {
        'quantum': run_quantum_example(language, epochs),
        'vision': run_vision_example(language),
        'multimodal': run_multimodal_example(language)
    }
    
    total_time = time.time() - start_time
    
    successful = sum(1 for result in results.values() if result)
    print(f"{'='*60}")
    print(f"LANGUAGE {language.upper()} SUMMARY")
    print(f"{'='*60}")
    print(f"Successful examples: {successful}/3")
    print(f"Total time: {total_time:.2f} seconds")
    print()
    
    return results

def run_all_examples(languages: List[str] = None, epochs: int = 2) -> Dict[str, Dict[str, bool]]:
    """
    Run all examples in all specified languages.
    
    Args:
        languages (list): List of languages to run examples in
        epochs (int): Number of epochs for quantum training
        
    Returns:
        dict: Results for each language
    """
    if languages is None:
        languages = ['en', 'ru', 'zh', 'ar']
    
    print("=" * 60)
    print("RUNNING ALL AIPLATFORM SDK EXAMPLES")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    all_results = {}
    for language in languages:
        all_results[language] = run_all_examples_in_language(language, epochs)
    
    total_time = time.time() - start_time
    
    # Generate final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    total_examples = 0
    successful_examples = 0
    
    for language, results in all_results.items():
        lang_total = len(results)
        lang_successful = sum(1 for result in results.values() if result)
        total_examples += lang_total
        successful_examples += lang_successful
        print(f"{language.upper()}: {lang_successful}/{lang_total} successful")
    
    print()
    print(f"Overall: {successful_examples}/{total_examples} examples successful")
    print(f"Success rate: {(successful_examples/total_examples*100):.1f}%")
    print(f"Total execution time: {total_time:.2f} seconds")
    print()
    
    return all_results

def main():
    """Main function to run examples based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run AIPlatform SDK Examples")
    parser.add_argument(
        '--languages', 
        nargs='+', 
        default=['en', 'ru', 'zh', 'ar'],
        help='Languages to run examples in (default: en ru zh ar)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of epochs for quantum training (default: 2)'
    )
    parser.add_argument(
        '--quantum-only',
        action='store_true',
        help='Run only quantum example'
    )
    parser.add_argument(
        '--vision-only',
        action='store_true',
        help='Run only vision example'
    )
    parser.add_argument(
        '--multimodal-only',
        action='store_true',
        help='Run only multimodal example'
    )
    
    args = parser.parse_args()
    
    # Determine which examples to run
    if args.quantum_only:
        print("Running Quantum-Classical Hybrid AI Example...")
        for language in args.languages:
            run_quantum_example(language, args.epochs)
    elif args.vision_only:
        print("Running Vision Demo Example...")
        for language in args.languages:
            run_vision_example(language)
    elif args.multimodal_only:
        print("Running Multimodal AI Example...")
        for language in args.languages:
            run_multimodal_example(language)
    else:
        # Run all examples
        run_all_examples(args.languages, args.epochs)

if __name__ == "__main__":
    main()