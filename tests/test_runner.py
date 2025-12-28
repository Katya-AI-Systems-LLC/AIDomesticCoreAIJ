"""
AIPlatform SDK - Comprehensive Test Runner

This script runs all tests for the AIPlatform SDK including:
- Core functionality tests
- Multilingual support tests
- Integration tests
- Performance tests
- Security tests
"""

import sys
import os
import unittest
import argparse
import time
from typing import List, Dict, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import AIPlatform modules for testing
from aiplatform.core import AIPlatform
from aiplatform.testing import run_comprehensive_tests

# Import test modules
try:
    from tests.test_multilingual import create_test_suite as create_multilingual_suite
    MULTILINGUAL_TESTS_AVAILABLE = True
except ImportError:
    MULTILINGUAL_TESTS_AVAILABLE = False
    print("Warning: Multilingual tests not available")

# Import example test modules
try:
    from aiplatform.examples.integration_test import AIPlatformIntegrationTest
    INTEGRATION_TESTS_AVAILABLE = True
except ImportError:
    INTEGRATION_TESTS_AVAILABLE = False
    print("Warning: Integration tests not available")


class AIPlatformTestRunner:
    """Comprehensive test runner for AIPlatform SDK."""
    
    def __init__(self, verbose: bool = True, languages: List[str] = None):
        """
        Initialize the test runner.
        
        Args:
            verbose (bool): Whether to show verbose output
            languages (list): List of languages to test (None for all supported)
        """
        self.verbose = verbose
        self.languages = languages or ['en', 'ru', 'zh', 'ar']
        self.results = {}
        self.start_time = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all available tests.
        
        Returns:
            dict: Test results summary
        """
        self.start_time = time.time()
        print("=" * 80)
        print("AIPPLATFORM SDK COMPREHENSIVE TEST RUNNER")
        print("=" * 80)
        print(f"Testing languages: {', '.join(self.languages)}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run component tests
        component_results = self._run_component_tests()
        
        # Run multilingual tests
        multilingual_results = self._run_multilingual_tests()
        
        # Run integration tests
        integration_results = self._run_integration_tests()
        
        # Run performance tests
        performance_results = self._run_performance_tests()
        
        # Run example tests
        example_results = self._run_example_tests()
        
        # Compile final results
        total_time = time.time() - self.start_time
        final_results = {
            "component_tests": component_results,
            "multilingual_tests": multilingual_results,
            "integration_tests": integration_results,
            "performance_tests": performance_results,
            "example_tests": example_results,
            "total_time": total_time,
            "languages_tested": self.languages,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self._print_final_summary(final_results)
        return final_results
    
    def _run_component_tests(self) -> Dict[str, Any]:
        """
        Run core component tests.
        
        Returns:
            dict: Component test results
        """
        print("Running Core Component Tests...")
        print("-" * 40)
        
        start_time = time.time()
        results = {}
        
        try:
            # Test core platform
            platform = AIPlatform()
            results["core_platform"] = {
                "initialized": platform is not None,
                "status": "passed" if platform else "failed"
            }
            
            # Test quantum components
            from aiplatform.quantum import create_quantum_circuit
            circuit = create_quantum_circuit(2)
            results["quantum_components"] = {
                "initialized": circuit is not None,
                "status": "passed" if circuit else "failed"
            }
            
            # Test QIZ components
            from aiplatform.qiz import create_qiz_infrastructure
            qiz = create_qiz_infrastructure()
            results["qiz_components"] = {
                "initialized": qiz is not None,
                "status": "passed" if qiz else "failed"
            }
            
            # Test federated components
            from aiplatform.federated import create_federated_coordinator
            coordinator = create_federated_coordinator()
            results["federated_components"] = {
                "initialized": coordinator is not None,
                "status": "passed" if coordinator else "failed"
            }
            
            # Test vision components
            from aiplatform.vision import create_object_detector
            detector = create_object_detector()
            results["vision_components"] = {
                "initialized": detector is not None,
                "status": "passed" if detector else "failed"
            }
            
            # Test GenAI components
            from aiplatform.genai import create_genai_model
            genai = create_genai_model("gigachat3-702b")
            results["genai_components"] = {
                "initialized": genai is not None,
                "status": "passed" if genai else "failed"
            }
            
            # Test security components
            from aiplatform.security import create_didn
            didn = create_didn()
            results["security_components"] = {
                "initialized": didn is not None,
                "status": "passed" if didn else "failed"
            }
            
            # Test protocol components
            from aiplatform.protocols import create_qmp_protocol
            qmp = create_qmp_protocol()
            results["protocol_components"] = {
                "initialized": qmp is not None,
                "status": "passed" if qmp else "failed"
            }
            
            # Calculate component test results
            passed_tests = sum(1 for r in results.values() if r["status"] == "passed")
            total_tests = len(results)
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            results["summary"] = {
                "passed": passed_tests,
                "total": total_tests,
                "success_rate": success_rate,
                "status": "passed" if success_rate >= 0.8 else "failed"
            }
            
            print(f"Component tests: {passed_tests}/{total_tests} passed ({success_rate:.1%} success rate)")
            
        except Exception as e:
            print(f"Component tests error: {e}")
            results["error"] = str(e)
            results["summary"] = {"status": "failed"}
        
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        print(f"Component tests completed in {execution_time:.2f} seconds")
        print()
        
        return results
    
    def _run_multilingual_tests(self) -> Dict[str, Any]:
        """
        Run multilingual support tests.
        
        Returns:
            dict: Multilingual test results
        """
        print("Running Multilingual Support Tests...")
        print("-" * 40)
        
        start_time = time.time()
        results = {}
        
        if not MULTILINGUAL_TESTS_AVAILABLE:
            print("Multilingual tests not available, skipping...")
            results["summary"] = {"status": "skipped"}
            results["execution_time"] = 0
            print()
            return results
        
        try:
            # Run multilingual test suite
            suite = create_multilingual_suite()
            runner = unittest.TextTestRunner(verbosity=0)
            test_result = runner.run(suite)
            
            # Calculate results
            passed_tests = test_result.testsRun - len(test_result.failures) - len(test_result.errors)
            total_tests = test_result.testsRun
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            results["summary"] = {
                "passed": passed_tests,
                "total": total_tests,
                "success_rate": success_rate,
                "failures": len(test_result.failures),
                "errors": len(test_result.errors),
                "status": "passed" if success_rate >= 0.95 else "failed"
            }
            
            print(f"Multilingual tests: {passed_tests}/{total_tests} passed ({success_rate:.1%} success rate)")
            
        except Exception as e:
            print(f"Multilingual tests error: {e}")
            results["error"] = str(e)
            results["summary"] = {"status": "failed"}
        
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        print(f"Multilingual tests completed in {execution_time:.2f} seconds")
        print()
        
        return results
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """
        Run integration tests.
        
        Returns:
            dict: Integration test results
        """
        print("Running Integration Tests...")
        print("-" * 40)
        
        start_time = time.time()
        results = {}
        
        if not INTEGRATION_TESTS_AVAILABLE:
            print("Integration tests not available, skipping...")
            results["summary"] = {"status": "skipped"}
            results["execution_time"] = 0
            print()
            return results
        
        try:
            # Run integration test suite
            integration_test = AIPlatformIntegrationTest()
            test_result = integration_test.run_comprehensive_integration_test()
            
            # Calculate results
            overall_score = getattr(test_result, "overall_score", 0.0)
            processing_time = getattr(test_result, "processing_time", 0.0)
            
            results["summary"] = {
                "overall_score": overall_score,
                "processing_time": processing_time,
                "status": "passed" if overall_score >= 0.8 else "failed"
            }
            
            print(f"Integration tests: overall score {overall_score:.2f}")
            
        except Exception as e:
            print(f"Integration tests error: {e}")
            results["error"] = str(e)
            results["summary"] = {"status": "failed"}
        
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        print(f"Integration tests completed in {execution_time:.2f} seconds")
        print()
        
        return results
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """
        Run performance tests.
        
        Returns:
            dict: Performance test results
        """
        print("Running Performance Tests...")
        print("-" * 40)
        
        start_time = time.time()
        results = {}
        
        try:
            # Run comprehensive performance tests
            test_results = run_comprehensive_tests(languages=self.languages)
            
            # Calculate results
            overall_score = getattr(test_results, "overall_score", 0.0)
            performance_score = getattr(test_results, "performance_score", 0.0)
            
            results["summary"] = {
                "overall_score": overall_score,
                "performance_score": performance_score,
                "status": "passed" if overall_score >= 0.7 else "failed"
            }
            
            print(f"Performance tests: overall score {overall_score:.2f}")
            
        except Exception as e:
            print(f"Performance tests error: {e}")
            results["error"] = str(e)
            results["summary"] = {"status": "failed"}
        
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        print(f"Performance tests completed in {execution_time:.2f} seconds")
        print()
        
        return results
    
    def _run_example_tests(self) -> Dict[str, Any]:
        """
        Run example tests.
        
        Returns:
            dict: Example test results
        """
        print("Running Example Tests...")
        print("-" * 40)
        
        start_time = time.time()
        results = {}
        
        try:
            # Test comprehensive multimodal example
            from aiplatform.examples.comprehensive_multimodal_example import MultimodalAI
            multimodal_example = MultimodalAI()
            multimodal_result = multimodal_example.process_multimodal_data(
                text="Test text",
                image=None,  # Will be generated
                audio=b"test_audio",
                video=b"test_video"
            )
            
            # Test quantum vision example
            from aiplatform.examples.quantum_vision_example import QuantumVisionAI
            quantum_vision_example = QuantumVisionAI()
            quantum_vision_result = quantum_vision_example.process_quantum_vision_data(None)
            
            # Test federated quantum example
            from aiplatform.examples.federated_quantum_example import FederatedQuantumAI
            federated_example = FederatedQuantumAI()
            node_ids = federated_example.setup_federated_network(num_nodes=2)
            federated_result = federated_example.train_federated_quantum_model(node_ids, rounds=1)
            
            # Calculate results
            successful_examples = sum([
                multimodal_result is not None,
                quantum_vision_result is not None,
                federated_result is not None
            ])
            total_examples = 3
            success_rate = successful_examples / total_examples
            
            results["summary"] = {
                "successful": successful_examples,
                "total": total_examples,
                "success_rate": success_rate,
                "status": "passed" if success_rate >= 0.8 else "failed"
            }
            
            print(f"Example tests: {successful_examples}/{total_examples} passed ({success_rate:.1%} success rate)")
            
        except Exception as e:
            print(f"Example tests error: {e}")
            results["error"] = str(e)
            results["summary"] = {"status": "failed"}
        
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        print(f"Example tests completed in {execution_time:.2f} seconds")
        print()
        
        return results
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """
        Print final test results summary.
        
        Args:
            results (dict): Test results
        """
        print("=" * 80)
        print("AIPPLATFORM SDK TEST RESULTS SUMMARY")
        print("=" * 80)
        
        # Component tests summary
        component_summary = results.get("component_tests", {}).get("summary", {})
        print(f"Component Tests: {component_summary.get('passed', 0)}/{component_summary.get('total', 0)} "
              f"({component_summary.get('success_rate', 0):.1%}) - {component_summary.get('status', 'unknown').upper()}")
        
        # Multilingual tests summary
        multilingual_summary = results.get("multilingual_tests", {}).get("summary", {})
        print(f"Multilingual Tests: {multilingual_summary.get('status', 'unknown').upper()}")
        
        # Integration tests summary
        integration_summary = results.get("integration_tests", {}).get("summary", {})
        print(f"Integration Tests: {integration_summary.get('status', 'unknown').upper()}")
        
        # Performance tests summary
        performance_summary = results.get("performance_tests", {}).get("summary", {})
        print(f"Performance Tests: {performance_summary.get('status', 'unknown').upper()}")
        
        # Example tests summary
        example_summary = results.get("example_tests", {}).get("summary", {})
        print(f"Example Tests: {example_summary.get('successful', 0)}/{example_summary.get('total', 0)} "
              f"({example_summary.get('success_rate', 0):.1%}) - {example_summary.get('status', 'unknown').upper()}")
        
        # Overall summary
        total_time = results.get("total_time", 0)
        languages = results.get("languages_tested", [])
        
        print()
        print(f"Total Testing Time: {total_time:.2f} seconds")
        print(f"Languages Tested: {', '.join(languages)}")
        print(f"Test Completion Time: {results.get('timestamp', 'unknown')}")
        print("=" * 80)


def main():
    """Main function to run the test suite."""
    parser = argparse.ArgumentParser(description="AIPlatform SDK Comprehensive Test Runner")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--languages", nargs="+", default=['en', 'ru', 'zh', 'ar'],
                       help="Languages to test (default: en ru zh ar)")
    parser.add_argument("--components", action="store_true", help="Run only component tests")
    parser.add_argument("--multilingual", action="store_true", help="Run only multilingual tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--examples", action="store_true", help="Run only example tests")
    
    args = parser.parse_args()
    
    # Create test runner
    test_runner = AIPlatformTestRunner(verbose=args.verbose, languages=args.languages)
    
    # Run tests based on arguments
    if args.components or args.multilingual or args.integration or args.performance or args.examples:
        # Run specific test types
        results = {}
        if args.components:
            results["component_tests"] = test_runner._run_component_tests()
        if args.multilingual:
            results["multilingual_tests"] = test_runner._run_multilingual_tests()
        if args.integration:
            results["integration_tests"] = test_runner._run_integration_tests()
        if args.performance:
            results["performance_tests"] = test_runner._run_performance_tests()
        if args.examples:
            results["example_tests"] = test_runner._run_example_tests()
    else:
        # Run all tests
        results = test_runner.run_all_tests()
    
    # Return exit code based on results
    overall_status = "passed"
    for test_type, test_results in results.items():
        if test_type != "total_time" and test_type != "languages_tested" and test_type != "timestamp":
            summary = test_results.get("summary", {})
            if summary.get("status") == "failed":
                overall_status = "failed"
                break
    
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())