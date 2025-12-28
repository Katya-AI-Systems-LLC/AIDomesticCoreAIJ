"""
AIPlatform SDK - Multilingual Support Test Suite

Comprehensive tests for internationalization features across all modules.
"""

import sys
import os
import unittest
import numpy as np
from typing import Dict, List, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import AIPlatform modules
from aiplatform.core import AIPlatform
from aiplatform.quantum import (
    create_quantum_circuit, create_vqe_solver, create_qaoa_solver
)
from aiplatform.qiz import (
    create_qiz_infrastructure, create_zero_server,
    create_post_dns_layer, create_zero_trust_security
)
from aiplatform.federated import (
    create_federated_coordinator, create_federated_node, create_model_marketplace,
    create_hybrid_model, create_collaborative_evolution
)
from aiplatform.vision import (
    create_object_detector, create_face_recognizer, create_gesture_processor,
    create_video_analyzer, create_3d_vision_engine
)
from aiplatform.genai import (
    create_genai_model, create_diffusion_model, create_speech_processor,
    create_multimodal_model
)
from aiplatform.security import (
    create_didn, create_zero_trust_model, create_quantum_safe_crypto,
    create_kyber_crypto, create_dilithium_crypto
)
from aiplatform.protocols import (
    create_qmp_protocol, create_post_dns, create_zero_dns,
    create_quantum_signature, create_mesh_network
)
from aiplatform.i18n import TranslationManager, VocabularyManager, LanguageDetector, ResourceManager


class TestMultilingualSupport(unittest.TestCase):
    """Test suite for multilingual support across all AIPlatform components."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.supported_languages = ['en', 'ru', 'zh', 'ar']
        self.test_data = b"Test data for multilingual testing"
        
    def test_translation_manager_initialization(self):
        """Test TranslationManager initialization for all languages."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                translator = TranslationManager(language)
                self.assertEqual(translator.language, language)
                self.assertIsNotNone(translator.resource_manager)
    
    def test_vocabulary_manager_initialization(self):
        """Test VocabularyManager initialization for all languages."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                vocabulary = VocabularyManager(language)
                self.assertEqual(vocabulary.language, language)
                self.assertIsNotNone(vocabulary.resource_manager)
    
    def test_language_detector_initialization(self):
        """Test LanguageDetector initialization."""
        detector = LanguageDetector()
        self.assertIsNotNone(detector)
    
    def test_resource_manager_initialization(self):
        """Test ResourceManager initialization."""
        manager = ResourceManager()
        self.assertIsNotNone(manager)
    
    def test_translation_functionality(self):
        """Test translation functionality for all supported languages."""
        test_keys = [
            "welcome_message",
            "initialization_complete",
            "error_occurred",
            "processing_started",
            "processing_completed"
        ]
        
        for language in self.supported_languages:
            with self.subTest(language=language):
                translator = TranslationManager(language)
                for key in test_keys:
                    with self.subTest(key=key):
                        translation = translator.translate(key, language)
                        self.assertIsNotNone(translation)
                        self.assertIsInstance(translation, str)
                        self.assertNotEqual(translation, key)  # Should be translated
    
    def test_vocabulary_translation(self):
        """Test technical vocabulary translation for all domains."""
        technical_terms = [
            "quantum_computing",
            "artificial_intelligence",
            "machine_learning",
            "neural_network",
            "quantum_entanglement",
            "superposition",
            "qubit",
            "algorithm",
            "optimization",
            "cryptography",
            "zero_trust",
            "federated_learning",
            "computer_vision",
            "object_detection",
            "generative_ai"
        ]
        
        domains = [
            "quantum", "ai", "security", "networking", 
            "vision", "computing", "qiz", "protocols"
        ]
        
        for language in self.supported_languages:
            with self.subTest(language=language):
                vocabulary = VocabularyManager(language)
                for term in technical_terms:
                    with self.subTest(term=term):
                        translation = vocabulary.translate_term(term, language)
                        self.assertIsNotNone(translation)
                        self.assertIsInstance(translation, str)
                
                # Test domain-specific translations
                for domain in domains:
                    with self.subTest(domain=domain):
                        domain_translation = vocabulary.translate_term("quantum_computing", language, domain)
                        self.assertIsNotNone(domain_translation)
    
    def test_language_detection(self):
        """Test language detection functionality."""
        detector = LanguageDetector()
        
        test_texts = {
            'en': "This is a test of English language detection.",
            'ru': "Это тест русского языка для определения языка.",
            'zh': "这是中文语言检测的测试。",
            'ar': "هذا اختبار لكشف اللغة العربية."
        }
        
        for expected_lang, text in test_texts.items():
            with self.subTest(expected_lang=expected_lang):
                detected_lang = detector.detect_language(text)
                # Should detect the correct language or a similar one
                self.assertIn(detected_lang, self.supported_languages)
    
    def test_multilingual_quantum_components(self):
        """Test multilingual support in quantum computing components."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                # Test quantum circuit
                circuit = create_quantum_circuit(4, language=language)
                self.assertIsNotNone(circuit)
                
                # Test VQE solver
                vqe_solver = create_vqe_solver(None, language=language)
                self.assertIsNotNone(vqe_solver)
                
                # Test QAOA solver
                qaoa_solver = create_qaoa_solver(None, max_depth=2, language=language)
                self.assertIsNotNone(qaoa_solver)
    
    def test_multilingual_qiz_components(self):
        """Test multilingual support in QIZ components."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                # Test QIZ infrastructure
                qiz = create_qiz_infrastructure(language=language)
                self.assertIsNotNone(qiz)
                
                # Test zero server
                zero_server = create_zero_server(language=language)
                self.assertIsNotNone(zero_server)
                
                # Test post-DNS layer
                post_dns_layer = create_post_dns_layer(language=language)
                self.assertIsNotNone(post_dns_layer)
                
                # Test zero-trust security
                zero_trust = create_zero_trust_security(language=language)
                self.assertIsNotNone(zero_trust)
    
    def test_multilingual_federated_components(self):
        """Test multilingual support in federated learning components."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                # Test federated coordinator
                coordinator = create_federated_coordinator(language=language)
                self.assertIsNotNone(coordinator)
                
                # Test model marketplace
                marketplace = create_model_marketplace(language=language)
                self.assertIsNotNone(marketplace)
                
                # Test collaborative evolution
                evolution = create_collaborative_evolution(language=language)
                self.assertIsNotNone(evolution)
    
    def test_multilingual_vision_components(self):
        """Test multilingual support in computer vision components."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                # Test object detector
                detector = create_object_detector(language=language)
                self.assertIsNotNone(detector)
                
                # Test face recognizer
                face_recognizer = create_face_recognizer(language=language)
                self.assertIsNotNone(face_recognizer)
                
                # Test gesture processor
                gesture_processor = create_gesture_processor(language=language)
                self.assertIsNotNone(gesture_processor)
                
                # Test video analyzer
                video_analyzer = create_video_analyzer(language=language)
                self.assertIsNotNone(video_analyzer)
                
                # Test 3D vision engine
                vision_3d = create_3d_vision_engine(language=language)
                self.assertIsNotNone(vision_3d)
    
    def test_multilingual_genai_components(self):
        """Test multilingual support in generative AI components."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                # Test GenAI model
                genai_model = create_genai_model("gigachat3-702b", language=language)
                self.assertIsNotNone(genai_model)
                
                # Test diffusion model
                diffusion_model = create_diffusion_model(language=language)
                self.assertIsNotNone(diffusion_model)
                
                # Test speech processor
                speech_processor = create_speech_processor(language=language)
                self.assertIsNotNone(speech_processor)
                
                # Test multimodal model
                multimodal_model = create_multimodal_model(language=language)
                self.assertIsNotNone(multimodal_model)
    
    def test_multilingual_security_components(self):
        """Test multilingual support in security components."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                # Test DIDN
                didn = create_didn(language=language)
                self.assertIsNotNone(didn)
                
                # Test zero-trust model
                zero_trust = create_zero_trust_model(language=language)
                self.assertIsNotNone(zero_trust)
                
                # Test quantum-safe crypto
                quantum_safe = create_quantum_safe_crypto(language=language)
                self.assertIsNotNone(quantum_safe)
                
                # Test Kyber crypto
                kyber_crypto = create_kyber_crypto(language=language)
                self.assertIsNotNone(kyber_crypto)
                
                # Test Dilithium crypto
                dilithium_crypto = create_dilithium_crypto(language=language)
                self.assertIsNotNone(dilithium_crypto)
    
    def test_multilingual_protocol_components(self):
        """Test multilingual support in protocol components."""
        for language in self.supported_languages:
            with self.subTest(language=language):
                # Test QMP protocol
                qmp = create_qmp_protocol(language=language)
                self.assertIsNotNone(qmp)
                
                # Test Post-DNS
                post_dns = create_post_dns(language=language)
                self.assertIsNotNone(post_dns)
                
                # Test Zero-DNS
                zero_dns = create_zero_dns(language=language)
                self.assertIsNotNone(zero_dns)
                
                # Test quantum signature
                quantum_signature = create_quantum_signature(language=language)
                self.assertIsNotNone(quantum_signature)
                
                # Test mesh network
                mesh_network = create_mesh_network(language=language)
                self.assertIsNotNone(mesh_network)
    
    def test_multilingual_performance_caching(self):
        """Test performance optimization with caching for multilingual features."""
        import time
        
        # Test translation caching
        translator = TranslationManager('ru')
        
        # First access (no cache)
        start_time = time.time()
        translation1 = translator.translate("welcome_message", 'ru')
        first_time = time.time() - start_time
        
        # Second access (cached)
        start_time = time.time()
        translation2 = translator.translate("welcome_message", 'ru')
        second_time = time.time() - start_time
        
        # Second access should be faster (cached)
        self.assertLess(second_time, first_time * 0.5)  # At least 2x faster
        self.assertEqual(translation1, translation2)
        
        # Test vocabulary caching
        vocabulary = VocabularyManager('zh')
        
        # First access (no cache)
        start_time = time.time()
        term1 = vocabulary.translate_term("quantum_computing", 'zh')
        first_time = time.time() - start_time
        
        # Second access (cached)
        start_time = time.time()
        term2 = vocabulary.translate_term("quantum_computing", 'zh')
        second_time = time.time() - start_time
        
        # Second access should be faster (cached)
        self.assertLess(second_time, first_time * 0.5)  # At least 2x faster
        self.assertEqual(term1, term2)
    
    def test_multilingual_resource_loading(self):
        """Test resource loading for multilingual features."""
        manager = ResourceManager()
        
        # Test loading translation resources
        for language in self.supported_languages:
            with self.subTest(language=language):
                translations = manager.load_translations(language)
                self.assertIsNotNone(translations)
                self.assertIsInstance(translations, dict)
                self.assertGreater(len(translations), 0)
        
        # Test loading vocabulary resources
        for language in self.supported_languages:
            with self.subTest(language=language):
                vocabularies = manager.load_vocabulary(language)
                self.assertIsNotNone(vocabularies)
                self.assertIsInstance(vocabularies, dict)
                self.assertGreater(len(vocabularies), 0)
    
    def test_multilingual_error_handling(self):
        """Test error handling in multilingual components."""
        # Test translation with invalid language
        translator = TranslationManager('en')
        translation = translator.translate("welcome_message", 'invalid_lang')
        # Should fall back to English
        self.assertIsNotNone(translation)
        
        # Test vocabulary translation with invalid term
        vocabulary = VocabularyManager('ru')
        term_translation = vocabulary.translate_term("nonexistent_term", 'ru')
        # Should return the original term
        self.assertEqual(term_translation, "nonexistent_term")
        
        # Test language detection with empty text
        detector = LanguageDetector()
        detected_lang = detector.detect_language("")
        # Should return default language
        self.assertEqual(detected_lang, 'en')
    
    def test_multilingual_thread_safety(self):
        """Test thread safety of multilingual components."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(language, worker_id):
            try:
                # Test various multilingual components
                translator = TranslationManager(language)
                vocabulary = VocabularyManager(language)
                
                # Perform translations
                translation = translator.translate("welcome_message", language)
                term = vocabulary.translate_term("quantum_computing", language)
                
                results.append((worker_id, language, translation, term))
                time.sleep(0.01)  # Small delay
                
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create multiple threads with different languages
        threads = []
        languages = ['en', 'ru', 'zh', 'ar'] * 5  # 20 threads total
        
        for i, language in enumerate(languages):
            thread = threading.Thread(target=worker, args=(language, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred in threads: {errors}")
        self.assertEqual(len(results), len(languages), f"Expected {len(languages)} results, got {len(results)}")
        
        # Verify all translations are valid
        for worker_id, language, translation, term in results:
            self.assertIsInstance(translation, str)
            self.assertIsInstance(term, str)
            self.assertNotEqual(translation, "welcome_message")  # Should be translated
            self.assertNotEqual(term, "quantum_computing")  # Should be translated
    
    def test_multilingual_character_support(self):
        """Test character support for all supported languages."""
        # Test Cyrillic characters (Russian)
        ru_translator = TranslationManager('ru')
        ru_translation = ru_translator.translate("welcome_message", 'ru')
        self.assertTrue(any('\u0400' <= char <= '\u04FF' for char in ru_translation))  # Cyrillic range
        
        # Test Chinese characters (Chinese)
        zh_vocabulary = VocabularyManager('zh')
        zh_term = zh_vocabulary.translate_term("quantum_computing", 'zh')
        self.assertTrue(any('\u4e00' <= char <= '\u9fff' for char in zh_term))  # CJK range
        
        # Test Arabic characters (Arabic)
        ar_translator = TranslationManager('ar')
        ar_translation = ar_translator.translate("welcome_message", 'ar')
        self.assertTrue(any('\u0600' <= char <= '\u06ff' for char in ar_translation))  # Arabic range
    
    def test_multilingual_integration_with_examples(self):
        """Test integration of multilingual features with example modules."""
        from aiplatform.examples.comprehensive_multimodal_example import MultimodalAI
        from aiplatform.examples.quantum_vision_example import QuantumVisionAI
        from aiplatform.examples.federated_quantum_example import FederatedQuantumAI
        from aiplatform.examples.security_example import SecurityExample
        from aiplatform.examples.protocols_example import ProtocolsExample
        
        # Test each example with all supported languages
        example_classes = [
            MultimodalAI,
            QuantumVisionAI,
            FederatedQuantumAI,
            SecurityExample,
            ProtocolsExample
        ]
        
        for language in self.supported_languages:
            with self.subTest(language=language):
                for example_class in example_classes:
                    with self.subTest(example_class=example_class.__name__):
                        # Initialize example with specific language
                        example = example_class(language=language)
                        self.assertIsNotNone(example)
                        
                        # Verify language is set correctly
                        self.assertEqual(example.language, language)


class TestMultilingualPerformance(unittest.TestCase):
    """Performance tests for multilingual features."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.translator = TranslationManager('en')
        self.vocabulary = VocabularyManager('en')
        self.test_keys = ["welcome_message", "initialization_complete", "error_occurred"] * 100
    
    def test_translation_performance(self):
        """Test translation performance with caching."""
        import time
        
        # Warm up cache
        for key in self.test_keys[:10]:
            self.translator.translate(key, 'ru')
        
        # Measure cached performance
        start_time = time.time()
        for key in self.test_keys:
            translation = self.translator.translate(key, 'ru')
        cached_time = time.time() - start_time
        
        # Performance should be reasonable (less than 100ms for 300 translations)
        self.assertLess(cached_time, 0.1)
    
    def test_vocabulary_performance(self):
        """Test vocabulary translation performance."""
        import time
        
        # Warm up cache
        for i in range(10):
            self.vocabulary.translate_term("quantum_computing", 'zh')
        
        # Measure cached performance
        start_time = time.time()
        for i in range(100):
            term = self.vocabulary.translate_term("quantum_computing", 'zh')
        cached_time = time.time() - start_time
        
        # Performance should be reasonable
        self.assertLess(cached_time, 0.05)
    
    def test_memory_usage(self):
        """Test memory usage of multilingual components."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple multilingual components
        translators = []
        vocabularies = []
        
        for i in range(100):
            translator = TranslationManager('en')
            vocabulary = VocabularyManager('en')
            translators.append(translator)
            vocabularies.append(vocabulary)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 100 components)
        self.assertLess(memory_increase, 100)


def create_test_suite():
    """Create and return the test suite."""
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestMultilingualSupport))
    suite.addTest(unittest.makeSuite(TestMultilingualPerformance))
    
    return suite


def main():
    """Run the multilingual test suite."""
    # Create test suite
    suite = create_test_suite()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())