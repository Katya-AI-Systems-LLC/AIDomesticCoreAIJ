"""
Testing Framework for AIPlatform SDK Internationalization

This module provides a comprehensive testing framework for multilingual features.
"""

import unittest
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to the path to import aiplatform modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import i18n components
from aiplatform.i18n import (
    detect_language,
    translate,
    load_resource,
    get_vocabulary_manager
)
from aiplatform.i18n.language_detector import LanguageDetector
from aiplatform.i18n.translation_manager import TranslationManager
from aiplatform.i18n.resource_manager import ResourceManager
from aiplatform.i18n.vocabulary_manager import VocabularyManager

# Set up logging
logger = logging.getLogger(__name__)


class I18nTestCase(unittest.TestCase):
    """Base test case for internationalization features."""
    
    def setUp(self):
        """Set up test case."""
        self.supported_languages = ['en', 'ru', 'zh', 'ar']
        self.test_domains = ['quantum', 'qiz', 'federated', 'vision', 'genai', 'security', 'protocols']
        self.sample_texts = {
            'en': 'Hello, World!',
            'ru': 'Привет, мир!',
            'zh': '你好，世界！',
            'ar': 'مرحبا بالعالم!'
        }


class LanguageDetectionTest(I18nTestCase):
    """Test language detection functionality."""
    
    def setUp(self):
        """Set up language detection test."""
        super().setUp()
        self.detector = LanguageDetector()
    
    def test_english_detection(self):
        """Test English language detection."""
        text = "This is a sample English text for testing."
        detected_lang = self.detector.detect(text)
        self.assertEqual(detected_lang, 'en')
    
    def test_russian_detection(self):
        """Test Russian language detection."""
        text = "Это пример русского текста для тестирования."
        detected_lang = self.detector.detect(text)
        self.assertEqual(detected_lang, 'ru')
    
    def test_chinese_detection(self):
        """Test Chinese language detection."""
        text = "这是一个用于测试的中文示例文本。"
        detected_lang = self.detector.detect(text)
        self.assertEqual(detected_lang, 'zh')
    
    def test_arabic_detection(self):
        """Test Arabic language detection."""
        text = "هذا نص عربي عينة للاختبار."
        detected_lang = self.detector.detect(text)
        self.assertEqual(detected_lang, 'ar')
    
    def test_mixed_text_detection(self):
        """Test mixed text language detection."""
        text = "Hello 你好 مرحبا"
        detected_lang = self.detector.detect(text)
        # Should detect based on dominant script
        self.assertIn(detected_lang, ['en', 'zh', 'ar'])


class TranslationTest(I18nTestCase):
    """Test translation functionality."""
    
    def setUp(self):
        """Set up translation test."""
        super().setUp()
        self.translation_manager = TranslationManager()
    
    def test_english_to_russian_translation(self):
        """Test English to Russian translation."""
        text = "Quantum Computing"
        translated = translate(text, 'ru')
        self.assertIsInstance(translated, str)
        self.assertNotEqual(translated, text)
        # Should contain Cyrillic characters
        self.assertTrue(any('\u0400' <= char <= '\u04FF' for char in translated))
    
    def test_english_to_chinese_translation(self):
        """Test English to Chinese translation."""
        text = "Artificial Intelligence"
        translated = translate(text, 'zh')
        self.assertIsInstance(translated, str)
        self.assertNotEqual(translated, text)
        # Should contain Chinese characters
        self.assertTrue(any('\u4e00' <= char <= '\u9fff' for char in translated))
    
    def test_english_to_arabic_translation(self):
        """Test English to Arabic translation."""
        text = "Machine Learning"
        translated = translate(text, 'ar')
        self.assertIsInstance(translated, str)
        self.assertNotEqual(translated, text)
        # Should contain Arabic characters
        self.assertTrue(any('\u0600' <= char <= '\u06ff' for char in translated))
    
    def test_reverse_translation_consistency(self):
        """Test reverse translation consistency."""
        original = "Neural Network"
        # Translate to Russian and back
        russian = translate(original, 'ru')
        back_to_english = translate(russian, 'en')
        # Should be similar to original (not exact due to translation nuances)
        self.assertIsInstance(back_to_english, str)
    
    def test_translation_with_parameters(self):
        """Test translation with parameters."""
        text = "Hello {name}, welcome to {platform}!"
        # This would be handled by the translation system
        # For now, we test that it doesn't crash
        translated = translate(text, 'ru')
        self.assertIsInstance(translated, str)


class ResourceManagerTest(I18nTestCase):
    """Test resource management functionality."""
    
    def setUp(self):
        """Set up resource manager test."""
        super().setUp()
        self.resource_manager = ResourceManager()
    
    def test_resource_loading(self):
        """Test resource loading."""
        # Test loading a resource
        resource = self.resource_manager.load_resource('quick_start', 'en')
        self.assertIsNotNone(resource)
        self.assertIsInstance(resource, dict)
    
    def test_multilingual_resource_loading(self):
        """Test multilingual resource loading."""
        for lang in self.supported_languages:
            resource = self.resource_manager.load_resource('quick_start', lang)
            self.assertIsNotNone(resource)
            self.assertIsInstance(resource, dict)
            self.assertEqual(resource.get('language'), lang)
    
    def test_resource_caching(self):
        """Test resource caching."""
        # Load the same resource twice
        resource1 = self.resource_manager.load_resource('quick_start', 'en')
        resource2 = self.resource_manager.load_resource('quick_start', 'en')
        
        # Should be the same object (cached)
        self.assertIs(resource1, resource2)
    
    def test_resource_preloading(self):
        """Test resource preloading."""
        # Preload resources
        self.resource_manager.preload_resources(self.supported_languages, ['quick_start', 'api_reference'])
        
        # Check cache statistics
        stats = self.resource_manager.get_cache_stats()
        self.assertGreater(stats['hits'] + stats['misses'], 0)


class VocabularyManagerTest(I18nTestCase):
    """Test vocabulary management functionality."""
    
    def setUp(self):
        """Set up vocabulary manager test."""
        super().setUp()
        self.vocabulary_manager = get_vocabulary_manager()
    
    def test_technical_term_translation(self):
        """Test technical term translation."""
        term = "Quantum Computing"
        for lang in self.supported_languages:
            translated = self.vocabulary_manager.translate_term(term, 'quantum', lang)
            self.assertIsInstance(translated, str)
            self.assertNotEqual(translated, term if lang != 'en' else term)
    
    def test_domain_vocabulary_access(self):
        """Test domain vocabulary access."""
        for domain in self.test_domains:
            vocabulary = self.vocabulary_manager.get_domain_vocabulary(domain)
            self.assertIsNotNone(vocabulary)
            self.assertIsInstance(vocabulary, dict)
    
    def test_vocabulary_consistency(self):
        """Test vocabulary consistency across languages."""
        term = "Artificial Intelligence"
        translations = {}
        
        for lang in self.supported_languages:
            translated = self.vocabulary_manager.translate_term(term, 'genai', lang)
            translations[lang] = translated
            self.assertIsInstance(translated, str)
        
        # All translations should be different from the original (except English)
        non_english_translations = [t for lang, t in translations.items() if lang != 'en']
        self.assertTrue(all(t != term for t in non_english_translations))


class PerformanceTest(I18nTestCase):
    """Test performance of internationalization features."""
    
    def setUp(self):
        """Set up performance test."""
        super().setUp()
        self.resource_manager = ResourceManager()
    
    def test_translation_performance(self):
        """Test translation performance."""
        import time
        
        # Measure translation time
        start_time = time.time()
        for i in range(100):
            translate("Quantum Computing", 'ru')
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second for 100 translations)
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)
    
    def test_resource_loading_performance(self):
        """Test resource loading performance."""
        import time
        
        # Measure resource loading time
        start_time = time.time()
        for i in range(50):
            self.resource_manager.load_resource('quick_start', 'en')
        end_time = time.time()
        
        # Should be fast due to caching (less than 0.5 seconds for 50 loads)
        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.5)
    
    def test_batch_translation_performance(self):
        """Test batch translation performance."""
        import time
        
        # Test batch translation
        texts = ["Quantum Computing", "Artificial Intelligence", "Machine Learning"] * 10
        
        start_time = time.time()
        # This would use the batch translation feature
        for text in texts:
            translate(text, 'zh')
        end_time = time.time()
        
        # Should complete within reasonable time
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)


class IntegrationTest(I18nTestCase):
    """Test integration of internationalization features."""
    
    def test_full_i18n_workflow(self):
        """Test full internationalization workflow."""
        # Detect language
        text = "Привет, это русский текст!"
        detected_lang = detect_language(text)
        self.assertEqual(detected_lang, 'ru')
        
        # Translate to English
        translated = translate(text, 'en')
        self.assertIsInstance(translated, str)
        self.assertNotEqual(translated, text)
        
        # Load resources in detected language
        resource_manager = ResourceManager()
        resource = resource_manager.load_resource('quick_start', detected_lang)
        self.assertIsNotNone(resource)
    
    def test_multilingual_resource_access(self):
        """Test multilingual resource access."""
        resource_manager = ResourceManager()
        
        # Access resources in all supported languages
        for lang in self.supported_languages:
            quick_start = resource_manager.load_resource('quick_start', lang)
            api_reference = resource_manager.load_resource('api_reference', lang)
            
            self.assertIsNotNone(quick_start)
            self.assertIsNotNone(api_reference)
            self.assertEqual(quick_start.get('language'), lang)
            self.assertEqual(api_reference.get('language'), lang)
    
    def test_vocabulary_integration(self):
        """Test vocabulary integration with translation."""
        vocabulary_manager = get_vocabulary_manager()
        
        # Test technical terms in different contexts
        term = "Neural Network"
        for domain in self.test_domains[:3]:  # Test first 3 domains
            for lang in self.supported_languages:
                translated = vocabulary_manager.translate_term(term, domain, lang)
                self.assertIsInstance(translated, str)


class ErrorHandlingTest(I18nTestCase):
    """Test error handling in internationalization features."""
    
    def test_invalid_language_handling(self):
        """Test handling of invalid languages."""
        # Should handle gracefully
        result = translate("Hello World", 'invalid_lang')
        # Should return original text or a default translation
        self.assertIsInstance(result, str)
    
    def test_missing_resource_handling(self):
        """Test handling of missing resources."""
        resource_manager = ResourceManager()
        # Should handle gracefully
        resource = resource_manager.load_resource('nonexistent_resource', 'en')
        # Should return a default resource or None
        self.assertIsNotNone(resource)  # Should at least return a default
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        result = translate("", 'ru')
        self.assertIsInstance(result, str)
        self.assertEqual(result, "")
    
    def test_none_text_handling(self):
        """Test handling of None text."""
        result = translate(None, 'ru')
        self.assertIsNone(result)


class CompatibilityTest(I18nTestCase):
    """Test compatibility with different environments."""
    
    def test_unicode_support(self):
        """Test Unicode support."""
        # Test with various Unicode characters
        unicode_texts = [
            "🚀 Quantum Computing 🚀",
            "🌟 Искусственный Интеллект 🌟",
            "💫 人工智能 💫",
            "✨ الذكاء الاصطناعي ✨"
        ]
        
        for text in unicode_texts:
            for lang in self.supported_languages:
                translated = translate(text, lang)
                self.assertIsInstance(translated, str)
    
    def test_special_character_handling(self):
        """Test special character handling."""
        special_texts = [
            "Quantum Computing & AI",
            "Machine Learning @ 2025",
            "Neural Networks + Deep Learning",
            "Data Science = Statistics + Programming"
        ]
        
        for text in special_texts:
            for lang in self.supported_languages:
                translated = translate(text, lang)
                self.assertIsInstance(translated, str)


def create_test_suite():
    """Create a test suite with all internationalization tests."""
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_cases = [
        LanguageDetectionTest,
        TranslationTest,
        ResourceManagerTest,
        VocabularyManagerTest,
        PerformanceTest,
        IntegrationTest,
        ErrorHandlingTest,
        CompatibilityTest
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    
    return suite


def run_i18n_tests():
    """Run all internationalization tests."""
    print("Running AIPlatform Internationalization Tests...")
    print("=" * 50)
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 50)
    
    return result.wasSuccessful()


# Example usage
if __name__ == "__main__":
    # Run the tests
    success = run_i18n_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)