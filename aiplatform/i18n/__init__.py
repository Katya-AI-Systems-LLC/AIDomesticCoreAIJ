"""
Internationalization package for AIPlatform SDK

This module provides internationalization support for the AIPlatform Quantum Infrastructure Zero SDK,
enabling multilingual capabilities for Russian, Chinese, and Arabic languages.
"""

from .language_detector import LanguageDetector
from .translation_manager import TranslationManager
from .resource_manager import ResourceManager
from .vocabulary_manager import VocabularyManager

__all__ = [
    'LanguageDetector',
    'TranslationManager',
    'ResourceManager',
    'VocabularyManager',
    'get_translator',
    'translate',
    'detect_language'
]

# Global instances
_language_detector = None
_translation_manager = None
_resource_manager = None
_vocabulary_manager = None


def get_translator():
    """
    Get the global translation manager instance.
    
    Returns:
        TranslationManager: Global translation manager instance
    """
    global _translation_manager
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    return _translation_manager


def translate(key, language=None, **kwargs):
    """
    Translate a key to the specified language.
    
    Args:
        key (str): Translation key
        language (str, optional): Target language code
        **kwargs: Additional parameters for translation
        
    Returns:
        str: Translated text
    """
    translator = get_translator()
    return translator.translate(key, language, **kwargs)


def detect_language(text):
    """
    Detect the language of the given text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Detected language code
    """
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    return _language_detector.detect_language(text)


def initialize_i18n():
    """
    Initialize the internationalization system.
    """
    global _language_detector, _translation_manager, _resource_manager, _vocabulary_manager
    
    if _language_detector is None:
        _language_detector = LanguageDetector()
    
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    
    if _vocabulary_manager is None:
        _vocabulary_manager = VocabularyManager()


# Initialize on import
initialize_i18n()