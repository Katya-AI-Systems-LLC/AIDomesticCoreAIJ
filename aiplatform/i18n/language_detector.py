"""
Language Detection System for AIPlatform SDK

This module provides language detection capabilities for the AIPlatform Quantum Infrastructure Zero SDK,
supporting automatic detection of Russian, Chinese, and Arabic languages.
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter


class LanguageDetector:
    """Language detection system for multilingual support."""
    
    def __init__(self):
        """Initialize the language detector."""
        self.supported_languages = ['en', 'ru', 'zh', 'ar']
        self.default_language = 'en'
        
        # Language-specific character sets
        self.character_patterns = {
            'ru': re.compile(r'[а-яА-ЯёЁ]'),
            'zh': re.compile(r'[\u4e00-\u9fff]'),
            'ar': re.compile(r'[\u0600-\u06ff\u0750-\u077f]'),
            'en': re.compile(r'[a-zA-Z]')
        }
        
        # Language-specific word patterns
        self.word_patterns = {
            'ru': re.compile(r'\b[а-яА-ЯёЁ]{2,}\b'),
            'zh': re.compile(r'[\u4e00-\u9fff]{1,}'),
            'ar': re.compile(r'[\u0600-\u06ff\u0750-\u077f]{2,}'),
            'en': re.compile(r'\b[a-zA-Z]{2,}\b')
        }
        
        # Common words for each language (simplified for demonstration)
        self.common_words = {
            'en': {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'},
            'ru': {'и', 'в', 'не', 'на', 'с', 'как', 'а', 'то', 'что', 'это'},
            'zh': {'的', '一', '是', '在', '不', '了', '有', '和', '人', '这'},
            'ar': {'الذي', 'التي', 'كان', 'هذا', 'هذه', 'ذلك', 'تلك', 'من', 'على', 'في'}
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code
        """
        if not text or not isinstance(text, str):
            return self.default_language
        
        # Clean text for analysis
        clean_text = text.strip()
        if not clean_text:
            return self.default_language
        
        # Try multiple detection methods
        detection_methods = [
            self._detect_by_characters,
            self._detect_by_words,
            self._detect_by_common_words
        ]
        
        # Collect results from all methods
        results = []
        for method in detection_methods:
            try:
                result = method(clean_text)
                if result:
                    results.append(result)
            except Exception:
                # Continue with other methods if one fails
                continue
        
        # Return the most common result or default
        if results:
            counter = Counter(results)
            return counter.most_common(1)[0][0]
        
        return self.default_language
    
    def _detect_by_characters(self, text: str) -> Optional[str]:
        """
        Detect language by analyzing character sets.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code or None
        """
        # Count characters for each language
        char_counts = {}
        for lang, pattern in self.character_patterns.items():
            matches = pattern.findall(text)
            char_counts[lang] = len(matches)
        
        # Find language with most matching characters
        if char_counts:
            total_chars = sum(char_counts.values())
            if total_chars > 0:
                # Calculate percentages
                percentages = {lang: (count / total_chars) for lang, count in char_counts.items()}
                
                # Return language with highest percentage if significant
                max_lang = max(percentages, key=percentages.get)
                if percentages[max_lang] > 0.3:  # At least 30% of characters match
                    return max_lang
        
        return None
    
    def _detect_by_words(self, text: str) -> Optional[str]:
        """
        Detect language by analyzing word patterns.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code or None
        """
        # Find words for each language
        word_matches = {}
        for lang, pattern in self.word_patterns.items():
            matches = pattern.findall(text)
            word_matches[lang] = len(matches)
        
        # Find language with most matching words
        if word_matches:
            max_lang = max(word_matches, key=word_matches.get)
            if word_matches[max_lang] > 0:
                return max_lang
        
        return None
    
    def _detect_by_common_words(self, text: str) -> Optional[str]:
        """
        Detect language by matching common words.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code or None
        """
        # Convert to lowercase for English
        lower_text = text.lower()
        
        # Check each language's common words
        word_scores = {}
        for lang, words in self.common_words.items():
            score = 0
            if lang == 'en':
                # For English, use lowercase text
                text_to_check = lower_text
            else:
                # For other languages, use original text
                text_to_check = text
            
            # Count matches
            for word in words:
                if lang == 'en':
                    # Simple word boundary check for English
                    score += text_to_check.count(f" {word} ") + text_to_check.count(f" {word}.") + text_to_check.count(f" {word},")
                else:
                    # Simple substring check for other languages
                    score += text_to_check.count(word)
            
            word_scores[lang] = score
        
        # Find language with highest score
        if word_scores:
            max_lang = max(word_scores, key=word_scores.get)
            if word_scores[max_lang] > 0:
                return max_lang
        
        return None
    
    def get_confidence(self, text: str) -> Dict[str, float]:
        """
        Get confidence scores for all supported languages.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Language codes mapped to confidence scores (0.0 to 1.0)
        """
        if not text or not isinstance(text, str):
            return {lang: 0.0 for lang in self.supported_languages}
        
        # Clean text
        clean_text = text.strip()
        if not clean_text:
            return {lang: 0.0 for lang in self.supported_languages}
        
        # Get scores from different methods
        char_scores = self._get_character_scores(clean_text)
        word_scores = self._get_word_scores(clean_text)
        common_word_scores = self._get_common_word_scores(clean_text)
        
        # Combine scores (weighted average)
        final_scores = {}
        for lang in self.supported_languages:
            # Weighted combination of different methods
            final_score = (
                char_scores.get(lang, 0) * 0.4 +
                word_scores.get(lang, 0) * 0.3 +
                common_word_scores.get(lang, 0) * 0.3
            )
            final_scores[lang] = min(1.0, final_score)  # Cap at 1.0
        
        return final_scores
    
    def _get_character_scores(self, text: str) -> Dict[str, float]:
        """Get character-based language scores."""
        char_counts = {}
        for lang, pattern in self.character_patterns.items():
            matches = pattern.findall(text)
            char_counts[lang] = len(matches)
        
        # Convert to percentages
        total_chars = sum(char_counts.values())
        if total_chars > 0:
            return {lang: count / total_chars for lang, count in char_counts.items()}
        
        return {lang: 0.0 for lang in self.supported_languages}
    
    def _get_word_scores(self, text: str) -> Dict[str, float]:
        """Get word-based language scores."""
        word_counts = {}
        for lang, pattern in self.word_patterns.items():
            matches = pattern.findall(text)
            word_counts[lang] = len(matches)
        
        # Convert to percentages
        total_words = sum(word_counts.values())
        if total_words > 0:
            return {lang: count / total_words for lang, count in word_counts.items()}
        
        return {lang: 0.0 for lang in self.supported_languages}
    
    def _get_common_word_scores(self, text: str) -> Dict[str, float]:
        """Get common word-based language scores."""
        word_scores = {}
        lower_text = text.lower()
        
        for lang, words in self.common_words.items():
            score = 0
            if lang == 'en':
                text_to_check = lower_text
            else:
                text_to_check = text
            
            for word in words:
                if lang == 'en':
                    score += text_to_check.count(f" {word} ") + text_to_check.count(f" {word}.") + text_to_check.count(f" {word},")
                else:
                    score += text_to_check.count(word)
            
            # Normalize by number of common words
            word_scores[lang] = score / len(words) if words else 0.0
        
        # Convert to relative scores
        max_score = max(word_scores.values()) if word_scores else 0
        if max_score > 0:
            return {lang: score / max_score for lang, score in word_scores.items()}
        
        return {lang: 0.0 for lang in self.supported_languages}
    
    def is_supported_language(self, language: str) -> bool:
        """
        Check if a language is supported.
        
        Args:
            language (str): Language code to check
            
        Returns:
            bool: True if language is supported
        """
        return language in self.supported_languages


# Global instance
_detector = None


def get_language_detector() -> LanguageDetector:
    """
    Get the global language detector instance.
    
    Returns:
        LanguageDetector: Global language detector instance
    """
    global _detector
    if _detector is None:
        _detector = LanguageDetector()
    return _detector