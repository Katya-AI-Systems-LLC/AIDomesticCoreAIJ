"""
Text Processor
==============

Advanced text processing and understanding.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextAnalysis:
    """Text analysis result."""
    language: str
    sentiment: str
    sentiment_score: float
    entities: List[Dict[str, Any]]
    keywords: List[str]
    summary: Optional[str]
    embedding: np.ndarray


class TextProcessor:
    """
    Advanced text processing.
    
    Features:
    - Language detection
    - Sentiment analysis
    - Named entity recognition
    - Keyword extraction
    - Text summarization
    - Embedding generation
    
    Example:
        >>> processor = TextProcessor()
        >>> result = processor.analyze("This is a great product!")
        >>> print(f"Sentiment: {result.sentiment}")
    """
    
    SUPPORTED_LANGUAGES = ["en", "ru", "zh", "ar", "es", "fr", "de", "ja"]
    
    def __init__(self, model: str = "default",
                 language: str = "en"):
        """
        Initialize text processor.
        
        Args:
            model: Model name
            language: Default language
        """
        self.model = model
        self.default_language = language
        
        self._embedding_dim = 768
        
        logger.info(f"Text processor initialized: {model}")
    
    def analyze(self, text: str) -> TextAnalysis:
        """
        Analyze text comprehensively.
        
        Args:
            text: Input text
            
        Returns:
            TextAnalysis result
        """
        # Detect language
        language = self.detect_language(text)
        
        # Sentiment analysis
        sentiment, score = self.analyze_sentiment(text)
        
        # Entity extraction
        entities = self.extract_entities(text)
        
        # Keyword extraction
        keywords = self.extract_keywords(text)
        
        # Generate embedding
        embedding = self.get_embedding(text)
        
        # Summarize if long enough
        summary = None
        if len(text.split()) > 50:
            summary = self.summarize(text)
        
        return TextAnalysis(
            language=language,
            sentiment=sentiment,
            sentiment_score=score,
            entities=entities,
            keywords=keywords,
            summary=summary,
            embedding=embedding
        )
    
    def detect_language(self, text: str) -> str:
        """
        Detect text language.
        
        Args:
            text: Input text
            
        Returns:
            Language code
        """
        # Simple heuristic-based detection
        text_lower = text.lower()
        
        # Check for Cyrillic
        if re.search(r'[а-яё]', text_lower):
            return "ru"
        
        # Check for Chinese
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh"
        
        # Check for Arabic
        if re.search(r'[\u0600-\u06ff]', text):
            return "ar"
        
        # Check for Japanese
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "ja"
        
        # Default to English
        return "en"
    
    def analyze_sentiment(self, text: str) -> tuple:
        """
        Analyze sentiment.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment, score)
        """
        text_lower = text.lower()
        
        # Simple keyword-based sentiment
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", 
                         "fantastic", "love", "best", "happy", "beautiful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "hate",
                         "worst", "poor", "sad", "ugly", "disappointing"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return "neutral", 0.5
        
        score = (positive_count - negative_count + total) / (2 * total)
        
        if score > 0.6:
            sentiment = "positive"
        elif score < 0.4:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return sentiment, score
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities.
        
        Args:
            text: Input text
            
        Returns:
            List of entities
        """
        entities = []
        
        # Simple pattern-based extraction
        
        # Email
        emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
        for email in emails:
            entities.append({"type": "EMAIL", "text": email})
        
        # URL
        urls = re.findall(r'https?://\S+', text)
        for url in urls:
            entities.append({"type": "URL", "text": url})
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        for num in numbers[:5]:  # Limit
            entities.append({"type": "NUMBER", "text": num})
        
        # Capitalized words (potential names/organizations)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for cap in caps[:10]:
            if len(cap) > 2:
                entities.append({"type": "ENTITY", "text": cap})
        
        return entities
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum keywords to return
            
        Returns:
            List of keywords
        """
        # Simple TF-based keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "being", "have", "has", "had", "do", "does", "did", "will",
                    "would", "could", "should", "may", "might", "must", "shall",
                    "can", "need", "dare", "ought", "used", "to", "of", "in",
                    "for", "on", "with", "at", "by", "from", "as", "into",
                    "through", "during", "before", "after", "above", "below",
                    "between", "under", "again", "further", "then", "once",
                    "and", "but", "or", "nor", "so", "yet", "both", "either",
                    "neither", "not", "only", "own", "same", "than", "too",
                    "very", "just", "also", "now", "here", "there", "when",
                    "where", "why", "how", "all", "each", "every", "both",
                    "few", "more", "most", "other", "some", "such", "no",
                    "any", "this", "that", "these", "those", "what", "which"}
        
        words = [w for w in words if w not in stopwords]
        
        # Count frequencies
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        Summarize text.
        
        Args:
            text: Input text
            max_sentences: Maximum sentences in summary
            
        Returns:
            Summary text
        """
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text[:200] + "..."
        
        # Score sentences by keyword overlap
        keywords = set(self.extract_keywords(text, 20))
        
        scored = []
        for sent in sentences:
            words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower()))
            score = len(words & keywords)
            scored.append((score, sent))
        
        # Get top sentences
        scored.sort(reverse=True)
        top_sentences = [s for _, s in scored[:max_sentences]]
        
        return ". ".join(top_sentences) + "."
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Simulated embedding
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self._embedding_dim).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        return float(np.dot(emb1, emb2))
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def __repr__(self) -> str:
        return f"TextProcessor(model='{self.model}')"
