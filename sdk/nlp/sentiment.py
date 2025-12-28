"""
Sentiment Analyzer
==================

Analyze sentiment and emotions in text.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class Sentiment(Enum):
    """Sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Emotion(Enum):
    """Emotion labels."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    sentiment: Sentiment
    confidence: float
    scores: Dict[str, float]
    emotions: Dict[Emotion, float]


class SentimentAnalyzer:
    """
    Sentiment and emotion analyzer.
    
    Features:
    - Sentiment classification
    - Emotion detection
    - Aspect-based sentiment
    - Multi-language support
    - Fine-grained scoring
    
    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.analyze("I love this product!")
        >>> print(result.sentiment)  # POSITIVE
    """
    
    # Lexicon-based word lists
    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'like', 'best', 'happy', 'joy', 'beautiful', 'awesome',
        'perfect', 'brilliant', 'outstanding', 'superb', 'delightful',
        'pleasant', 'positive', 'nice', 'enjoy', 'satisfied', 'recommend'
    }
    
    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst',
        'poor', 'disappointing', 'frustrating', 'annoying', 'angry', 'sad',
        'unhappy', 'negative', 'fail', 'broken', 'useless', 'waste',
        'problem', 'issue', 'bug', 'error', 'crash', 'slow', 'expensive'
    }
    
    INTENSIFIERS = {'very', 'really', 'extremely', 'absolutely', 'totally'}
    NEGATORS = {'not', "n't", 'never', 'no', 'without', 'hardly'}
    
    EMOTION_WORDS = {
        Emotion.JOY: {'happy', 'joy', 'excited', 'delighted', 'pleased', 'glad'},
        Emotion.SADNESS: {'sad', 'unhappy', 'depressed', 'disappointed', 'sorry'},
        Emotion.ANGER: {'angry', 'furious', 'annoyed', 'frustrated', 'mad'},
        Emotion.FEAR: {'afraid', 'scared', 'worried', 'anxious', 'nervous'},
        Emotion.SURPRISE: {'surprised', 'amazed', 'shocked', 'astonished'},
        Emotion.DISGUST: {'disgusted', 'sick', 'gross', 'awful', 'nasty'},
        Emotion.TRUST: {'trust', 'believe', 'confident', 'reliable', 'honest'},
        Emotion.ANTICIPATION: {'expect', 'hope', 'await', 'anticipate', 'eager'}
    }
    
    def __init__(self, use_ml: bool = False,
                 model_name: str = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_ml: Use ML-based analysis
            model_name: Model name
        """
        self.use_ml = use_ml
        self.model_name = model_name
        
        self._ml_model = None
        
        if use_ml:
            self._load_ml_model()
        
        logger.info(f"Sentiment Analyzer initialized (ML={use_ml})")
    
    def _load_ml_model(self):
        """Load ML model."""
        try:
            from transformers import pipeline
            self._ml_model = pipeline(
                "sentiment-analysis",
                model=self.model_name or "distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Transformers model loaded")
        except ImportError:
            logger.warning("Transformers not installed, using lexicon-based analysis")
            self.use_ml = False
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            SentimentResult
        """
        if self.use_ml and self._ml_model:
            return self._analyze_ml(text)
        else:
            return self._analyze_lexicon(text)
    
    def _analyze_ml(self, text: str) -> SentimentResult:
        """ML-based analysis."""
        result = self._ml_model(text[:512])[0]
        
        label = result['label'].lower()
        confidence = result['score']
        
        if label == 'positive':
            sentiment = Sentiment.POSITIVE
            scores = {'positive': confidence, 'negative': 1 - confidence, 'neutral': 0}
        elif label == 'negative':
            sentiment = Sentiment.NEGATIVE
            scores = {'positive': 1 - confidence, 'negative': confidence, 'neutral': 0}
        else:
            sentiment = Sentiment.NEUTRAL
            scores = {'positive': 0, 'negative': 0, 'neutral': confidence}
        
        emotions = self._detect_emotions(text)
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            scores=scores,
            emotions=emotions
        )
    
    def _analyze_lexicon(self, text: str) -> SentimentResult:
        """Lexicon-based analysis."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_score = 0
        negative_score = 0
        
        negation = False
        intensifier = 1.0
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.NEGATORS:
                negation = True
                continue
            
            # Check for intensifier
            if word in self.INTENSIFIERS:
                intensifier = 1.5
                continue
            
            # Score word
            if word in self.POSITIVE_WORDS:
                if negation:
                    negative_score += 1 * intensifier
                else:
                    positive_score += 1 * intensifier
            elif word in self.NEGATIVE_WORDS:
                if negation:
                    positive_score += 1 * intensifier
                else:
                    negative_score += 1 * intensifier
            
            # Reset modifiers after word
            negation = False
            intensifier = 1.0
        
        total = positive_score + negative_score
        
        if total == 0:
            sentiment = Sentiment.NEUTRAL
            confidence = 0.5
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        else:
            pos_ratio = positive_score / total
            neg_ratio = negative_score / total
            
            if pos_ratio > 0.6:
                sentiment = Sentiment.POSITIVE
                confidence = pos_ratio
            elif neg_ratio > 0.6:
                sentiment = Sentiment.NEGATIVE
                confidence = neg_ratio
            elif abs(pos_ratio - neg_ratio) < 0.2:
                sentiment = Sentiment.MIXED
                confidence = 0.5
            else:
                sentiment = Sentiment.NEUTRAL
                confidence = 0.5
            
            scores = {
                'positive': pos_ratio,
                'negative': neg_ratio,
                'neutral': 1 - (pos_ratio + neg_ratio) / 2
            }
        
        emotions = self._detect_emotions(text)
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            scores=scores,
            emotions=emotions
        )
    
    def _detect_emotions(self, text: str) -> Dict[Emotion, float]:
        """Detect emotions in text."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        emotions = {}
        
        for emotion, emotion_words in self.EMOTION_WORDS.items():
            overlap = len(words & emotion_words)
            if overlap > 0:
                emotions[emotion] = min(overlap / 3, 1.0)
        
        return emotions
    
    def analyze_aspects(self, text: str,
                        aspects: List[str]) -> Dict[str, SentimentResult]:
        """
        Aspect-based sentiment analysis.
        
        Args:
            text: Input text
            aspects: Aspects to analyze
            
        Returns:
            Dict of aspect -> SentimentResult
        """
        results = {}
        
        sentences = re.split(r'[.!?]', text)
        
        for aspect in aspects:
            aspect_lower = aspect.lower()
            relevant_sentences = [
                s for s in sentences if aspect_lower in s.lower()
            ]
            
            if relevant_sentences:
                combined = ' '.join(relevant_sentences)
                results[aspect] = self.analyze(combined)
            else:
                results[aspect] = SentimentResult(
                    sentiment=Sentiment.NEUTRAL,
                    confidence=0.0,
                    scores={'positive': 0, 'negative': 0, 'neutral': 1},
                    emotions={}
                )
        
        return results
    
    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def __repr__(self) -> str:
        return f"SentimentAnalyzer(ml={self.use_ml})"
