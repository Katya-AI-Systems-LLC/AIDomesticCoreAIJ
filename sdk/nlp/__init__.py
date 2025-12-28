"""
NLP Module
==========

Natural Language Processing capabilities.

Features:
- Text embeddings
- Semantic search
- Named Entity Recognition
- Sentiment analysis
- Language translation
- RAG (Retrieval Augmented Generation)
"""

from .embeddings import EmbeddingEngine
from .semantic_search import SemanticSearch
from .ner import NERExtractor
from .sentiment import SentimentAnalyzer
from .translation import Translator
from .rag import RAGPipeline

__all__ = [
    "EmbeddingEngine",
    "SemanticSearch",
    "NERExtractor",
    "SentimentAnalyzer",
    "Translator",
    "RAGPipeline"
]
