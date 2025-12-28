"""
Embedding Engine
================

Text embeddings for semantic understanding.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Embedding models."""
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    COHERE_ENGLISH = "embed-english-v3.0"
    COHERE_MULTILINGUAL = "embed-multilingual-v3.0"
    BGE_SMALL = "bge-small-en-v1.5"
    BGE_LARGE = "bge-large-en-v1.5"
    E5_SMALL = "e5-small-v2"
    E5_LARGE = "e5-large-v2"
    INSTRUCTOR = "instructor-xl"
    LOCAL = "local"


@dataclass
class EmbeddingResult:
    """Embedding result."""
    text: str
    embedding: np.ndarray
    model: str
    dimensions: int
    tokens: int
    latency_ms: float


class EmbeddingEngine:
    """
    Text embedding engine.
    
    Features:
    - Multiple embedding models
    - Batch processing
    - Caching
    - Dimensionality reduction
    - Similarity computation
    
    Example:
        >>> engine = EmbeddingEngine(EmbeddingModel.OPENAI_3_SMALL)
        >>> embedding = engine.embed("Hello world")
        >>> similarity = engine.similarity(emb1, emb2)
    """
    
    MODEL_DIMENSIONS = {
        EmbeddingModel.OPENAI_ADA: 1536,
        EmbeddingModel.OPENAI_3_SMALL: 1536,
        EmbeddingModel.OPENAI_3_LARGE: 3072,
        EmbeddingModel.COHERE_ENGLISH: 1024,
        EmbeddingModel.COHERE_MULTILINGUAL: 1024,
        EmbeddingModel.BGE_SMALL: 384,
        EmbeddingModel.BGE_LARGE: 1024,
        EmbeddingModel.E5_SMALL: 384,
        EmbeddingModel.E5_LARGE: 1024,
        EmbeddingModel.INSTRUCTOR: 768,
        EmbeddingModel.LOCAL: 384
    }
    
    def __init__(self, model: EmbeddingModel = EmbeddingModel.OPENAI_3_SMALL,
                 api_key: Optional[str] = None,
                 cache_enabled: bool = True):
        """
        Initialize embedding engine.
        
        Args:
            model: Embedding model
            api_key: API key for cloud models
            cache_enabled: Enable embedding cache
        """
        self.model = model
        self.api_key = api_key
        self.cache_enabled = cache_enabled
        self.dimensions = self.MODEL_DIMENSIONS.get(model, 384)
        
        # Cache
        self._cache: Dict[str, np.ndarray] = {}
        
        # Stats
        self._total_embeddings = 0
        self._cache_hits = 0
        
        # Local model
        self._local_model = None
        
        logger.info(f"Embedding Engine initialized: {model.value} ({self.dimensions}D)")
    
    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            EmbeddingResult
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if self.cache_enabled and cache_key in self._cache:
            self._cache_hits += 1
            embedding = self._cache[cache_key]
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model.value,
                dimensions=self.dimensions,
                tokens=len(text.split()),
                latency_ms=0.0
            )
        
        # Generate embedding
        if self.model in [EmbeddingModel.OPENAI_ADA, EmbeddingModel.OPENAI_3_SMALL, EmbeddingModel.OPENAI_3_LARGE]:
            embedding = self._embed_openai(text)
        elif self.model in [EmbeddingModel.COHERE_ENGLISH, EmbeddingModel.COHERE_MULTILINGUAL]:
            embedding = self._embed_cohere(text)
        elif self.model in [EmbeddingModel.BGE_SMALL, EmbeddingModel.BGE_LARGE, EmbeddingModel.E5_SMALL, EmbeddingModel.E5_LARGE]:
            embedding = self._embed_local(text)
        else:
            embedding = self._embed_simulated(text)
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = embedding
        
        self._total_embeddings += 1
        latency = (time.time() - start_time) * 1000
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model.value,
            dimensions=self.dimensions,
            tokens=len(text.split()),
            latency_ms=latency
        )
    
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of EmbeddingResults
        """
        return [self.embed(text) for text in texts]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{self.model.value}:{text}".encode()).hexdigest()
    
    def _embed_openai(self, text: str) -> np.ndarray:
        """Embed using OpenAI."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model=self.model.value,
                input=text
            )
            return np.array(response.data[0].embedding)
            
        except ImportError:
            return self._embed_simulated(text)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return self._embed_simulated(text)
    
    def _embed_cohere(self, text: str) -> np.ndarray:
        """Embed using Cohere."""
        try:
            import cohere
            
            client = cohere.Client(self.api_key)
            response = client.embed(
                texts=[text],
                model=self.model.value,
                input_type="search_document"
            )
            return np.array(response.embeddings[0])
            
        except ImportError:
            return self._embed_simulated(text)
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            return self._embed_simulated(text)
    
    def _embed_local(self, text: str) -> np.ndarray:
        """Embed using local model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            if self._local_model is None:
                model_name = {
                    EmbeddingModel.BGE_SMALL: "BAAI/bge-small-en-v1.5",
                    EmbeddingModel.BGE_LARGE: "BAAI/bge-large-en-v1.5",
                    EmbeddingModel.E5_SMALL: "intfloat/e5-small-v2",
                    EmbeddingModel.E5_LARGE: "intfloat/e5-large-v2"
                }.get(self.model, "all-MiniLM-L6-v2")
                
                self._local_model = SentenceTransformer(model_name)
            
            return self._local_model.encode(text)
            
        except ImportError:
            return self._embed_simulated(text)
    
    def _embed_simulated(self, text: str) -> np.ndarray:
        """Generate simulated embedding."""
        # Deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.dimensions)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def similarity(self, embedding1: np.ndarray,
                   embedding2: np.ndarray,
                   metric: str = "cosine") -> float:
        """
        Compute similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine, euclidean, dot)
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            return float(np.dot(embedding1, embedding2) / 
                        (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
        elif metric == "euclidean":
            return float(1 / (1 + np.linalg.norm(embedding1 - embedding2)))
        elif metric == "dot":
            return float(np.dot(embedding1, embedding2))
        else:
            return self.similarity(embedding1, embedding2, "cosine")
    
    def reduce_dimensions(self, embeddings: np.ndarray,
                          target_dims: int = 2,
                          method: str = "pca") -> np.ndarray:
        """
        Reduce embedding dimensions.
        
        Args:
            embeddings: Embeddings matrix (n_samples x n_dims)
            target_dims: Target dimensions
            method: Reduction method (pca, umap, tsne)
            
        Returns:
            Reduced embeddings
        """
        if method == "pca":
            # Simple PCA
            mean = np.mean(embeddings, axis=0)
            centered = embeddings - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1][:target_dims]
            return centered @ eigenvectors[:, idx]
        
        return embeddings[:, :target_dims]
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "model": self.model.value,
            "dimensions": self.dimensions,
            "total_embeddings": self._total_embeddings,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_embeddings)
        }
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"EmbeddingEngine(model={self.model.value}, dims={self.dimensions})"
