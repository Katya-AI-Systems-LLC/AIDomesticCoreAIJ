"""
Semantic Search
===============

Vector-based semantic search.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Searchable document."""
    doc_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SemanticSearch:
    """
    Semantic search engine.
    
    Features:
    - Vector similarity search
    - Hybrid search (vector + keyword)
    - Filtering and facets
    - Index management
    - Multiple similarity metrics
    
    Example:
        >>> search = SemanticSearch(embedding_engine)
        >>> search.add_documents(docs)
        >>> results = search.search("find relevant documents")
    """
    
    def __init__(self, embedding_engine: Any = None,
                 similarity_metric: str = "cosine"):
        """
        Initialize semantic search.
        
        Args:
            embedding_engine: EmbeddingEngine instance
            similarity_metric: Similarity metric
        """
        self.embedding_engine = embedding_engine
        self.similarity_metric = similarity_metric
        
        # Document index
        self._documents: Dict[str, Document] = {}
        
        # Embedding matrix
        self._embeddings: Optional[np.ndarray] = None
        self._doc_ids: List[str] = []
        
        # Index needs rebuild
        self._index_dirty = True
        
        logger.info("Semantic Search initialized")
    
    def add_document(self, doc_id: str,
                     content: str,
                     embedding: np.ndarray = None,
                     metadata: Dict = None):
        """
        Add document to index.
        
        Args:
            doc_id: Document ID
            content: Document content
            embedding: Pre-computed embedding
            metadata: Document metadata
        """
        if embedding is None and self.embedding_engine:
            result = self.embedding_engine.embed(content)
            embedding = result.embedding
        
        doc = Document(
            doc_id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self._documents[doc_id] = doc
        self._index_dirty = True
    
    def add_documents(self, documents: List[Dict]):
        """
        Add multiple documents.
        
        Args:
            documents: List of doc dicts with id, content, metadata
        """
        for doc in documents:
            self.add_document(
                doc["id"],
                doc["content"],
                doc.get("embedding"),
                doc.get("metadata")
            )
    
    def remove_document(self, doc_id: str):
        """Remove document from index."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._index_dirty = True
    
    def _rebuild_index(self):
        """Rebuild embedding index."""
        if not self._index_dirty:
            return
        
        embeddings = []
        self._doc_ids = []
        
        for doc_id, doc in self._documents.items():
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
                self._doc_ids.append(doc_id)
        
        if embeddings:
            self._embeddings = np.array(embeddings)
        else:
            self._embeddings = None
        
        self._index_dirty = False
        logger.debug(f"Index rebuilt with {len(self._doc_ids)} documents")
    
    def search(self, query: str,
               top_k: int = 10,
               filters: Dict = None,
               min_score: float = 0.0) -> List[SearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            min_score: Minimum similarity score
            
        Returns:
            List of SearchResults
        """
        self._rebuild_index()
        
        if self._embeddings is None or len(self._doc_ids) == 0:
            return []
        
        # Get query embedding
        if self.embedding_engine:
            query_result = self.embedding_engine.embed(query)
            query_embedding = query_result.embedding
        else:
            # Simulated query embedding
            np.random.seed(hash(query) % 2**32)
            query_embedding = np.random.randn(self._embeddings.shape[1])
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        scores = self._compute_similarities(query_embedding)
        
        # Apply filters
        if filters:
            mask = self._apply_filters(filters)
            scores = scores * mask
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            
            if score < min_score:
                continue
            
            doc_id = self._doc_ids[idx]
            doc = self._documents[doc_id]
            
            results.append(SearchResult(
                doc_id=doc_id,
                content=doc.content,
                score=float(score),
                metadata=doc.metadata
            ))
        
        return results
    
    def _compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute similarity scores."""
        if self.similarity_metric == "cosine":
            # Normalize
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            normalized = self._embeddings / (norms + 1e-8)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            return normalized @ query_norm
        
        elif self.similarity_metric == "dot":
            return self._embeddings @ query_embedding
        
        elif self.similarity_metric == "euclidean":
            distances = np.linalg.norm(self._embeddings - query_embedding, axis=1)
            return 1 / (1 + distances)
        
        return self._embeddings @ query_embedding
    
    def _apply_filters(self, filters: Dict) -> np.ndarray:
        """Apply metadata filters."""
        mask = np.ones(len(self._doc_ids))
        
        for idx, doc_id in enumerate(self._doc_ids):
            doc = self._documents[doc_id]
            
            for key, value in filters.items():
                if key not in doc.metadata:
                    mask[idx] = 0
                    break
                
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        mask[idx] = 0
                        break
                elif doc.metadata[key] != value:
                    mask[idx] = 0
                    break
        
        return mask
    
    def hybrid_search(self, query: str,
                      top_k: int = 10,
                      alpha: float = 0.7) -> List[SearchResult]:
        """
        Hybrid search (semantic + keyword).
        
        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for semantic (1-alpha for keyword)
            
        Returns:
            List of SearchResults
        """
        # Semantic search
        semantic_results = self.search(query, top_k * 2)
        
        # Keyword search
        keyword_scores = {}
        query_terms = query.lower().split()
        
        for doc_id, doc in self._documents.items():
            content_lower = doc.content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            keyword_scores[doc_id] = score / max(len(query_terms), 1)
        
        # Combine scores
        combined = {}
        
        for result in semantic_results:
            combined[result.doc_id] = alpha * result.score
        
        for doc_id, score in keyword_scores.items():
            if doc_id in combined:
                combined[doc_id] += (1 - alpha) * score
            else:
                combined[doc_id] = (1 - alpha) * score
        
        # Sort and return
        sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_k]
        
        results = []
        for doc_id in sorted_ids:
            doc = self._documents[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                content=doc.content,
                score=combined[doc_id],
                metadata=doc.metadata
            ))
        
        return results
    
    def find_similar(self, doc_id: str,
                     top_k: int = 5) -> List[SearchResult]:
        """Find documents similar to given document."""
        if doc_id not in self._documents:
            return []
        
        doc = self._documents[doc_id]
        
        if doc.embedding is None:
            return []
        
        self._rebuild_index()
        
        scores = self._compute_similarities(doc.embedding)
        
        # Exclude self
        self_idx = self._doc_ids.index(doc_id) if doc_id in self._doc_ids else -1
        if self_idx >= 0:
            scores[self_idx] = -1
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] < 0:
                continue
            
            result_id = self._doc_ids[idx]
            result_doc = self._documents[result_id]
            
            results.append(SearchResult(
                doc_id=result_id,
                content=result_doc.content,
                score=float(scores[idx]),
                metadata=result_doc.metadata
            ))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)
    
    def count(self) -> int:
        """Get document count."""
        return len(self._documents)
    
    def __repr__(self) -> str:
        return f"SemanticSearch(documents={len(self._documents)})"
