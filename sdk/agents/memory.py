"""
Agent Memory
============

Memory systems for AI agents.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
import time
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Memory item."""
    content: str
    role: str  # user, assistant, system, tool
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5


class AgentMemory(ABC):
    """
    Base agent memory.
    
    Features:
    - Store and retrieve memories
    - Relevance-based retrieval
    - Memory consolidation
    """
    
    @abstractmethod
    def add(self, role: str, content: str, metadata: Dict = None):
        """Add memory."""
        pass
    
    @abstractmethod
    def get_recent(self, k: int = 10) -> List[MemoryItem]:
        """Get recent memories."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Search memories."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all memories."""
        pass


class ConversationMemory(AgentMemory):
    """
    Conversation-based memory.
    
    Stores conversation history with optional summarization.
    
    Example:
        >>> memory = ConversationMemory(max_messages=100)
        >>> memory.add("user", "Hello!")
        >>> memory.add("assistant", "Hi there!")
    """
    
    def __init__(self, max_messages: int = 100,
                 summarize_threshold: int = 50):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum messages to store
            summarize_threshold: When to summarize
        """
        self.max_messages = max_messages
        self.summarize_threshold = summarize_threshold
        
        self._messages: deque = deque(maxlen=max_messages)
        self._summary: Optional[str] = None
        
        logger.info(f"Conversation Memory initialized (max={max_messages})")
    
    def add(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation."""
        item = MemoryItem(
            content=content,
            role=role,
            metadata=metadata or {}
        )
        
        self._messages.append(item)
        
        # Check if summarization needed
        if len(self._messages) >= self.summarize_threshold:
            self._maybe_summarize()
    
    def _maybe_summarize(self):
        """Summarize older messages if needed."""
        # Keep recent messages, summarize older ones
        pass
    
    def get_recent(self, k: int = 10) -> List[MemoryItem]:
        """Get recent messages."""
        return list(self._messages)[-k:]
    
    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Search messages by keyword."""
        query_lower = query.lower()
        
        matches = []
        for item in self._messages:
            if query_lower in item.content.lower():
                matches.append(item)
        
        return matches[-k:]
    
    def get_context_window(self, max_tokens: int = 4000) -> str:
        """Get messages that fit in context window."""
        messages = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate
        
        for item in reversed(list(self._messages)):
            if total_chars + len(item.content) > max_chars:
                break
            messages.insert(0, item)
            total_chars += len(item.content)
        
        return self._format_messages(messages)
    
    def _format_messages(self, messages: List[MemoryItem]) -> str:
        """Format messages for prompt."""
        lines = []
        for msg in messages:
            lines.append(f"{msg.role.capitalize()}: {msg.content}")
        return "\n".join(lines)
    
    def to_chat_messages(self) -> List[Dict]:
        """Convert to chat message format."""
        return [
            {"role": item.role, "content": item.content}
            for item in self._messages
        ]
    
    def clear(self):
        """Clear conversation."""
        self._messages.clear()
        self._summary = None
    
    def __len__(self) -> int:
        return len(self._messages)


class VectorMemory(AgentMemory):
    """
    Vector-based semantic memory.
    
    Uses embeddings for semantic retrieval.
    """
    
    def __init__(self, embedding_engine: Any = None,
                 max_items: int = 1000):
        """
        Initialize vector memory.
        
        Args:
            embedding_engine: Embedding engine
            max_items: Maximum items
        """
        self.embedding_engine = embedding_engine
        self.max_items = max_items
        
        self._items: List[MemoryItem] = []
        self._embeddings: List = []
    
    def add(self, role: str, content: str, metadata: Dict = None):
        """Add memory with embedding."""
        item = MemoryItem(
            content=content,
            role=role,
            metadata=metadata or {}
        )
        
        self._items.append(item)
        
        # Generate embedding
        if self.embedding_engine:
            result = self.embedding_engine.embed(content)
            self._embeddings.append(result.embedding)
        
        # Trim if needed
        if len(self._items) > self.max_items:
            self._items.pop(0)
            if self._embeddings:
                self._embeddings.pop(0)
    
    def get_recent(self, k: int = 10) -> List[MemoryItem]:
        """Get recent memories."""
        return self._items[-k:]
    
    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Semantic search."""
        if not self.embedding_engine or not self._embeddings:
            # Fallback to keyword search
            return [item for item in self._items if query.lower() in item.content.lower()][-k:]
        
        import numpy as np
        
        query_emb = self.embedding_engine.embed(query).embedding
        
        # Compute similarities
        similarities = []
        for emb in self._embeddings:
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            similarities.append(sim)
        
        # Get top-k
        indices = np.argsort(similarities)[-k:][::-1]
        
        return [self._items[i] for i in indices]
    
    def clear(self):
        """Clear memory."""
        self._items.clear()
        self._embeddings.clear()


class WorkingMemory:
    """
    Short-term working memory.
    
    Stores task-specific information.
    """
    
    def __init__(self, capacity: int = 10):
        """
        Initialize working memory.
        
        Args:
            capacity: Maximum items
        """
        self.capacity = capacity
        self._items: Dict[str, Any] = {}
        self._order: deque = deque(maxlen=capacity)
    
    def set(self, key: str, value: Any):
        """Set item in working memory."""
        if key not in self._items and len(self._items) >= self.capacity:
            # Remove oldest
            oldest = self._order.popleft()
            del self._items[oldest]
        
        self._items[key] = value
        
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from working memory."""
        return self._items.get(key, default)
    
    def remove(self, key: str):
        """Remove item."""
        if key in self._items:
            del self._items[key]
            self._order.remove(key)
    
    def clear(self):
        """Clear working memory."""
        self._items.clear()
        self._order.clear()
    
    def items(self) -> Dict[str, Any]:
        """Get all items."""
        return self._items.copy()
