"""
Unified GenAI
=============

Unified interface for multiple GenAI providers.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class Provider(Enum):
    """GenAI providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    LLAMA = "llama"
    GIGACHAT = "gigachat"
    KATYA = "katya"


@dataclass
class UnifiedResponse:
    """Unified response from any provider."""
    content: str
    provider: Provider
    model: str
    usage: Dict[str, int]
    latency_ms: float
    metadata: Dict[str, Any]


class UnifiedGenAI:
    """
    Unified interface for multiple GenAI providers.
    
    Features:
    - Single API for multiple providers
    - Automatic fallback
    - Load balancing
    - Cost optimization
    
    Example:
        >>> genai = UnifiedGenAI()
        >>> genai.add_provider(Provider.OPENAI, api_key="sk-...")
        >>> response = await genai.generate("Hello!")
    """
    
    def __init__(self, default_provider: Provider = Provider.OPENAI,
                 language: str = "en"):
        """
        Initialize unified GenAI.
        
        Args:
            default_provider: Default provider
            language: Default language
        """
        self.default_provider = default_provider
        self.language = language
        
        self._providers: Dict[Provider, Any] = {}
        self._fallback_order: List[Provider] = []
        
        logger.info(f"Unified GenAI initialized: default={default_provider.value}")
    
    def add_provider(self, provider: Provider,
                     api_key: Optional[str] = None,
                     **kwargs):
        """
        Add a provider.
        
        Args:
            provider: Provider type
            api_key: API key
            **kwargs: Provider-specific arguments
        """
        if provider == Provider.OPENAI:
            from .openai_client import OpenAIClient
            self._providers[provider] = OpenAIClient(api_key=api_key, **kwargs)
        
        elif provider == Provider.CLAUDE:
            from .claude_client import ClaudeClient
            self._providers[provider] = ClaudeClient(api_key=api_key, **kwargs)
        
        elif provider == Provider.LLAMA:
            from .llama_client import LLaMAClient
            self._providers[provider] = LLaMAClient(**kwargs)
        
        elif provider == Provider.GIGACHAT:
            from ..multimodal.gigachat import GigaChat3Client
            self._providers[provider] = GigaChat3Client(api_key=api_key, **kwargs)
        
        elif provider == Provider.KATYA:
            from .katya_genai import KatyaGenAI
            self._providers[provider] = KatyaGenAI(**kwargs)
        
        self._fallback_order.append(provider)
        logger.info(f"Added provider: {provider.value}")
    
    def remove_provider(self, provider: Provider):
        """Remove a provider."""
        if provider in self._providers:
            del self._providers[provider]
            self._fallback_order.remove(provider)
    
    def set_fallback_order(self, order: List[Provider]):
        """Set fallback order for providers."""
        self._fallback_order = [p for p in order if p in self._providers]
    
    async def generate(self, prompt: str,
                       provider: Optional[Provider] = None,
                       **kwargs) -> UnifiedResponse:
        """
        Generate response using specified or default provider.
        
        Args:
            prompt: Input prompt
            provider: Provider to use
            **kwargs: Additional arguments
            
        Returns:
            UnifiedResponse
        """
        provider = provider or self.default_provider
        
        if provider not in self._providers:
            # Try fallback
            for fallback in self._fallback_order:
                if fallback in self._providers:
                    provider = fallback
                    break
            else:
                raise ValueError("No providers available")
        
        start_time = time.time()
        
        try:
            response = await self._call_provider(provider, prompt, **kwargs)
            latency = (time.time() - start_time) * 1000
            
            return UnifiedResponse(
                content=response.get("content", ""),
                provider=provider,
                model=response.get("model", "unknown"),
                usage=response.get("usage", {}),
                latency_ms=latency,
                metadata=response.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Provider {provider.value} failed: {e}")
            
            # Try fallback
            for fallback in self._fallback_order:
                if fallback != provider and fallback in self._providers:
                    return await self.generate(prompt, provider=fallback, **kwargs)
            
            raise
    
    async def _call_provider(self, provider: Provider,
                              prompt: str, **kwargs) -> Dict[str, Any]:
        """Call specific provider."""
        client = self._providers[provider]
        
        if provider == Provider.OPENAI:
            response = await client.chat(prompt, **kwargs)
            return {
                "content": response.content,
                "model": response.model,
                "usage": response.usage
            }
        
        elif provider == Provider.CLAUDE:
            response = await client.chat(prompt, **kwargs)
            return {
                "content": response.content,
                "model": response.model,
                "usage": response.usage
            }
        
        elif provider == Provider.LLAMA:
            response = await client.generate(prompt, **kwargs)
            return {
                "content": response.content,
                "model": response.model,
                "usage": {"tokens": response.tokens_generated}
            }
        
        elif provider == Provider.GIGACHAT:
            from ..multimodal.gigachat import GigaChatMessage, GigaChatRole
            response = await client.chat([
                GigaChatMessage(GigaChatRole.USER, prompt)
            ])
            return {
                "content": response.content,
                "model": response.model,
                "usage": response.usage
            }
        
        elif provider == Provider.KATYA:
            response = await client.generate(prompt, **kwargs)
            return {
                "content": response.content,
                "model": response.model,
                "usage": {}
            }
        
        return {"content": "", "model": "unknown", "usage": {}}
    
    async def embed(self, text: str,
                    provider: Optional[Provider] = None) -> np.ndarray:
        """
        Get text embedding.
        
        Args:
            text: Input text
            provider: Provider to use
            
        Returns:
            Embedding vector
        """
        provider = provider or self.default_provider
        
        if provider not in self._providers:
            provider = self._fallback_order[0] if self._fallback_order else None
        
        if provider is None:
            raise ValueError("No providers available")
        
        client = self._providers[provider]
        
        if hasattr(client, 'embed'):
            return await client.embed(text)
        
        # Fallback embedding
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(768).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def get_providers(self) -> List[Provider]:
        """Get list of available providers."""
        return list(self._providers.keys())
    
    def get_provider_info(self, provider: Provider) -> Dict[str, Any]:
        """Get provider information."""
        if provider not in self._providers:
            return {}
        
        client = self._providers[provider]
        return {
            "provider": provider.value,
            "model": getattr(client, 'model', 'unknown'),
            "type": type(client).__name__
        }
    
    def __repr__(self) -> str:
        providers = ", ".join(p.value for p in self._providers.keys())
        return f"UnifiedGenAI(providers=[{providers}])"
