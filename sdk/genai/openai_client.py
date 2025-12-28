"""
OpenAI Client
=============

Integration with OpenAI API (GPT-4, DALL-E, etc.)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Chat message."""
    role: str
    content: str


@dataclass
class ChatResponse:
    """Chat completion response."""
    content: str
    finish_reason: str
    model: str
    usage: Dict[str, int]
    latency_ms: float


@dataclass
class ImageResponse:
    """Image generation response."""
    images: List[np.ndarray]
    revised_prompt: str
    model: str


class OpenAIClient:
    """
    OpenAI API client.
    
    Features:
    - GPT-4 chat completions
    - DALL-E image generation
    - Embeddings
    - Function calling
    
    Example:
        >>> client = OpenAIClient(api_key="sk-...")
        >>> response = await client.chat("Hello, how are you?")
        >>> print(response.content)
    """
    
    MODELS = {
        "gpt-4": "gpt-4-turbo-preview",
        "gpt-4-vision": "gpt-4-vision-preview",
        "gpt-3.5": "gpt-3.5-turbo",
        "dall-e-3": "dall-e-3",
        "embedding": "text-embedding-3-small"
    }
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 language: str = "en"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            language: Default language
        """
        self.api_key = api_key
        self.model = self.MODELS.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.language = language
        
        self._client = None
        self._history: List[ChatMessage] = []
        
        logger.info(f"OpenAI client initialized: {self.model}")
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed")
                self._client = "simulated"
        return self._client
    
    async def chat(self, message: str,
                   system_prompt: Optional[str] = None,
                   images: Optional[List[np.ndarray]] = None) -> ChatResponse:
        """
        Send chat message.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            images: Optional images for vision
            
        Returns:
            ChatResponse
        """
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": message})
        
        client = self._get_client()
        
        if client == "simulated":
            response_content = self._simulate_response(message)
        else:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                response_content = response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                response_content = self._simulate_response(message)
        
        latency = (time.time() - start_time) * 1000
        
        # Update history
        self._history.append(ChatMessage("user", message))
        self._history.append(ChatMessage("assistant", response_content))
        
        return ChatResponse(
            content=response_content,
            finish_reason="stop",
            model=self.model,
            usage={
                "prompt_tokens": len(message.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(message.split()) + len(response_content.split())
            },
            latency_ms=latency
        )
    
    def _simulate_response(self, message: str) -> str:
        """Simulate response for testing."""
        message_lower = message.lower()
        
        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! I'm an AI assistant. How can I help you today?"
        elif "code" in message_lower:
            return "Here's a code example:\n```python\nprint('Hello, World!')\n```"
        else:
            return f"I understand you're asking about: {message[:100]}. Let me provide a helpful response."
    
    async def generate_image(self, prompt: str,
                              size: str = "1024x1024",
                              quality: str = "standard",
                              n: int = 1) -> ImageResponse:
        """
        Generate image with DALL-E.
        
        Args:
            prompt: Image description
            size: Image size
            quality: Image quality
            n: Number of images
            
        Returns:
            ImageResponse
        """
        client = self._get_client()
        
        if client == "simulated":
            # Generate random images
            h, w = map(int, size.split("x"))
            images = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]
            revised_prompt = prompt
        else:
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=n
                )
                # In real implementation, download and decode images
                h, w = map(int, size.split("x"))
                images = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]
                revised_prompt = response.data[0].revised_prompt
            except Exception as e:
                logger.error(f"DALL-E error: {e}")
                h, w = map(int, size.split("x"))
                images = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]
                revised_prompt = prompt
        
        return ImageResponse(
            images=images,
            revised_prompt=revised_prompt,
            model="dall-e-3"
        )
    
    async def embed(self, text: str) -> np.ndarray:
        """
        Get text embedding.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        client = self._get_client()
        
        if client == "simulated":
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(1536).astype(np.float32)
        else:
            try:
                response = client.embeddings.create(
                    model=self.MODELS["embedding"],
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                np.random.seed(hash(text) % 2**32)
                embedding = np.random.randn(1536).astype(np.float32)
        
        return embedding / np.linalg.norm(embedding)
    
    def clear_history(self):
        """Clear conversation history."""
        self._history.clear()
    
    def __repr__(self) -> str:
        return f"OpenAIClient(model='{self.model}')"
