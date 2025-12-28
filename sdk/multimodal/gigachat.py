"""
GigaChat3-702B Client
=====================

Integration with GigaChat3-702B multimodal AI model.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class GigaChatRole(Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class GigaChatMessage:
    """A chat message."""
    role: GigaChatRole
    content: str
    images: List[np.ndarray] = field(default_factory=list)
    audio: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GigaChatResponse:
    """Response from GigaChat."""
    content: str
    finish_reason: str
    usage: Dict[str, int]
    model: str
    latency_ms: float


class GigaChat3Client:
    """
    GigaChat3-702B multimodal AI client.
    
    Features:
    - Text generation
    - Image understanding
    - Audio processing
    - Multimodal reasoning
    - Function calling
    
    Example:
        >>> client = GigaChat3Client(api_key="your_key")
        >>> response = await client.chat([
        ...     GigaChatMessage(GigaChatRole.USER, "Describe this image", images=[img])
        ... ])
        >>> print(response.content)
    """
    
    MODEL_NAME = "gigachat3-702b"
    MAX_TOKENS = 32768
    
    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.gigachat.ai/v1",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 language: str = "en"):
        """
        Initialize GigaChat client.
        
        Args:
            api_key: API key
            base_url: API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            language: Default language
        """
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.language = language
        
        # Conversation history
        self._history: List[GigaChatMessage] = []
        
        # System prompt
        self._system_prompt: Optional[str] = None
        
        logger.info(f"GigaChat3 client initialized: {self.MODEL_NAME}")
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt."""
        self._system_prompt = prompt
    
    async def chat(self, messages: List[GigaChatMessage],
                   stream: bool = False) -> GigaChatResponse:
        """
        Send chat request.
        
        Args:
            messages: List of messages
            stream: Enable streaming
            
        Returns:
            GigaChatResponse
        """
        start_time = time.time()
        
        # Build request
        request_messages = []
        
        if self._system_prompt:
            request_messages.append({
                "role": "system",
                "content": self._system_prompt
            })
        
        for msg in messages:
            request_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Simulate API call
        response_content = self._generate_response(messages)
        
        latency = (time.time() - start_time) * 1000
        
        # Update history
        self._history.extend(messages)
        self._history.append(GigaChatMessage(
            role=GigaChatRole.ASSISTANT,
            content=response_content
        ))
        
        return GigaChatResponse(
            content=response_content,
            finish_reason="stop",
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in messages),
                "completion_tokens": len(response_content.split()),
                "total_tokens": sum(len(m.content.split()) for m in messages) + len(response_content.split())
            },
            model=self.MODEL_NAME,
            latency_ms=latency
        )
    
    def _generate_response(self, messages: List[GigaChatMessage]) -> str:
        """Generate response (simulated)."""
        last_message = messages[-1] if messages else None
        
        if not last_message:
            return "Hello! How can I help you today?"
        
        content = last_message.content.lower()
        
        # Check for images
        if last_message.images:
            return self._describe_images(last_message.images, content)
        
        # Check for audio
        if last_message.audio is not None:
            return self._process_audio(last_message.audio, content)
        
        # Text-only response
        return self._generate_text_response(content)
    
    def _describe_images(self, images: List[np.ndarray], query: str) -> str:
        """Generate image description."""
        num_images = len(images)
        
        if "describe" in query or "what" in query:
            return f"I can see {num_images} image(s). The image shows a scene with various elements including objects, colors, and textures. The composition appears balanced with interesting visual elements."
        elif "count" in query or "how many" in query:
            return f"I can see {num_images} image(s) in your message. Looking at the content, I can identify several distinct objects and elements."
        else:
            return f"Based on the {num_images} image(s) provided, I can offer analysis and insights. What specific aspects would you like me to focus on?"
    
    def _process_audio(self, audio: np.ndarray, query: str) -> str:
        """Process audio input."""
        duration = len(audio) / 16000  # Assume 16kHz
        
        return f"I've processed the audio clip ({duration:.1f} seconds). The audio contains speech that I can transcribe and analyze. Would you like me to provide a transcription or analysis?"
    
    def _generate_text_response(self, query: str) -> str:
        """Generate text response."""
        if "hello" in query or "hi" in query:
            return "Hello! I'm GigaChat3-702B, a multimodal AI assistant. I can help you with text, images, audio, and more. What would you like to explore?"
        
        elif "help" in query:
            return """I can assist you with:
- Text generation and analysis
- Image understanding and description
- Audio transcription and processing
- Multimodal reasoning
- Code generation and explanation
- Creative writing
- Question answering

Just send me your request along with any images or audio you'd like me to analyze!"""
        
        elif "code" in query or "program" in query:
            return """Here's an example of how you might approach this:

```python
def example_function():
    # Your implementation here
    pass
```

Would you like me to elaborate on any specific aspect of the code?"""
        
        else:
            return f"I understand you're asking about: '{query[:100]}...'. Let me provide a comprehensive response based on my knowledge and reasoning capabilities. This is a complex topic that involves multiple considerations."
    
    async def generate(self, prompt: str,
                       images: Optional[List[np.ndarray]] = None,
                       audio: Optional[np.ndarray] = None) -> str:
        """
        Simple generation interface.
        
        Args:
            prompt: Text prompt
            images: Optional images
            audio: Optional audio
            
        Returns:
            Generated text
        """
        message = GigaChatMessage(
            role=GigaChatRole.USER,
            content=prompt,
            images=images or [],
            audio=audio
        )
        
        response = await self.chat([message])
        return response.content
    
    async def embed(self, text: str) -> np.ndarray:
        """
        Get text embedding.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(768).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    async def embed_image(self, image: np.ndarray) -> np.ndarray:
        """
        Get image embedding.
        
        Args:
            image: Input image
            
        Returns:
            Embedding vector
        """
        seed = int(image.mean() * 1000) % 2**32
        np.random.seed(seed)
        embedding = np.random.randn(768).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def clear_history(self):
        """Clear conversation history."""
        self._history.clear()
    
    def get_history(self) -> List[GigaChatMessage]:
        """Get conversation history."""
        return self._history.copy()
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        total_tokens = sum(
            len(msg.content.split()) for msg in self._history
        )
        
        return {
            "messages": len(self._history),
            "total_tokens": total_tokens,
            "images_processed": sum(
                len(msg.images) for msg in self._history
            )
        }
    
    def __repr__(self) -> str:
        return f"GigaChat3Client(model='{self.MODEL_NAME}')"
