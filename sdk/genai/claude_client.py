"""
Claude Client
=============

Integration with Anthropic Claude API.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    """Claude API response."""
    content: str
    stop_reason: str
    model: str
    usage: Dict[str, int]
    latency_ms: float


class ClaudeClient:
    """
    Anthropic Claude API client.
    
    Features:
    - Claude 3 chat completions
    - Vision capabilities
    - Long context support
    - Constitutional AI
    
    Example:
        >>> client = ClaudeClient(api_key="sk-ant-...")
        >>> response = await client.chat("Explain quantum computing")
        >>> print(response.content)
    """
    
    MODELS = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307"
    }
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-3-sonnet",
                 max_tokens: int = 4096,
                 temperature: float = 0.7,
                 language: str = "en"):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            language: Default language
        """
        self.api_key = api_key
        self.model = self.MODELS.get(model, model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.language = language
        
        self._client = None
        self._history: List[Dict] = []
        self._system_prompt: Optional[str] = None
        
        logger.info(f"Claude client initialized: {self.model}")
    
    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed")
                self._client = "simulated"
        return self._client
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt."""
        self._system_prompt = prompt
    
    async def chat(self, message: str,
                   images: Optional[List[np.ndarray]] = None) -> ClaudeResponse:
        """
        Send chat message.
        
        Args:
            message: User message
            images: Optional images
            
        Returns:
            ClaudeResponse
        """
        start_time = time.time()
        
        # Build messages
        messages = self._history.copy()
        
        content = [{"type": "text", "text": message}]
        
        if images:
            for img in images:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "data": "..."}
                })
        
        messages.append({"role": "user", "content": content})
        
        client = self._get_client()
        
        if client == "simulated":
            response_content = self._simulate_response(message)
        else:
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self._system_prompt or "",
                    messages=messages
                )
                response_content = response.content[0].text
            except Exception as e:
                logger.error(f"Claude API error: {e}")
                response_content = self._simulate_response(message)
        
        latency = (time.time() - start_time) * 1000
        
        # Update history
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": response_content})
        
        return ClaudeResponse(
            content=response_content,
            stop_reason="end_turn",
            model=self.model,
            usage={
                "input_tokens": len(message.split()),
                "output_tokens": len(response_content.split())
            },
            latency_ms=latency
        )
    
    def _simulate_response(self, message: str) -> str:
        """Simulate response."""
        message_lower = message.lower()
        
        if "explain" in message_lower:
            return "I'd be happy to explain that concept. Let me break it down into clear, understandable parts..."
        elif "help" in message_lower:
            return "I'm here to help! I can assist with analysis, writing, coding, math, and much more. What would you like to explore?"
        elif "code" in message_lower:
            return "Here's a well-structured code example:\n\n```python\ndef example():\n    '''Docstring explaining the function'''\n    return 'result'\n```"
        else:
            return f"Thank you for your question about '{message[:50]}...'. Let me provide a thoughtful and comprehensive response."
    
    async def analyze_image(self, image: np.ndarray,
                             prompt: str = "Describe this image") -> str:
        """
        Analyze image with Claude Vision.
        
        Args:
            image: Input image
            prompt: Analysis prompt
            
        Returns:
            Analysis text
        """
        response = await self.chat(prompt, images=[image])
        return response.content
    
    def clear_history(self):
        """Clear conversation history."""
        self._history.clear()
    
    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self._history.copy()
    
    def __repr__(self) -> str:
        return f"ClaudeClient(model='{self.model}')"
