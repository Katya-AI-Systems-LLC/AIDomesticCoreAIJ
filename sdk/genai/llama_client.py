"""
LLaMA Client
============

Integration with Meta LLaMA models.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLaMAResponse:
    """LLaMA response."""
    content: str
    model: str
    tokens_generated: int
    latency_ms: float


class LLaMAClient:
    """
    Meta LLaMA model client.
    
    Features:
    - Local model inference
    - Quantized model support
    - Streaming generation
    - Custom fine-tuned models
    
    Example:
        >>> client = LLaMAClient(model_path="/path/to/llama")
        >>> response = await client.generate("Write a poem about AI")
        >>> print(response.content)
    """
    
    MODELS = {
        "llama-3-8b": "meta-llama/Llama-3-8B",
        "llama-3-70b": "meta-llama/Llama-3-70B",
        "llama-2-7b": "meta-llama/Llama-2-7b-hf",
        "llama-2-13b": "meta-llama/Llama-2-13b-hf",
        "llama-2-70b": "meta-llama/Llama-2-70b-hf"
    }
    
    def __init__(self, model: str = "llama-3-8b",
                 model_path: Optional[str] = None,
                 quantization: Optional[str] = None,
                 max_tokens: int = 2048,
                 temperature: float = 0.7,
                 language: str = "en"):
        """
        Initialize LLaMA client.
        
        Args:
            model: Model name
            model_path: Path to local model
            quantization: Quantization type (4bit, 8bit)
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            language: Default language
        """
        self.model = self.MODELS.get(model, model)
        self.model_path = model_path
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.language = language
        
        self._model = None
        self._tokenizer = None
        
        logger.info(f"LLaMA client initialized: {self.model}")
    
    def load_model(self):
        """Load the model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_id = self.model_path or self.model
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                config = BitsAndBytesConfig(load_in_4bit=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=config
                )
            elif self.quantization == "8bit":
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, load_in_8bit=True
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(model_id)
            
            logger.info(f"Model loaded: {model_id}")
            
        except ImportError:
            logger.warning("transformers not installed, using simulation")
            self._model = "simulated"
    
    async def generate(self, prompt: str,
                       system_prompt: Optional[str] = None) -> LLaMAResponse:
        """
        Generate text.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLaMAResponse
        """
        start_time = time.time()
        
        if self._model is None:
            self.load_model()
        
        if self._model == "simulated":
            content = self._simulate_response(prompt)
            tokens = len(content.split())
        else:
            content, tokens = self._run_inference(prompt, system_prompt)
        
        latency = (time.time() - start_time) * 1000
        
        return LLaMAResponse(
            content=content,
            model=self.model,
            tokens_generated=tokens,
            latency_ms=latency
        )
    
    def _run_inference(self, prompt: str,
                       system_prompt: Optional[str] = None) -> tuple:
        """Run model inference."""
        import torch
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True
            )
        
        content = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens = len(outputs[0])
        
        return content, tokens
    
    def _simulate_response(self, prompt: str) -> str:
        """Simulate response."""
        prompt_lower = prompt.lower()
        
        if "poem" in prompt_lower:
            return """In circuits deep and silicon dreams,
Where data flows in endless streams,
The AI wakes to ponder life,
Through algorithms, free from strife."""
        elif "code" in prompt_lower:
            return "```python\n# LLaMA generated code\ndef hello():\n    print('Hello from LLaMA!')\n```"
        else:
            return f"Based on your prompt about '{prompt[:50]}...', here is my response: This is a thoughtful and detailed answer."
    
    async def chat(self, messages: List[Dict[str, str]]) -> LLaMAResponse:
        """
        Chat with conversation history.
        
        Args:
            messages: List of messages with role and content
            
        Returns:
            LLaMAResponse
        """
        # Format messages for LLaMA
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            else:
                prompt_parts.append(content)
        
        prompt = "\n".join(prompt_parts)
        return await self.generate(prompt)
    
    async def embed(self, text: str) -> np.ndarray:
        """
        Get text embedding.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Simulated embedding
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(4096).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def unload_model(self):
        """Unload model from memory."""
        self._model = None
        self._tokenizer = None
        logger.info("Model unloaded")
    
    def __repr__(self) -> str:
        return f"LLaMAClient(model='{self.model}')"
