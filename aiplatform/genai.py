"""
GenAI Integration Module for AIPlatform SDK

This module provides integration with various generative AI models with internationalization support
for Russian, Chinese, and Arabic languages.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import logging
import json
from datetime import datetime
import base64
import numpy as np

# Import i18n components
from aiplatform.i18n import translate
from aiplatform.i18n.vocabulary_manager import get_vocabulary_manager

# Import exceptions
from aiplatform.exceptions import GenAIError

# Set up logging
logger = logging.getLogger(__name__)


class GenAIModel:
    """Generic GenAI model interface with multilingual support and real API capabilities."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, language: str = 'en'):
        """
        Initialize GenAI model.
        
        Args:
            model_name: Name of the model
            api_key: API key for cloud models
            language: Language code for internationalization
        """
        self.model_name = model_name
        self.api_key = api_key
        self.language = language
        self.vocabulary_manager = get_vocabulary_manager()
        
        # Get localized terms
        model_term = self.vocabulary_manager.translate_term('Generative AI Model', 'genai', self.language)
        logger.info(f"{model_term} '{model_name}' initialized")
    
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text with real API capabilities.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating text', 'genai', self.language)
        logger.info(f"{generating_term} with {self.model_name}")
        
        # Try real API calls for known models
        if self.model_name.startswith('gpt-') and self.api_key:
            return self._call_openai_api(prompt, max_tokens)
        elif self.model_name.startswith('claude-') and self.api_key:
            return self._call_claude_api(prompt, max_tokens)
        elif self.api_key:
            return self._call_generic_api(prompt, max_tokens)
        else:
            # Fallback to simulation
            return self._simulate_text_generation(prompt, max_tokens)
    
    def _call_openai_api(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI API for text generation."""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content']
                logger.info(translate('text_generation_completed', self.language) or "Text generation completed")
                return generated_text
            else:
                raise GenAIError(f"OpenAI API error: {response.status_code}")
                
        except ImportError:
            logger.warning("Requests library not available, using simulation")
            return self._simulate_text_generation(prompt, max_tokens)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return self._simulate_text_generation(prompt, max_tokens)
    
    def _call_claude_api(self, prompt: str, max_tokens: int) -> str:
        """Call Claude API for text generation."""
        try:
            import requests
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['content'][0]['text']
                logger.info(translate('text_generation_completed', self.language) or "Text generation completed")
                return generated_text
            else:
                raise GenAIError(f"Claude API error: {response.status_code}")
                
        except ImportError:
            logger.warning("Requests library not available, using simulation")
            return self._simulate_text_generation(prompt, max_tokens)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return self._simulate_text_generation(prompt, max_tokens)
    
    def _call_generic_api(self, prompt: str, max_tokens: int) -> str:
        """Generic API call for other models."""
        # This could be extended for other APIs
        return self._simulate_text_generation(prompt, max_tokens)
    
    def _simulate_text_generation(self, prompt: str, max_tokens: int) -> str:
        """Fallback simulation for text generation."""
        generated_text = f"Generated response to: {prompt}\n\n"
        generated_text += f"This is a simulated response from {self.model_name}.\n"
        generated_text += f"Maximum tokens requested: {max_tokens}\n"
        generated_text += f"Language: {self.language}\n"
        generated_text += f"[SIMULATED - No API key provided or API unavailable]"
        
        logger.info(translate('text_generation_completed', self.language) or "Text generation completed (simulation)")
        return generated_text
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding with real API capabilities.
        
        Args:
            text: Input text
            
        Returns:
            list: Embedding vector
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating embedding', 'genai', self.language)
        logger.info(f"{generating_term} with {self.model_name}")
        
        # Try real API calls for embedding models
        if self.model_name in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'] and self.api_key:
            return self._call_openai_embedding_api(text)
        else:
            # Fallback to simulation
            return self._simulate_embedding_generation(text)
    
    def _call_openai_embedding_api(self, text: str) -> List[float]:
        """Call OpenAI embedding API."""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "input": text
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result['data'][0]['embedding']
                logger.info(translate('embedding_generation_completed', self.language) or "Embedding generation completed")
                return embedding
            else:
                raise GenAIError(f"OpenAI embedding API error: {response.status_code}")
                
        except ImportError:
            logger.warning("Requests library not available, using simulation")
            return self._simulate_embedding_generation(text)
        except Exception as e:
            logger.error(f"Embedding API call failed: {str(e)}")
            return self._simulate_embedding_generation(text)
    
    def _simulate_embedding_generation(self, text: str) -> List[float]:
        """Fallback simulation for embedding generation."""
        # Generate deterministic but pseudo-random embedding based on text hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        np.random.seed(seed)
        embedding = np.random.random(768).tolist()  # 768-dim embedding (common size)
        
        logger.info(translate('embedding_generation_completed', self.language) or "Embedding generation completed (simulation)")
        return embedding


class OpenAIIntegration(GenAIModel):
    """OpenAI integration with multilingual support and real API calls."""
    
    def __init__(self, api_key: str, model: str = 'gpt-4', language: str = 'en'):
        """
        Initialize OpenAI integration.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name
            language: Language code for internationalization
        """
        super().__init__(f"OpenAI-{model}", api_key, language)
        self.model = model
        self.vocabulary_manager = get_vocabulary_manager()
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Get localized terms
        openai_term = self.vocabulary_manager.translate_term('OpenAI Integration', 'genai', self.language)
        logger.info(f"{openai_term} initialized")
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate chat completion with real OpenAI API calls.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            
        Returns:
            dict: Chat completion result
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating chat completion', 'genai', self.language)
        logger.info(generating_term)
        
        try:
            import requests
            
            # Prepare API request
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(translate('chat_completion_completed', self.language) or "Chat completion completed")
                return result
            else:
                error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise GenAIError(error_msg)
                
        except ImportError:
            # Fallback to simulation if requests not available
            logger.warning("Requests library not available, using simulation")
            return self._simulate_chat_completion(messages, temperature)
        except Exception as e:
            error_msg = f"OpenAI API call failed: {str(e)}"
            logger.error(error_msg)
            # Fallback to simulation on error
            return self._simulate_chat_completion(messages, temperature)
    
    def _simulate_chat_completion(self, messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
        """Fallback simulation for chat completion."""
        response = {
            'id': f"chatcmpl-{np.random.randint(1000000, 9999999)}",
            'object': 'chat.completion',
            'created': int(datetime.now().timestamp()),
            'model': self.model,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': f"Simulated response to messages: {len(messages)} messages processed\nTemperature: {temperature}\nLanguage: {self.language}"
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': sum(len(msg.get('content', '')) for msg in messages),
                'completion_tokens': 100,
                'total_tokens': sum(len(msg.get('content', '')) for msg in messages) + 100
            },
            'language': self.language,
            'simulated': True
        }
        
        logger.info(translate('chat_completion_completed', self.language) or "Chat completion completed (simulation)")
        return response


class ClaudeIntegration(GenAIModel):
    """Claude integration with multilingual support."""
    
    def __init__(self, api_key: str, model: str = 'claude-3-opus', language: str = 'en'):
        """
        Initialize Claude integration.
        
        Args:
            api_key: Claude API key
            model: Claude model name
            language: Language code for internationalization
        """
        super().__init__(f"Claude-{model}", api_key, language)
        self.model = model
        self.vocabulary_manager = get_vocabulary_manager()
        
        # Get localized terms
        claude_term = self.vocabulary_manager.translate_term('Claude Integration', 'genai', self.language)
        logger.info(f"{claude_term} initialized")
    
    def generate_with_context(self, prompt: str, context: str, max_tokens: int = 1000) -> str:
        """
        Generate text with context with localized logging.
        
        Args:
            prompt: Input prompt
            context: Context information
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating text with context', 'genai', self.language)
        logger.info(generating_term)
        
        # Simulate context-aware generation
        # In a real implementation, this would call Claude API
        generated_text = f"Context-aware response:\nContext: {context}\nPrompt: {prompt}\n\n"
        generated_text += f"Generated by {self.model} with {max_tokens} max tokens\n"
        generated_text += f"Language: {self.language}"
        
        logger.info(translate('context_generation_completed', self.language) or "Context-aware generation completed")
        return generated_text


class LLaMAIntegration(GenAIModel):
    """LLaMA integration with multilingual support."""
    
    def __init__(self, model_path: str, language: str = 'en'):
        """
        Initialize LLaMA integration.
        
        Args:
            model_path: Path to LLaMA model
            language: Language code for internationalization
        """
        super().__init__(f"LLaMA-{model_path.split('/')[-1]}", None, language)
        self.model_path = model_path
        self.vocabulary_manager = get_vocabulary_manager()
        
        # Get localized terms
        llama_term = self.vocabulary_manager.translate_term('LLaMA Integration', 'genai', self.language)
        logger.info(f"{llama_term} initialized")
    
    def generate_with_sampling(self, prompt: str, temperature: float = 0.8, top_p: float = 0.9) -> str:
        """
        Generate text with sampling parameters with localized logging.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            str: Generated text
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating text with sampling', 'genai', self.language)
        logger.info(generating_term)
        
        # Simulate sampling-based generation
        # In a real implementation, this would call LLaMA model
        generated_text = f"Sampling-based response:\nPrompt: {prompt}\n\n"
        generated_text += f"Temperature: {temperature}, Top-p: {top_p}\n"
        generated_text += f"Model: {self.model_name}\n"
        generated_text += f"Language: {self.language}"
        
        logger.info(translate('sampling_generation_completed', self.language) or "Sampling-based generation completed")
        return generated_text


class GigaChat3Integration(GenAIModel):
    """GigaChat3-702B integration with multilingual support."""
    
    def __init__(self, api_key: str, language: str = 'en'):
        """
        Initialize GigaChat3 integration.
        
        Args:
            api_key: GigaChat3 API key
            language: Language code for internationalization
        """
        super().__init__("GigaChat3-702B", api_key, language)
        self.vocabulary_manager = get_vocabulary_manager()
        
        # Get localized terms
        giga_term = self.vocabulary_manager.translate_term('GigaChat3 Integration', 'genai', self.language)
        logger.info(f"{giga_term} initialized")
    
    def generate_multilingual(self, prompt: str, target_language: str) -> str:
        """
        Generate multilingual text with localized logging.
        
        Args:
            prompt: Input prompt
            target_language: Target language for generation
            
        Returns:
            str: Generated text in target language
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating multilingual text', 'genai', self.language)
        logger.info(f"{generating_term} in {target_language}")
        
        # Simulate multilingual generation
        # In a real implementation, this would call GigaChat3 API
        generated_text = f"Multilingual response:\nPrompt: {prompt}\n\n"
        generated_text += f"Target language: {target_language}\n"
        generated_text += f"Generated by: {self.model_name}\n"
        generated_text += f"Request language: {self.language}"
        
        logger.info(translate('multilingual_generation_completed', self.language) or "Multilingual generation completed")
        return generated_text


class KatyaAIIntegration(GenAIModel):
    """Katya AI integration with multilingual support."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize Katya AI integration.
        
        Args:
            language: Language code for internationalization
        """
        super().__init__("Katya AI", None, language)
        self.vocabulary_manager = get_vocabulary_manager()
        
        # Get localized terms
        katya_term = self.vocabulary_manager.translate_term('Katya AI Integration', 'genai', self.language)
        logger.info(f"{katya_term} initialized")
    
    def generate_with_personality(self, prompt: str, personality: str = 'helpful') -> str:
        """
        Generate text with personality with localized logging.
        
        Args:
            prompt: Input prompt
            personality: Personality type
            
        Returns:
            str: Generated text with personality
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating text with personality', 'genai', self.language)
        logger.info(f"{generating_term}: {personality}")
        
        # Simulate personality-based generation
        # In a real implementation, this would use Katya AI personality system
        generated_text = f"Personality-based response ({personality}):\nPrompt: {prompt}\n\n"
        generated_text += f"Generated by: {self.model_name}\n"
        generated_text += f"Language: {self.language}"
        
        logger.info(translate('personality_generation_completed', self.language) or "Personality-based generation completed")
        return generated_text


class SpeechProcessor:
    """Speech processing system with multilingual support."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize speech processor.
        
        Args:
            language: Language code for internationalization
        """
        self.language = language
        self.vocabulary_manager = get_vocabulary_manager()
        
        # Get localized terms
        processor_term = self.vocabulary_manager.translate_term('Speech Processor', 'genai', self.language)
        logger.info(f"{processor_term} initialized")
    
    def text_to_speech(self, text: str, voice: str = 'default') -> bytes:
        """
        Convert text to speech with localized logging.
        
        Args:
            text: Text to convert
            voice: Voice type
            
        Returns:
            bytes: Audio data
        """
        # Get localized terms
        converting_term = self.vocabulary_manager.translate_term('Converting text to speech', 'genai', self.language)
        logger.info(f"{converting_term} with voice: {voice}")
        
        # Simulate TTS
        # In a real implementation, this would call actual TTS system
        # For demonstration, we'll create mock audio data
        audio_data = f"Mock TTS audio for: {text}".encode('utf-8')
        
        logger.info(translate('tts_completed', self.language) or "Text-to-speech conversion completed")
        return audio_data
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """
        Convert speech to text with localized logging.
        
        Args:
            audio_data: Audio data
            
        Returns:
            str: Transcribed text
        """
        # Get localized terms
        converting_term = self.vocabulary_manager.translate_term('Converting speech to text', 'genai', self.language)
        logger.info(converting_term)
        
        # Simulate STT
        # In a real implementation, this would call actual STT system
        try:
            text = audio_data.decode('utf-8')
            transcribed_text = f"Transcribed: {text}"
        except:
            transcribed_text = "Transcribed: [Audio data]"
        
        logger.info(translate('stt_completed', self.language) or "Speech-to-text conversion completed")
        return transcribed_text


class DiffusionAI:
    """Diffusion AI for image generation with multilingual support."""
    
    def __init__(self, model_type: str = 'stable_diffusion', language: str = 'en'):
        """
        Initialize diffusion AI.
        
        Args:
            model_type: Type of diffusion model
            language: Language code for internationalization
        """
        self.model_type = model_type
        self.language = language
        self.vocabulary_manager = get_vocabulary_manager()
        
        # Get localized terms
        diffusion_term = self.vocabulary_manager.translate_term('Diffusion AI', 'genai', self.language)
        logger.info(f"{diffusion_term} ({model_type}) initialized")
    
    def generate_image(self, prompt: str, size: Tuple[int, int] = (512, 512)) -> bytes:
        """
        Generate image with localized logging.
        
        Args:
            prompt: Image generation prompt
            size: Image size (width, height)
            
        Returns:
            bytes: Generated image data
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating image', 'genai', self.language)
        logger.info(f"{generating_term}: {prompt}")
        
        # Simulate image generation
        # In a real implementation, this would call actual diffusion model
        width, height = size
        image_data = f"Mock image data ({width}x{height}) for prompt: {prompt}".encode('utf-8')
        
        logger.info(translate('image_generation_completed', self.language) or "Image generation completed")
        return image_data
    
    def generate_3d_model(self, prompt: str) -> Dict[str, Any]:
        """
        Generate 3D model with localized logging.
        
        Args:
            prompt: 3D model generation prompt
            
        Returns:
            dict: Generated 3D model data
        """
        # Get localized terms
        generating_term = self.vocabulary_manager.translate_term('Generating 3D model', 'genai', self.language)
        logger.info(f"{generating_term}: {prompt}")
        
        # Simulate 3D model generation
        # In a real implementation, this would call actual 3D diffusion model
        model_data = {
            'model_type': 'mesh',
            'vertices': np.random.randint(100, 1000),
            'faces': np.random.randint(50, 500),
            'prompt': prompt,
            'generated': datetime.now().isoformat(),
            'language': self.language
        }
        
        logger.info(translate('model_generation_completed', self.language) or "3D model generation completed")
        return model_data


class MCPIntegration:
    """MCP (Model Coordination Protocol) integration with multilingual support."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize MCP integration.
        
        Args:
            language: Language code for internationalization
        """
        self.language = language
        self.vocabulary_manager = get_vocabulary_manager()
        self.models = {}
        
        # Get localized terms
        mcp_term = self.vocabulary_manager.translate_term('MCP Integration', 'genai', self.language)
        logger.info(f"{mcp_term} initialized")
    
    def register_model(self, model_id: str, model: GenAIModel) -> None:
        """
        Register model with MCP with localized logging.
        
        Args:
            model_id: Model identifier
            model: Model instance
        """
        # Get localized terms
        registering_term = self.vocabulary_manager.translate_term('Registering model with MCP', 'genai', self.language)
        logger.info(f"{registering_term}: {model_id}")
        
        self.models[model_id] = model
        logger.info(translate('model_registered', self.language) or "Model registered with MCP")
    
    def coordinate_generation(self, task: str, models: List[str]) -> Dict[str, Any]:
        """
        Coordinate generation across multiple models with localized logging.
        
        Args:
            task: Generation task
            models: List of model IDs to use
            
        Returns:
            dict: Coordination results
        """
        # Get localized terms
        coordinating_term = self.vocabulary_manager.translate_term('Coordinating generation', 'genai', self.language)
        logger.info(f"{coordinating_term}: {task}")
        
        results = {}
        
        for model_id in models:
            if model_id in self.models:
                model = self.models[model_id]
                try:
                    # Get localized model term
                    model_term = self.vocabulary_manager.translate_term('Generating with model', 'genai', self.language)
                    logger.debug(f"{model_term}: {model_id}")
                    
                    # Generate response (simplified)
                    response = model.generate_text(task)
                    results[model_id] = {
                        'success': True,
                        'response': response,
                        'timestamp': datetime.now().isoformat()
                    }
                except Exception as e:
                    results[model_id] = {
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                results[model_id] = {
                    'success': False,
                    'error': 'Model not found',
                    'timestamp': datetime.now().isoformat()
                }
        
        results['task'] = task
        results['coordinated'] = datetime.now().isoformat()
        results['language'] = self.language
        
        logger.info(translate('coordination_completed', self.language) or "Generation coordination completed")
        return results


# Convenience functions for multilingual GenAI
def create_genai_model(model_name: str, api_key: Optional[str] = None, language: str = 'en') -> GenAIModel:
    """
    Create generic GenAI model with specified language.
    
    Args:
        model_name: Name of the model
        api_key: API key for cloud models
        language: Language code
        
    Returns:
        GenAIModel: Created GenAI model
    """
    return GenAIModel(model_name, api_key, language=language)


def create_openai_integration(api_key: str, model: str = 'gpt-4', language: str = 'en') -> OpenAIIntegration:
    """
    Create OpenAI integration with specified language.
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model name
        language: Language code
        
    Returns:
        OpenAIIntegration: Created OpenAI integration
    """
    return OpenAIIntegration(api_key, model, language=language)


def create_claude_integration(api_key: str, model: str = 'claude-3-opus', language: str = 'en') -> ClaudeIntegration:
    """
    Create Claude integration with specified language.
    
    Args:
        api_key: Claude API key
        model: Claude model name
        language: Language code
        
    Returns:
        ClaudeIntegration: Created Claude integration
    """
    return ClaudeIntegration(api_key, model, language=language)


def create_llama_integration(model_path: str, language: str = 'en') -> LLaMAIntegration:
    """
    Create LLaMA integration with specified language.
    
    Args:
        model_path: Path to LLaMA model
        language: Language code
        
    Returns:
        LLaMAIntegration: Created LLaMA integration
    """
    return LLaMAIntegration(model_path, language=language)


def create_gigachat3_integration(api_key: str, language: str = 'en') -> GigaChat3Integration:
    """
    Create GigaChat3 integration with specified language.
    
    Args:
        api_key: GigaChat3 API key
        language: Language code
        
    Returns:
        GigaChat3Integration: Created GigaChat3 integration
    """
    return GigaChat3Integration(api_key, language=language)


def create_katya_ai_integration(language: str = 'en') -> KatyaAIIntegration:
    """
    Create Katya AI integration with specified language.
    
    Args:
        language: Language code
        
    Returns:
        KatyaAIIntegration: Created Katya AI integration
    """
    return KatyaAIIntegration(language=language)


def create_speech_processor(language: str = 'en') -> SpeechProcessor:
    """
    Create speech processor with specified language.
    
    Args:
        language: Language code
        
    Returns:
        SpeechProcessor: Created speech processor
    """
    return SpeechProcessor(language=language)


def create_diffusion_ai(model_type: str = 'stable_diffusion', language: str = 'en') -> DiffusionAI:
    """
    Create diffusion AI with specified language.
    
    Args:
        model_type: Type of diffusion model
        language: Language code
        
    Returns:
        DiffusionAI: Created diffusion AI
    """
    return DiffusionAI(model_type, language=language)


def create_mcp_integration(language: str = 'en') -> MCPIntegration:
    """
    Create MCP integration with specified language.
    
    Args:
        language: Language code
        
    Returns:
        MCPIntegration: Created MCP integration
    """
    return MCPIntegration(language=language)