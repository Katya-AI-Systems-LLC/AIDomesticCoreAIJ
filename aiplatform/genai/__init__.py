"""
GenAI Integration module for AIPlatform SDK

This module provides integration with various Generative AI models
including OpenAI, Claude, LLaMA, GigaChat3-702B, and Katya AI systems.
"""

from .models import GenAIModel, ModelProvider, ModelConfig
from .integrations import OpenAIIntegration, ClaudeIntegration, LLaMAIntegration
from .katya import KatyaAI, KatyaSpeech
from .diffusion import DiffusionModel, Diffusion3D
from .mcp import MCPInterface
from ..exceptions import GenAIError

__all__ = [
    'GenAIModel',
    'ModelProvider',
    'ModelConfig',
    'OpenAIIntegration',
    'ClaudeIntegration', 
    'LLaMAIntegration',
    'KatyaAI',
    'KatyaSpeech',
    'DiffusionModel',
    'Diffusion3D',
    'MCPInterface',
    'GenAIError'
]

__version__ = '1.0.0'
__author__ = 'REChain Network Solutions & Katya AI Systems'