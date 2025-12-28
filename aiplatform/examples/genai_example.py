"""
GenAI Integration Example for AIPlatform SDK

This example demonstrates generative AI capabilities with multilingual support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiplatform.genai import (
    create_genai_model,
    create_openai_integration,
    create_claude_integration,
    create_llama_integration,
    create_gigachat3_integration,
    create_katya_ai_integration,
    create_speech_processor,
    create_diffusion_ai,
    create_mcp_integration
)
import numpy as np


def genai_model_example(language='en'):
    """Demonstrate generic GenAI model."""
    print(f"=== {translate('genai_model_example', language) or 'Generic GenAI Model Example'} ===")
    
    # Create generic GenAI model
    model = create_genai_model("GenericAI-1.0", language=language)
    
    # Generate text
    prompt = "Explain quantum computing in simple terms"
    generated_text = model.generate_text(prompt, max_tokens=200)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text[:100]}...")
    print()


def openai_integration_example(language='en'):
    """Demonstrate OpenAI integration."""
    print(f"=== {translate('openai_integration_example', language) or 'OpenAI Integration Example'} ===")
    
    # Create OpenAI integration (simulated)
    openai = create_openai_integration("fake-api-key", "gpt-4", language=language)
    
    # Chat completion
    messages = [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
    
    response = openai.chat_completion(messages, temperature=0.7)
    print(f"Chat completion: {response['choices'][0]['message']['content'][:100]}...")
    print()


def claude_integration_example(language='en'):
    """Demonstrate Claude integration."""
    print(f"=== {translate('claude_integration_example', language) or 'Claude Integration Example'} ===")
    
    # Create Claude integration (simulated)
    claude = create_claude_integration("fake-api-key", "claude-3-opus", language=language)
    
    # Generate with context
    prompt = "Summarize the benefits of quantum computing"
    context = "The audience is high school students with basic physics knowledge"
    
    generated_text = claude.generate_with_context(prompt, context, max_tokens=300)
    print(f"Context-aware generation: {generated_text[:100]}...")
    print()


def llama_integration_example(language='en'):
    """Demonstrate LLaMA integration."""
    print(f"=== {translate('llama_integration_example', language) or 'LLaMA Integration Example'} ===")
    
    # Create LLaMA integration (simulated)
    llama = create_llama_integration("/path/to/llama/model", language=language)
    
    # Generate with sampling
    prompt = "Write a poem about artificial intelligence"
    
    generated_text = llama.generate_with_sampling(prompt, temperature=0.8, top_p=0.9)
    print(f"Sampling-based generation: {generated_text[:100]}...")
    print()


def gigachat3_integration_example(language='en'):
    """Demonstrate GigaChat3 integration."""
    print(f"=== {translate('gigachat3_integration_example', language) or 'GigaChat3 Integration Example'} ===")
    
    # Create GigaChat3 integration (simulated)
    gigachat3 = create_gigachat3_integration("fake-api-key", language=language)
    
    # Generate multilingual text
    prompt = "Hello, how are you?"
    target_language = "ru"  # Russian
    
    generated_text = gigachat3.generate_multilingual(prompt, target_language)
    print(f"Multilingual generation: {generated_text[:100]}...")
    print()


def katya_ai_integration_example(language='en'):
    """Demonstrate Katya AI integration."""
    print(f"=== {translate('katya_ai_integration_example', language) or 'Katya AI Integration Example'} ===")
    
    # Create Katya AI integration
    katya_ai = create_katya_ai_integration(language=language)
    
    # Generate with personality
    prompt = "Tell me a joke"
    personality = "humorous"
    
    generated_text = katya_ai.generate_with_personality(prompt, personality)
    print(f"Personality-based generation: {generated_text[:100]}...")
    print()


def speech_processing_example(language='en'):
    """Demonstrate speech processing."""
    print(f"=== {translate('speech_processing_example', language) or 'Speech Processing Example'} ===")
    
    # Create speech processor
    speech_processor = create_speech_processor(language=language)
    
    # Text to speech
    text = "Hello, this is a text to speech demonstration"
    audio_data = speech_processor.text_to_speech(text, voice="default")
    
    print(f"TTS result: {len(audio_data)} bytes of audio data")
    
    # Speech to text
    transcribed_text = speech_processor.speech_to_text(audio_data)
    print(f"STT result: {transcribed_text}")
    print()


def diffusion_ai_example(language='en'):
    """Demonstrate Diffusion AI."""
    print(f"=== {translate('diffusion_ai_example', language) or 'Diffusion AI Example'} ===")
    
    # Create Diffusion AI
    diffusion = create_diffusion_ai("stable_diffusion", language=language)
    
    # Generate image
    prompt = "A beautiful landscape with mountains and a lake"
    image_data = diffusion.generate_image(prompt, size=(256, 256))
    
    print(f"Generated image: {len(image_data)} bytes")
    
    # Generate 3D model
    model_prompt = "A 3D model of a futuristic car"
    model_data = diffusion.generate_3d_model(model_prompt)
    
    print(f"Generated 3D model: {model_data}")
    print()


def mcp_integration_example(language='en'):
    """Demonstrate MCP integration."""
    print(f"=== {translate('mcp_integration_example', language) or 'MCP Integration Example'} ===")
    
    # Create MCP integration
    mcp = create_mcp_integration(language=language)
    
    # Register models
    model1 = create_genai_model("Model-1", language=language)
    model2 = create_genai_model("Model-2", language=language)
    
    mcp.register_model("model_1", model1)
    mcp.register_model("model_2", model2)
    
    # Coordinate generation
    task = "Write a short story about a robot learning to paint"
    models = ["model_1", "model_2"]
    
    results = mcp.coordinate_generation(task, models)
    print(f"MCP coordination results:")
    for model_id, result in results.items():
        if model_id != 'task' and model_id != 'coordinated' and model_id != 'language':
            print(f"  {model_id}: {'Success' if result['success'] else 'Failed'}")
    print()


def translate(key, language):
    """Simple translation function for example titles."""
    translations = {
        'genai_model_example': {
            'ru': 'Пример общей модели GenAI',
            'zh': '通用GenAI模型示例',
            'ar': 'مثال نموذج GenAI العام'
        },
        'openai_integration_example': {
            'ru': 'Пример интеграции OpenAI',
            'zh': 'OpenAI集成示例',
            'ar': 'مثال تكامل OpenAI'
        },
        'claude_integration_example': {
            'ru': 'Пример интеграции Claude',
            'zh': 'Claude集成示例',
            'ar': 'مثال تكامل Claude'
        },
        'llama_integration_example': {
            'ru': 'Пример интеграции LLaMA',
            'zh': 'LLaMA集成示例',
            'ar': 'مثال تكامل LLaMA'
        },
        'gigachat3_integration_example': {
            'ru': 'Пример интеграции GigaChat3',
            'zh': 'GigaChat3集成示例',
            'ar': 'مثال تكامل GigaChat3'
        },
        'katya_ai_integration_example': {
            'ru': 'Пример интеграции Katya AI',
            'zh': 'Katya AI集成示例',
            'ar': 'مثال تكامل Katya AI'
        },
        'speech_processing_example': {
            'ru': 'Пример обработки речи',
            'zh': '语音处理示例',
            'ar': 'مثال معالجة الكلام'
        },
        'diffusion_ai_example': {
            'ru': 'Пример Diffusion AI',
            'zh': 'Diffusion AI示例',
            'ar': 'مثال Diffusion AI'
        },
        'mcp_integration_example': {
            'ru': 'Пример интеграции MCP',
            'zh': 'MCP集成示例',
            'ar': 'مثال تكامل MCP'
        }
    }
    
    if key in translations and language in translations[key]:
        return translations[key][language]
    return None


def main():
    """Run all GenAI examples."""
    languages = ['en', 'ru', 'zh', 'ar']
    
    for language in languages:
        print(f"\n{'='*50}")
        print(f"GENAI EXAMPLES - {language.upper()}")
        print(f"{'='*50}\n")
        
        try:
            genai_model_example(language)
            openai_integration_example(language)
            claude_integration_example(language)
            llama_integration_example(language)
            gigachat3_integration_example(language)
            katya_ai_integration_example(language)
            speech_processing_example(language)
            diffusion_ai_example(language)
            mcp_integration_example(language)
        except Exception as e:
            print(f"Error in {language} examples: {e}")


if __name__ == "__main__":
    main()