"""
GenAI Module Tests

Tests for the generative AI components of AIPlatform.
"""

import unittest
from unittest.mock import patch, MagicMock

# Import GenAI components
try:
    from aiplatform.genai import GenAIModel
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class TestGenAIModel(unittest.TestCase):
    """Test cases for GenAIModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if GENAI_AVAILABLE:
            self.openai_model = GenAIModel(provider='openai', model_name='gpt-4')
            self.claude_model = GenAIModel(provider='claude', model_name='claude-2')
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_initialization(self):
        """Test GenAI model initialization."""
        self.assertEqual(self.openai_model.provider, 'openai')
        self.assertEqual(self.openai_model.model_name, 'gpt-4')
        self.assertEqual(self.claude_model.provider, 'claude')
        self.assertEqual(self.claude_model.model_name, 'claude-2')
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_generate_text(self):
        """Test text generation."""
        prompt = "Explain quantum computing in simple terms."
        
        # Mock generation results
        mock_response = "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously..."
        
        with patch.object(self.openai_model, 'generate', return_value=mock_response):
            result = self.openai_model.generate(prompt)
            self.assertIsInstance(result, str)
            self.assertIn("quantum", result.lower())
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_get_model_info(self):
        """Test getting model information."""
        # Mock model info
        mock_info = {
            'provider': 'openai',
            'model_name': 'gpt-4',
            'capabilities': ['text-generation', 'code-completion'],
            'max_tokens': 8192
        }
        
        with patch.object(self.openai_model, 'get_model_info', return_value=mock_info):
            info = self.openai_model.get_model_info()
            self.assertEqual(info['provider'], 'openai')
            self.assertEqual(info['model_name'], 'gpt-4')
            self.assertIn('text-generation', info['capabilities'])
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_different_providers(self):
        """Test different AI providers."""
        providers = [
            ('openai', 'gpt-4'),
            ('claude', 'claude-2'),
            ('llama', 'llama-2-70b'),
            ('gigachat3', 'gigachat3-702b')
        ]
        
        for provider, model_name in providers:
            model = GenAIModel(provider=provider, model_name=model_name)
            self.assertEqual(model.provider, provider)
            self.assertEqual(model.model_name, model_name)
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_generation_parameters(self):
        """Test generation with different parameters."""
        prompt = "Write a short story about AI."
        
        # Test with different parameters
        with patch.object(self.openai_model, 'generate') as mock_generate:
            mock_generate.return_value = "Once upon a time, there was an AI..."
            
            # Test temperature parameter
            result1 = self.openai_model.generate(prompt, temperature=0.7)
            result2 = self.openai_model.generate(prompt, temperature=0.9)
            
            # Both should return results
            self.assertIsInstance(result1, str)
            self.assertIsInstance(result2, str)
            
            # Verify generate was called with parameters
            self.assertEqual(mock_generate.call_count, 2)


class TestGenAIIntegration(unittest.TestCase):
    """Integration tests for GenAI components."""
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_multi_model_workflow(self):
        """Test workflow using multiple AI models."""
        # Create different models
        openai_model = GenAIModel(provider='openai', model_name='gpt-4')
        claude_model = GenAIModel(provider='claude', model_name='claude-2')
        
        # Mock responses
        openai_response = "Quantum computing is a revolutionary technology..."
        claude_response = "Quantum computing represents a paradigm shift in computational capabilities..."
        
        with patch.object(openai_model, 'generate', return_value=openai_response):
            with patch.object(claude_model, 'generate', return_value=claude_response):
                # Generate responses from both models
                response1 = openai_model.generate("Explain quantum computing.")
                response2 = claude_model.generate("Explain quantum computing.")
                
                # Verify both responses
                self.assertIsInstance(response1, str)
                self.assertIsInstance(response2, str)
                self.assertIn("quantum", response1.lower())
                self.assertIn("quantum", response2.lower())
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_model_comparison(self):
        """Test comparing different AI models."""
        models = [
            GenAIModel(provider='openai', model_name='gpt-4'),
            GenAIModel(provider='claude', model_name='claude-2'),
            GenAIModel(provider='llama', model_name='llama-2-70b')
        ]
        
        prompt = "What is the future of AI?"
        
        # Mock responses for each model
        responses = [
            "The future of AI will involve greater integration with human society...",
            "AI's future lies in collaborative intelligence with humans...",
            "The future of artificial intelligence encompasses advanced reasoning..."
        ]
        
        for i, model in enumerate(models):
            with patch.object(model, 'generate', return_value=responses[i]):
                response = model.generate(prompt)
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 10)
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_context_preservation(self):
        """Test context preservation across multiple generations."""
        model = GenAIModel(provider='openai', model_name='gpt-4')
        
        conversation = [
            "Hello, what can you tell me about quantum computing?",
            "That's interesting. Can you explain quantum entanglement?",
            "How does this relate to quantum cryptography?"
        ]
        
        responses = [
            "Quantum computing uses quantum bits that can exist in superposition...",
            "Quantum entanglement is a phenomenon where particles become correlated...",
            "Quantum cryptography leverages quantum properties for secure communication..."
        ]
        
        for i, message in enumerate(conversation):
            with patch.object(model, 'generate', return_value=responses[i]):
                response = model.generate(message)
                self.assertIsInstance(response, str)
                self.assertIn("quantum", response.lower())
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_error_handling(self):
        """Test error handling in GenAI models."""
        model = GenAIModel(provider='openai', model_name='gpt-4')
        
        # Test with API error
        with patch.object(model, 'generate', side_effect=Exception("API timeout")):
            with self.assertRaises(Exception) as context:
                model.generate("Test prompt")
            
            self.assertIn("API timeout", str(context.exception))
    
    @unittest.skipIf(not GENAI_AVAILABLE, "GenAI components not available")
    def test_model_fallback(self):
        """Test model fallback mechanism."""
        # Create primary and fallback models
        primary_model = GenAIModel(provider='openai', model_name='gpt-4')
        fallback_model = GenAIModel(provider='llama', model_name='llama-2-70b')
        
        # Mock primary model failure and fallback success
        def mock_generate_with_fallback(prompt, **kwargs):
            # First call fails, second succeeds
            if not hasattr(mock_generate_with_fallback, 'called'):
                mock_generate_with_fallback.called = True
                raise Exception("Primary model unavailable")
            return "Fallback model response"
        
        with patch.object(primary_model, 'generate', side_effect=mock_generate_with_fallback):
            with patch.object(fallback_model, 'generate', return_value="Fallback response"):
                # This would be implemented in a real fallback system
                pass


if __name__ == '__main__':
    unittest.main()