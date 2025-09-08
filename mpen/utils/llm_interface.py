"""
Language Model Interface for MPEN agents.
"""

import os
from typing import Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, config: Dict[str, Any]):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed")
        
        self.client = openai.OpenAI(
            api_key=config.get('api_key', os.getenv('OPENAI_API_KEY'))
        )
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.default_params = config.get('default_params', {})
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI API."""
        params = {**self.default_params, **kwargs}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return "Error: Failed to generate response"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, config: Dict[str, Any]):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed")
        
        self.client = anthropic.Anthropic(
            api_key=config.get('api_key', os.getenv('ANTHROPIC_API_KEY'))
        )
        self.model = config.get('model', 'claude-3-sonnet-20240229')
        self.default_params = config.get('default_params', {})
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Anthropic API."""
        params = {**self.default_params, **kwargs}
        
        try:
            # Convert messages format for Anthropic
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    user_messages.append(msg)
            
            response = self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=user_messages,
                max_tokens=params.get('max_tokens', 1000),
                **{k: v for k, v in params.items() if k != 'max_tokens'}
            )
            
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            return "Error: Failed to generate response"


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.responses = config.get('mock_responses', [
            "This is a mock response for testing.",
            "Another mock response with different content.",
            "Mock response focusing on the specific request."
        ])
        self.response_index = 0
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Return mock response."""
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response


class LLMInterface:
    """
    Interface for interacting with various language models.
    
    Supports OpenAI GPT, Anthropic Claude, and mock providers for testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM interface.
        
        Args:
            config: Configuration dictionary with provider, model, and parameters
        """
        self.config = config
        self.provider_name = config.get('provider', 'mock')
        self.logger = logging.getLogger(f"mpen.llm.{self.provider_name}")
        
        # Initialize provider
        if self.provider_name == 'openai':
            self.provider = OpenAIProvider(config)
        elif self.provider_name == 'anthropic':
            self.provider = AnthropicProvider(config)
        elif self.provider_name == 'mock':
            self.provider = MockProvider(config)
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")
        
        self.logger.info(f"Initialized LLM interface with {self.provider_name} provider")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        Generate a response from the language model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Generated text response
        """
        try:
            response = self.provider.generate(messages, **kwargs)
            self.logger.debug(f"Generated response: {response[:100]}...")
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "Error: Unable to generate response"
    
    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """
        Generate response with system and user prompts.
        
        Args:
            system_prompt: System-level instruction
            user_prompt: User query or instruction
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.generate(messages, **kwargs)
    
    def batch_generate(
        self,
        message_batches: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple message batches.
        
        Args:
            message_batches: List of message lists
            **kwargs: Additional parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for messages in message_batches:
            try:
                response = self.generate(messages, **kwargs)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Error in batch generation: {e}")
                responses.append("Error: Batch generation failed")
        
        return responses
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'provider': self.provider_name,
            'model': getattr(self.provider, 'model', 'unknown'),
            'config': self.config
        }


def create_llm_interface(
    provider: str = 'mock',
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMInterface:
    """
    Factory function to create LLM interface.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'mock')
        model: Model name (provider-specific)
        api_key: API key for the provider
        **kwargs: Additional configuration
        
    Returns:
        Configured LLMInterface instance
    """
    config = {
        'provider': provider,
        'api_key': api_key,
        **kwargs
    }
    
    if model:
        config['model'] = model
    
    return LLMInterface(config)
