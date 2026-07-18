"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMProvider(ABC):
    """Abstract interface for LLM providers.

    All LLM providers (OpenAI, DeepSeek, etc.) must implement this interface.
    """

    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 2048):
        """Initialize the provider.

        Args:
            model_name: The model identifier string.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in the response.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text response from a prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text response.
        """
        ...

    @abstractmethod
    def generate_structured(
        self, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate a structured (JSON) response conforming to a schema.

        Args:
            prompt: The input prompt string.
            schema: JSON Schema dict the response must conform to.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Parsed dict matching the schema.
        """
        ...
