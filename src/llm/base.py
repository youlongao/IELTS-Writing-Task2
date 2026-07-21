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
        self.last_usage: Dict[str, int] = {}
        self.usage_total: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.call_count = 0

    def reset_usage(self):
        """Reset accumulated token usage before a harness run."""
        self.last_usage = {}
        self.usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.call_count = 0

    def record_usage(self, usage: Optional[Dict[str, int]]):
        """Record provider token usage when the API exposes it."""
        self.call_count += 1
        if not usage:
            self.last_usage = {}
            return

        normalized = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        self.last_usage = normalized
        for key, value in normalized.items():
            self.usage_total[key] = self.usage_total.get(key, 0) + value

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
