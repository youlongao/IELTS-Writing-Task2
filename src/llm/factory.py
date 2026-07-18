"""LLM provider factory for creating and switching between providers."""

from typing import Optional

from .base import LLMProvider


class LLMFactory:
    """Factory for creating LLM provider instances.

    Usage:
        config = Config()
        llm = LLMFactory.create("openai", config)
        response = llm.generate("Hello!")
    """

    @staticmethod
    def create(provider: str, config, model_overrides: Optional[dict] = None) -> LLMProvider:
        """Create an LLM provider instance.

        Args:
            provider: Provider name ("openai" or "deepseek").
            config: Config instance for API keys and model settings.
            model_overrides: Optional dict to override model settings.
                Keys: model_name, temperature, max_tokens.

        Returns:
            An LLMProvider instance.

        Raises:
            ValueError: If provider is not supported.
        """
        model_config = config.get_model_config(provider)

        if model_overrides:
            model_config = {**model_config, **model_overrides}

        model_name = model_config.get("model_name", "gpt-4o")
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 2048)

        if provider == "openai":
            from .openai_provider import OpenAIProvider

            api_key = config.openai_api_key
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Please set it in .env file or environment variable."
                )
            return OpenAIProvider(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=model_config.get("base_url"),
            )

        elif provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            from .openai_provider import OpenAIProvider

            api_key = config.deepseek_api_key
            if not api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY not set. Please set it in .env file or environment variable."
                )
            return OpenAIProvider(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=model_config.get("base_url", "https://api.deepseek.com"),
            )

        else:
            raise ValueError(
                f"Unknown provider: '{provider}'. Supported: 'openai', 'deepseek'."
            )
