"""Configuration loader using YAML files and environment variables."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


class Config:
    """Loads and manages application configuration.

    Priority: env vars > YAML config > defaults.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_dir: Path to the config directory. Defaults to project_root/config/.
        """
        # Load .env file
        load_dotenv()

        # Determine project root
        self.project_root = Path(__file__).parent.parent.parent
        if config_dir is None:
            config_dir = str(self.project_root / "config")

        self.config_dir = Path(config_dir)

        # Load YAML configs
        self._defaults = self._load_yaml("default.yaml")
        self._models = self._load_yaml("models.yaml")

    def _load_yaml(self, filename: str) -> dict:
        """Load a YAML config file."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a config value by dot-separated path.

        Args:
            key_path: Dot-separated key path (e.g. "rag.chunk_size").
            default: Default value if key not found.

        Returns:
            The config value, or default.
        """
        keys = key_path.split(".")
        value = self._defaults
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    def get_model_config(self, provider: str) -> dict:
        """Get model configuration for a specific provider.

        Args:
            provider: Provider name ("openai" or "deepseek").

        Returns:
            Dict with model_name, temperature, max_tokens, etc.
        """
        return self._models.get(provider, {})

    @property
    def default_provider(self) -> str:
        """Get the default LLM provider name."""
        return self.get("app.default_provider", "openai")

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")

    @property
    def deepseek_api_key(self) -> Optional[str]:
        """Get DeepSeek API key from environment."""
        return os.getenv("DEEPSEEK_API_KEY")

    @property
    def chroma_persist_dir(self) -> str:
        """Get ChromaDB persistence directory (absolute path)."""
        relative = self.get("rag.chroma_persist_dir", "./data/chroma_db")
        return str(self.project_root / relative)

    @property
    def chunk_size(self) -> int:
        """Get RAG chunk size."""
        return self.get("rag.chunk_size", 500)

    @property
    def chunk_overlap(self) -> int:
        """Get RAG chunk overlap."""
        return self.get("rag.chunk_overlap", 50)
