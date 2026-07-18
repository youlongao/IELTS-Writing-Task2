"""Prompt template manager — loads and renders versioned prompts."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Represents a single prompt template with system and user parts."""

    def __init__(self, name: str, version: str, system: str, user: str):
        self.name = name
        self.version = version
        self.system = system
        self.user = user

    def render(self, **kwargs) -> Dict[str, str]:
        """Render the template with variable substitution.

        Args:
            **kwargs: Variables to substitute in the template.
                      Uses {variable_name} placeholders.

        Returns:
            Dict with 'system' and 'user' keys containing rendered strings.
        """
        system_msg = self.system
        user_msg = self.user

        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            system_msg = system_msg.replace(placeholder, str(value))
            user_msg = user_msg.replace(placeholder, str(value))

        return {"system": system_msg.strip(), "user": user_msg.strip()}


class PromptManager:
    """Manages loading and caching of versioned prompt templates.

    Usage:
        manager = PromptManager()
        prompt = manager.get("question_analyzer", "v1")
        rendered = prompt.render(question="...")
        # rendered = {"system": "...", "user": "..."}
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize the prompt manager.

        Args:
            prompts_dir: Path to prompts/versions/ directory.
        """
        if prompts_dir is None:
            prompts_dir = str(
                Path(__file__).parent / "versions"
            )
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, PromptTemplate] = {}

    def get(self, name: str, version: str = "v1") -> PromptTemplate:
        """Get a prompt template by name and version.

        Args:
            name: Template name (e.g. 'question_analyzer').
            version: Version string (e.g. 'v1').

        Returns:
            PromptTemplate instance.

        Raises:
            FileNotFoundError: If the template file doesn't exist.
            ValueError: If the template format is invalid.
        """
        cache_key = f"{version}_{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try the versioned filename format: v1_question_analyzer.yaml
        filepath = self.prompts_dir / f"{version}_{name}.yaml"
        if not filepath.exists():
            # Try alternative: v1/question_analyzer.yaml
            filepath = self.prompts_dir / version / f"{name}.yaml"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {name} (version: {version}). "
                f"Searched: {self.prompts_dir}"
            )

        template = self._load_yaml(filepath, name, version)
        self._cache[cache_key] = template
        return template

    def _load_yaml(
        self, filepath: Path, name: str, version: str
    ) -> PromptTemplate:
        """Load a prompt template from a YAML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid prompt template format: {filepath}")

        system = data.get("system", "")
        user = data.get("user", "")

        if not user:
            raise ValueError(
                f"Prompt template missing 'user' section: {filepath}"
            )

        return PromptTemplate(
            name=name,
            version=version,
            system=system.strip(),
            user=user.strip(),
        )

    def list_templates(self) -> Dict[str, list]:
        """List all available templates and versions.

        Returns:
            Dict mapping template name -> list of available versions.
        """
        templates: Dict[str, list] = {}

        for filepath in self.prompts_dir.glob("*.yaml"):
            # Parse filename: v1_question_analyzer.yaml
            stem = filepath.stem
            parts = stem.split("_", 1)
            if len(parts) == 2:
                version, name = parts
                if name not in templates:
                    templates[name] = []
                templates[name].append(version)

        return templates

    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()
