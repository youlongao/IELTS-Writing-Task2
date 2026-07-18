"""OpenAI GPT provider implementation.

Handles both standard models (GPT-4o) and reasoning models (GPT-5, o-series).
"""

import json
import time
import logging
from typing import Any, Dict, Optional

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI GPT models.

    Supports:
    - Standard models: GPT-4o, GPT-4-turbo, etc.
    - Reasoning models: GPT-5, o1, o3, o4 (no temperature, no system role)
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens)

        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text via OpenAI Chat Completion API.

        Handles both standard and reasoning models automatically.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        system_message = kwargs.get(
            "system_message",
            "You are an IELTS writing tutor. Respond in Chinese with English writing examples.",
        )

        reasoning = self._is_reasoning_model()

        # Reasoning models use a single user message with the system guidance merged in.
        if reasoning:
            # For reasoning models, merge system into user message
            messages = [
                {"role": "user", "content": f"{system_message}\n\n---\n\n{prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]

        # Build parameters
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }

        if not reasoning:
            params["temperature"] = temperature

        # Retry up to 2 times for empty responses
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**params)
                content = response.choices[0].message.content

                if content:
                    return content

                # If content is empty, log and retry
                logger.warning(
                    f"Empty response from {self.model_name} (attempt {attempt + 1}/3)"
                )
                if attempt < 2:
                    time.sleep(2)

            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise

        return "(The model returned an empty response. Please retry.)"

    def _is_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model."""
        reasoning_prefixes = ("gpt-5", "o1", "o3", "o4")
        return self.model_name.lower().startswith(reasoning_prefixes)

    def generate_structured(
        self, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        schema_instruction = (
            f"\n\nReturn ONLY valid JSON matching this schema, nothing else:\n"
            f"```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"
        )
        full_prompt = prompt + schema_instruction

        response_text = self.generate(full_prompt, **kwargs)
        return self._parse_json_response(response_text)

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text, "parse_error": True}
