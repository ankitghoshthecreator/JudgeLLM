"""
Groq API backend implementation for JudgeLLM.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from evaluation.backends.base import BackendPricing, JudgeBackend

logger = logging.getLogger(__name__)


class GroqJudgeBackend(JudgeBackend):
    """Judge backend utilizing the official Groq SDK."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        pricing: Optional[BackendPricing] = None,
        **kwargs: Any,
    ):
        model = (
            model_name
            or os.getenv("GROQ_JUDGE_MODEL")
            or os.getenv("GROQ_MODEL")
            or "llama-3.3-70b-versatile"
        )
        # Default pricing for Llama-3.3-70B on Groq (~$0.00059 / 1k prompt, $0.00079 / 1k completion)
        default_pricing = pricing or BackendPricing(prompt_per_1k=0.00059, completion_per_1k=0.00079)
        super().__init__(model_name=model, pricing=default_pricing, **kwargs)

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Cannot initialize GroqJudgeBackend.")

        from groq import Groq
        self.client = Groq(api_key=self.api_key)

    @property
    def backend_type(self) -> str:
        return "groq"

    def generate_judgment(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        json_mode: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
