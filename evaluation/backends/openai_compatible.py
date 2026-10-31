"""
OpenAI-compatible HTTP backend for vLLM, Ollama, LM Studio, or any OpenAI-compatible API.
Uses standard library urllib (thin adapter) so no extra dependencies are required.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from evaluation.backends.base import BackendPricing, JudgeBackend

logger = logging.getLogger(__name__)


class OpenAICompatibleBackend(JudgeBackend):
    """
    Connects to any standard /v1/chat/completions endpoint.
    Ideal for local air-gapped on-prem deployments (vLLM, Ollama, LocalAI, TGI, LM Studio).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        pricing: Optional[BackendPricing] = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        model = (
            model_name
            or os.getenv("OPENAI_MODEL")
            or os.getenv("LOCAL_MODEL")
            or "llama3.3:70b"
        )
        # On-prem default is zero marginal cost per token
        default_pricing = pricing or BackendPricing(prompt_per_1k=0.0, completion_per_1k=0.0)
        super().__init__(model_name=model, pricing=default_pricing, **kwargs)

        # Normalize base_url
        raw_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LOCAL_BASE_URL") or "http://localhost:11434/v1"
        raw_url = raw_url.rstrip("/")
        if not raw_url.endswith("/chat/completions"):
            self.endpoint = f"{raw_url}/chat/completions"
        else:
            self.endpoint = raw_url

        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LOCAL_API_KEY") or "EMPTY"
        self.timeout = timeout

    @property
    def backend_type(self) -> str:
        return "openai_compatible"

    def generate_judgment(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        json_mode: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "JudgeLLM/1.0",
        }

        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
                choices = parsed.get("choices", [])
                if not choices:
                    raise ValueError(f"No choices in OpenAI-compatible response: {body}")
                message = choices[0].get("message", {})
                return message.get("content", "") or ""
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8", errors="replace")
            # If server doesn't support response_format json_object (e.g. older Ollama), retry without it
            if json_mode and ("response_format" in error_msg or e.code == 400):
                logger.warning("Endpoint rejected response_format, retrying without json_mode: %s", error_msg)
                payload.pop("response_format", None)
                retry_req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(retry_req, timeout=self.timeout) as resp:
                    p = json.loads(resp.read().decode("utf-8"))
                    return p["choices"][0]["message"]["content"]
            raise RuntimeError(f"OpenAI compatible request failed ({e.code}): {error_msg}") from e
        except Exception as exc:
            raise RuntimeError(f"OpenAI compatible connection failed to {self.endpoint}: {exc}") from exc
