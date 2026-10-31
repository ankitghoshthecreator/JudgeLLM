"""
Deterministic mock backend for zero-network testing and fallback simulation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from evaluation.backends.base import BackendPricing, JudgeBackend


class MockJudgeBackend(JudgeBackend):
    """
    Mock LLM judge backend that yields valid JSON responses without network access.
    Can be configured with fixed responses or dynamic generation.
    """

    def __init__(
        self,
        model_name: str = "mock-judge-70b",
        fixed_evaluation: Optional[Dict[str, Any]] = None,
        pricing: Optional[BackendPricing] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, pricing=pricing or BackendPricing(), **kwargs)
        self.fixed_evaluation = fixed_evaluation
        self.call_count = 0
        self.history: List[List[Dict[str, str]]] = []

    @property
    def backend_type(self) -> str:
        return "mock"

    def generate_judgment(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        json_mode: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.call_count += 1
        self.history.append(messages)

        if self.fixed_evaluation:
            return json.dumps(self.fixed_evaluation)

        # Inspect prompt text for any heuristic cues
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        # Default fallback response
        response = {
            "fluency": 0.85,
            "coherence": 0.80,
            "instruction_adherence": 0.82,
            "no_hallucination": 0.88,
            "reasoning": f"MOCK judge evaluation on backend '{self.model_name}'. High fluency and sound reasoning.",
        }

        if "fail" in user_msg.lower() or "hallucination" in user_msg.lower():
            response["no_hallucination"] = 0.35
            response["reasoning"] = "MOCK detected severe hallucination or failure."

        return json.dumps(response)
