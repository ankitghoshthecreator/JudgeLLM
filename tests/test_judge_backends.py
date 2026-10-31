"""
Unit tests for provider-agnostic judge backends and on-prem integration.
Uses fake and mock backends to guarantee zero network access during testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from evaluation.backends import (
    BackendPricing,
    JudgeBackend,
    MockJudgeBackend,
    OpenAICompatibleBackend,
    create_backend,
    load_backend_from_config,
)
from evaluation.teacher_judge import JudgeEvaluation, TeacherJudge


class FakeNetworkBackend(JudgeBackend):
    """Fake backend that simulates network responses and errors deterministically."""

    def __init__(self, response_queue: List[str], model_name: str = "fake-judge-model"):
        super().__init__(model_name=model_name)
        self.response_queue = list(response_queue)
        self.call_history: List[List[Dict[str, str]]] = []

    @property
    def backend_type(self) -> str:
        return "fake"

    def generate_judgment(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        json_mode: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        self.call_history.append(messages)
        if not self.response_queue:
            raise RuntimeError("Fake backend ran out of scripted responses")
        resp = self.response_queue.pop(0)
        if resp.startswith("ERROR:"):
            raise RuntimeError(resp[6:])
        return resp


def test_mock_backend_interface():
    """Verify that MockJudgeBackend returns valid JSON conforming to expectations."""
    backend = MockJudgeBackend(model_name="test-mock-70b")
    messages = [{"role": "user", "content": "Evaluate this financial summary."}]
    raw = backend.generate_judgment(messages)
    data = json.loads(raw)

    assert "fluency" in data
    assert "coherence" in data
    assert "instruction_adherence" in data
    assert "no_hallucination" in data
    assert "reasoning" in data
    assert backend.call_count == 1


def test_backend_pricing_estimation():
    """Verify cost projection formula."""
    pricing = BackendPricing(prompt_per_1k=0.0015, completion_per_1k=0.0020)
    cost = pricing.estimate_cost(prompt_tokens=2000, completion_tokens=1000)
    # (2000/1000 * 0.0015) + (1000/1000 * 0.0020) = 0.0030 + 0.0020 = 0.0050
    assert pytest.approx(cost, 1e-6) == 0.0050


def test_factory_creates_correct_backend():
    """Verify backend factory parses arguments and configurations appropriately."""
    mock_b = create_backend(backend_type="mock")
    assert isinstance(mock_b, MockJudgeBackend)

    openai_b = create_backend(
        backend_type="openai_compatible",
        base_url="http://localhost:11434/v1",
        model_name="llama3.3:70b",
        api_key="test-key",
    )
    assert isinstance(openai_b, OpenAICompatibleBackend)
    assert openai_b.endpoint == "http://localhost:11434/v1/chat/completions"
    assert openai_b.model_name == "llama3.3:70b"


def test_load_backend_from_yaml_config(tmp_path: Path):
    """Verify loading backend configuration from YAML file."""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(
        """
judge_backend:
  type: openai_compatible
  base_url: http://127.0.0.1:8000/v1
  model_name: meta-llama/Llama-3.3-70B-Instruct
  api_key: secret-airgap-token
  pricing:
    prompt_per_1k: 0.0
    completion_per_1k: 0.0
""",
        encoding="utf-8",
    )

    loaded = load_backend_from_config(cfg_file)
    assert isinstance(loaded, OpenAICompatibleBackend)
    assert loaded.endpoint == "http://127.0.0.1:8000/v1/chat/completions"
    assert loaded.api_key == "secret-airgap-token"
    assert loaded.pricing.prompt_per_1k == 0.0


def test_teacher_judge_with_fake_backend_retry_and_recovery():
    """
    Verify that TeacherJudge retries on invalid output, feeds error back,
    and successfully validates once a valid schema JSON is returned.
    """
    rubric_path = Path(__file__).resolve().parent.parent / "evaluation" / "rubrics" / "rubric_b.yaml"

    invalid_json = '{"fluency": "invalid_str", "reasoning": "bad"}'
    valid_json = json.dumps({
        "fluency": 0.95,
        "coherence": 0.90,
        "instruction_adherence": 0.85,
        "no_hallucination": 0.88,
        "reasoning": "Recovered from initial formatting error.",
    })

    fake = FakeNetworkBackend(response_queue=[invalid_json, valid_json])
    judge = TeacherJudge(rubric_path=rubric_path, backend=fake, max_retries=3)

    result = judge.evaluate_single(
        prompt="State company revenue growth.",
        reference_context="Revenue expanded 14% year-over-year.",
        model_output="Revenue increased 14% compared to prior year.",
    )

    assert isinstance(result, JudgeEvaluation)
    assert result.fluency == 0.95
    assert result.coherence == 0.90
    assert len(fake.call_history) == 2
    # Second prompt must contain schema error feedback
    second_call_msgs = fake.call_history[1]
    assert any("failed validation" in m["content"] for m in second_call_msgs)


def test_teacher_judge_clamp_and_schema_enforcement():
    """Verify that scores outside 0.0-1.0 or on 0-10/0-100 scales get normalized cleanly."""
    eval_obj = JudgeEvaluation(
        fluency=9.5,  # 0-10 scale
        coherence=85.0,  # 0-100 scale
        instruction_adherence=0.75,
        no_hallucination=2.5,  # 0-3 scale
        reasoning="Testing schema normalization.",
    )
    assert eval_obj.fluency == 0.95
    assert eval_obj.coherence == 0.85
    assert eval_obj.instruction_adherence == 0.75
    assert pytest.approx(eval_obj.no_hallucination, 0.01) == 0.833
