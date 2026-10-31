"""
Unit and integration tests for FastAPI judgment service.
Tests:
  - GET /health
  - GET /metrics (Prometheus format)
  - POST /judge (single evaluation)
  - POST /judge/batch (bounded concurrent evaluation)
  - Cache hit vs cache miss behavior
  - Retry & exponential backoff on transient errors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from fastapi.testclient import TestClient

from evaluation.backends.base import JudgeBackend
from evaluation.teacher_judge import TeacherJudge
from judgellm.api import app, set_active_judge
from judgellm.cache import JudgeDiskCache


class MockFlakyBackend(JudgeBackend):
    """Simulates a backend that fails on the first attempt and succeeds on retry."""

    def __init__(self, model_name: str = "flaky-test-judge"):
        super().__init__(model_name=model_name)
        self.call_count = 0

    @property
    def backend_type(self) -> str:
        return "mock_flaky"

    def generate_judgment(
        self,
        messages: List[dict],
        temperature: float = 0.1,
        json_mode: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        self.call_count += 1
        if self.call_count == 1:
            raise ConnectionResetError("Simulated temporary connection reset")
        return json.dumps({
            "fluency": 0.92,
            "coherence": 0.88,
            "instruction_adherence": 0.85,
            "no_hallucination": 0.90,
            "reasoning": "Recovered successfully on retry attempt.",
        })


@pytest.fixture(autouse=True)
def setup_test_judge(tmp_path: Path):
    """Sets a deterministic fake judge and isolated temporary cache for tests."""
    rubric_path = Path(__file__).resolve().parent.parent / "evaluation" / "rubrics" / "rubric_b.yaml"
    backend = MockFlakyBackend()
    judge = TeacherJudge(rubric_path=rubric_path, backend=backend, max_retries=2)
    cache = JudgeDiskCache(tmp_path / "cache")
    set_active_judge(judge, cache=cache)


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "judgellm"
    assert "git_commit_hash" in data
    assert "rubric_version" in data


def test_metrics_endpoint():
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "judgellm_requests_total" in resp.text
    assert "judgellm_cache_hits_total" in resp.text


def test_single_judgment_and_retry_recovery():
    """Verify endpoint recovers from transient error and returns compliant schema."""
    client = TestClient(app)
    payload = {
        "prompt": "State CET1 ratio requirement.",
        "reference_context": "The regulatory CET1 minimum is 10.5%.",
        "model_output": "The minimum requirement is 10.5%.",
    }
    resp = client.post("/judge", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fluency"] == 0.92
    assert data["cache_hit"] is False
    assert "git_commit_hash" in data


def test_disk_cache_hit_on_subsequent_request():
    """Verify that identical request hits disk cache with cache_hit=True."""
    client = TestClient(app)
    payload = {
        "prompt": "Explain repo margin calls.",
        "reference_context": "Margin calls occur when collateral value drops below haircut.",
        "model_output": "A margin call occurs when haircut boundary is breached.",
    }

    # First request: cache miss
    resp1 = client.post("/judge", json=payload)
    assert resp1.status_code == 200
    assert resp1.json()["cache_hit"] is False

    # Second request: cache hit
    resp2 = client.post("/judge", json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["cache_hit"] is True
    assert resp2.json()["estimated_cost_usd"] == 0.0


def test_batch_judgment_endpoint():
    """Verify concurrent batch evaluation endpoint."""
    client = TestClient(app)
    items = [
        {"prompt": f"Financial QA #{i}", "model_output": f"Answer #{i}"}
        for i in range(3)
    ]
    resp = client.post("/judge/batch", json={"items": items, "max_concurrency": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    for item in data:
        assert "fluency" in item
        assert "no_hallucination" in item
