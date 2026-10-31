"""
Tests for LLM Judge drift monitoring, CI calculation, and CI-aware gate thresholds.
Verifies:
  1. Identical runs produce zero false alarm drift flags.
  2. Injected score perturbations / degradation trigger statistically significant drift alerts.
  3. CI-aware gate logic properly assigns PASS, WARN, and FAIL based on confidence intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from evaluation.backends.base import JudgeBackend
from evaluation.eval_gate import evaluate_ci_gate, run_evaluation_gates
from evaluation.teacher_judge import TeacherJudge
from monitoring.drift import bootstrap_score_difference, run_drift_check


class ScriptedEvaluationBackend(JudgeBackend):
    """Returns scripted evaluation scores with an optional shift offset."""

    def __init__(self, score_offset: float = 0.0, model_name: str = "scripted-test-judge"):
        super().__init__(model_name=model_name)
        self.score_offset = score_offset

    @property
    def backend_type(self) -> str:
        return "scripted"

    def generate_judgment(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        json_mode: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        base_f = min(1.0, max(0.0, 0.85 + self.score_offset))
        base_c = min(1.0, max(0.0, 0.80 + self.score_offset))
        base_a = min(1.0, max(0.0, 0.82 + self.score_offset))
        base_h = min(1.0, max(0.0, 0.88 + self.score_offset))

        return json.dumps({
            "fluency": round(base_f, 3),
            "coherence": round(base_c, 3),
            "instruction_adherence": round(base_a, 3),
            "no_hallucination": round(base_h, 3),
            "reasoning": f"Scripted evaluation with offset {self.score_offset:+0.2f}",
        })


@pytest.fixture
def sample_drift_probes() -> List[Dict[str, Any]]:
    return [
        {
            "id": f"probe_{i:03d}",
            "prompt": f"Financial question #{i}",
            "reference_context": "Financial reference context statement.",
            "model_output": f"Financial model response #{i}",
        }
        for i in range(25)
    ]


@pytest.fixture
def sample_baseline_and_human() -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    baseline = {}
    human = {}
    for i in range(25):
        pid = f"probe_{i:03d}"
        baseline[pid] = {
            "fluency": 0.85,
            "coherence": 0.80,
            "instruction_adherence": 0.82,
            "no_hallucination": 0.88,
        }
        human[pid] = {
            "fluency": 0.85,
            "coherence": 0.80,
            "instruction_adherence": 0.82,
            "no_hallucination": 0.88,
        }
    return baseline, human


def test_no_false_alarm_on_identical_run(sample_drift_probes, sample_baseline_and_human, tmp_path: Path):
    """Verify identical scores against baseline yield drift_detected=False."""
    baseline, human = sample_baseline_and_human
    rubric_p = Path(__file__).resolve().parent.parent / "evaluation" / "rubrics" / "rubric_b.yaml"

    backend = ScriptedEvaluationBackend(score_offset=0.0)
    judge = TeacherJudge(rubric_path=rubric_p, backend=backend)

    report = run_drift_check(
        golden_probes=sample_drift_probes,
        baseline_records=baseline,
        human_consensus=human,
        judge=judge,
        history_file=tmp_path / "drift.jsonl",
        mean_shift_threshold=0.08,
        min_kappa_floor=0.65,
        n_resamples=500,
        seed=42,
    )

    assert report.overall_drift_detected is False
    assert report.axis_metrics["fluency"].drift_detected is False
    assert abs(report.axis_metrics["fluency"].mean_shift) < 0.05


def test_drift_detected_on_injected_score_shift(sample_drift_probes, sample_baseline_and_human, tmp_path: Path):
    """Verify significant score degradation (+0.25 offset shift) triggers drift alert."""
    baseline, human = sample_baseline_and_human
    rubric_p = Path(__file__).resolve().parent.parent / "evaluation" / "rubrics" / "rubric_b.yaml"

    # Inject large negative degradation
    backend = ScriptedEvaluationBackend(score_offset=-0.35)
    judge = TeacherJudge(rubric_path=rubric_p, backend=backend)

    report = run_drift_check(
        golden_probes=sample_drift_probes,
        baseline_records=baseline,
        human_consensus=human,
        judge=judge,
        history_file=tmp_path / "drift.jsonl",
        mean_shift_threshold=0.08,
        min_kappa_floor=0.65,
        n_resamples=500,
        seed=42,
    )

    assert report.overall_drift_detected is True
    # The negative mean shift must be captured
    f_metric = report.axis_metrics["fluency"]
    assert f_metric.mean_shift < -0.20
    assert f_metric.drift_detected is True


def test_ci_aware_gate_evaluation_logic():
    """Verify PASS, WARN, and FAIL states based on confidence interval positioning."""
    # Case 1: Strictly clears floor -> PASS
    passed, status, _ = evaluate_ci_gate(observed=0.78, ci_95=[0.72, 0.84], floor=0.70, is_lower_bound=True)
    assert passed is True
    assert status == "PASS"

    # Case 2: Point estimate >= floor, but lower CI overlaps below floor -> WARN
    passed, status, _ = evaluate_ci_gate(observed=0.72, ci_95=[0.67, 0.77], floor=0.70, is_lower_bound=True)
    assert passed is True
    assert status == "WARN"

    # Case 3: Entire CI lies below floor -> FAIL
    passed, status, _ = evaluate_ci_gate(observed=0.64, ci_95=[0.58, 0.69], floor=0.70, is_lower_bound=True)
    assert passed is False
    assert status == "FAIL"


def test_bootstrap_score_difference_math():
    """Verify bootstrap mean shift computation on synthetic arrays."""
    curr = np.array([0.9, 0.8, 0.85, 0.95, 0.88])
    base = np.array([0.7, 0.6, 0.65, 0.75, 0.68])
    # Exact shift is 0.20
    est, lo, hi = bootstrap_score_difference(curr, base, n_resamples=500, seed=42)
    assert pytest.approx(est, 0.01) == 0.20
    assert lo > 0.15
    assert hi < 0.25
