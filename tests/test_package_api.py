"""
Unit and integration tests for public API and packaging.
Tests audit_judge, route, detect_bias, and package exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import judgellm
from judgellm import AuditReport, audit_judge, detect_bias, route


def test_package_exports():
    """Verify core public API functions and classes are accessible at top-level."""
    assert hasattr(judgellm, "audit_judge")
    assert hasattr(judgellm, "route")
    assert hasattr(judgellm, "detect_bias")
    assert hasattr(judgellm, "TeacherJudge")
    assert hasattr(judgellm, "JudgeBackend")
    assert hasattr(judgellm, "JudgeDiskCache")


def test_audit_judge_api_with_mock_config(tmp_path: Path):
    """Verify that audit_judge generates a valid AuditReport with CIs and Markdown output."""
    traces = [
        {
            "id": "probe_001",
            "prompt": "What is the CET1 capital ratio?",
            "reference_context": "CET1 ratio is 14.8%.",
            "model_output": "The reported CET1 ratio is 14.8%.",
            "category": "high_quality",
        },
        {
            "id": "probe_002",
            "prompt": "List drivers of commercial real estate stress.",
            "reference_context": "Office vacancies above 22%.",
            "model_output": "Office vacancies above 22%.",
            "category": "hallucination",
        },
    ]

    labels = [
        {
            "id": "probe_001",
            "ground_truth_quality": {
                "fluency": 0.9,
                "coherence": 0.9,
                "instruction_adherence": 0.9,
                "no_hallucination": 0.9,
            },
        },
        {
            "id": "probe_002",
            "ground_truth_quality": {
                "fluency": 0.8,
                "coherence": 0.8,
                "instruction_adherence": 0.8,
                "no_hallucination": 0.3,
            },
        },
    ]

    mock_cfg = {"backend_type": "mock", "mock_mode": True}
    report = audit_judge(
        traces=traces,
        labels=labels,
        judge_config=mock_cfg,
        n_resamples=100,
        seed=42,
    )

    assert isinstance(report, AuditReport)
    assert report.n_evaluated == 2
    assert report.judge_model == "mock-judge"
    assert "fluency" in report.axis_results
    assert len(report.overall_kappa_ci_95) == 2

    # Verify Markdown serialization
    md = report.to_markdown()
    assert "# LLM Judge Reliability Audit Report" in md
    assert "Overall Cohen's Kappa" in md


def test_public_route_api():
    """Verify route function returns decisions for batch traces."""
    sample_traces = [
        {"id": "p1", "model_output": "Short response."},
        {"id": "p2", "model_output": "Very long output " * 50},
    ]
    decisions = route(sample_traces, policy={"threshold": 0.5})
    assert len(decisions) == 2
    assert "route_to_judge" in decisions[0]
    assert "risk_score" in decisions[0]


def test_public_detect_bias_api():
    """Verify detect_bias returns correlation and bias indicators."""
    traces = [
        {"model_output": "Short answer.", "fluency": 0.5},
        {"model_output": "A moderately longer answer.", "fluency": 0.7},
        {"model_output": "An extremely verbose and lengthy explanation of financial terms.", "fluency": 0.9},
    ]
    res = detect_bias(traces)
    assert "verbosity_bias" in res
    assert "pearson_r" in res["verbosity_bias"]
    assert "position_bias" in res
    assert "self_preference_bias" in res
