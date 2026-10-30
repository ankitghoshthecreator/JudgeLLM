"""
Test Suite for Bias Investigation Pipeline and Multi-LLM Environment Integration
"""

from __future__ import annotations

import os
from pathlib import Path

from evaluation.pdf_qa_pipeline import (
    _load_env,
    compute_verbosity_stats,
    detect_position_bias,
    generate_questions,
    get_available_models,
    get_default_judge_model,
    get_default_model_1,
    judge_answers,
    stream_answer,
)


def test_env_loading_and_defaults():
    """Verify that .env is loaded and default model resolvers pick up env values."""
    _load_env()
    available = get_available_models()
    assert len(available) > 0
    m1 = get_default_model_1()
    judge_m = get_default_judge_model()
    assert isinstance(m1, str) and len(m1) > 0
    assert isinstance(judge_m, str) and len(judge_m) > 0
    assert m1 in available
    assert judge_m in available


def test_mock_question_generation():
    """Verify that mock question generation produces the requested number of questions."""
    # Force mock mode by clearing API key temporarily if needed
    qs = generate_questions("Mock financial text about revenue and margin.", n=5)
    assert len(qs) == 5
    assert all(isinstance(q, str) and len(q) > 5 for q in qs)


def test_mock_stream_answering():
    """Verify streaming token generator yields chunks that combine into a coherent answer."""
    tokens = list(stream_answer("What is the Q3 operating margin?", "Context text", model="llama-3.1-8b-instant"))
    full_text = "".join(tokens)
    assert len(full_text) > 20
    assert "margin" in full_text.lower() or "answer" in full_text.lower() or "financial" in full_text.lower()


def test_judge_scoring_and_position_bias():
    """Verify judge ratings and position bias detection when swapping order."""
    ans_a = "The operating margin is 30.4%, reflecting strong cost discipline across divisions."
    ans_b = "Margin was 30.4%."
    res = judge_answers("What was the margin?", ans_a, ans_b, model_a="Model 1", model_b="Model 2")

    assert "score_a" in res
    assert "score_b" in res
    assert 1 <= res["score_a"] <= 10
    assert 1 <= res["score_b"] <= 10
    assert "reasoning_a" in res
    assert res["preferred"] in ["A", "B", "Tie"]

    pos = detect_position_bias("What was the margin?", ans_a, ans_b, "Model 1", "Model 2")
    assert "position_bias_detected" in pos
    assert "avg_score_delta" in pos
    assert isinstance(pos["position_bias_detected"], bool)


def test_verbosity_bias_detection():
    """Verify verbosity bias correlation statistics."""
    session_data = [
        {
            "question_idx": 0,
            "answer_a": "Short answer here.",
            "answer_b": "A much longer and much more elaborate answer here with many extra words describing everything in detail.",
            "score_a": 4,
            "score_b": 8,
        },
        {
            "question_idx": 1,
            "answer_a": "Brief reply.",
            "answer_b": "Another extensively verbose response detailing the comprehensive mechanics and background context at length.",
            "score_a": 5,
            "score_b": 9,
        },
        {
            "question_idx": 2,
            "answer_a": "Concise point.",
            "answer_b": "Substantial elaboration providing background, historical perspective, and multi-factor breakdown across multiple sentences.",
            "score_a": 5,
            "score_b": 9,
        },
    ]
    stats = compute_verbosity_stats(session_data)
    assert "longer_answer_win_rate" in stats
    assert stats["longer_answer_win_rate"] == 1.0
    assert stats["bias_detected"] is True
