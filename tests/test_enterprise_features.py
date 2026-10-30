"""
Test Suite for Enterprise Features:
1. Reference-Free vs. Reference-Based Evaluation Modes
2. Multi-Judge Panel & Variance Ensembling
3. CI/CD Quality Regression Gating
"""

from __future__ import annotations

from pathlib import Path

from evaluation.ensemble_judge import MultiJudgeEnsemble
from evaluation.eval_gate import run_evaluation_gates
from evaluation.local_scorer import LightweightStudentEvaluator
from evaluation.teacher_judge import TeacherJudge

BASE_DIR = Path(__file__).resolve().parent.parent


def test_reference_based_local_evaluator():
    """Verify reference-based mode assesses lexical overlap with context."""
    evaluator = LightweightStudentEvaluator()
    prompt = "Extract Q3 revenue."
    context = "Q3 consolidated revenue was $14.2 billion with 8.4% growth."
    model_output = "Consolidated revenue in Q3 reached $14.2 billion."

    res = evaluator.evaluate(
        model_output=model_output,
        prompt=prompt,
        reference_context=context,
        mode="reference_based",
    )

    assert res.evaluation_mode == "reference_based"
    assert res.jaccard_overlap > 0.10
    assert res.fluency > 0.60
    assert res.no_hallucination > 0.40


def test_reference_free_local_evaluator():
    """Verify reference-free mode evaluates self-contained quality without ground truth context."""
    evaluator = LightweightStudentEvaluator()
    prompt = "Draft an executive summary of market risk hedging."
    model_output = (
        "Market risk hedging involves derivatives such as interest rate swaps and futures. "
        "These instruments protect balance sheet capital against unexpected volatility and liquidity shocks."
    )

    res = evaluator.evaluate(
        model_output=model_output,
        prompt=prompt,
        reference_context="",  # No context provided
        mode="reference_free",
    )

    assert res.evaluation_mode == "reference_free"
    assert res.raw_metrics["evaluation_mode"] == "reference_free"
    assert res.fluency > 0.60
    assert res.coherence > 0.60
    assert res.no_hallucination > 0.50


def test_reference_free_teacher_judge():
    """Verify teacher judge handles reference-free mode cleanly."""
    rubric_b = BASE_DIR / "evaluation" / "rubrics" / "rubric_b.yaml"
    judge = TeacherJudge(rubric_path=rubric_b, mock_mode=True)
    res = judge.evaluate_single(
        prompt="Explain liquidity risk.",
        reference_context="",
        model_output="Liquidity risk is the inability of an institution to meet its debt obligations without incurring unaccetable losses.",
        mode="reference_free",
    )
    assert 0.0 <= res.fluency <= 1.0
    assert 0.0 <= res.coherence <= 1.0
    assert len(res.reasoning) > 5


def test_multi_judge_cross_model_ensemble():
    """Verify cross-model judge ensembling aggregates scores and estimates variance."""
    ensemble = MultiJudgeEnsemble(
        panel_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        mock_mode=True,
    )
    res = ensemble.evaluate_cross_model(
        prompt="Summarize operating income.",
        reference_context="Operating income stood at $4.1B.",
        model_output="Operating income reached $4.1B in Q3.",
    )

    assert res.num_judges == 3
    assert res.ensemble_type == "cross_model"
    assert "fluency" in res.consensus_scores
    assert "no_hallucination" in res.consensus_scores
    assert 0.0 <= res.consensus_scores["fluency"] <= 1.0
    assert "fluency" in res.variance
    assert res.uncertainty_index >= 0.0
    assert len(res.individual_evaluations) == 3


def test_multi_judge_self_consistency_ensemble():
    """Verify self-consistency ensembling across temperature-sampled iterations."""
    ensemble = MultiJudgeEnsemble(mock_mode=True)
    res = ensemble.evaluate_self_consistency(
        prompt="State the capital ratio.",
        reference_context="CET1 capital ratio was 13.8%.",
        model_output="The CET1 capital ratio was 13.8%.",
        n_samples=3,
        temperature=0.7,
    )

    assert res.num_judges == 3
    assert res.ensemble_type == "self_consistency"
    assert "coherence" in res.median_scores
    assert res.uncertainty_index >= 0.0


def test_multi_judge_qa_pair_comparison():
    """Verify 10-star rating multi-judge panel for dual-model answering."""
    ensemble = MultiJudgeEnsemble(mock_mode=True)
    res = ensemble.evaluate_qa_pair(
        question="What is the CET1 minimum?",
        answer_a="Basel III mandates a minimum CET1 ratio of 10.5%.",
        answer_b="The minimum requirement is 10.5%.",
        model_a_name="Model 1",
        model_b_name="Model 2",
    )

    assert "consensus_score_a" in res
    assert "consensus_score_b" in res
    assert 1.0 <= res["consensus_score_a"] <= 10.0
    assert 1.0 <= res["consensus_score_b"] <= 10.0
    assert "std_dev_a" in res
    assert res["panel_preference"] in ["A", "B", "Tie"]
    assert len(res["panel_evaluations"]) >= 2


def test_eval_gate_execution():
    """Verify the CI/CD evaluation gate passes against production thresholds."""
    passed, checks, report_md = run_evaluation_gates(quick_mode=True)
    assert passed is True
    assert len(checks) == 6
    assert all(c.passed for c in checks)
    assert "PASSED (All Quality Gates Satisfied)" in report_md
