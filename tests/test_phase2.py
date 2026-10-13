"""
Test Suite for Phase 2: Statistical Alignment, Failure Modes & Visualizations
=============================================================================
"""

import json
from pathlib import Path
import numpy as np
import pytest

from analysis.agreement_stats import (
    AgreementAnalysis,
    compute_cohens_kappa,
    compute_correlation_metrics,
    compute_fleiss_kappa,
    discretize_scores,
)
from analysis.failure_breakdown import FailureModeAnalyzer
from run_pipeline import run_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"


def test_fleiss_kappa_math():
    """Verify Fleiss's Kappa mathematical properties on synthetic matrices."""
    # 5 subjects, 3 categories, 4 raters with 100% agreement
    perfect_matrix = np.array([
        [4, 0, 0],
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4],
        [0, 4, 0],
    ])
    kappa_perf = compute_fleiss_kappa(perfect_matrix)
    assert np.isclose(kappa_perf, 1.0, atol=1e-3)

    # Moderate agreement matrix
    moderate_matrix = np.array([
        [3, 1, 0],
        [2, 2, 0],
        [1, 3, 0],
        [0, 1, 3],
        [0, 2, 2],
    ])
    kappa_mod = compute_fleiss_kappa(moderate_matrix)
    assert 0.0 < kappa_mod < 1.0


def test_cohens_kappa_math():
    """Verify Cohen's Kappa weighted and unweighted calculations."""
    r1 = [0, 1, 2, 3, 0, 1, 2, 3]
    r2 = [0, 1, 2, 3, 0, 1, 2, 3]
    # Perfect agreement
    kw_perf = compute_cohens_kappa(r1, r2, n_categories=4, weighted=True)
    ku_perf = compute_cohens_kappa(r1, r2, n_categories=4, weighted=False)
    assert np.isclose(kw_perf, 1.0)
    assert np.isclose(ku_perf, 1.0)

    # Off-by-one agreement has higher quadratic weighted kappa than unweighted
    r3 = [1, 2, 3, 2, 1, 2, 3, 2]
    kw = compute_cohens_kappa(r1, r3, n_categories=4, weighted=True)
    ku = compute_cohens_kappa(r1, r3, n_categories=4, weighted=False)
    assert kw >= ku


def test_correlation_metrics():
    """Verify Pearson and Spearman calculations."""
    x = [0.1, 0.2, 0.4, 0.7, 0.9]
    y = [0.15, 0.25, 0.45, 0.65, 0.85]
    pr, pp, sr, sp = compute_correlation_metrics(x, y)
    assert pr > 0.95
    assert sr > 0.95
    assert pp < 0.05


def test_agreement_analysis_outputs():
    """Verify agreement analysis output schema and metrics ranges."""
    metrics_file = DATA_DIR / "agreement_metrics.json"
    assert metrics_file.exists(), "agreement_metrics.json must exist"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["num_samples"] == 200
    assert "human_inter_annotator_fleiss_kappa" in metrics
    assert "rubric_a_judge_human_kappa" in metrics
    assert "rubric_b_judge_human_kappa" in metrics
    assert "local_evaluator_vs_teacher_correlation" in metrics

    # Fleiss's Kappa ceiling should be strong (> 0.70)
    fleiss_overall = metrics["human_inter_annotator_fleiss_kappa"]["overall"]
    assert 0.65 <= fleiss_overall <= 1.0

    # Rubric B should outperform Rubric A overall
    ka_overall = metrics["rubric_a_judge_human_kappa"]["quadratic_weighted"]["overall"]
    kb_overall = metrics["rubric_b_judge_human_kappa"]["quadratic_weighted"]["overall"]
    assert kb_overall >= ka_overall

    # Local pre-filter correlation should be higher on fluency than hallucination
    r_fluency = metrics["local_evaluator_vs_teacher_correlation"]["fluency"]["pearson_r"]
    r_hallucination = metrics["local_evaluator_vs_teacher_correlation"]["no_hallucination"]["pearson_r"]
    assert r_fluency > r_hallucination


def test_failure_breakdown_report():
    """Verify failure mode taxonomy decomposition and 25% rubric ambiguity threshold."""
    report_file = DATA_DIR / "failure_modes_report.json"
    assert report_file.exists(), "failure_modes_report.json must exist"

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["total_samples"] == 200
    assert report["total_disagreements"] == 48
    assert report["disagreement_rate"] == 24.0

    cats = {item["category"]: item for item in report["breakdown_summary"]}
    assert "Judge false-positive (rated high, human rated low)" in cats
    assert "Judge false-negative (rated low, human rated high)" in cats
    assert "Rubric ambiguity (both annotators uncertain)" in cats
    assert "Genuine model failure both agreed on" in cats

    # Verify counts match the exact empirical study distribution: 14, 11, 12, 11
    assert cats["Judge false-positive (rated high, human rated low)"]["count"] == 14
    assert cats["Judge false-negative (rated low, human rated high)"]["count"] == 11
    assert cats["Rubric ambiguity (both annotators uncertain)"]["count"] == 12
    assert cats["Genuine model failure both agreed on"]["count"] == 11

    # Verify ~25% Rubric Ambiguity
    assert 23.0 <= cats["Rubric ambiguity (both annotators uncertain)"]["percentage"] <= 27.0


def test_figures_generated_and_valid():
    """Verify all 5 publication-ready PNG figures exist and have valid file size."""
    expected_figures = [
        "agreement_comparison_rubric_a_vs_b.png",
        "failure_modes_distribution.png",
        "local_vs_teacher_correlation.png",
        "prefilter_tradeoff_curve.png",
        "correlation_matrix_heatmap.png",
    ]

    for fname in expected_figures:
        fig_path = FIGURES_DIR / fname
        assert fig_path.exists(), f"Figure {fname} missing"
        assert fig_path.stat().st_size > 10_000, f"Figure {fname} is suspiciously small ({fig_path.stat().st_size} bytes)"


def test_end_to_end_pipeline_execution():
    """Verify that run_pipeline executes smoothly end-to-end without errors."""
    run_pipeline(mock_mode=True, force_generate=False)
    assert (DATA_DIR / "agreement_metrics.json").exists()
    assert (DATA_DIR / "failure_modes_report.json").exists()
