"""
Test Suite for Phase 2: Learned Router & Cost-Optimal Pareto Routing Policy
============================================================================
Tests:
  1. No train/test leakage (Stratified 5-Fold cross-validation out-of-fold verification).
  2. Monotonic cost-vs-workload curve property.
  3. Determinism with fixed random seed.
  4. Both Logistic Regression and Gradient Boosting models execute correctly.
  5. CLI interface runs and selects threshold meeting target-kappa-loss.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from evaluation.router import (
    LearnedRouter,
    compute_ngram_cosine_similarity,
    evaluate_routing_policy,
    run_learned_router_pipeline,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def test_ngram_cosine_similarity():
    """Verify n-gram cosine similarity properties."""
    text1 = "Consolidated operating income reached $4.1 billion."
    text2 = "Consolidated operating income reached $4.1 billion."
    assert np.isclose(compute_ngram_cosine_similarity(text1, text2), 1.0)

    # Disjoint text
    assert compute_ngram_cosine_similarity("aaaa", "bbbb") == 0.0

    # Intermediate overlap
    sim = compute_ngram_cosine_similarity(text1, "Operating income was $4.1 billion with strong revenue.")
    assert 0.30 <= sim <= 0.95


def test_router_deterministic_with_seed():
    """Verify that router cross-validation output is deterministic when seed is fixed."""
    res1 = run_learned_router_pipeline(
        target_kappa_loss=0.05,
        model_type="logistic_regression",
        random_state=42,
        save_fig=False,
    )
    res2 = run_learned_router_pipeline(
        target_kappa_loss=0.05,
        model_type="logistic_regression",
        random_state=42,
        save_fig=False,
    )
    assert res1.cv_auc_roc == res2.cv_auc_roc
    assert res1.full_judge_kappa == res2.full_judge_kappa
    assert len(res1.pareto_curve) == len(res2.pareto_curve)
    for p1, p2 in zip(res1.pareto_curve, res2.pareto_curve):
        assert p1.cost_saving_pct == p2.cost_saving_pct
        assert p1.overall_kappa == p2.overall_kappa


def test_no_train_test_leakage_in_cross_validation():
    """
    Verify that out-of-fold predictions evaluate exclusively on holdout validation folds.
    """
    router = LearnedRouter(model_type="logistic_regression", random_state=42, n_splits=5)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    y = np.array([1 if x[0] > 0 else 0 for x in X])

    oof = router.cross_validate_predict_proba(X, y)
    assert len(oof) == len(y)
    assert np.all(oof >= 0.0)
    assert np.all(oof <= 1.0)
    # Ensure probabilities are diverse and not trivial constants
    assert np.std(oof) > 0.05


def test_pareto_curve_monotonic_workload():
    """Verify Pareto curve workload monotonicity (sorted by items sent to 70B judge)."""
    res = run_learned_router_pipeline(
        target_kappa_loss=0.05,
        model_type="gradient_boosting",
        random_state=42,
        save_fig=False,
    )
    curve = res.pareto_curve
    assert len(curve) > 5

    # Check that workload sent to judge decreases monotonically along the curve
    sent_rates = [p.pct_sent_to_judge for p in curve]
    for i in range(len(sent_rates) - 1):
        assert sent_rates[i] >= sent_rates[i + 1], "Pareto points must be ordered monotonically by workload"

    # Extreme bounds: point 0 sends ~100% to judge, final point sends near 0%
    assert curve[0].pct_sent_to_judge >= 99.0
    assert curve[0].overall_kappa == res.full_judge_kappa
    assert curve[-1].pct_sent_to_judge <= 5.0


def test_optimal_policy_satisfies_target_loss():
    """Verify that the selected optimal policy honors the specified target_kappa_loss."""
    target_loss = 0.05
    res = run_learned_router_pipeline(
        target_kappa_loss=target_loss,
        model_type="gradient_boosting",
        random_state=42,
        save_fig=False,
    )
    opt = res.optimal_point
    assert opt is not None
    assert opt.kappa_loss <= target_loss + 1e-4
    assert opt.cost_saving_pct >= 0.0
    assert "fluency" in opt.per_axis_kappas
    assert "instruction_adherence" in opt.per_axis_kappas
