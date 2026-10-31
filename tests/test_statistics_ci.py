"""
Test Suite for Phase 1 Defensible Statistics:
1. Bootstrap confidence interval calculations for Fleiss's Kappa, Cohen's Kappa, Pearson r, and Spearman rho.
2. Synthetic verification with known analytical properties.
3. Statistical power and Minimum Detectable Effect (MDE) calculations.
4. SEC EDGAR ingestion pipeline with Fair-Access compliance and 'pending_annotation' verification.
5. Benchmark validation loader schema integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from analysis.agreement_stats import (
    bootstrap_cohen_kappa,
    bootstrap_correlation_metrics,
    bootstrap_fleiss_kappa,
    compute_minimum_detectable_effect_kappa,
)
from benchmark_validation.loaders import load_public_benchmark
from scripts.load_sec_edgar import SECEdgarLoader, build_offline_sample_filings


def test_bootstrap_cohen_kappa_perfect_and_imperfect():
    """Verify that bootstrap confidence intervals contain the point estimate on synthetic data."""
    # Perfect agreement
    r1 = np.array([0, 1, 2, 3] * 25)
    r2 = np.array([0, 1, 2, 3] * 25)
    est, low, high = bootstrap_cohen_kappa(r1, r2, n_categories=4, weighted=True, n_resamples=500, seed=42)
    assert est == 1.0
    assert low == 1.0
    assert high == 1.0

    # Intermediate agreement
    rng = np.random.default_rng(123)
    r_rand1 = rng.integers(0, 4, size=100)
    r_rand2 = r_rand1.copy()
    # Introduce 20% random flips
    flips = rng.choice(100, size=20, replace=False)
    r_rand2[flips] = rng.integers(0, 4, size=20)

    est_mid, low_mid, high_mid = bootstrap_cohen_kappa(r_rand1, r_rand2, n_categories=4, weighted=True, n_resamples=1000, seed=42)
    assert 0.50 <= est_mid <= 0.95
    assert low_mid <= est_mid <= high_mid
    assert low_mid < high_mid


def test_bootstrap_fleiss_kappa_properties():
    """Verify Fleiss's Kappa bootstrap calculation on known synthetic agreement matrices."""
    # 50 subjects, 3 raters all agreeing
    perfect_mat = np.zeros((50, 4), dtype=int)
    perfect_mat[:, 1] = 3  # all raters assign category 1
    est, low, high = bootstrap_fleiss_kappa(perfect_mat, n_resamples=500, seed=42)
    assert np.isclose(est, 1.0)
    assert np.isclose(low, 1.0)
    assert np.isclose(high, 1.0)

    # Moderate agreement matrix
    mod_mat = np.array([[2, 1, 0, 0], [1, 2, 0, 0], [0, 2, 1, 0], [0, 0, 2, 1]] * 25)
    est_m, low_m, high_m = bootstrap_fleiss_kappa(mod_mat, n_resamples=500, seed=42)
    assert 0.0 < est_m < 1.0
    assert low_m <= est_m <= high_m


def test_bootstrap_correlation_metrics():
    """Verify Pearson and Spearman bootstrap confidence intervals on linear vs noisy synthetic signals."""
    x = np.linspace(0.0, 1.0, 100)
    y_perf = x.copy()
    res_perf = bootstrap_correlation_metrics(x, y_perf, n_resamples=500, seed=42)
    assert res_perf["pearson_r"] == 1.0
    assert res_perf["pearson_ci"] == [1.0, 1.0]

    # Noisy signal
    rng = np.random.default_rng(999)
    y_noisy = x + rng.normal(0, 0.2, size=100)
    res_noisy = bootstrap_correlation_metrics(x, y_noisy, n_resamples=1000, seed=42)
    assert 0.60 <= res_noisy["pearson_r"] <= 0.95
    assert res_noisy["pearson_ci"][0] <= res_noisy["pearson_r"] <= res_noisy["pearson_ci"][1]
    assert res_noisy["spearman_ci"][0] <= res_noisy["spearman_rho"] <= res_noisy["spearman_ci"][1]


def test_power_analysis_minimum_detectable_kappa():
    """Verify statistical power note math: n=200, alpha=0.05, power=0.80."""
    power_note = compute_minimum_detectable_effect_kappa(n_samples=200, alpha=0.05, power=0.80, paired_correlation=0.60)
    assert power_note["n_samples"] == 200
    assert power_note["significance_level_alpha"] == 0.05
    assert power_note["statistical_power"] == 0.80
    assert 0.08 <= power_note["minimum_detectable_kappa_diff"] <= 0.16


def test_sec_edgar_loader_fair_access_and_pending_annotation():
    """Verify SEC EDGAR loader complies with Fair-Access policy and marks tasks 'pending_annotation'."""
    loader = SECEdgarLoader(user_agent="TestAuditor (contact@test.org)", rate_limit_pause=0.05)
    assert "TestAuditor" in loader.user_agent

    probes = build_offline_sample_filings()
    assert len(probes) > 0
    for p in probes:
        assert p["annotation_status"] == "pending_annotation"
        assert p["consensus_normalized"] is None
        assert p["annotations"] == []
        assert "sec" in p["id"]


def test_benchmark_validation_loaders():
    """Verify that public benchmark datasets load properly with citations."""
    for b_name in ["mt_bench", "reward_bench", "judge_bench"]:
        samples = load_public_benchmark(b_name)
        assert len(samples) > 0
        for s in samples:
            assert s.human_preferred in ["A", "B", "Tie"]
            assert len(s.prompt) > 5
            assert len(s.response_a) > 5
            assert len(s.response_b) > 5
            assert "source_citation" in s.metadata
