"""
Statistical Agreement & Correlation Engine
===========================================
Computes:
1. Inter-Annotator Agreement Ceiling: Fleiss's Kappa across 3 human annotators.
2. Judge-Human Alignment: Cohen's Kappa (Quadratic Weighted & Unweighted) for Rubric A and Rubric B.
3. Local vs. Teacher Scorer Correlation: Pearson correlation (r) and Spearman rank (rho) per axis.
4. Summary metrics export to `data/agreement_metrics.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Computes Fleiss's Kappa for inter-annotator agreement.
    ratings_matrix shape: (N, k) where N is number of subjects,
    k is number of categories, and entries are counts of annotators assigning that category.
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = np.sum(ratings_matrix[0, :])
    if n_raters <= 1:
        return 1.0

    # Proportion of all assignments to each category
    p_j = np.sum(ratings_matrix, axis=0) / (n_subjects * n_raters)

    # Extent of agreement for the i-th subject
    p_i = (np.sum(ratings_matrix**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)
    p_e = np.sum(p_j**2)

    if np.isclose(1.0 - p_e, 0.0):
        return 1.0

    kappa = (p_bar - p_e) / (1.0 - p_e)
    return float(kappa)


def discretize_scores(scores: List[float], n_bins: int = 4) -> List[int]:
    """
    Discretizes continuous [0.0, 1.0] scores into discrete ordinal bins (e.g., 0, 1, 2, 3)
    matching Likert scales for Cohen's / Fleiss's Kappa computation.
    """
    arr = np.array(scores, dtype=float)
    # Scaled to [0, n_bins - 1] and rounded
    binned = np.clip(np.round(arr * (n_bins - 1)), 0, n_bins - 1).astype(int)
    return binned.tolist()



def compute_cohens_kappa(
    rater1: List[int],
    rater2: List[int],
    n_categories: int = 4,
    weighted: bool = True,
) -> float:
    """
    Computes Cohen's Kappa between two raters.
    If weighted is True, applies Quadratic Weighting for ordinal ratings.
    """
    conf_matrix = np.zeros((n_categories, n_categories), dtype=float)
    n = len(rater1)
    if n == 0:
        return 0.0

    for r1, r2 in zip(rater1, rater2):
        conf_matrix[r1, r2] += 1

    total = float(n)
    conf_matrix /= total

    # Marginals
    hist1 = np.sum(conf_matrix, axis=1)
    hist2 = np.sum(conf_matrix, axis=0)
    expected_matrix = np.outer(hist1, hist2)

    if not weighted:
        p_o = np.trace(conf_matrix)
        p_e = np.trace(expected_matrix)
        if np.isclose(1.0 - p_e, 0.0):
            return 1.0
        return float((p_o - p_e) / (1.0 - p_e))

    # Quadratic weighting matrix: w_ij = 1 - (i - j)^2 / (k - 1)^2
    weights = np.zeros((n_categories, n_categories))
    denom = (n_categories - 1) ** 2
    for i in range(n_categories):
        for j in range(n_categories):
            weights[i, j] = 1.0 - ((i - j) ** 2) / denom

    p_o = np.sum(weights * conf_matrix)
    p_e = np.sum(weights * expected_matrix)

    if np.isclose(1.0 - p_e, 0.0):
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)


def bootstrap_cohen_kappa(
    rater1: np.ndarray | List[int],
    rater2: np.ndarray | List[int],
    n_categories: int = 4,
    weighted: bool = True,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes point estimate and empirical 95% bootstrap confidence interval
    for Cohen's Kappa using seeded subject-level resampling.

    Returns:
        (estimate, ci_lower, ci_upper)
    """
    r1 = np.asarray(rater1, dtype=int)
    r2 = np.asarray(rater2, dtype=int)
    n = len(r1)
    if n == 0:
        return 0.0, 0.0, 0.0

    point_est = compute_cohens_kappa(r1.tolist(), r2.tolist(), n_categories=n_categories, weighted=weighted)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_resamples, n))
    pair_ids = r1 * n_categories + r2
    boot_pairs = pair_ids[boot_idx]

    denom = (n_categories - 1) ** 2 if weighted else 1.0
    w = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            w[i, j] = 1.0 - ((i - j) ** 2) / denom if weighted else (1.0 if i == j else 0.0)
    w_flat = w.flatten()

    counts = np.zeros((n_resamples, n_categories * n_categories), dtype=float)
    for k in range(n_categories * n_categories):
        counts[:, k] = np.sum(boot_pairs == k, axis=1)

    conf = counts / n
    p_o = np.dot(conf, w_flat)
    conf_3d = conf.reshape((n_resamples, n_categories, n_categories))
    hist1 = np.sum(conf_3d, axis=2)
    hist2 = np.sum(conf_3d, axis=1)
    exp = np.einsum("bi,bj,ij->b", hist1, hist2, w)

    denom_k = 1.0 - exp
    valid = np.abs(denom_k) > 1e-6
    boot_kappas = np.where(valid, (p_o - exp) / denom_k, 1.0)

    ci_lower = float(np.percentile(boot_kappas, 2.5))
    ci_upper = float(np.percentile(boot_kappas, 97.5))
    return round(point_est, 3), round(ci_lower, 3), round(ci_upper, 3)


def bootstrap_fleiss_kappa(
    ratings_matrix: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes point estimate and empirical 95% bootstrap confidence interval
    for Fleiss's Kappa across subjects using seeded resampling.

    Returns:
        (estimate, ci_lower, ci_upper)
    """
    matrix = np.asarray(ratings_matrix, dtype=int)
    n_subjects, n_categories = matrix.shape
    n_raters = np.sum(matrix[0, :])
    point_est = compute_fleiss_kappa(matrix)
    if n_subjects <= 1 or n_raters <= 1:
        return round(point_est, 3), round(point_est, 3), round(point_est, 3)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_subjects, size=(n_resamples, n_subjects))

    sum_sq_orig = np.sum(matrix**2, axis=1)
    p_i_orig = (sum_sq_orig - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i_orig[boot_idx], axis=1)

    cat_totals = np.sum(matrix[boot_idx], axis=1)
    p_j = cat_totals / (n_subjects * n_raters)
    p_e = np.sum(p_j**2, axis=1)

    denom = 1.0 - p_e
    valid = np.abs(denom) > 1e-6
    safe_denom = np.where(valid, denom, 1.0)
    boot_kappas = np.where(valid, (p_bar - p_e) / safe_denom, 1.0)

    ci_lower = float(np.percentile(boot_kappas, 2.5))
    ci_upper = float(np.percentile(boot_kappas, 97.5))
    return round(point_est, 3), round(ci_lower, 3), round(ci_upper, 3)


def bootstrap_correlation_metrics(
    scores_x: List[float] | np.ndarray,
    scores_y: List[float] | np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes Pearson r and Spearman rho along with 95% bootstrap confidence intervals.
    """
    x = np.asarray(scores_x, dtype=float)
    y = np.asarray(scores_y, dtype=float)
    n = len(x)

    pr, pp, sr, sp = compute_correlation_metrics(x.tolist(), y.tolist())
    if n < 4:
        return {
            "pearson_r": round(pr, 3),
            "pearson_ci": [round(pr, 3), round(pr, 3)],
            "pearson_p": pp,
            "spearman_rho": round(sr, 3),
            "spearman_ci": [round(sr, 3), round(sr, 3)],
            "spearman_p": sp,
        }

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_resamples, n))
    x_boot = x[boot_idx]
    y_boot = y[boot_idx]

    # Vectorized Pearson r
    x_mean = np.mean(x_boot, axis=1, keepdims=True)
    y_mean = np.mean(y_boot, axis=1, keepdims=True)
    dx = x_boot - x_mean
    dy = y_boot - y_mean
    cov = np.sum(dx * dy, axis=1)
    denom_p = np.sqrt(np.sum(dx**2, axis=1) * np.sum(dy**2, axis=1))
    valid_p = denom_p > 1e-8
    boot_pearson = np.where(valid_p, cov / denom_p, 0.0)

    # Vectorized Spearman rho via double argsort rank
    ranks_x = np.argsort(np.argsort(x_boot, axis=1), axis=1).astype(float)
    ranks_y = np.argsort(np.argsort(y_boot, axis=1), axis=1).astype(float)
    rx_mean = np.mean(ranks_x, axis=1, keepdims=True)
    ry_mean = np.mean(ranks_y, axis=1, keepdims=True)
    drx = ranks_x - rx_mean
    dry = ranks_y - ry_mean
    cov_r = np.sum(drx * dry, axis=1)
    denom_r = np.sqrt(np.sum(drx**2, axis=1) * np.sum(dry**2, axis=1))
    valid_r = denom_r > 1e-8
    boot_spearman = np.where(valid_r, cov_r / denom_r, 0.0)

    p_lower = float(np.percentile(boot_pearson, 2.5))
    p_upper = float(np.percentile(boot_pearson, 97.5))
    s_lower = float(np.percentile(boot_spearman, 2.5))
    s_upper = float(np.percentile(boot_spearman, 97.5))

    return {
        "pearson_r": round(pr, 3),
        "pearson_ci": [round(p_lower, 3), round(p_upper, 3)],
        "pearson_p": pp,
        "spearman_rho": round(sr, 3),
        "spearman_ci": [round(s_lower, 3), round(s_upper, 3)],
        "spearman_p": sp,
    }


def compute_minimum_detectable_effect_kappa(
    n_samples: int = 200,
    alpha: float = 0.05,
    power: float = 0.80,
    paired_correlation: float = 0.60,
    pooled_se: float = 0.045,
) -> Dict[str, float]:
    """
    Power analysis: computes the minimum detectable difference (MDE) in Cohen's Kappa
    between Rubric A and Rubric B given sample size n under a two-tailed paired design.

    Formula:
        MDE = (z_{1 - alpha/2} + z_{power}) * SE(Delta_kappa)
        where SE(Delta_kappa) = sqrt(2 * (1 - r_paired)) * SE_pooled
    """
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_beta = float(stats.norm.ppf(power))
    se_diff = float(np.sqrt(2.0 * (1.0 - paired_correlation)) * pooled_se)
    mde = float((z_alpha + z_beta) * se_diff)

    return {
        "n_samples": n_samples,
        "significance_level_alpha": alpha,
        "statistical_power": power,
        "paired_correlation": paired_correlation,
        "se_difference": round(se_diff, 4),
        "minimum_detectable_kappa_diff": round(mde, 3),
    }


def compute_correlation_metrics(
    scores_x: List[float], scores_y: List[float]
) -> Tuple[float, float, float, float]:
    """Computes Pearson r and Spearman rho along with their p-values."""
    x = np.array(scores_x, dtype=float)
    y = np.array(scores_y, dtype=float)

    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0, 0.0, 1.0

    pearson_r, p_pearson = stats.pearsonr(x, y)
    spearman_rho, p_spearman = stats.spearmanr(x, y)
    return float(pearson_r), float(p_pearson), float(spearman_rho), float(p_spearman)


class AgreementAnalysis:
    def __init__(
        self,
        human_path: str | Path,
        rubric_a_path: str | Path,
        rubric_b_path: str | Path,
        local_scores_path: str | Path,
    ):
        with open(human_path, "r", encoding="utf-8") as f:
            self.human_data = {item["id"]: item for item in json.load(f)}
        with open(rubric_a_path, "r", encoding="utf-8") as f:
            self.rubric_a_data = {item["id"]: item for item in json.load(f)}
        with open(rubric_b_path, "r", encoding="utf-8") as f:
            self.rubric_b_data = {item["id"]: item for item in json.load(f)}
        with open(local_scores_path, "r", encoding="utf-8") as f:
            self.local_data = {item["id"]: item for item in json.load(f)}

        self.probe_ids = sorted(self.human_data.keys())
        self.axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination"]

    def compute_human_fleiss_kappa(self) -> Dict[str, Any]:
        """Calculates Fleiss's Kappa across the 3 human annotators per axis with 95% bootstrap CIs.
        
        The 'overall' kappa is calculated as the arithmetic mean across the 4 orthogonal axes:
        overall = mean(fluency, coherence, instruction_adherence, no_hallucination)
        """
        results: Dict[str, Any] = {}
        matrices: Dict[str, np.ndarray] = {}
        for axis in self.axes:
            # Build ratings count matrix: N subjects x 4 Likert categories (0, 1, 2, 3)
            matrix = np.zeros((len(self.probe_ids), 4), dtype=int)
            for i, pid in enumerate(self.probe_ids):
                ratings = self.human_data[pid]["ratings"]
                for ann_name in ["annotator_1", "annotator_2", "annotator_3"]:
                    score = ratings[ann_name][axis]
                    matrix[i, score] += 1
            matrices[axis] = matrix
            est, low, high = bootstrap_fleiss_kappa(matrix, n_resamples=10000, seed=42)
            results[axis] = est
            results[f"{axis}_ci"] = [low, high]

        # Overall across axes
        overall_est = round(float(np.mean([results[a] for a in self.axes])), 3)
        results["overall"] = overall_est

        # Joint subject-level bootstrap for overall mean
        rng = np.random.default_rng(42)
        boot_idx = rng.integers(0, len(self.probe_ids), size=(10000, len(self.probe_ids)))
        n_raters = 3
        axis_boot_kappas = []
        for axis in self.axes:
            mat = matrices[axis]
            sum_sq = np.sum(mat**2, axis=1)
            p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
            p_bar = np.mean(p_i[boot_idx], axis=1)
            cat_tot = np.sum(mat[boot_idx], axis=1)
            p_j = cat_tot / (len(self.probe_ids) * n_raters)
            p_e = np.sum(p_j**2, axis=1)
            denom = 1.0 - p_e
            valid = np.abs(denom) > 1e-6
            k_boot = np.where(valid, (p_bar - p_e) / denom, 1.0)
            axis_boot_kappas.append(k_boot)
        overall_boots = np.mean(axis_boot_kappas, axis=0)
        results["overall_ci"] = [
            round(float(np.percentile(overall_boots, 2.5)), 3),
            round(float(np.percentile(overall_boots, 97.5)), 3),
        ]
        return results

    def compute_judge_human_alignment(self, judge_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Calculates Cohen's Kappa (Quadratic Weighted & Unweighted) between consensus
        human ratings and Teacher Judge ratings with 95% bootstrap CIs.

        'overall' is the arithmetic mean across the 4 orthogonal axes.
        """
        res_weighted: Dict[str, Any] = {}
        res_unweighted: Dict[str, Any] = {}
        kw_boots_all: List[np.ndarray] = []
        ku_boots_all: List[np.ndarray] = []

        for axis in self.axes:
            human_raw = [self.human_data[pid]["consensus_normalized"][axis] for pid in self.probe_ids]
            judge_raw = [judge_dict[pid]["scores"][axis] for pid in self.probe_ids]

            h_bins = discretize_scores(human_raw, n_bins=4)
            j_bins = discretize_scores(judge_raw, n_bins=4)

            kw, kw_low, kw_high = bootstrap_cohen_kappa(h_bins, j_bins, n_categories=4, weighted=True, n_resamples=10000, seed=42)
            ku, ku_low, ku_high = bootstrap_cohen_kappa(h_bins, j_bins, n_categories=4, weighted=False, n_resamples=10000, seed=42)

            res_weighted[axis] = kw
            res_weighted[f"{axis}_ci"] = [kw_low, kw_high]
            res_unweighted[axis] = ku
            res_unweighted[f"{axis}_ci"] = [ku_low, ku_high]

        # Overall calculation (arithmetic mean across dimensions)
        overall_kw = round(float(np.mean([res_weighted[a] for a in self.axes])), 3)
        overall_ku = round(float(np.mean([res_unweighted[a] for a in self.axes])), 3)
        res_weighted["overall"] = overall_kw
        res_unweighted["overall"] = overall_ku

        # Joint subject-level bootstrap for overall
        rng = np.random.default_rng(42)
        n = len(self.probe_ids)
        boot_idx = rng.integers(0, n, size=(10000, n))
        w_mat = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                w_mat[i, j] = 1.0 - ((i - j) ** 2) / 9.0
        w_flat = w_mat.flatten()

        w_axis_boots = []
        u_axis_boots = []
        for axis in self.axes:
            h_bins = np.array(discretize_scores([self.human_data[pid]["consensus_normalized"][axis] for pid in self.probe_ids], 4))
            j_bins = np.array(discretize_scores([judge_dict[pid]["scores"][axis] for pid in self.probe_ids], 4))
            pair_ids = h_bins * 4 + j_bins
            boot_pairs = pair_ids[boot_idx]
            counts = np.zeros((10000, 16), dtype=float)
            for k in range(16):
                counts[:, k] = np.sum(boot_pairs == k, axis=1)
            conf = counts / n
            conf_3d = conf.reshape((10000, 4, 4))
            hist1 = np.sum(conf_3d, axis=2)
            hist2 = np.sum(conf_3d, axis=1)

            # weighted
            p_o_w = np.dot(conf, w_flat)
            exp_w = np.einsum("bi,bj,ij->b", hist1, hist2, w_mat)
            valid_w = np.abs(1.0 - exp_w) > 1e-6
            w_axis_boots.append(np.where(valid_w, (p_o_w - exp_w) / (1.0 - exp_w), 1.0))

            # unweighted
            p_o_u = np.trace(conf_3d, axis1=1, axis2=2)
            exp_u = np.sum(hist1 * hist2, axis=1)
            valid_u = np.abs(1.0 - exp_u) > 1e-6
            u_axis_boots.append(np.where(valid_u, (p_o_u - exp_u) / (1.0 - exp_u), 1.0))

        overall_w_boots = np.mean(w_axis_boots, axis=0)
        overall_u_boots = np.mean(u_axis_boots, axis=0)

        res_weighted["overall_ci"] = [
            round(float(np.percentile(overall_w_boots, 2.5)), 3),
            round(float(np.percentile(overall_w_boots, 97.5)), 3),
        ]
        res_unweighted["overall_ci"] = [
            round(float(np.percentile(overall_u_boots, 2.5)), 3),
            round(float(np.percentile(overall_u_boots, 97.5)), 3),
        ]

        return {"quadratic_weighted": res_weighted, "unweighted": res_unweighted}

    def compute_local_vs_teacher_correlation(self) -> Dict[str, Dict[str, Any]]:
        """Calculates Pearson r and Spearman rho between Local Scorer and Rubric B Teacher Judge with 95% bootstrap CIs."""
        corr_results = {}
        for axis in self.axes:
            local_vals = [self.local_data[pid]["scores"][axis] for pid in self.probe_ids]
            teacher_vals = [self.rubric_b_data[pid]["scores"][axis] for pid in self.probe_ids]

            corr_results[axis] = bootstrap_correlation_metrics(local_vals, teacher_vals, n_resamples=10000, seed=42)

        # Mean across axes
        avg_r = float(np.mean([corr_results[a]["pearson_r"] for a in self.axes]))
        avg_rho = float(np.mean([corr_results[a]["spearman_rho"] for a in self.axes]))
        r_low = float(np.mean([corr_results[a]["pearson_ci"][0] for a in self.axes]))
        r_high = float(np.mean([corr_results[a]["pearson_ci"][1] for a in self.axes]))
        rho_low = float(np.mean([corr_results[a]["spearman_ci"][0] for a in self.axes]))
        rho_high = float(np.mean([corr_results[a]["spearman_ci"][1] for a in self.axes]))

        corr_results["overall"] = {
            "pearson_r": round(avg_r, 3),
            "pearson_ci": [round(r_low, 3), round(r_high, 3)],
            "spearman_rho": round(avg_rho, 3),
            "spearman_ci": [round(rho_low, 3), round(rho_high, 3)],
        }
        return corr_results

    def run_full_analysis(self) -> Dict[str, Any]:
        fleiss = self.compute_human_fleiss_kappa()
        alignment_a = self.compute_judge_human_alignment(self.rubric_a_data)
        alignment_b = self.compute_judge_human_alignment(self.rubric_b_data)
        local_corr = self.compute_local_vs_teacher_correlation()
        power_note = compute_minimum_detectable_effect_kappa(
            n_samples=len(self.probe_ids),
            alpha=0.05,
            power=0.80,
            paired_correlation=0.60,
        )

        report = {
            "num_samples": len(self.probe_ids),
            "bootstrap_settings": {
                "n_resamples": 10000,
                "confidence_level": 0.95,
                "random_seed": 42,
            },
            "statistical_power_analysis": power_note,
            "human_inter_annotator_fleiss_kappa": fleiss,
            "rubric_a_judge_human_kappa": alignment_a,
            "rubric_b_judge_human_kappa": alignment_b,
            "local_evaluator_vs_teacher_correlation": local_corr,
        }
        return report


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Compute LLM-as-a-Judge statistical alignment metrics")
    parser.add_argument("--human", default="data/human_annotations.json", help="Human annotations JSON")
    parser.add_argument("--rubric-a", default="data/judge_scores/rubric_a.json", help="Rubric A judge scores JSON")
    parser.add_argument("--rubric-b", default="data/judge_scores/rubric_b.json", help="Rubric B judge scores JSON")
    parser.add_argument("--local", default="data/local_scores.json", help="Local scorer JSON")
    parser.add_argument("--output", default="data/agreement_metrics.json", help="Output metrics path")
    args = parser.parse_args()

    analyzer = AgreementAnalysis(
        human_path=args.human,
        rubric_a_path=args.rubric_a,
        rubric_b_path=args.rubric_b,
        local_scores_path=args.local,
    )
    results = analyzer.run_full_analysis()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print publication-ready table
    print("=" * 90)
    print("      LLM-AS-A-JUDGE STATISTICAL ALIGNMENT & AGREEMENT REPORT (WITH 95% BOOTSTRAP CIs)")
    print("=" * 90)
    print("\n[1] Inter-Annotator Agreement Ceiling (Fleiss's Kappa across 3 Human Annotators, 10k Bootstrap):")
    axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination", "overall"]
    for axis in axes:
        k = results["human_inter_annotator_fleiss_kappa"][axis]
        ci = results["human_inter_annotator_fleiss_kappa"].get(f"{axis}_ci", [k, k])
        print(f"  * {axis.replace('_', ' ').capitalize():<24}: Kappa = {k:.3f}  [95% CI: {ci[0]:.3f}, {ci[1]:.3f}]")

    print("\n[2] Human-Judge Alignment (Quadratic Weighted Cohen's Kappa, 10k Bootstrap):")
    print(f"{'Axis':<24} | {'Rubric A (Zero-Shot) [95% CI]':<32} | {'Rubric B (CoT + Exemplars) [95% CI]':<36}")
    print("-" * 96)
    for axis in axes:
        ka = results["rubric_a_judge_human_kappa"]["quadratic_weighted"][axis]
        cia = results["rubric_a_judge_human_kappa"]["quadratic_weighted"].get(f"{axis}_ci", [ka, ka])
        kb = results["rubric_b_judge_human_kappa"]["quadratic_weighted"][axis]
        cib = results["rubric_b_judge_human_kappa"]["quadratic_weighted"].get(f"{axis}_ci", [kb, kb])
        str_a = f"{ka:.3f} [{cia[0]:.3f}, {cia[1]:.3f}]"
        str_b = f"{kb:.3f} [{cib[0]:.3f}, {cib[1]:.3f}]"
        print(f"{axis.replace('_', ' ').capitalize():<24} | {str_a:<32} | {str_b:<36}")

    print("\n[3] Local Scorer vs. Teacher Judge Correlation (Pearson r & Spearman rho, 10k Bootstrap):")
    print(f"{'Axis':<24} | {'Pearson r [95% CI]':<26} | {'Spearman rho [95% CI]':<26}")
    print("-" * 80)
    for axis in axes:
        r_val = results["local_evaluator_vs_teacher_correlation"][axis]["pearson_r"]
        r_ci = results["local_evaluator_vs_teacher_correlation"][axis].get("pearson_ci", [r_val, r_val])
        rho_val = results["local_evaluator_vs_teacher_correlation"][axis]["spearman_rho"]
        rho_ci = results["local_evaluator_vs_teacher_correlation"][axis].get("spearman_ci", [rho_val, rho_val])
        str_r = f"{r_val:.3f} [{r_ci[0]:.3f}, {r_ci[1]:.3f}]"
        str_rho = f"{rho_val:.3f} [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]"
        print(f"{axis.replace('_', ' ').capitalize():<24} | {str_r:<26} | {str_rho:<26}")

    p_note = results.get("statistical_power_analysis", {})
    if p_note:
        print("\n[4] Statistical Power Analysis:")
        print(f"  * Sample Size: n = {p_note.get('n_samples')}")
        print(f"  * Alpha = {p_note.get('significance_level_alpha')}, Power = {p_note.get('statistical_power')}")
        print(f"  * Minimum Detectable Kappa Difference (MDE): Delta_kappa >= {p_note.get('minimum_detectable_kappa_diff')}")

    print("=" * 90)
    print(f"[OK] Full agreement statistics saved to: {out_p}")


if __name__ == "__main__":
    main()

