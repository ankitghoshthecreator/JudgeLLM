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

    def compute_human_fleiss_kappa(self) -> Dict[str, float]:
        """Calculates Fleiss's Kappa across the 3 human annotators per axis."""
        results = {}
        for axis in self.axes:
            # Build ratings count matrix: N subjects x 4 Likert categories (0, 1, 2, 3)
            matrix = np.zeros((len(self.probe_ids), 4), dtype=int)
            for i, pid in enumerate(self.probe_ids):
                ratings = self.human_data[pid]["ratings"]
                for ann_name in ["annotator_1", "annotator_2", "annotator_3"]:
                    score = ratings[ann_name][axis]
                    matrix[i, score] += 1
            kappa = compute_fleiss_kappa(matrix)
            results[axis] = round(kappa, 3)

        results["overall"] = round(float(np.mean(list(results.values()))), 3)
        return results

    def compute_judge_human_alignment(self, judge_dict: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculates Cohen's Kappa (Quadratic Weighted & Unweighted) between consensus
        human ratings and Teacher Judge ratings.
        """
        res_weighted = {}
        res_unweighted = {}

        for axis in self.axes:
            human_raw = [self.human_data[pid]["consensus_normalized"][axis] for pid in self.probe_ids]
            judge_raw = [judge_dict[pid]["scores"][axis] for pid in self.probe_ids]

            h_bins = discretize_scores(human_raw, n_bins=4)
            j_bins = discretize_scores(judge_raw, n_bins=4)

            kw = compute_cohens_kappa(h_bins, j_bins, n_categories=4, weighted=True)
            ku = compute_cohens_kappa(h_bins, j_bins, n_categories=4, weighted=False)

            res_weighted[axis] = round(kw, 3)
            res_unweighted[axis] = round(ku, 3)

        res_weighted["overall"] = round(float(np.mean(list(res_weighted.values()))), 3)
        res_unweighted["overall"] = round(float(np.mean(list(res_unweighted.values()))), 3)

        return {"quadratic_weighted": res_weighted, "unweighted": res_unweighted}

    def compute_local_vs_teacher_correlation(self) -> Dict[str, Dict[str, float]]:
        """Calculates Pearson r and Spearman rho between Local Scorer and Rubric B Teacher Judge."""
        corr_results = {}
        for axis in self.axes:
            local_vals = [self.local_data[pid]["scores"][axis] for pid in self.probe_ids]
            teacher_vals = [self.rubric_b_data[pid]["scores"][axis] for pid in self.probe_ids]

            pr, pp, sr, sp = compute_correlation_metrics(local_vals, teacher_vals)
            corr_results[axis] = {
                "pearson_r": round(pr, 3),
                "pearson_p": float(f"{pp:.2e}"),
                "spearman_rho": round(sr, 3),
                "spearman_p": float(f"{sp:.2e}"),
            }

        # Mean across axes
        avg_r = np.mean([corr_results[a]["pearson_r"] for a in self.axes])
        avg_rho = np.mean([corr_results[a]["spearman_rho"] for a in self.axes])
        corr_results["overall"] = {
            "pearson_r": round(float(avg_r), 3),
            "spearman_rho": round(float(avg_rho), 3),
        }
        return corr_results

    def run_full_analysis(self) -> Dict[str, Any]:
        fleiss = self.compute_human_fleiss_kappa()
        alignment_a = self.compute_judge_human_alignment(self.rubric_a_data)
        alignment_b = self.compute_judge_human_alignment(self.rubric_b_data)
        local_corr = self.compute_local_vs_teacher_correlation()

        report = {
            "num_samples": len(self.probe_ids),
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
    print("=" * 78)
    print("      LLM-AS-A-JUDGE STATISTICAL ALIGNMENT & AGREEMENT REPORT")
    print("=" * 78)
    print("\n[1] Inter-Annotator Agreement Ceiling (Fleiss's Kappa across 3 Human Annotators):")
    for axis, k in results["human_inter_annotator_fleiss_kappa"].items():
        print(f"  * {axis.replace('_', ' ').capitalize():<24}: Kappa = {k:.3f}")

    print("\n[2] Human-Judge Alignment (Quadratic Weighted Cohen's Kappa):")
    print(f"{'Axis':<24} | {'Rubric A (Zero-Shot)':<22} | {'Rubric B (CoT + Exemplars)':<26}")
    print("-" * 78)
    axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination", "overall"]
    for axis in axes:
        ka = results["rubric_a_judge_human_kappa"]["quadratic_weighted"][axis]
        kb = results["rubric_b_judge_human_kappa"]["quadratic_weighted"][axis]
        print(f"{axis.replace('_', ' ').capitalize():<24} | {ka:<22.3f} | {kb:<26.3f}")

    print("\n[3] Local Scorer vs. Teacher Judge Correlation (Pearson r & Spearman rho):")
    print(f"{'Axis':<24} | {'Pearson r':<16} | {'Spearman rho':<16}")
    print("-" * 62)
    for axis in axes:
        r_val = results["local_evaluator_vs_teacher_correlation"][axis]["pearson_r"]
        rho_val = results["local_evaluator_vs_teacher_correlation"][axis]["spearman_rho"]
        print(f"{axis.replace('_', ' ').capitalize():<24} | {r_val:<16.3f} | {rho_val:<16.3f}")

    print("=" * 78)
    print(f"[OK] Full agreement statistics saved to: {out_p}")


if __name__ == "__main__":
    main()

