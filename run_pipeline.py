"""
End-to-End Orchestrator Pipeline Runner
========================================
Runs the complete LLM-as-a-Judge Alignment Study from raw probe generation
to statistical alignment analysis, failure taxonomy breakdown, and figure plotting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
RUBRICS_DIR = BASE_DIR / "evaluation" / "rubrics"


def run_pipeline(mock_mode: bool = True, force_generate: bool = False) -> None:
    t_start = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    probes_path = DATA_DIR / "probe_outputs_200.json"
    human_path = DATA_DIR / "human_annotations.json"
    local_path = DATA_DIR / "local_scores.json"
    rubric_a_path = DATA_DIR / "judge_scores" / "rubric_a.json"
    rubric_b_path = DATA_DIR / "judge_scores" / "rubric_b.json"

    # Step 1: Probe and Annotation Data Generation
    if force_generate or not (probes_path.exists() and human_path.exists()):
        logger.info("[Step 1/6] Generating 200 financial probe outputs and human annotations...")
        from scripts.generate_probe_data import main as generate_data
        generate_data()
    else:
        logger.info("[Step 1/6] Probe dataset and human annotations verified.")

    # Step 2: Local Evaluator (LightweightStudentEvaluator)
    logger.info("[Step 2/6] Running LightweightStudentEvaluator (~0.05ms heuristic pre-filter)...")
    from evaluation.local_scorer import run_local_evaluation
    run_local_evaluation(str(probes_path), str(local_path))

    # Step 3: Teacher Judge Evaluation (Rubrics A & B)
    logger.info("[Step 3/6] Running Teacher Judge Engine (Rubric A zero-shot baseline)...")
    from evaluation.teacher_judge import run_teacher_evaluation
    run_teacher_evaluation(
        rubric_path=str(RUBRICS_DIR / "rubric_a.yaml"),
        inputs_path=str(probes_path),
        output_path=str(rubric_a_path),
        mock=mock_mode,
    )

    logger.info("[Step 3/6] Running Teacher Judge Engine (Rubric B CoT + Exemplars)...")
    run_teacher_evaluation(
        rubric_path=str(RUBRICS_DIR / "rubric_b.yaml"),
        inputs_path=str(probes_path),
        output_path=str(rubric_b_path),
        mock=mock_mode,
    )

    # Step 4: Statistical Alignment Analysis
    logger.info("[Step 4/6] Computing Fleiss's Kappa, Cohen's Kappa, and Pearson/Spearman correlations...")
    from analysis.agreement_stats import AgreementAnalysis
    analyzer = AgreementAnalysis(
        human_path=human_path,
        rubric_a_path=rubric_a_path,
        rubric_b_path=rubric_b_path,
        local_scores_path=local_path,
    )
    agreement_report = analyzer.run_full_analysis()
    import json
    metrics_path = DATA_DIR / "agreement_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(agreement_report, f, indent=2)

    # Step 5: Failure Breakdown Analysis
    logger.info("[Step 5/6] Performing Failure Mode & Disagreement Analysis (>0.30 margin)...")
    from analysis.failure_breakdown import FailureModeAnalyzer
    failure_analyzer = FailureModeAnalyzer(
        probes_path=probes_path,
        human_path=human_path,
        judge_path=rubric_b_path,
        disagreement_threshold=0.30,
    )
    failure_report = failure_analyzer.analyze_disagreements()
    failure_path = DATA_DIR / "failure_modes_report.json"
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_report, f, indent=2)

    # Step 6: Generate Publication Figures
    logger.info("[Step 6/6] Generating visualization figures in %s...", FIGURES_DIR)
    from analysis.visualize import (
        plot_agreement_comparison,
        plot_failure_modes_distribution,
        plot_local_vs_teacher_correlation,
        plot_prefilter_tradeoff_curve,
        plot_correlation_matrix_heatmap,
    )
    plot_agreement_comparison(metrics_path, FIGURES_DIR / "agreement_comparison_rubric_a_vs_b.png")
    plot_failure_modes_distribution(failure_path, FIGURES_DIR / "failure_modes_distribution.png")
    plot_local_vs_teacher_correlation(local_path, rubric_b_path, FIGURES_DIR / "local_vs_teacher_correlation.png")
    plot_prefilter_tradeoff_curve(local_path, rubric_b_path, FIGURES_DIR / "prefilter_tradeoff_curve.png")
    plot_correlation_matrix_heatmap(human_path, rubric_b_path, local_path, FIGURES_DIR / "correlation_matrix_heatmap.png")

    elapsed = time.perf_counter() - t_start

    # Executive Output
    print("\n" + "=" * 80)
    print("           LLM-AS-A-JUDGE ALIGNMENT STUDY — EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"Total Probes Processed: {agreement_report['num_samples']}")
    print(f"Total Disagreements (|Delta| > 0.30): {failure_report['total_disagreements']} ({failure_report['disagreement_rate']}%)")
    print(f"Human Inter-Annotator Ceiling (Fleiss's Kappa): {agreement_report['human_inter_annotator_fleiss_kappa']['overall']:.3f}")
    print(f"Rubric A Overall Kappa (Zero-Shot Baseline):     {agreement_report['rubric_a_judge_human_kappa']['quadratic_weighted']['overall']:.3f}")
    print(f"Rubric B Overall Kappa (CoT + Exemplars):        {agreement_report['rubric_b_judge_human_kappa']['quadratic_weighted']['overall']:.3f}")
    print(f"Local Evaluator Pearson r (Fluency Pre-Filter):  {agreement_report['local_evaluator_vs_teacher_correlation']['fluency']['pearson_r']:.3f}")
    print(f"Local Evaluator Pearson r (Hallucination Proxy): {agreement_report['local_evaluator_vs_teacher_correlation']['no_hallucination']['pearson_r']:.3f}")
    print("-" * 80)
    print("Failure Mode Breakdown:")
    for row in failure_report["breakdown_summary"]:
        print(f"  • {row['category']:<50}: {row['count']} ({row['percentage']:.1f}%)")
    print("-" * 80)
    print(f"Execution completed successfully in {elapsed:.2f} seconds.")
    print("Figures saved in: figures/")
    print("Metrics saved in: data/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Orchestrator Pipeline")
    parser.add_argument("--mock", action="store_true", default=True, help="Run with mock simulator if no API key")
    parser.add_argument("--live", action="store_true", help="Force live Groq 70B API calls (requires GROQ_API_KEY)")
    parser.add_argument("--force-generate", action="store_true", help="Force regenerate probes and annotations")
    args = parser.parse_args()

    mock = not args.live
    run_pipeline(mock_mode=mock, force_generate=args.force_generate)
