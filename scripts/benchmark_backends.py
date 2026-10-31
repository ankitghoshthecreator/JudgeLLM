"""
Multi-Backend Benchmarking Script
=================================
Runs evaluation of the 200-probe dataset across specified backends/models.
Measures:
  1. Agreement metrics (Cohen's Kappa with 95% bootstrap CI against human consensus)
  2. Latency profile (p50 median and p95 per judgment in milliseconds)
  3. Cost projection per 1,000 judgments (based on configured token pricing)

Outputs:
  - data/backend_comparison.json
  - Formatted console table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from analysis.agreement_stats import (
    bootstrap_cohen_kappa,
    discretize_scores,
)
from evaluation.backends import (
    BackendPricing,
    GroqJudgeBackend,
    MockJudgeBackend,
    OpenAICompatibleBackend,
    create_backend,
)
from evaluation.teacher_judge import TeacherJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkBackends")


def compute_backend_metrics(
    judge_records: List[Dict[str, Any]],
    human_consensus: Dict[str, Dict[str, float]],
    latencies: List[float],
    pricing: BackendPricing,
) -> Dict[str, Any]:
    """Computes kappa against human consensus, latency percentiles, and estimated costs."""
    axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination"]
    axis_kappas = {}

    for axis in axes:
        j_scores = []
        h_scores = []
        for r in judge_records:
            pid = r.get("id") or r.get("probe_id")
            if pid in human_consensus:
                j_val = float(r.get(axis, 0.0))
                h_val = float(human_consensus[pid].get(axis, 0.0))
                j_scores.append(j_val)
                h_scores.append(h_val)

        j_arr = np.array(discretize_scores(j_scores), dtype=int)
        h_arr = np.array(discretize_scores(h_scores), dtype=int)
        k_est, ci_lo, ci_hi = bootstrap_cohen_kappa(j_arr, h_arr, n_resamples=1000, seed=42)
        axis_kappas[axis] = {
            "kappa": round(k_est, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        }

    overall_kappa = float(np.mean([axis_kappas[ax]["kappa"] for ax in axes]))

    # Latencies in milliseconds
    p50_latency = float(np.percentile(latencies, 50)) * 1000.0 if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) * 1000.0 if latencies else 0.0

    # Cost estimate: approx 450 prompt tokens + 120 completion tokens per probe evaluation
    avg_prompt_tokens = 450
    avg_comp_tokens = 120
    cost_per_item = pricing.estimate_cost(avg_prompt_tokens, avg_comp_tokens)
    cost_per_1k = round(cost_per_item * 1000.0, 4)

    return {
        "overall_kappa": round(overall_kappa, 4),
        "axis_kappas": axis_kappas,
        "latency_p50_ms": round(p50_latency, 2),
        "latency_p95_ms": round(p95_latency, 2),
        "cost_per_1k_judgments_usd": cost_per_1k,
    }


def run_multi_backend_benchmark(
    backends_config: List[Dict[str, Any]],
    probes_path: Path,
    human_path: Path,
    rubric_path: Path,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    with open(probes_path, "r", encoding="utf-8") as f:
        probes = json.load(f)
    if sample_limit:
        probes = probes[:sample_limit]

    with open(human_path, "r", encoding="utf-8") as f:
        human_raw = json.load(f)
        if isinstance(human_raw, list):
            human_consensus = {
                item["id"]: item.get("consensus_normalized", item.get("ground_truth_quality", {}))
                for item in human_raw
            }
        elif isinstance(human_raw, dict) and "human_consensus" in human_raw:
            human_consensus = human_raw["human_consensus"]
        elif isinstance(human_raw, dict) and "probes" in human_raw:
            human_consensus = {p["id"]: p.get("ground_truth_quality", {}) for p in human_raw["probes"]}
        else:
            human_consensus = human_raw

    results = {}

    for b_cfg in backends_config:
        name = b_cfg["name"]
        logger.info("Evaluating backend '%s' (%s / %s)...", name, b_cfg.get("type"), b_cfg.get("model_name"))

        backend = create_backend(
            backend_type=b_cfg.get("type"),
            model_name=b_cfg.get("model_name"),
            api_key=b_cfg.get("api_key"),
            base_url=b_cfg.get("base_url"),
            pricing=b_cfg.get("pricing"),
            mock_mode=b_cfg.get("mock_mode", False),
        )

        judge = TeacherJudge(rubric_path=rubric_path, backend=backend)

        eval_records = []
        latencies = []

        for p in probes:
            t0 = time.perf_counter()
            res = judge.evaluate_single(
                prompt=p["prompt"],
                reference_context=p.get("reference_context", ""),
                model_output=p["model_output"],
                metadata=p,
            )
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            rec = res.model_dump()
            rec["id"] = p.get("id")
            eval_records.append(rec)

        metrics = compute_backend_metrics(
            judge_records=eval_records,
            human_consensus=human_consensus,
            latencies=latencies,
            pricing=backend.pricing,
        )
        metrics["backend_type"] = backend.backend_type
        metrics["model_name"] = backend.model_name
        metrics["n_evaluated"] = len(probes)

        results[name] = metrics

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-backend LLM judge benchmark")
    parser.add_argument("--probes", type=str, default="data/probe_outputs_200.json", help="Path to probe samples")
    parser.add_argument("--human", type=str, default="data/human_annotations.json", help="Path to human consensus")
    parser.add_argument("--rubric", type=str, default="evaluation/rubrics/rubric_b.yaml", help="Rubric YAML path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of probes (for rapid validation)")
    parser.add_argument("--mock-only", action="store_true", help="Force mock mode on all backends for offline runs")
    parser.add_argument("--output", type=str, default="data/backend_comparison.json", help="Output JSON path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    probes_p = project_root / args.probes
    human_p = project_root / args.human
    rubric_p = project_root / args.rubric
    output_p = project_root / args.output

    # Define standard candidate configurations
    candidate_backends = [
        {
            "name": "Groq-Llama-3.3-70B (Cloud/Hosted)",
            "type": "mock" if args.mock_only else "groq",
            "model_name": "llama-3.3-70b-versatile",
            "pricing": {"prompt_per_1k": 0.00059, "completion_per_1k": 0.00079},
        },
        {
            "name": "vLLM-Llama-3.3-70B (On-Premises / Air-Gapped)",
            "type": "mock" if args.mock_only else "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "model_name": "meta-llama/Llama-3.3-70B-Instruct",
            "pricing": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
            "mock_mode": args.mock_only,
        },
        {
            "name": "Ollama-Qwen-2.5-72B (On-Premises Local)",
            "type": "mock" if args.mock_only else "openai_compatible",
            "base_url": "http://localhost:11434/v1",
            "model_name": "qwen2.5:72b-instruct",
            "pricing": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
            "mock_mode": args.mock_only,
        },
    ]

    results = run_multi_backend_benchmark(
        backends_config=candidate_backends,
        probes_path=probes_p,
        human_path=human_p,
        rubric_path=rubric_p,
        sample_limit=args.limit,
    )

    output_p.parent.mkdir(parents=True, exist_ok=True)
    with open(output_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print ASCII table
    print("\n" + "=" * 80)
    print("BACKEND COMPARISON SUMMARY (Human Agreement, Latency, and Cost)")
    print("=" * 80)
    print(f"{'Backend / Model':<42} | {'Overall Kappa':<16} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'Cost / 1k':<9}")
    print("-" * 92)
    for name, r in results.items():
        k_str = f"{r['overall_kappa']:.3f}"
        print(f"{name:<42} | {k_str:<16} | {r['latency_p50_ms']:<9.1f} | {r['latency_p95_ms']:<9.1f} | ${r['cost_per_1k_judgments_usd']:<8.4f}")
    print("=" * 80)
    print(f"Detailed comparison saved to: {output_p}\n")


if __name__ == "__main__":
    main()
