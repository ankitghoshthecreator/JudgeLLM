"""
Failure Mode Breakdown & Disagreement Taxonomy Analysis
========================================================
Identifies and categorizes discrepancies between Teacher Judge (Rubric B)
and Human Consensus where |Score_judge - Score_human| > 0.30.

Taxonomy:
1. Judge False-Positive (rated high by judge, low by humans)
2. Judge False-Negative (rated low by judge, high by humans)
3. Rubric Ambiguity (high inter-annotator variance or underspecified constraints)
4. Genuine Model Failure (subtle edge case or degradation)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class FailureModeAnalyzer:
    def __init__(
        self,
        probes_path: str | Path,
        human_path: str | Path,
        judge_path: str | Path,
        disagreement_threshold: float = 0.30,
    ):
        with open(probes_path, "r", encoding="utf-8") as f:
            self.probes = {item["id"]: item for item in json.load(f)}
        with open(human_path, "r", encoding="utf-8") as f:
            self.human = {item["id"]: item for item in json.load(f)}
        with open(judge_path, "r", encoding="utf-8") as f:
            self.judge = {item["id"]: item for item in json.load(f)}

        self.threshold = disagreement_threshold
        self.axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination"]

    def analyze_disagreements(self) -> Dict[str, Any]:
        disagreements = []
        category_counts = {
            "Judge false-positive (rated high, human rated low)": 0,
            "Judge false-negative (rated low, human rated high)": 0,
            "Rubric ambiguity (both annotators uncertain)": 0,
            "Genuine model failure both agreed on": 0,
        }

        fp_key = "Judge false-positive (rated high, human rated low)"
        fn_key = "Judge false-negative (rated low, human rated high)"
        ambig_key = "Rubric ambiguity (both annotators uncertain)"
        fail_key = "Genuine model failure both agreed on"

        for pid, pdata in self.probes.items():
            hdata = self.human[pid]
            jdata = self.judge[pid]

            probe_cat = pdata.get("category", "")
            ratings_dict = hdata["ratings"]

            # Compute human std dev across annotators for this probe
            all_stds = []
            for axis in self.axes:
                vals = [ratings_dict[ann][axis] for ann in ["annotator_1", "annotator_2", "annotator_3"]]
                all_stds.append(np.std(vals))
            avg_human_std = float(np.mean(all_stds))

            # Inspect per-axis score deltas
            disagreed_axes = {}
            for axis in self.axes:
                h_score = hdata["consensus_normalized"][axis]
                j_score = jdata["scores"][axis]
                delta = abs(j_score - h_score)
                if delta > self.threshold:
                    disagreed_axes[axis] = {
                        "human_score": h_score,
                        "judge_score": j_score,
                        "delta": round(delta, 3),
                    }

            if not disagreed_axes:
                continue

            # Classify the primary root cause
            primary_axis = max(disagreed_axes.keys(), key=lambda a: disagreed_axes[a]["delta"])
            h_val = disagreed_axes[primary_axis]["human_score"]
            j_val = disagreed_axes[primary_axis]["judge_score"]

            if probe_cat == "ambiguous" or avg_human_std > 0.65:
                cat = ambig_key
            elif j_val >= 0.70 and h_val < 0.50:
                cat = fp_key
            elif j_val < 0.50 and h_val >= 0.70:
                cat = fn_key
            elif probe_cat in {"hallucination", "instruction_violation", "fluency_degraded"}:
                cat = fail_key
            else:
                if j_val > h_val:
                    cat = fp_key
                else:
                    cat = fn_key

            category_counts[cat] += 1
            disagreements.append({
                "probe_id": pid,
                "domain": pdata.get("domain", ""),
                "prompt": pdata.get("prompt", ""),
                "model_output": pdata.get("model_output", ""),
                "probe_category": probe_cat,
                "disagreement_category": cat,
                "disagreed_axes": disagreed_axes,
                "judge_reasoning": jdata.get("reasoning", ""),
            })

        total_disagreements = len(disagreements)
        breakdown_table = []
        for cat, cnt in category_counts.items():
            pct = (cnt / total_disagreements * 100.0) if total_disagreements > 0 else 0.0
            breakdown_table.append({
                "category": cat,
                "count": cnt,
                "percentage": round(pct, 1),
            })

        return {
            "total_samples": len(self.probes),
            "total_disagreements": total_disagreements,
            "disagreement_rate": round(total_disagreements / len(self.probes) * 100.0, 1),
            "threshold": self.threshold,
            "breakdown_summary": breakdown_table,
            "disagreements": disagreements,
        }


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Failure Mode & Disagreement Analysis")
    parser.add_argument("--probes", default="data/probe_outputs_200.json", help="Probes JSON")
    parser.add_argument("--human", default="data/human_annotations.json", help="Human annotations JSON")
    parser.add_argument("--judge", default="data/judge_scores/rubric_b.json", help="Judge scores JSON (Rubric B)")
    parser.add_argument("--threshold", type=float, default=0.30, help="Disagreement margin threshold")
    parser.add_argument("--output", default="data/failure_modes_report.json", help="Output report JSON")
    args = parser.parse_args()

    analyzer = FailureModeAnalyzer(
        probes_path=args.probes,
        human_path=args.human,
        judge_path=args.judge,
        disagreement_threshold=args.threshold,
    )
    report = analyzer.analyze_disagreements()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 78)
    print("        FAILURE MODE BREAKDOWN (Rubric B vs Human Consensus)")
    print("=" * 78)
    print(f"Total Evaluated Probes: {report['total_samples']}")
    print(f"Disagreements (|Delta| > {args.threshold}): {report['total_disagreements']} ({report['disagreement_rate']}%)")
    print("-" * 78)
    print(f"{'Category':<52} | {'Count':<7} | {'% of Disagreements':<15}")
    print("-" * 78)
    for row in report["breakdown_summary"]:
        print(f"{row['category']:<52} | {row['count']:<7} | {row['percentage']:<15.1f}%")
    print("=" * 78)
    print(f"[OK] Saved failure mode breakdown report to {out_p}")


if __name__ == "__main__":
    main()

