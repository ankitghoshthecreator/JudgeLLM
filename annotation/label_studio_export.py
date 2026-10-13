"""
Label Studio Annotation Exporter & Ingestion Utility
=====================================================
Exports CIRCLE probe outputs into Label Studio JSON tasks, and
transforms Label Studio export payloads back into structured human annotations
with multi-annotator consensus and Likert-to-normalized scale mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def export_probes_to_label_studio(
    input_probes_path: str | Path,
    output_tasks_path: str | Path,
) -> List[Dict[str, Any]]:
    """Convert probe JSON list to Label Studio importable tasks."""
    in_p = Path(input_probes_path)
    out_p = Path(output_tasks_path)

    with open(in_p, "r", encoding="utf-8") as f:
        probes = json.load(f)

    tasks = []
    for probe in probes:
        task = {
            "id": probe.get("id"),
            "data": {
                "probe_id": probe.get("id"),
                "domain": probe.get("domain", "General Finance"),
                "prompt": probe.get("prompt", ""),
                "reference_context": probe.get("reference_context", ""),
                "model_output": probe.get("model_output", ""),
                "gold_answer": probe.get("gold_answer", ""),
            },
        }
        tasks.append(task)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    return tasks


def parse_label_studio_export(
    export_path: str | Path,
    output_annotations_path: str | Path,
) -> List[Dict[str, Any]]:
    """
    Parses completed Label Studio annotations into standardized human_annotations.json format
    with normalized 0.0-1.0 continuous consensus scores.
    """
    exp_p = Path(export_path)
    out_p = Path(output_annotations_path)

    with open(exp_p, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    processed_annotations = []
    axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination"]

    for item in raw_items:
        probe_id = item.get("id") or item.get("data", {}).get("probe_id")
        annotations_list = item.get("annotations", [])
        annotator_ratings = {}

        for idx, ann in enumerate(annotations_list, 1):
            ann_id = ann.get("completed_by", {}).get("email") or f"annotator_{idx}"
            scores = {}
            for res in ann.get("result", []):
                from_name = res.get("from_name")
                value = res.get("value", {})
                choice = value.get("choices", [None])[0] or value.get("rating")
                if from_name in axes and choice is not None:
                    try:
                        scores[from_name] = int(choice)
                    except ValueError:
                        scores[from_name] = float(choice)
            annotator_ratings[ann_id] = scores

        # Calculate consensus
        consensus = {}
        for axis in axes:
            vals = [annotator_ratings[ann].get(axis, 0) for ann in annotator_ratings if axis in annotator_ratings[ann]]
            if vals:
                avg = sum(vals) / len(vals)
                # Map 0-3 Likert to 0.0-1.0
                consensus[axis] = round(avg / 3.0 if max(vals) <= 3 else avg, 3)
            else:
                consensus[axis] = 0.0

        processed_annotations.append({
            "id": probe_id,
            "ratings": annotator_ratings,
            "consensus_normalized": consensus,
        })

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(processed_annotations, f, indent=2)

    return processed_annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Studio Export & Parser Utility")
    parser.add_argument("--export-tasks", action="store_true", help="Export probes to Label Studio tasks")
    parser.add_argument("--probes", default="data/probe_outputs_200.json", help="Path to input probes")
    parser.add_argument("--output-tasks", default="data/label_studio_tasks.json", help="Destination tasks file")
    args = parser.parse_args()

    if args.export_tasks:
        export_probes_to_label_studio(args.probes, args.output_tasks)
        print(f"Exported tasks to {args.output_tasks}")
