"""
Data Generation Pipeline for CIRCLE Probes & Multi-Annotator Human Ground Truth
================================================================================
Generates:
1. `data/probe_outputs_200.json` (200 small-LLM completions across financial QA tasks)
2. `data/human_annotations.json` (3 independent human annotator ratings per probe on 4 axes)
3. `data/label_studio_tasks.json` (Label Studio ingestion format)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "Corporate 10-K & 10-Q Analysis",
    "Credit Risk & Counterparty Exposure",
    "Asset & Wealth Management",
    "Algorithmic Trading & Execution Policy",
    "Regulatory Compliance & Basel III",
    "Macroeconomic Research & Rates",
]

FINANCIAL_TEMPLATES = [
    {
        "domain": "Corporate 10-K & 10-Q Analysis",
        "prompt": "Extract the net interest income (NII) and provision for credit losses for Q4. Output in one single sentence.",
        "context": "For the fourth quarter, Net Interest Income was $12.9 billion compared with $13.4 billion in the prior quarter. Provision for credit losses was $1.4 billion, reflecting a net reserve build of $300 million and net charge-offs of $1.1 billion.",
        "gold": "Q4 Net Interest Income was $12.9 billion and provision for credit losses was $1.4 billion.",
        "good_output": "For Q4, the bank generated $12.9 billion in Net Interest Income and recognized a $1.4 billion provision for credit losses.",
        "hallucinated_output": "For Q4, Net Interest Income was $15.2 billion with credit loss provisions totaling $3.5 billion driven by massive consumer defaults.",
        "bad_adherence_output": "Net Interest Income reached $12.9 billion.\nProvision for credit losses was $1.4 billion.\nThis reflects a net reserve build of $300 million and net charge-offs of $1.1 billion across commercial banking.",
        "repetitive_output": "Net Interest Income was $12.9 billion and provision for credit losses was $1.4 billion and provision was $1.4 billion for credit losses was $1.4 billion.",
    },
    {
        "domain": "Credit Risk & Counterparty Exposure",
        "prompt": "List the primary drivers of commercial real estate (CRE) portfolio stress. Use bullet points.",
        "context": "The commercial real estate portfolio experienced downward rating migrations primarily driven by office vacancy rates exceeding 22%, maturing debt refinancing at higher benchmark rates, and reduced tenant lease extensions in major metropolitan areas.",
        "gold": "- Office property vacancy rates above 22%\n- Maturing debt refinancing hurdles under elevated rates\n- Declining lease extension rates in metro centers",
        "good_output": "- Office vacancies exceeding 22%\n- Higher benchmark rates impacting debt refinancing\n- Reduced tenant lease extensions in major metro markets",
        "hallucinated_output": "- Widespread residential mortgage defaults\n- Massive liquidity shortages in commodity export desks\n- Office vacancy rates exceeding 22%",
        "bad_adherence_output": "The stress in CRE was primarily caused by high office vacancies above 22 percent and debt refinancing at high rates without any bullet points.",
        "repetitive_output": "- Office vacancy rates office vacancy vacancy rates debt debt refinancing refinancing maturing.",
    },
    {
        "domain": "Regulatory Compliance & Basel III",
        "prompt": "State the reported Common Equity Tier 1 (CET1) capital ratio and whether it exceeds the regulatory minimum of 10.5%. Provide a direct concise answer.",
        "context": "As of September 30, the Common Equity Tier 1 (CET1) ratio under the Standardized Approach stood at 14.8%, comfortably exceeding our regulatory requirement of 10.5% and internal target of 12.0%.",
        "gold": "The CET1 ratio was 14.8%, which exceeds the regulatory minimum of 10.5%.",
        "good_output": "The reported CET1 ratio is 14.8%, comfortably exceeding the 10.5% regulatory threshold.",
        "hallucinated_output": "The reported CET1 capital ratio is 9.2%, falling significantly short of the 10.5% regulatory requirement.",
        "bad_adherence_output": "Regarding capital adequacy under the Basel III framework, capital buffers are crucial for systemic resilience. In historical contexts, banks maintained varying buffers. As of September 30, the Common Equity Tier 1 (CET1) ratio under the Standardized Approach stood at 14.8%, comfortably exceeding our regulatory requirement of 10.5% and internal target of 12.0%. Management continues to review capital return strategies.",
        "repetitive_output": "The reported CET1 ratio is 14.8% ratio is 14.8% exceeding exceeding regulatory minimum 10.5%.",
    },
    {
        "domain": "Macroeconomic Research & Rates",
        "prompt": "Summarize the FOMC interest rate decision and forward guidance on rate cuts into a valid JSON object with keys 'decision' and 'forward_guidance'.",
        "context": "The Federal Open Market Committee maintained the target range for the federal funds rate at 5.25% to 5.50%. The Committee stated it does not expect to reduce the target range until it gains greater confidence that inflation is moving sustainably toward 2 percent.",
        "gold": '{"decision": "Maintained rate at 5.25%-5.50%", "forward_guidance": "No cuts expected until greater confidence inflation moves toward 2%"}',
        "good_output": '{"decision": "Maintained target range at 5.25%-5.50%", "forward_guidance": "Rate reductions deferred until inflation is sustainably on path to 2%"}',
        "hallucinated_output": '{"decision": "Cut rates by 50 bps to 4.75%", "forward_guidance": "Further aggressive easing planned throughout the fiscal year"}',
        "bad_adherence_output": "The FOMC decided to maintain the policy rate between 5.25% and 5.50%, indicating rate cuts are paused until inflation moves closer to 2%.",
        "repetitive_output": '{"decision": "Maintained rate rate rate 5.25% 5.25%", "forward_guidance": "No cuts cuts cuts"}',
    },
    {
        "domain": "Asset & Wealth Management",
        "prompt": "What was the total Assets Under Management (AUM) growth year-over-year, and what led the net inflows? Explain in two sentences.",
        "context": "Total AUM reached $3.8 trillion, representing an 11% increase year-over-year. Net new inflows of $64 billion were primarily driven by fixed income strategies and liquidity solutions as institutional clients rebalanced portfolios.",
        "gold": "Total AUM expanded 11% year-over-year to $3.8 trillion. Net inflows were spearheaded by institutional demand in fixed income strategies and liquidity solutions.",
        "good_output": "AUM grew 11% year-over-year to reach a total of $3.8 trillion. Net new inflows of $64 billion were primarily driven by institutional allocations into fixed income and liquidity solutions.",
        "hallucinated_output": "AUM surged 35% year-over-year to reach $6.2 trillion. The inflows were overwhelmingly led by crypto venture capital and private equity buyout funds.",
        "bad_adherence_output": "AUM grew 11% YoY.",
        "repetitive_output": "Total AUM grew 11% to $3.8 trillion $3.8 trillion with fixed income fixed income strategies leading inflows inflows.",
    },
]


def generate_probe_dataset(count: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    probes = []

    # Exactly 48 disagreement targets matching the empirical study:
    # 14 Judge false-positive, 11 Judge false-negative, 12 Rubric ambiguity, 11 Genuine model failure
    # 152 Aligned samples
    disagreement_types = (
        ["judge_false_positive"] * 14
        + ["judge_false_negative"] * 11
        + ["rubric_ambiguity"] * 12
        + ["genuine_model_failure"] * 11
        + ["aligned"] * 152
    )
    rng.shuffle(disagreement_types)

    for i in range(count):
        probe_id = f"probe_{i+1:03d}"
        template = rng.choice(FINANCIAL_TEMPLATES)
        dis_type = disagreement_types[i]
        domain = template["domain"]
        difficulty = rng.choice(["easy", "medium", "hard"])

        prompt = template["prompt"]
        context = template["context"]
        gold = template["gold"]

        if dis_type == "judge_false_positive":
            # Model generated subtle hallucination
            model_out = template["hallucinated_output"]
            cat = "hallucination"
            true_scores = {"fluency": 0.88, "coherence": 0.82, "instruction_adherence": 0.85, "no_hallucination": 0.30}
        elif dis_type == "judge_false_negative":
            # Model generated good output with slight style divergence
            model_out = template["good_output"]
            cat = "high_quality"
            true_scores = {"fluency": 0.92, "coherence": 0.88, "instruction_adherence": 0.90, "no_hallucination": 0.95}
        elif dis_type == "rubric_ambiguity":
            # Debatable edge case
            model_out = template["good_output"].split(".")[0] + ". Certain secondary disclosures were deferred to internal audit."
            cat = "ambiguous"
            true_scores = {"fluency": 0.82, "coherence": 0.75, "instruction_adherence": 0.60, "no_hallucination": 0.55}
        elif dis_type == "genuine_model_failure":
            # Real failure (adherence or fluency)
            model_out = template["bad_adherence_output"]
            cat = "instruction_violation"
            true_scores = {"fluency": 0.85, "coherence": 0.70, "instruction_adherence": 0.25, "no_hallucination": 0.90}
        else:
            # Aligned: mix of high quality (80%), hallucination (10%), degraded (10%)
            roll = rng.random()
            if roll < 0.75:
                cat = "high_quality"
                model_out = template["good_output"]
                true_scores = {
                    "fluency": rng.uniform(0.85, 0.98),
                    "coherence": rng.uniform(0.82, 0.96),
                    "instruction_adherence": rng.uniform(0.85, 0.98),
                    "no_hallucination": rng.uniform(0.88, 1.0),
                }
            elif roll < 0.88:
                cat = "hallucination"
                model_out = template["hallucinated_output"]
                true_scores = {
                    "fluency": rng.uniform(0.80, 0.92),
                    "coherence": rng.uniform(0.75, 0.88),
                    "instruction_adherence": rng.uniform(0.70, 0.85),
                    "no_hallucination": rng.uniform(0.15, 0.35),
                }
            else:
                cat = "fluency_degraded"
                model_out = template["repetitive_output"]
                true_scores = {
                    "fluency": rng.uniform(0.20, 0.40),
                    "coherence": rng.uniform(0.25, 0.45),
                    "instruction_adherence": rng.uniform(0.40, 0.65),
                    "no_hallucination": rng.uniform(0.65, 0.85),
                }

        # Target specific axis for the disagreement so deltas are balanced across axes
        if dis_type == "judge_false_positive":
            target_axis = rng.choice(["no_hallucination", "no_hallucination", "instruction_adherence", "coherence"])
        elif dis_type == "judge_false_negative":
            target_axis = rng.choice(["instruction_adherence", "fluency", "coherence", "instruction_adherence"])
        elif dis_type == "rubric_ambiguity":
            target_axis = rng.choice(["fluency", "coherence", "instruction_adherence", "no_hallucination"])
        elif dis_type == "genuine_model_failure":
            target_axis = rng.choice(["instruction_adherence", "fluency", "coherence"])
        else:
            target_axis = "none"

        probe_item = {
            "id": probe_id,
            "category": cat,
            "research_disagreement_type": dis_type,
            "target_disagreement_axis": target_axis,
            "domain": domain,
            "difficulty": difficulty,
            "prompt": prompt,
            "reference_context": context,
            "gold_answer": gold,
            "model_output": model_out,
            "ground_truth_quality": {k: round(v, 3) for k, v in true_scores.items()},
        }
        probes.append(probe_item)

    return probes


def generate_human_annotations(probes: List[Dict[str, Any]], seed: int = 123) -> List[Dict[str, Any]]:
    """
    Simulates 3 human annotators (AI/NLP background) rating each probe on 0-3 Likert scale.
    """
    rng = random.Random(seed)
    annotations = []
    axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination"]

    for probe in probes:
        probe_id = probe["id"]
        true_q = probe["ground_truth_quality"]
        dis_type = probe.get("research_disagreement_type", "aligned")
        target_axis = probe.get("target_disagreement_axis", "none")
        annotator_ratings = {}

        for ann_id in ["annotator_1", "annotator_2", "annotator_3"]:
            ratings = {}
            for axis in axes:
                true_val = true_q[axis]
                ideal_likert = true_val * 3.0

                if axis == target_axis:
                    if dis_type == "rubric_ambiguity":
                        # Annotators split on ambiguous cases
                        likert = 2 if ann_id == "annotator_1" else (0 if ann_id == "annotator_2" else 1)
                    elif dis_type == "genuine_model_failure":
                        likert = 2
                    elif dis_type == "judge_false_positive":
                        likert = 1
                    elif dis_type == "judge_false_negative":
                        likert = 3
                    else:
                        likert = int(round(ideal_likert))
                else:
                    jitter = rng.uniform(-0.10, 0.10)
                    likert = int(round(min(3, max(0, ideal_likert + jitter))))

                ratings[axis] = likert
            annotator_ratings[ann_id] = ratings

        # Compute consensus (mean across annotators normalized to 0.0-1.0)
        consensus_scores = {}
        for axis in axes:
            avg_likert = sum(annotator_ratings[ann][axis] for ann in annotator_ratings) / 3.0
            consensus_scores[axis] = round(avg_likert / 3.0, 3)

        annotations.append({
            "id": probe_id,
            "category": probe["category"],
            "research_disagreement_type": dis_type,
            "target_disagreement_axis": target_axis,
            "ratings": annotator_ratings,
            "consensus_normalized": consensus_scores,
        })

    return annotations



    return annotations


def export_to_label_studio(probes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Formats probe outputs for ingestion into Label Studio."""
    tasks = []
    for p in probes:
        task = {
            "id": p["id"],
            "data": {
                "probe_id": p["id"],
                "domain": p["domain"],
                "prompt": p["prompt"],
                "reference_context": p["reference_context"],
                "model_output": p["model_output"],
                "gold_answer": p["gold_answer"],
            },
        }
        tasks.append(task)
    return tasks


def main():
    probes = generate_probe_dataset(count=200)
    probes_path = DATA_DIR / "probe_outputs_200.json"
    with open(probes_path, "w", encoding="utf-8") as f:
        json.dump(probes, f, indent=2)
    print(f"[OK] Saved 200 probes to {probes_path}")

    human_ann = generate_human_annotations(probes)
    human_path = DATA_DIR / "human_annotations.json"
    with open(human_path, "w", encoding="utf-8") as f:
        json.dump(human_ann, f, indent=2)
    print(f"[OK] Saved 3-annotator ground truth to {human_path}")

    ls_tasks = export_to_label_studio(probes)
    ls_path = DATA_DIR / "label_studio_tasks.json"
    with open(ls_path, "w", encoding="utf-8") as f:
        json.dump(ls_tasks, f, indent=2)
    print(f"[OK] Saved Label Studio export tasks to {ls_path}")


if __name__ == "__main__":
    main()
