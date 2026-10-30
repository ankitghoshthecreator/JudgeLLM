"""
Multi-Judge Ensembling & Variance Quantification Engine
========================================================
Enterprise-grade evaluation harness that reduces single-judge bias and quantifies
evaluator uncertainty using:
1. Cross-Model Ensembling: Aggregates judgments from multiple distinct LLM architectures.
2. Self-Consistency Resampling: Samples temperature-perturbed judge responses to quantify variance.
3. Outlier Judge Identification: Flags evaluators deviating from panel consensus by >1.5 sigma.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.teacher_judge import JudgeEvaluation, TeacherJudge

logger = logging.getLogger("EnsembleJudge")


@dataclass
class EnsembleResult:
    consensus_scores: Dict[str, float]
    median_scores: Dict[str, float]
    variance: Dict[str, float]
    std_dev: Dict[str, float]
    uncertainty_index: float
    num_judges: int
    ensemble_type: str
    individual_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    outlier_judges: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiJudgeEnsemble:
    """
    Orchestrates a panel of AI judges to reach consensus ratings,
    measure decision variance, and isolate idiosyncratic model bias.
    """

    DEFAULT_PANEL = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
    ]

    def __init__(
        self,
        panel_models: Optional[List[str]] = None,
        rubric_path: Optional[str | Path] = None,
        mock_mode: Optional[bool] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.rubric_path = rubric_path or (base_dir / "evaluation" / "rubrics" / "rubric_b.yaml")
        self.panel_models = panel_models or list(self.DEFAULT_PANEL)
        
        api_key = os.getenv("GROQ_API_KEY", "")
        if mock_mode is not None:
            self.mock_mode = mock_mode
        else:
            self.mock_mode = not bool(api_key)

    def _calculate_statistics(
        self,
        scores_by_axis: Dict[str, List[float]],
        judge_names: List[str],
        individual_evals: List[Dict[str, Any]],
        ensemble_type: str,
    ) -> EnsembleResult:
        consensus: Dict[str, float] = {}
        medians: Dict[str, float] = {}
        variances: Dict[str, float] = {}
        std_devs: Dict[str, float] = {}

        axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination"]
        total_std = 0.0

        for axis in axes:
            vals = scores_by_axis.get(axis, [0.7])
            if not vals:
                vals = [0.7]

            mean_val = statistics.mean(vals)
            med_val = statistics.median(vals)
            var_val = statistics.variance(vals) if len(vals) > 1 else 0.0
            sd_val = math.sqrt(var_val)

            consensus[axis] = round(mean_val, 3)
            medians[axis] = round(med_val, 3)
            variances[axis] = round(var_val, 4)
            std_devs[axis] = round(sd_val, 4)
            total_std += sd_val

        avg_uncertainty = round(total_std / len(axes), 4)

        # Detect outlier judges (> 1.5 sigma distance from consensus across all axes)
        outliers: List[str] = []
        if len(judge_names) > 2 and avg_uncertainty > 0.01:
            for idx, j_name in enumerate(judge_names):
                diffs = [
                    abs(scores_by_axis[axis][idx] - consensus[axis]) / (std_devs[axis] if std_devs[axis] > 1e-4 else 1.0)
                    for axis in axes if len(scores_by_axis[axis]) > idx
                ]
                avg_z = sum(diffs) / len(diffs) if diffs else 0.0
                if avg_z > 1.5:
                    outliers.append(j_name)

        return EnsembleResult(
            consensus_scores=consensus,
            median_scores=medians,
            variance=variances,
            std_dev=std_devs,
            uncertainty_index=avg_uncertainty,
            num_judges=len(judge_names),
            ensemble_type=ensemble_type,
            individual_evaluations=individual_evals,
            outlier_judges=outliers,
        )

    def evaluate_cross_model(
        self,
        prompt: str,
        reference_context: str,
        model_output: str,
        models: Optional[List[str]] = None,
        mode: str = "auto",
    ) -> EnsembleResult:
        """
        Queries multiple distinct LLM judges and aggregates their evaluations.
        """
        active_models = models or self.panel_models
        scores_by_axis: Dict[str, List[float]] = {
            "fluency": [],
            "coherence": [],
            "instruction_adherence": [],
            "no_hallucination": [],
        }
        judge_names: List[str] = []
        individual_evals: List[Dict[str, Any]] = []

        for m_name in active_models:
            judge = TeacherJudge(
                rubric_path=self.rubric_path,
                model_name=m_name,
                mock_mode=self.mock_mode,
            )
            try:
                # Add minor simulated variance across architectures in mock mode
                meta = {}
                if self.mock_mode:
                    import random
                    seed = abs(hash(m_name + prompt[:20])) % (10**6)
                    rng = random.Random(seed)
                    jitter = rng.uniform(-0.06, 0.06)
                    meta["simulated_jitter"] = jitter

                res: JudgeEvaluation = judge.evaluate_single(
                    prompt=prompt,
                    reference_context=reference_context,
                    model_output=model_output,
                    metadata=meta,
                    mode=mode,
                )

                # Collect scores
                for axis in scores_by_axis:
                    val = getattr(res, axis)
                    if self.mock_mode and "simulated_jitter" in meta:
                        val = min(1.0, max(0.0, val + meta["simulated_jitter"]))
                    scores_by_axis[axis].append(round(val, 3))

                judge_names.append(m_name)
                individual_evals.append({
                    "judge_model": m_name,
                    "scores": {
                        "fluency": res.fluency,
                        "coherence": res.coherence,
                        "instruction_adherence": res.instruction_adherence,
                        "no_hallucination": res.no_hallucination,
                    },
                    "reasoning": res.reasoning,
                })
            except Exception as e:
                logger.warning("Judge '%s' evaluation failed: %s", m_name, e)

        return self._calculate_statistics(scores_by_axis, judge_names, individual_evals, "cross_model")

    def evaluate_self_consistency(
        self,
        prompt: str,
        reference_context: str,
        model_output: str,
        model_name: Optional[str] = None,
        n_samples: int = 3,
        temperature: float = 0.7,
        mode: str = "auto",
    ) -> EnsembleResult:
        """
        Samples the same judge N times with temperature sampling to measure variance.
        """
        target_model = model_name or self.panel_models[0]
        scores_by_axis: Dict[str, List[float]] = {
            "fluency": [],
            "coherence": [],
            "instruction_adherence": [],
            "no_hallucination": [],
        }
        judge_names: List[str] = []
        individual_evals: List[Dict[str, Any]] = []

        judge = TeacherJudge(
            rubric_path=self.rubric_path,
            model_name=target_model,
            mock_mode=self.mock_mode,
        )

        for i in range(n_samples):
            try:
                meta = {}
                if self.mock_mode:
                    import random
                    seed = abs(hash(f"{target_model}_{i}_{prompt[:15]}")) % (10**6)
                    rng = random.Random(seed)
                    meta["simulated_jitter"] = rng.uniform(-0.05, 0.05)

                res: JudgeEvaluation = judge.evaluate_single(
                    prompt=prompt,
                    reference_context=reference_context,
                    model_output=model_output,
                    metadata=meta,
                    mode=mode,
                )

                for axis in scores_by_axis:
                    val = getattr(res, axis)
                    if self.mock_mode and "simulated_jitter" in meta:
                        val = min(1.0, max(0.0, val + meta["simulated_jitter"]))
                    scores_by_axis[axis].append(round(val, 3))

                sample_label = f"{target_model} (Sample #{i+1}, T={temperature})"
                judge_names.append(sample_label)
                individual_evals.append({
                    "judge_sample": sample_label,
                    "scores": {
                        "fluency": res.fluency,
                        "coherence": res.coherence,
                        "instruction_adherence": res.instruction_adherence,
                        "no_hallucination": res.no_hallucination,
                    },
                    "reasoning": res.reasoning,
                })
            except Exception as e:
                logger.warning("Sample %d failed: %s", i + 1, e)

        return self._calculate_statistics(scores_by_axis, judge_names, individual_evals, "self_consistency")

    def evaluate_qa_pair(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ensemble 10-star rating comparison for dual-model answering.
        Returns consensus stars, variance, and panel breakdown.
        """
        from evaluation.pdf_qa_pipeline import judge_answers

        active_models = models or self.panel_models
        scores_a: List[int] = []
        scores_b: List[int] = []
        panel_records: List[Dict[str, Any]] = []

        for m_judge in active_models:
            res = judge_answers(
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
                model_a=model_a_name,
                model_b=model_b_name,
                judge_model=m_judge,
            )
            scores_a.append(res["score_a"])
            scores_b.append(res["score_b"])
            panel_records.append({
                "judge_model": m_judge,
                "score_a": res["score_a"],
                "score_b": res["score_b"],
                "preferred": res.get("preferred", "Tie"),
                "reasoning_a": res.get("reasoning_a", ""),
                "reasoning_b": res.get("reasoning_b", ""),
            })

        mean_a = round(statistics.mean(scores_a), 2)
        mean_b = round(statistics.mean(scores_b), 2)
        med_a = round(statistics.median(scores_a), 1)
        med_b = round(statistics.median(scores_b), 1)
        var_a = round(statistics.variance(scores_a) if len(scores_a) > 1 else 0.0, 3)
        var_b = round(statistics.variance(scores_b) if len(scores_b) > 1 else 0.0, 3)
        sd_a = round(math.sqrt(var_a), 2)
        sd_b = round(math.sqrt(var_b), 2)

        # Majority preference
        prefs = [r["preferred"] for r in panel_records]
        majority_pref = max(set(prefs), key=prefs.count) if prefs else "Tie"

        return {
            "consensus_score_a": mean_a,
            "consensus_score_b": mean_b,
            "median_score_a": med_a,
            "median_score_b": med_b,
            "std_dev_a": sd_a,
            "std_dev_b": sd_b,
            "panel_preference": majority_pref,
            "panel_evaluations": panel_records,
            "num_judges": len(active_models),
        }
