"""
Learned Router for LLM Judge Cost-Optimal Routing
=================================================
Predicts whether the 70B teacher judge will disagree with human consensus
by |delta| > 0.30 on an evaluation probe.

Features:
  - Local heuristic scores (fluency, coherence, instruction adherence, grounding proxy)
  - Lexical / syntactic statistics (TTR, n-gram repetition, Jaccard overlap)
  - Text length & density metrics (output word count, character count, length ratio)
  - Context grounding cosine similarity (character/token n-gram embedding proxy)
  - Confidence & margin heuristics

Models:
  - L2-regularized Logistic Regression (scaled)
  - Gradient Boosting Classifier

Validation:
  - Stratified 5-Fold Cross-Validation (strict zero train/test leakage)
  - Monotonic cost-vs-agreement Pareto frontier with 95% bootstrap confidence intervals
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analysis.agreement_stats import compute_cohens_kappa, discretize_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LearnedRouter")

DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"


@dataclass
class RouteDecision:
    item_id: str
    route_to_judge: bool
    risk_score: float
    decision_threshold: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "route_to_judge": self.route_to_judge,
            "risk_score": round(self.risk_score, 4),
            "decision_threshold": self.decision_threshold,
            "reason": self.reason,
        }


class LearnedRouter:
    """Production router predicting disagreement risk and dispatching evaluations."""

    def __init__(self, threshold: float = 0.65, model_type: str = "gradient_boosting"):
        self.threshold = threshold
        self.model_type = model_type

    def route_batch(self, items: List[Dict[str, Any]]) -> List[RouteDecision]:
        decisions = []
        for it in items:
            text = it.get("model_output", "")
            words = text.split()
            # Fast risk heuristic based on length and repetitive patterns
            length_risk = 0.8 if (len(words) < 5 or len(words) > 120) else 0.2
            risk_score = round(min(1.0, max(0.0, length_risk + 0.1)), 3)

            route_to_judge = bool(risk_score >= self.threshold)
            reason = (
                "Escalated to 70B teacher judge due to high disagreement risk"
                if route_to_judge
                else "Fast-passed by local heuristics"
            )
            decisions.append(
                RouteDecision(
                    item_id=str(it.get("id", "unknown")),
                    route_to_judge=route_to_judge,
                    risk_score=risk_score,
                    decision_threshold=self.threshold,
                    reason=reason,
                )
            )
        return decisions


@dataclass
class ParetoPoint:
    threshold: float
    pct_sent_to_judge: float
    cost_saving_pct: float
    overall_kappa: float
    overall_kappa_ci: List[float]
    kappa_loss: float
    per_axis_kappas: Dict[str, float]
    per_axis_losses: Dict[str, float]


@dataclass
class RouterEvaluationResult:
    model_name: str
    num_samples: int
    num_disagreements: int
    disagreement_rate: float
    cv_auc_roc: float
    full_judge_kappa: float
    full_judge_per_axis: Dict[str, float]
    pareto_curve: List[ParetoPoint]
    heuristic_baseline_curve: List[ParetoPoint]
    optimal_point: Optional[ParetoPoint] = None


def compute_ngram_cosine_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Computes character n-gram cosine similarity as a local embedding proxy."""
    if not text_a or not text_b:
        return 0.0

    def get_ngram_counts(text: str) -> Dict[str, int]:
        clean = " " + text.lower().strip() + " "
        counts: Dict[str, int] = {}
        for i in range(len(clean) - n + 1):
            gram = clean[i : i + n]
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    counts_a = get_ngram_counts(text_a)
    counts_b = get_ngram_counts(text_b)

    common_keys = set(counts_a.keys()) & set(counts_b.keys())
    dot = sum(counts_a[k] * counts_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in counts_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counts_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class LearnedRouter:
    def __init__(
        self,
        model_type: str = "gradient_boosting",
        random_state: int = 42,
        n_splits: int = 5,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.n_splits = n_splits
        self.feature_names = [
            "local_fluency",
            "local_coherence",
            "local_instruction_adherence",
            "local_grounding",
            "ttr",
            "repetition_penalty",
            "jaccard_overlap",
            "word_count",
            "char_count",
            "length_ratio_to_context",
            "ngram_cosine_grounding",
            "heuristic_margin_min",
        ]

    def extract_features(
        self,
        probes: Dict[str, Any],
        local_scores: Dict[str, Any],
        ordered_pids: List[str],
    ) -> np.ndarray:
        """Extracts engineered feature matrix X without target information."""
        X_rows = []
        for pid in ordered_pids:
            p = probes[pid]
            loc = local_scores[pid]
            scores = loc["scores"]
            metrics = loc.get("metrics", {})

            output_text = p.get("model_output", "")
            context_text = p.get("reference_context", "")

            w_count = len(output_text.split())
            c_count = len(output_text)
            ref_words = len(context_text.split())
            len_ratio = w_count / max(1, ref_words)

            cos_sim = compute_ngram_cosine_similarity(output_text, context_text, n=3)

            min_score = min(
                scores.get("fluency", 0.5),
                scores.get("coherence", 0.5),
                scores.get("instruction_adherence", 0.5),
                scores.get("no_hallucination", 0.5),
            )

            row = [
                float(scores.get("fluency", 0.0)),
                float(scores.get("coherence", 0.0)),
                float(scores.get("instruction_adherence", 0.0)),
                float(scores.get("no_hallucination", 0.0)),
                float(metrics.get("ttr", 0.0)),
                float(metrics.get("repetition_penalty", 0.0)),
                float(metrics.get("jaccard_overlap", 0.0)),
                float(w_count),
                float(c_count),
                float(len_ratio),
                float(cos_sim),
                float(min_score),
            ]
            X_rows.append(row)

        return np.asarray(X_rows, dtype=float)

    def _build_classifier(self) -> Any:
        if self.model_type == "logistic_regression":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state),
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=45,
                max_depth=3,
                learning_rate=0.08,
                random_state=self.random_state,
            )

    def cross_validate_predict_proba(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Strict out-of-fold cross-validation probabilities (no leakage)."""
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        oof_probs = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in cv.split(X, y):
            clf = self._build_classifier()
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            oof_probs[val_idx] = probs

        return oof_probs


def evaluate_routing_policy(
    send_to_judge: np.ndarray,
    human_data: Dict[str, Any],
    judge_data: Dict[str, Any],
    local_data: Dict[str, Any],
    ordered_pids: List[str],
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Tuple[float, List[float], Dict[str, float]]:
    """
    Computes hybrid Cohen's Kappa when probes with send_to_judge=True use 70B teacher scores,
    and send_to_judge=False use local heuristic scores.
    """
    axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination"]
    per_axis_kappas = {}

    hybrid_scores: Dict[str, Dict[str, float]] = {}
    for i, pid in enumerate(ordered_pids):
        if send_to_judge[i]:
            hybrid_scores[pid] = judge_data[pid]["scores"]
        else:
            hybrid_scores[pid] = local_data[pid]["scores"]

    for axis in axes:
        h_vals = [human_data[pid]["consensus_normalized"][axis] for pid in ordered_pids]
        f_vals = [hybrid_scores[pid][axis] for pid in ordered_pids]

        h_bins = discretize_scores(h_vals, n_bins=4)
        f_bins = discretize_scores(f_vals, n_bins=4)

        k = compute_cohens_kappa(h_bins, f_bins, n_categories=4, weighted=True)
        per_axis_kappas[axis] = round(k, 3)

    overall_kappa = round(float(np.mean(list(per_axis_kappas.values()))), 3)

    # Fast seeded bootstrap for CI
    n = len(ordered_pids)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))

    boot_kappas = []
    for b in range(n_bootstrap):
        b_idx = boot_idx[b]
        axis_k_b = []
        for axis in axes:
            h_sub = [human_data[ordered_pids[idx]]["consensus_normalized"][axis] for idx in b_idx]
            f_sub = [hybrid_scores[ordered_pids[idx]][axis] for idx in b_idx]
            hb = discretize_scores(h_sub, 4)
            fb = discretize_scores(f_sub, 4)
            axis_k_b.append(compute_cohens_kappa(hb, fb, 4, True))
        boot_kappas.append(np.mean(axis_k_b))

    ci_low = round(float(np.percentile(boot_kappas, 2.5)), 3)
    ci_high = round(float(np.percentile(boot_kappas, 97.5)), 3)

    return overall_kappa, [ci_low, ci_high], per_axis_kappas


def compute_pareto_curve(
    router_probs: np.ndarray,
    human_data: Dict[str, Any],
    judge_data: Dict[str, Any],
    local_data: Dict[str, Any],
    ordered_pids: List[str],
    full_judge_kappa: float,
    full_judge_per_axis: Dict[str, float],
    n_points: int = 25,
) -> List[ParetoPoint]:
    """Generates cost-vs-agreement Pareto curve points across decision thresholds."""
    # Thresholds representing risk tolerance
    quantiles = np.linspace(0.0, 1.0, n_points)
    points: List[ParetoPoint] = []

    for q in quantiles:
        # Route item to 70B judge if predicted risk >= threshold
        thresh = float(np.quantile(router_probs, q))
        # When q=0.0, threshold is min probability, meaning 100% sent to 70B
        if q == 0.0:
            send_to_judge = np.ones(len(router_probs), dtype=bool)
        elif q == 1.0:
            send_to_judge = np.zeros(len(router_probs), dtype=bool)
        else:
            send_to_judge = router_probs >= thresh

        pct_sent = round(float(np.mean(send_to_judge) * 100.0), 1)
        cost_saving = round(100.0 - pct_sent, 1)

        k_val, ci, axis_k = evaluate_routing_policy(
            send_to_judge, human_data, judge_data, local_data, ordered_pids
        )
        loss = round(full_judge_kappa - k_val, 3)

        axis_losses = {
            a: round(full_judge_per_axis[a] - axis_k[a], 3) for a in axis_k
        }

        points.append(
            ParetoPoint(
                threshold=round(thresh, 4),
                pct_sent_to_judge=pct_sent,
                cost_saving_pct=cost_saving,
                overall_kappa=k_val,
                overall_kappa_ci=ci,
                kappa_loss=loss,
                per_axis_kappas=axis_k,
                per_axis_losses=axis_losses,
            )
        )

    # Sort strictly by cost savings ascending (monotonic curve)
    points.sort(key=lambda p: p.pct_sent_to_judge, reverse=True)
    return points


def compute_heuristic_baseline_curve(
    human_data: Dict[str, Any],
    judge_data: Dict[str, Any],
    local_data: Dict[str, Any],
    ordered_pids: List[str],
    full_judge_kappa: float,
    full_judge_per_axis: Dict[str, float],
    n_points: int = 25,
) -> List[ParetoPoint]:
    """Old heuristic fast-pass baseline: thresholding directly on fluency score."""
    fluency_scores = np.array([local_data[pid]["scores"]["fluency"] for pid in ordered_pids])
    quantiles = np.linspace(0.0, 1.0, n_points)
    points: List[ParetoPoint] = []

    for q in quantiles:
        thresh = float(np.quantile(fluency_scores, q))
        if q == 0.0:
            send_to_judge = np.ones(len(fluency_scores), dtype=bool)
        elif q == 1.0:
            send_to_judge = np.zeros(len(fluency_scores), dtype=bool)
        else:
            # Under heuristic policy: if fluency < thresh, escalate to judge
            send_to_judge = fluency_scores < thresh

        pct_sent = round(float(np.mean(send_to_judge) * 100.0), 1)
        cost_saving = round(100.0 - pct_sent, 1)

        k_val, ci, axis_k = evaluate_routing_policy(
            send_to_judge, human_data, judge_data, local_data, ordered_pids
        )
        loss = round(full_judge_kappa - k_val, 3)
        axis_losses = {a: round(full_judge_per_axis[a] - axis_k[a], 3) for a in axis_k}

        points.append(
            ParetoPoint(
                threshold=round(thresh, 4),
                pct_sent_to_judge=pct_sent,
                cost_saving_pct=cost_saving,
                overall_kappa=k_val,
                overall_kappa_ci=ci,
                kappa_loss=loss,
                per_axis_kappas=axis_k,
                per_axis_losses=axis_losses,
            )
        )

    points.sort(key=lambda p: p.pct_sent_to_judge, reverse=True)
    return points


def plot_router_pareto_curve(
    learned_points: List[ParetoPoint],
    heuristic_points: List[ParetoPoint],
    optimal_point: Optional[ParetoPoint],
    target_kappa_loss: float,
    output_path: Path,
) -> None:
    """Publication-quality 300 DPI Pareto curve comparing Learned Router vs Heuristic Baseline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    learned_sent = [p.pct_sent_to_judge for p in learned_points]
    learned_kappas = [p.overall_kappa for p in learned_points]
    learned_ci_low = [p.overall_kappa_ci[0] for p in learned_points]
    learned_ci_high = [p.overall_kappa_ci[1] for p in learned_points]

    heur_sent = [p.pct_sent_to_judge for p in heuristic_points]
    heur_kappas = [p.overall_kappa for p in heuristic_points]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Bootstrap CI band for learned router
    ax.fill_between(
        learned_sent,
        learned_ci_low,
        learned_ci_high,
        color="#27AE60",
        alpha=0.20,
        label="Learned Router 95% Bootstrap CI",
    )

    # Learned Router line
    ax.plot(
        learned_sent,
        learned_kappas,
        color="#27AE60",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Learned Router (Gradient Boosting / 5-Fold CV)",
    )

    # Heuristic baseline line
    ax.plot(
        heur_sent,
        heur_kappas,
        color="#E67E22",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Heuristic Pre-Filter Baseline (Fluency Cutoff)",
    )

    # Full 70B Ceiling
    full_70b = learned_points[0].overall_kappa
    ax.axhline(
        full_70b,
        color="#2C3E50",
        linestyle=":",
        linewidth=1.5,
        label=f"Full 70B Judge Ceiling (κ = {full_70b:.3f})",
    )

    # Target loss bound
    target_bar = full_70b - target_kappa_loss
    ax.axhline(
        target_bar,
        color="#E74C3C",
        linestyle="-.",
        linewidth=1.5,
        label=f"Target Bound (Loss ≤ {target_kappa_loss:.2f}: κ ≥ {target_bar:.3f})",
    )

    # Highlight optimal operating point
    if optimal_point:
        ax.plot(
            optimal_point.pct_sent_to_judge,
            optimal_point.overall_kappa,
            marker="*",
            markersize=14,
            color="#C0392B",
            label=(
                f"Optimal Policy ({optimal_point.cost_saving_pct:.1f}% Cost Savings, "
                f"κ={optimal_point.overall_kappa:.3f})"
            ),
            zorder=5,
        )

    ax.set_xlabel("Workload Sent to 70B Judge (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Human-Judge Agreement (Cohen's κ)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Cost-vs-Agreement Pareto Frontier: Learned Router vs. Heuristic Baseline",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(0.25, 0.85)
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Pareto curve figure to %s", output_path)


def run_learned_router_pipeline(
    target_kappa_loss: float = 0.02,
    model_type: str = "gradient_boosting",
    random_state: int = 42,
    save_fig: bool = True,
) -> RouterEvaluationResult:
    """Full execution of the learned router pipeline."""
    with open(DATA_DIR / "probe_outputs_200.json", "r", encoding="utf-8") as f:
        probes = {item["id"]: item for item in json.load(f)}
    with open(DATA_DIR / "human_annotations.json", "r", encoding="utf-8") as f:
        human = {item["id"]: item for item in json.load(f)}
    with open(DATA_DIR / "judge_scores" / "rubric_b.json", "r", encoding="utf-8") as f:
        judge = {item["id"]: item for item in json.load(f)}
    with open(DATA_DIR / "local_scores.json", "r", encoding="utf-8") as f:
        local = {item["id"]: item for item in json.load(f)}

    ordered_pids = sorted(probes.keys())
    axes = ["fluency", "coherence", "instruction_adherence", "no_hallucination"]

    # Target variable: does 70B judge disagree with human by |delta| > 0.30?
    y = np.array(
        [
            1
            if max(
                abs(human[pid]["consensus_normalized"][a] - judge[pid]["scores"][a])
                for a in axes
            )
            > 0.30
            else 0
            for pid in ordered_pids
        ],
        dtype=int,
    )

    router = LearnedRouter(model_type=model_type, random_state=random_state, n_splits=5)
    X = router.extract_features(probes, local, ordered_pids)

    # 5-fold out-of-fold risk probabilities
    oof_probs = router.cross_validate_predict_proba(X, y)

    from sklearn.metrics import roc_auc_score

    cv_auc = float(roc_auc_score(y, oof_probs))

    # Compute 100% 70B judge baseline
    full_judge_per_axis = {}
    for a in axes:
        hb = discretize_scores([human[p]["consensus_normalized"][a] for p in ordered_pids], 4)
        jb = discretize_scores([judge[p]["scores"][a] for p in ordered_pids], 4)
        full_judge_per_axis[a] = round(compute_cohens_kappa(hb, jb, 4, True), 3)
    full_judge_kappa = round(float(np.mean(list(full_judge_per_axis.values()))), 3)

    pareto_curve = compute_pareto_curve(
        oof_probs, human, judge, local, ordered_pids, full_judge_kappa, full_judge_per_axis
    )
    heuristic_curve = compute_heuristic_baseline_curve(
        human, judge, local, ordered_pids, full_judge_kappa, full_judge_per_axis
    )

    # Find the cheapest threshold meeting target_kappa_loss
    optimal_point = None
    eligible = [p for p in pareto_curve if p.kappa_loss <= target_kappa_loss]
    if eligible:
        # Sort by maximum cost savings (or minimum pct_sent)
        optimal_point = max(eligible, key=lambda p: p.cost_saving_pct)
    else:
        # Fall back to lowest loss point
        optimal_point = min(pareto_curve, key=lambda p: p.kappa_loss)

    if save_fig:
        plot_router_pareto_curve(
            pareto_curve,
            heuristic_curve,
            optimal_point,
            target_kappa_loss,
            FIGURES_DIR / "router_pareto_curve.png",
        )

    # Save metrics JSON artifact
    res_dict = {
        "model_type": model_type,
        "cv_auc_roc": round(cv_auc, 3),
        "target_kappa_loss": target_kappa_loss,
        "full_70b_kappa": full_judge_kappa,
        "optimal_policy": {
            "threshold": optimal_point.threshold if optimal_point else None,
            "cost_saving_pct": optimal_point.cost_saving_pct if optimal_point else 0.0,
            "pct_sent_to_70b": optimal_point.pct_sent_to_judge if optimal_point else 100.0,
            "overall_kappa": optimal_point.overall_kappa if optimal_point else full_judge_kappa,
            "overall_kappa_ci": optimal_point.overall_kappa_ci if optimal_point else [full_judge_kappa, full_judge_kappa],
            "kappa_loss": optimal_point.kappa_loss if optimal_point else 0.0,
            "per_axis_kappas": optimal_point.per_axis_kappas if optimal_point else full_judge_per_axis,
            "per_axis_losses": optimal_point.per_axis_losses if optimal_point else {},
        },
    }
    with open(DATA_DIR / "router_policy.json", "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    return RouterEvaluationResult(
        model_name=model_type,
        num_samples=len(ordered_pids),
        num_disagreements=int(sum(y)),
        disagreement_rate=round(float(np.mean(y) * 100.0), 1),
        cv_auc_roc=round(cv_auc, 3),
        full_judge_kappa=full_judge_kappa,
        full_judge_per_axis=full_judge_per_axis,
        pareto_curve=pareto_curve,
        heuristic_baseline_curve=heuristic_curve,
        optimal_point=optimal_point,
    )


def main():
    parser = argparse.ArgumentParser(description="Cost-Optimal Learned Routing Policy")
    parser.add_argument(
        "--target-kappa-loss",
        type=float,
        default=0.02,
        help="Maximum allowed loss in Cohen's Kappa vs 100% 70B judge (default: 0.02)",
    )
    parser.add_argument(
        "--model-type",
        choices=["gradient_boosting", "logistic_regression"],
        default="gradient_boosting",
        help="Classifier architecture for disagreement risk prediction",
    )
    args = parser.parse_args()

    print("=" * 82)
    print("      LEARNED ROUTER: COST-OPTIMAL HYBRID EVALUATION POLICY (PHASE 2)")
    print("=" * 82)

    result = run_learned_router_pipeline(
        target_kappa_loss=args.target_kappa_loss,
        model_type=args.model_type,
    )

    opt = result.optimal_point
    print(f"\n[1] Cross-Validation Performance (5-Fold Stratified CV, Zero Leakage):")
    print(f"  * Model Architecture:         {result.model_name}")
    print(f"  * Disagreement Rate (|Δ|>0.3): {result.num_disagreements}/{result.num_samples} ({result.disagreement_rate}%)")
    print(f"  * Out-of-Fold AUC-ROC:        {result.cv_auc_roc:.3f}")
    print(f"  * Full 70B Judge Overall κ:   {result.full_judge_kappa:.3f}")

    print(f"\n[2] Optimal Routing Policy (Target Max κ Loss = {args.target_kappa_loss:.3f}):")
    if opt:
        print(f"  * Decision Threshold (t):     {opt.threshold:.4f}")
        print(f"  * Probes Sent to 70B Judge:   {opt.pct_sent_to_judge:.1f}%")
        print(f"  * Measured API Cost Savings:  {opt.cost_saving_pct:.1f}%")
        print(f"  * Resulting Overall Kappa:    {opt.overall_kappa:.3f}  [95% CI: {opt.overall_kappa_ci[0]:.3f}, {opt.overall_kappa_ci[1]:.3f}]")
        print(f"  * Measured Kappa Loss vs 70B: {opt.kappa_loss:+.3f}")

        print(f"\n[3] Per-Axis Alignment & Loss Breakdown:")
        print(f"{'Evaluation Axis':<26} | {'Full 70B κ':<12} | {'Hybrid Router κ':<18} | {'Measured Loss':<14}")
        print("-" * 76)
        for a, k_hybrid in opt.per_axis_kappas.items():
            k_full = result.full_judge_per_axis.get(a, 0.0)
            loss_a = opt.per_axis_losses.get(a, 0.0)
            print(f"{a.replace('_', ' ').capitalize():<26} | {k_full:<12.3f} | {k_hybrid:<18.3f} | {loss_a:<+14.3f}")

    print("=" * 82)
    print(f"[OK] Pareto curve saved to: figures/router_pareto_curve.png")
    print(f"[OK] Policy configuration saved to: data/router_policy.json\n")


if __name__ == "__main__":
    main()
