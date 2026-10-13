# Project Implementation Roadmap: LLM-as-a-Judge Alignment Study

This document divides the **LLM-as-a-Judge Alignment Study** into two rigorous engineering and research phases. Every aspect, component, metric, and deliverable is detailed below.

---

## Architecture Overview

```
                   ┌──────────────────────────────────────────────┐
                   │             Probe Outputs (200)              │
                   │    (CIRCLE Stage 3-5 Financial QA Bench)     │
                   └──────────────────────┬───────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌────────────────────────┐                     ┌───────────────────────────┐
     │ 3 Human Annotators     │                     │ Lightweight Local Scorer  │
     │ (Fluency, Coherence,   │                     │ (TTR, n-gram, Jaccard,    │
     │ Adherence, Factuality) │                     │ Regex Adherence @ ~0.01ms)│
     └───────────┬────────────┘                     └─────────────┬─────────────┘
                 │                                                │
                 │                ┌───────────────────────────────┘
                 │                │
                 ▼                ▼
   ┌────────────────────────────────────────────────────────┐
   │ Teacher Judge Engine (Groq 70B / Fallback Model)       │
   │  - Rubric A: Zero-shot baseline                        │
   │  - Rubric B: Chain-of-Thought + Few-Shot Exemplars     │
   │  - Pydantic v2 Schema Enforcement & Re-querying        │
   └──────────────────────────────┬─────────────────────────┘
                                  │
                 ═════════════════╪═════════════════ (Phase Boundary)
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────┐
   │ Phase 2: Alignment, Agreement & Decision Framework     │
   │  - Fleiss's Kappa (Inter-Annotator Agreement Ceiling)  │
   │  - Quadratic Cohen's Kappa (Judge vs Human Consensus)  │
   │  - Pearson & Spearman (Local Scorer vs Teacher Scorer) │
   │  - Disagreement & Failure Mode Decomposition (>0.3)    │
   │  - Visualization Suite & RLAIF Routing Policy          │
   └────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Infrastructure, Evaluation Engines & Annotation Pipeline

### Goal
Establish the complete data foundations, local high-throughput heuristics, Groq 70B structured teacher critique pipeline, and human ground-truth annotation ingestion mechanisms.

### Components & Technical Requirements

1. **Environment & Core Dependencies**:
   - Dedicated local virtual environment (`.venv`) isolating all dependencies.
   - Requirements: `pydantic>=2.0`, `groq`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`, `pytest`, `python-dotenv`.

2. **Probe Dataset (`data/probe_outputs_200.json`)**:
   - 200 standardized probe outputs simulating small-LLM completions (CIRCLE Stage 3-5 probes) across financial question answering, compliance reasoning, earnings summary, and risk assessments.
   - Fields: `id`, `prompt`, `reference_context`, `model_output`, `gold_answer`, `domain`, `difficulty`.
   - Balanced distribution of high-quality outputs, minor hallucinations, fluency breakdowns, and refusal/adherence violations to benchmark judges realistically.

3. **Human Annotation System & Label Studio Pipeline (`annotation/label_studio_export.py` & `data/human_annotations.json`)**:
   - Ingestion and export tool for Label Studio format.
   - 3 annotators rating each output across 4 axes:
     - `fluency` (0–3 Likert; normalized 0.0–1.0)
     - `coherence` (0–3 Likert; normalized 0.0–1.0)
     - `instruction_adherence` (0–3 Likert; normalized 0.0–1.0)
     - `no_hallucination` (0–3 Likert; normalized 0.0–1.0)
   - Schema validating individual ratings, consensus score, and inter-rater consistency.

4. **Rubric Architecture (`evaluation/rubrics/`)**:
   - `rubric_a.yaml`: Direct zero-shot prompt instructing the model to score each axis on a 0.0 to 1.0 continuous scale.
   - `rubric_b.yaml`: Chain-of-Thought (CoT) instructions requiring step-by-step reasoning per axis, explicit rubric anchors (Poor: 0.0, Acceptable: 0.5, Flawless: 1.0), and concrete exemplar demonstrations.

5. **Local Fast Scorer Engine (`evaluation/local_scorer.py`)**:
   - Ported and enhanced from CIRCLE's `LightweightStudentEvaluator` (`distilled_eval.py`).
   - Latency target: Sub-millisecond execution (~0.013ms per probe).
   - Metrics computed:
     - Lexical diversity via Type-Token Ratio (TTR) and Hapax Legomena ratio (fluency proxy).
     - Repetition penalties via n-gram frequency analysis (2-gram, 3-gram, 4-gram degradation).
     - Lexical overlap / Jaccard similarity against reference context and gold answer.
     - Structural constraint verification (length bounds, formatting, negative constraint matching).

6. **Teacher Judge Engine (`evaluation/teacher_judge.py`)**:
   - Asynchronous and batch evaluator querying Groq API (e.g. `llama-3.3-70b-versatile` / `llama-3-70b-8192`) with robust offline mock fallback when API keys are not present.
   - Strict Pydantic v2 validation enforcing schema output:
     - Numerical scores `[0.0, 1.0]` for all 4 axes.
     - Reasoning string explaining the rating.
     - Error category flags.
   - Automatic re-querying loop (up to 3 retries) with error feedback if the model emits invalid JSON or out-of-bound scores.
   - Output persistence to `data/judge_scores/rubric_a.json` and `data/judge_scores/rubric_b.json`.

7. **Phase 1 Validation & Unit Tests (`tests/test_phase1.py`)**:
   - Verify probe data loading and integrity (200 records, complete schemas).
   - Test human annotation schema validation and consensus calculations.
   - Test local evaluator speed, consistency, and boundary cases.
   - Test teacher judge schema enforcement, retry mechanics, and mock execution.

---

## Phase 2: Statistical Alignment Analysis, Failure Mode Deep-Dive, Visualization & RLAIF Decision System

### Goal
Empirically measure judge-human correlation, analyze discrepancy patterns, benchmark Rubric A vs Rubric B, test local pre-filtering viability, and provide actionable decision boundaries for financial post-training workflows.

### Components & Technical Requirements

1. **Statistical Agreement & Correlation Engine (`analysis/agreement_stats.py`)**:
   - **Inter-Annotator Agreement (Ceiling)**: Calculate Fleiss's Kappa ($\kappa$) across the 3 human annotators to establish the maximum expected human consistency ceiling.
   - **Judge-Human Alignment**: Compute Quadratic Weighted Cohen's Kappa ($\kappa$) and Unweighted Cohen's Kappa between consensus human ratings and Teacher Judge scores for both Rubric A and Rubric B across each axis.
   - **Local vs Teacher Correlation**: Compute Pearson correlation ($r$) and Spearman rank correlation ($\rho$) comparing `LightweightStudentEvaluator` metrics against Teacher Judge scores per axis.
   - Generate summary metrics tables mirroring the research study findings.

2. **Failure Mode Breakdown & Error Taxonomy (`analysis/failure_breakdown.py`)**:
   - Identify all evaluation samples where $|Score_{judge} - Score_{human}| > 0.30$.
   - Automatically classify disagreements into four key failure categories:
     - **Judge False-Positive**: Judge scored high ($\ge 0.7$), Human scored low ($< 0.5$).
     - **Judge False-Negative**: Judge scored low ($< 0.5$), Human scored high ($\ge 0.7$).
     - **Rubric Ambiguity**: High human variance ($std > 0.6$) or conflicting edge-cases.
     - **Genuine Model Failure**: Small LLM generated degenerate or subtly misleading text that fooled one evaluator.
   - Quantify relative percentages and root-cause drivers to direct rubric optimization.

3. **Visualization Suite (`analysis/visualize.py`)**:
   - Correlation heatmaps across metrics and axes.
   - Scatter plots with regression lines comparing Local Scorer vs Teacher Judge and Teacher Judge vs Human Annotators.
   - Bar charts comparing Rubric A vs Rubric B Cohen's $\kappa$ performance.
   - Failure mode distribution charts (pie/bar) highlighting the 25% rubric ambiguity threshold.
   - Pre-filter tradeoff curves (ROC/Precision-Recall) illustrating cost savings vs escalation safety for financial QA.

4. **End-to-End Orchestrator & CLI Runner (`run_pipeline.py`)**:
   - Single command execution to reproduce the entire experiment from raw data to full statistical report and charts.
   - Parameterized switches for offline mode, API key ingestion, and rubric selection.

5. **Phase 2 Validation & Verification (`tests/test_phase2.py`)**:
   - Test statistical mathematical formulations against known ground truths (`scipy.stats`).
   - Validate failure classification algorithms.
   - End-to-end integration test verifying complete pipeline execution.
