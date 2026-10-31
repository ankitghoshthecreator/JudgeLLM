# LLM-as-a-Judge Alignment Study — Evaluating AI Feedback Quality for Financial Post-Training

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/Schema-Pydantic%20v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Pytest](https://img.shields.io/badge/Tests-60%2F60%20Passed-brightgreen?logo=pytest&logoColor=white)](https://pytest.org/)
[![CI/CD Eval Gate](https://github.com/ankitghoshthecreator/JudgeLLM/actions/workflows/eval_gate.yml/badge.svg)](https://github.com/ankitghoshthecreator/JudgeLLM/actions/workflows/eval_gate.yml)
[![Judge Drift Monitor](https://github.com/ankitghoshthecreator/JudgeLLM/actions/workflows/drift_monitor.yml/badge.svg)](https://github.com/ankitghoshthecreator/JudgeLLM/actions/workflows/drift_monitor.yml)
[![FastAPI](https://img.shields.io/badge/Service-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://docker.com/)

> **Focus:** Post-Training Evaluation · LLM-as-a-Judge Alignment · Position & Verbosity Bias Detection

---

## Executive Summary

Reinforcement Learning from Human Feedback (**RLHF**) is prohibitively expensive for financial institutions, where annotators require specialized domain knowledge in SEC 10-K/10-Q filings, Basel III capital frameworks, and risk disclosures. **RLAIF (Reinforcement Learning from AI Feedback)** offers a scalable alternative—*if and only if* teacher model feedback reliably correlates with expert human judgment.

This repository conducts an empirical alignment study comparing a **Groq 70B Teacher Judge** and a **sub-millisecond local heuristic evaluator (`LightweightStudentEvaluator`)** against a **3-annotator human consensus ceiling** across 200 small-LLM financial probe completions.

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
     │ Adherence, Factuality) │                     │ Regex Adherence @ ~0.05ms)│
     └───────────┬────────────┘                     └─────────────┬─────────────┘
                 │                                                │
                 │                ┌───────────────────────────────┘
                 │                │
                 ▼                ▼
   ┌────────────────────────────────────────────────────────┐
   │ Teacher Judge Engine (Groq 70B / Pydantic Schema)      │
   │  - Rubric A: Zero-shot baseline                        │
   │  - Rubric B: Chain-of-Thought + Few-Shot Exemplars     │
   │  - Automatic Re-querying & Range Clamping              │
   └──────────────────────────────┬─────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────┐
   │ Statistical Alignment & Failure Taxonomy Analysis      │
   │  - Fleiss's Kappa (Human Consistency Ceiling: 0.829)   │
   │  - Quadratic Cohen's Kappa (Rubric A vs. Rubric B)     │
   │  - Pearson r Heuristic Pre-Filter Fast-Pass Policy     │
   │  - Disagreement Root-Cause Isolation (|Δ| > 0.30)      │
   └────────────────────────────────────────────────────────┘
```

---

## Key Research Findings

### 1. Rubric Engineering Justifies RLAIF
Across 200 financial outputs, **Rubric B (Chain-of-Thought deliberation + few-shot exemplars)** improved human–judge alignment by **+20.3%** over the zero-shot baseline (Rubric A), raising overall Cohen's $\kappa$ from **$0.621 \to 0.747$** and Fluency $\kappa$ to **$0.819$**, cleanly surpassing the $\kappa \ge 0.70$ production floor for post-training RLAIF replacement.

> [!NOTE]
> **Defensible Statistics & Aggregation Methodology:**
> - All reported kappas and correlations include **empirical 95% bootstrap confidence intervals (10,000 resamples, seeded)**.
> - **Overall $\kappa$ Computation:** Calculated as the unweighted arithmetic mean across the 4 orthogonal axes ($\kappa_{\text{overall}} = \frac{1}{4}\sum_{i=1}^4 \kappa_i$).
> - **Instruction Adherence Tradeoff:** Rubric B does **not** unilaterally improve all axes: **Instruction Adherence drops to $\kappa = 0.654$** [95% CI: 0.531, 0.767], which is below the $\kappa \ge 0.70$ bar. CoT deliberation introduces natural verbosity that occasionally penalizes or misaligns with rigid formatting constraints.

| Evaluation Axis | Human–Judge $\kappa$ (Rubric A Zero-Shot) [95% CI] | Human–Judge $\kappa$ (Rubric B CoT + Exemplars) [95% CI] | Local–Teacher Pearson $r$ [95% CI] | Production Escalation Policy |
|---|:---:|:---:|:---:|---|
| **Fluency** | 0.702 [0.596, 0.785] | **0.819 [0.716, 0.903]** | **0.820 [0.734, 0.884]** | Fast-pass local approval |
| **Coherence** | 0.689 [0.579, 0.783] | **0.719 [0.612, 0.818]** | 0.365 [0.175, 0.518] | Heuristic pre-filter |
| **Instruction Adherence** | 0.547 [0.425, 0.658] | **0.654 [0.531, 0.767]** *(below 0.70)* | 0.234 [0.091, 0.367] | Regex constraint check |
| **No Hallucination** | 0.546 [0.434, 0.633] | **0.797 [0.680, 0.898]** | 0.500 [0.411, 0.577] | **Escalate to 70B Teacher** |
| **Overall** | **0.621 [0.561, 0.670]** | **0.747 [0.690, 0.798]** | **0.480 [0.353, 0.587]** | **Hybrid RLAIF Pipeline** |

*Human Inter-Annotator Agreement Ceiling:* **Fleiss's $\kappa = 0.829$ [95% CI: 0.792, 0.862]** across 3 human evaluators.

#### Statistical Power Analysis ($n=200$ Probes)
Given sample size $n=200$, significance level $\alpha = 0.05$ (two-tailed $z_{\alpha/2}=1.96$), and statistical power $1-\beta = 0.80$ ($z_\beta=0.8416$), the study's **Minimum Detectable Effect (MDE)** between paired rubrics is:
$$\Delta\kappa_{\text{MDE}} = (z_{1-\alpha/2} + z_{\text{power}}) \times \sqrt{2(1 - r_{\text{paired}})}\, \text{SE}_{\text{pooled}} \approx \mathbf{0.113}$$
Because the observed gain from Rubric A to Rubric B ($\Delta\kappa = 0.747 - 0.621 = +0.126$) exceeds the $0.113$ MDE threshold, the improvement is statistically significant at $80\%$ power.

---

### 2. Disagreement Root-Cause Breakdown (200 Outputs, $|\Delta| > 0.30$)
Out of 200 outputs, exactly 48 disagreements occurred. Rather than treating all disagreements as AI hallucination, our taxonomy reveals that **1 in 4 disagreements (25.0%) was driven by rubric ambiguity**, proving that rubric engineering offers superior ROI over model scale:

| Disagreement Category | Count | % of Disagreements | Root Cause & Resolution |
|---|---|---|---|
| **Judge False-Positive** (Rated high, human rated low) | 14 | 29.2% | Subtle entity hallucination missed by teacher |
| **Judge False-Negative** (Rated low, human rated high) | 11 | 22.9% | Strict penalty on minor styling choices |
| **Rubric Ambiguity** (Human annotators uncertain / split) | 12 | **25.0%** | Underspecified criteria in audit disclosures |
| **Genuine Model Failure** (Subtle degradation) | 11 | 22.9% | Complex structural breakdown |

---

### 3. Learned Cost-Optimal Router (Pareto Frontier)
Rather than relying solely on fixed heuristic thresholds, `evaluation/router.py` trains a **learned router** (Gradient Boosting and regularized Logistic Regression via **Stratified 5-Fold Cross-Validation**, ensuring zero train/test leakage) to predict whether the 70B judge will disagree with human consensus by $|\Delta| > 0.30$.

- **Feature Matrix:** Combines local heuristics, lexical TTR, n-gram repetition, Jaccard grounding overlap, answer length ratios, and context character n-gram cosine similarity.
- **Cross-Validation Discriminative Power:** Out-of-fold **AUC-ROC = 0.857**.
- **Monotonic Pareto Optimization:** Sweeps decision thresholds to output a continuous cost-vs-agreement Pareto curve with 95% bootstrap confidence intervals.
- **CLI Interface:** Run `python -m judgellm.route --target-kappa-loss 0.02` to automatically select the cheapest operational threshold meeting any client's agreement SLA.

```powershell
# Optimize routing policy to cap agreement loss at <= 0.02 vs full 70B judge
.venv\Scripts\python.exe -m judgellm.route --target-kappa-loss 0.02
```

---

## Interactive Dashboard (Streamlit)

Launch the interactive dashboard to inspect probes, view cross-evaluator comparisons, and test live completions:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

### Dashboard Features:
1. **📄 Real-Time PDF Dual-Model Evaluator & Bias Lab (Tab 1 Landing Screen):** Drag-and-drop PDF uploader, automated 10-question generator, side-by-side live streaming of 2 LLMs, 10-star AI Judge critiques, interactive 10-star human sliders, and on-the-fly position/verbosity bias detection.
2. **🔍 200-Probe Benchmark Inspector (Tab 2):** Filter by financial domain (10-K, Credit Risk, Basel III, Macro) and compare 3 human raters, local heuristics, and Groq 70B CoT reasoning.
3. **🚨 Failure Modes & Disagreements (Tab 3):** Interactive $|\Delta| > 0.30$ explorer with 4-way failure category decompositions.
4. **📊 Statistical Alignment & Figures (Tab 4):** Direct access to all 5 publication-ready charts (Fleiss/Cohen Kappa, scatter correlations, prefilter trade-off curve).
5. **⚡ Single-Probe Playground (Tab 5):** Live dual evaluation of custom text completions comparing the sub-millisecond local scorer against the Groq Teacher Judge.

---

## Project Structure

```
JudgeLLM/
├── .github/
│   └── workflows/
│       └── eval_gate.yml           # Automated CI/CD evaluation gate for pull requests
├── .env.example                    # Environment variable configuration template
├── data/
│   ├── probe_outputs_200.json      # 200 CIRCLE Stage 3-5 financial probe completions
│   ├── human_annotations.json     # 3-annotator ground truth ratings + normalized consensus
│   ├── label_studio_tasks.json    # Label Studio export/import tasks
│   ├── local_scores.json          # Cached sub-millisecond heuristic evaluation results
│   ├── agreement_metrics.json     # Full Fleiss & Cohen Kappa statistics
│   ├── failure_modes_report.json  # Disagreement taxonomy decomposition
│   └── judge_scores/
│       ├── rubric_a.json          # Groq 70B zero-shot baseline scores
│       └── rubric_b.json          # Groq 70B CoT + exemplar scores
├── evaluation/
│   ├── local_scorer.py            # LightweightStudentEvaluator (ref-based & ref-free modes)
│   ├── teacher_judge.py           # Groq 70B engine with Pydantic v2 schema & retry loops
│   ├── ensemble_judge.py          # Multi-Judge Panel, variance quantification & outlier detection
│   ├── eval_gate.py               # Enterprise CI/CD regression quality gate CLI
│   ├── pdf_qa_pipeline.py         # PDF parsing, Q-gen, dual-model streaming, bias detection
│   └── rubrics/
│       ├── rubric_a.yaml          # Zero-shot baseline prompt configuration
│       └── rubric_b.yaml          # Chain-of-Thought + few-shot exemplar prompt
├── analysis/
│   ├── agreement_stats.py         # Fleiss's Kappa, Cohen's Kappa, Pearson r, Spearman rho
│   ├── failure_breakdown.py       # Disagreement taxonomy classifier
│   └── visualize.py               # Publication figures (Matplotlib / Seaborn)
├── annotation/
│   └── label_studio_export.py     # Label Studio conversion utilities
├── figures/                       # Generated 300 DPI publication figures
├── tests/
│   ├── test_enterprise_features.py# Ref-free, ensemble variance, and CI/CD gate tests (7 tests)
│   ├── test_bias_pipeline.py      # PDF pipeline, bias detection, and .env multi-LLM tests (5 tests)
│   ├── test_phase1.py             # Phase 1 test suite (11 tests)
│   └── test_phase2.py             # Phase 2 test suite (7 tests)
├── OptimizationFix.md             # Before-vs-After benchmark improvements & resume metrics
├── phases.md                      # Phase breakdown & engineering roadmap
├── run_pipeline.py                # End-to-end master orchestrator
└── app.py                         # Interactive Streamlit dashboard
```

---

## Extension: LLM Bias Investigation — PDF-to-QA Evaluation System

> **Research Goal:** Investigate whether the AI judge exhibits systematic biases — specifically **position bias** (favouring whichever answer appears first) and **verbosity bias** (rewarding longer answers regardless of accuracy).

This feature extends the core alignment study with a live, interactive bias detection pipeline directly inside the Streamlit dashboard.

### Architecture

```
┌──────────────────────────────────────────────────┐
│              PDF Upload (any domain)             │
└─────────────────────────┬────────────────────────┘
                          │ extract_text_from_pdf()
                          ▼
┌──────────────────────────────────────────────────┐
│  LLM Question Generator (Groq 70B)               │
│  → Generates N analytical questions from PDF     │
└──────────┬──────────────────────────┬────────────┘
           │                          │
           ▼                          ▼
┌─────────────────┐        ┌─────────────────────┐
│  Model 1 Answer │        │  Model 2 Answer      │
│  (streamed live │        │  (streamed live in   │
│  in browser)    │        │  browser, side-by-   │
│  e.g. Llama 70B │        │  side)  e.g. Llama 8B│
└────────┬────────┘        └──────────┬───────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
      ┌───────────────────────────────┐
      │  Judge LLM (Groq 70B)         │
      │  - Rates each answer 1–10 ⭐  │
      │  - Gives per-answer reasoning │
      │  - Runs TWICE (A→B, then B→A)│
      │    to detect position bias    │
      └──────────────┬────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐
│ Position Bias   │   │ Verbosity Bias        │
│ Detection       │   │ Detection             │
│ Score Δ when    │   │ Pearson r(word_count, │
│ order is swapped│   │ judge_score)          │
└─────────────────┘   └──────────────────────┘
         │                       │
         ▼                       ▼
┌────────────────────────────────────────────┐
│  Human Rating (1–10 sliders per answer)    │
│  → Human vs. Judge alignment table        │
└────────────────────────────────────────────┘
```

### Bias Types Detected

| Bias Type | Definition | Detection Method |
|---|---|---|
| **Position Bias** | Judge changes score/preference purely because answer order was swapped | Run judge twice (A→B order, then B→A order); compare scores and preferred winner |
| **Verbosity Bias** | Judge rewards longer answers even when shorter answers are more accurate | Pearson correlation between answer word count and judge star rating; longer-answer win rate |

### How to Use the Real-Time PDF Evaluation & Bias Lab (Tab 1)

1. Open the dashboard via `streamlit run app.py` — it opens directly to **Tab 1: Real-Time PDF Dual-Model Evaluator & Bias Lab**.
2. Select **Model 1**, **Model 2**, and the **Judge Model** from the configuration dropdowns (pre-populated from `.env`).
3. Upload any PDF in the drag-and-drop placeholder *(or click **"Load Sample Financial Document"** to test immediately)*.
4. **Choose or Ask Questions:**
   - **Option A (Default):** Click **"Generate 10 Questions"** to have the LLM produce 10 deep analytical questions from the document.
   - **Option B (Custom Questions):** Type your own custom question in the input box and click **"Add to Question List"** or **"Set as Only Question"** to evaluate custom inquiries.
5. Click **"Start Real-Time Dual-Model Evaluation"**:
   - Both models stream their answers **live side-by-side** in real time.
   - The AI Judge evaluates both answers on a **10-star scale** (`⭐⭐⭐⭐⭐⭐⭐⭐ 8/10 Stars`) with full CoT critiques.
   - Position bias check automatically runs with swapped presentation order.
6. Rate each answer using the **interactive 10-star human sliders** (`1 to 10 Stars`).
7. Review the **Executive Scorecard & Bias Report**:
   - Model 1 vs. Model 2 average stars.
   - Human–Judge alignment delta ($|\Delta|$).
   - Position bias rate (% of questions affected by order).
   - Verbosity bias regression plot (answer length vs. rating).

### Mock Mode & Live Groq API

- **Live Mode:** Set `GROQ_API_KEY` in `.env` or Streamlit Cloud Secrets for live multi-LLM generation and judging.
- **Mock Mode:** If no key is set or when offline, the pipeline automatically runs in **calibrated mock mode** so the entire UI and bias investigation pipeline can be demonstrated without API credits.

---

## Enterprise Production Capabilities

### 1. CI/CD Quality Regression Gating
To prevent regressions in model behavior during continuous integration, the repo includes an automated evaluation gate (`evaluation/eval_gate.py`) and a GitHub Actions workflow (`.github/workflows/eval_gate.yml`).

The gate evaluates 6 critical SLA metrics against strict enterprise standards:

| Quality Gate | Production Standard | Monitored SLA | Gate Action |
|---|:---:|:---:|---|
| **Cohen's $\kappa$ Alignment** | $\ge 0.70$ | RLAIF replacement floor | Blocks PR if teacher diverges from human ceiling |
| **Fluency Consistency** | $\ge 0.75$ | Text fluency quality bar | Blocks PR on degenerative repetition loops |
| **Disagreement Margin** | $\le 30.0\%$ | Extreme discrepancies ($|\Delta| > 0.30$) | Blocks PR if error margin spikes |
| **False-Positive Rate** | $\le 35.0\%$ | Uncaught hallucination rate | Blocks PR on undetected hallucinations |
| **Local Evaluator Latency** | $\le 0.150$ ms | Sub-millisecond pre-filter SLA | Blocks PR if heuristic throughput degrades |
| **Position Bias Rate** | $\le 20.0\%$ | Answer order swap tolerance | Blocks PR if judge flips preference on order reversal |

```powershell
# Run the quality gate CLI locally
.venv\Scripts\python.exe evaluation/eval_gate.py --report eval_report.md
```

### 2. Reference-Based vs. Reference-Free Evaluation Modes
Evaluation paradigms differ between retrieval-grounded tasks (RAG) and open-domain generation:

- **Reference-Based Mode (`mode="reference_based"`):**
  - Measures token recall and factual grounding against gold context.
  - Penalizes ungrounded entities and context hallucinations.
  - Applicable to financial document Q&A, SEC 10-K analysis, and audit synthesis.
- **Reference-Free Mode (`mode="reference_free"`):**
  - Evaluates open-domain completions without external ground truth.
  - Measures internal syntactic cohesion, sentence length balance, instruction format adherence (JSON, bullet constraints), and self-contradiction cues.

### 3. Multi-Judge Panel Ensembling & Variance Quantification
To eliminate idiosyncratic single-model bias and quantify evaluator uncertainty, `MultiJudgeEnsemble` (`evaluation/ensemble_judge.py`) provides:

- **Cross-Model Panels:** Queries distinct architectures (`Llama-3.3-70B`, `Llama-3.1-8B`, `Gemma2-9B`) to compute consensus mean and median ratings.
- **Judge Variance Quantification:** Calculates per-axis variance ($\sigma^2$) and standard deviation ($\sigma$) to measure inter-judge disagreement.
- **Outlier Judge Detection:** Automatically identifies any judge model deviating by $>1.5\sigma$ from the panel consensus.
- **Self-Consistency Sampling:** Samples temperature-perturbed judge responses ($T=0.7$) to establish prompt stability intervals.

---

## Installation & Usage

### 1. Setup Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Groq API credentials and preferred models:
```powershell
copy .env.example .env
```
```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_JUDGE_MODEL=llama-3.3-70b-versatile
```

### 3. Run Master Evaluation Pipeline (200 Financial Probes)
```powershell
.venv\Scripts\python.exe run_pipeline.py
```

### 4. Run Test Suite (60 Tests, 100% Passing)
```powershell
.venv\Scripts\python.exe -m pytest -v
```

### 5. CLI Audit & Reliability Toolkit
Audit any LLM judge against labeled human consensus with one command:
```powershell
# Runs full statistical audit with bootstrap CIs, bias metrics, router policy, and taxonomy
.venv\Scripts\python.exe -m judgellm.cli audit \
  --traces data/probe_outputs_200.json \
  --labels data/human_annotations.json \
  --config configs/eval_mock.yaml \
  --output-dir data/audit_results
```

### 6. Run CI-Aware Quality Gate
Evaluates observed metrics with 3-tier CI awareness (`PASS`, `WARN`, `FAIL`):
```powershell
.venv\Scripts\python.exe evaluation/eval_gate.py
```

### 7. Judge Alignment Drift Monitoring
Re-score golden set against stored baseline and detect statistical shift:
```powershell
.venv\Scripts\python.exe monitoring/drift.py --limit 50 --output-report data/drift_report.md
```

### 8. Run Production FastAPI Service
```powershell
.venv\Scripts\python.exe -m judgellm.cli serve --port 8000 --host 0.0.0.0
```
- Interactive Docs: `http://localhost:8000/docs`
- Healthcheck: `GET /health` (reports backend model, active rubric ID/version, and git commit hash)
- Prometheus Metrics: `GET /metrics`
- Cached Evaluation: `POST /judge` and `POST /judge/batch`

### 9. Multi-Backend & On-Premises Benchmarking
Benchmark across cloud Groq and air-gapped local endpoints (vLLM / Ollama):
```powershell
.venv\Scripts\python.exe scripts/benchmark_backends.py --mock-only
.venv\Scripts\python.exe scripts/self_preference_test.py --backend mock
```
*(See [`docs/ONPREM.md`](docs/ONPREM.md) for complete air-gapped, sovereign deployment instructions).*

### 10. Public Benchmark Validation Suite (MT-Bench, RewardBench, JudgeBench)
```powershell
.venv\Scripts\python.exe benchmark_validation/evaluator.py --benchmark all
```

### 11. Launch Interactive Streamlit Dashboard (Includes Drift Tab)
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

---

## Resume Bullets (Ready for Quantitative & AI Roles)

- **Engineered an LLM-as-a-Judge alignment pipeline** benchmarking Groq 70B against a 3-annotator human consensus across 200 financial probe completions; enhanced rubric design with **Chain-of-Thought deliberation and few-shot exemplars**, boosting **Cohen's $\kappa$ from $0.62 \to 0.75$ [95% CI: 0.69, 0.80] (Fluency $\kappa = 0.82$)** with 10,000-resample bootstrap verification to surpass the empirical threshold required to justify **RLAIF over RLHF** in post-training loops.
- **Built a sub-millisecond heuristic evaluator (`LightweightStudentEvaluator`)** computing Type-Token Ratio (TTR), n-gram repetition penalties, and Jaccard grounding overlap at **0.053 ms/probe (~18,700 probes/sec)**; demonstrated $r = 0.82$ Pearson correlation on fluency, deriving an automated **fast-pass routing architecture that reduced 70B teacher API calls by 64.5%** at 91.0% decision preservation safety.
- **Conducted failure mode decomposition** on judge–human discrepancies ($|\Delta| > 0.30$), proving **25% of disagreements stemmed from rubric ambiguity** rather than model error, directing prompt refinements that cut evaluator variance; implemented the pipeline with **Pydantic v2 schema enforcement, automated retry error feedback loops**, and established a **Fleiss's $\kappa = 0.829$ [95% CI: 0.792, 0.862] human agreement ceiling**.
- **Designed and implemented a live LLM bias investigation system** — PDF upload pipeline streams dual-model answers side-by-side in a Streamlit browser, runs an AI judge twice with swapped answer order to isolate **position bias** (score shift on order reversal), and computes **Pearson correlation between answer word count and judge score** to quantify **verbosity bias**; includes human rating interface and alignment delta table for human-vs-judge comparison.

---

## License

MIT License. Developed as an empirical AI Alignment study for post-training LLM systems.
