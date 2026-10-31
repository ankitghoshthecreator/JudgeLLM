# Benchmark Validation: Public Human-Judgment Datasets

This module evaluates **JudgeLLM** against standard, peer-reviewed public datasets containing ground-truth human judgments.

---

## 1. Supported Public Benchmarks

| Benchmark | Reference / Source | Evaluation Domain | Published Baseline Agreement |
|---|---|---|---|
| **MT-Bench** | Zheng et al., NeurIPS 2023 ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)) | Multi-turn reasoning, math, coding, professional writing | **83.2%** pairwise agreement with expert human judgment |
| **RewardBench** | Lambert et al., AI2 2024 ([arXiv:2403.13787](https://arxiv.org/abs/2403.13787)) | Factuality, safety refusals, subtle reasoning & hallucination | **81.4%** accuracy for frontier LLM judges / reward models |
| **JudgeBench** | Tan et al., 2024 ([arXiv:2410.12784](https://arxiv.org/abs/2410.12784)) | Objective mathematical proofs, logic fallacy identification | **72.3%** agreement on challenging reasoning tasks |

---

## 2. Running the Validation Suite

To run all public benchmarks against the JudgeLLM evaluation pipeline:

```powershell
.venv\Scripts\python.exe benchmark_validation/evaluator.py --benchmark all
```

To run a single benchmark:

```powershell
.venv\Scripts\python.exe benchmark_validation/evaluator.py --benchmark mt_bench
.venv\Scripts\python.exe benchmark_validation/evaluator.py --benchmark reward_bench
.venv\Scripts\python.exe benchmark_validation/evaluator.py --benchmark judge_bench
```

Results and detailed per-sample audit trails are saved as JSON artifacts in `data/benchmarks/`:
- `data/benchmarks/mt_bench_results.json`
- `data/benchmarks/reward_bench_results.json`
- `data/benchmarks/judge_bench_results.json`
- `data/benchmarks/all_benchmarks_summary.json`

---

## 3. Strict Scientific Rules

1. **Zero Fabrication:** If external network access is available, evaluations query live Groq judges. When offline, evaluations execute using deterministic heuristic evaluation pipelines without falsifying outputs.
2. **Explicit Baseline Citations:** Every published literature baseline includes paper authors, conference/venue, and exact citation notes in the source code comments.
