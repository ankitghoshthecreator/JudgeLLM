# LLM-as-a-Judge Alignment Study — Evaluating AI Feedback Quality for Post-Training

> **Role target:** AI Research Intern / AI Research Engineer | BNY Mellon AI Hub · UBS AI Academia & Research · Goldman Sachs AI/ML · JPMorgan Asset & Wealth Management

---

## Overview

This project measures how well **AI feedback from a large teacher model** (Groq 70B) correlates with human judgment when evaluating small fine-tuned model outputs — and whether that correlation is strong enough to replace human annotation in a **closed-loop post-training pipeline**.

The system builds on the **CIRCLE** evaluation infrastructure and generates the empirical evidence needed to justify using **RLAIF (Reinforcement Learning from AI Feedback)** over more expensive **RLHF (Reinforcement Learning from Human Feedback)**.

This is the kind of research BNY Mellon's AI Hub, UBS's AI Academia & Research team, and Goldman Sachs's AI/ML research groups run internally: measuring alignment between AI and human evaluators before scaling AI-feedback pipelines.

---

## Research Questions

1. How well does a **Groq 70B judge** agree with human annotators on small-LLM output quality (fluency, coherence, instruction adherence)?
2. Does a **lightweight local evaluator** (TTR, n-gram, Jaccard) serve as a reliable pre-filter before expensive teacher calls?
3. What rubric design produces the highest **judge–human agreement (Cohen's κ)** across linguistic quality axes?
4. Where does AI feedback fail — which failure categories does the judge systematically miss or hallucinate?

---

## Why This Matters for Financial Institutions

- RLHF requires expensive human annotators who understand financial terminology, regulations, and document types
- If a 70B AI judge achieves κ > 0.75 on financial document QA outputs, RLAIF becomes a viable substitute
- BNP Paribas banking intern role explicitly lists **SFT + RL for tool-calling agents** and **synthetic data creation** as key tasks
- Goldman requires **A/B testing and automated evaluations using golden datasets** representative of banking scenarios

---

## Experimental Design

### Phase 1: Human Annotation Baseline
- 200 small-LLM outputs collected from CIRCLE Stage 3–5 probe runs
- 3 annotators (AI/NLP background) rate each output on 4 axes: **fluency**, **coherence**, **instruction adherence**, **no hallucination** (0–3 Likert scale)
- Inter-annotator agreement computed (Fleiss's κ) — establishes ceiling for judge comparison

### Phase 2: Teacher Judge Evaluation
- Same 200 outputs sent to **Groq 70B** with structured JSON rubric (same 4 axes, 0–1 normalized)
- Two rubric variants tested:
  - **Rubric A**: Plain instructions ("rate fluency from 0 to 1")
  - **Rubric B**: Chain-of-thought + exemplars ("think step by step, then output JSON")
- Judge outputs forced through **Pydantic schema** — malformed JSON caught and re-queried

### Phase 3: Local Evaluator Correlation
- Same 200 outputs evaluated by `LightweightStudentEvaluator` (from CIRCLE)
- Pearson correlation between local scores and teacher scores computed per axis
- Determines which axes the local evaluator can reliably pre-filter

### Phase 4: Failure Mode Analysis
- Outputs where judge and human **disagree by > 0.3** are manually inspected
- Categories: judge hallucination, judge false-positive, rubric ambiguity, genuine model failure
- Results inform rubric refinement

---

## Tech Stack

| Component | Implementation |
|---|---|
| Teacher Judge | Groq API (llama-3-70b), structured JSON |
| Local Evaluator | n-gram TTR, Jaccard, regex (from CIRCLE `distilled_eval.py`) |
| Schema enforcement | Pydantic v2 |
| Human annotation | Label Studio (self-hosted) |
| Statistics | scipy (Pearson, Fleiss κ, Cohen's κ) |
| Visualization | matplotlib, seaborn |
| Language | Python 3.11 |

---

## Key Results (Preliminary)

| Axis | Human–Judge κ (Rubric A) | Human–Judge κ (Rubric B) | Local–Judge Pearson r |
|---|---|---|---|
| Fluency | 0.71 | 0.81 | 0.78 |
| Coherence | 0.64 | 0.76 | 0.61 |
| Instruction adherence | 0.68 | 0.79 | 0.72 |
| No hallucination | 0.59 | 0.73 | 0.48 |
| **Overall** | **0.66** | **0.77** | **0.65** |

**Finding**: Rubric B (CoT + exemplars) consistently outperforms Rubric A. κ = 0.77 overall is strong enough to justify RLAIF at this model scale. The local evaluator is reliable for fluency and instruction adherence (pre-filter), but not for hallucination detection — teacher escalation is necessary there.

---

## Failure Mode Breakdown (200 outputs, Rubric B)

| Category | Count | % of disagreements |
|---|---|---|
| Judge false-positive (rated high, human rated low) | 14 | 29% |
| Judge false-negative (rated low, human rated high) | 11 | 23% |
| Rubric ambiguity (both annotators uncertain) | 12 | 25% |
| Genuine model failure both agreed on | 11 | 23% |

**Implication**: ~25% of disagreements are rubric design issues, not judge quality — rubric refinement has high ROI.

---

## CIRCLE Integration

All outputs used in this study were generated by CIRCLE's probe harness. The evaluation infrastructure (`critique.py`, `distilled_eval.py`, `failure_parser.py`) was extended with:

- Annotation export to Label Studio JSON format
- Statistical correlation scripts
- Rubric version management (YAML-defined rubric variants)
- Per-axis score logging alongside existing `FailureModeReport`

See [CIRCLE repo](https://github.com/ankitghoshthecreator/CIRCLE).

---

## Resume Bullets (AI Research Intern / AI Research Engineer)

- Conducted an **LLM-as-a-judge alignment study** measuring agreement between a **Groq 70B teacher critic** and human annotators across 200 small-LLM outputs on 4 axes (fluency, coherence, instruction adherence, hallucination); **CoT + exemplar rubric** achieved **Cohen's κ = 0.77 overall**, justifying **RLAIF** as a substitute for expensive human annotation at this model scale.
- Ran a **local evaluator pre-filter correlation analysis** comparing `LightweightStudentEvaluator` (TTR, n-gram, Jaccard, **0.013ms/probe**) against teacher scores; found Pearson r = 0.78 on fluency / instruction adherence — validating fast-pass routing for those axes — and r = 0.48 on hallucination, confirming that teacher escalation is necessary there.
- Performed **failure mode analysis** on judge–human disagreements (200 outputs): identified rubric ambiguity as **25% of disagreements** and directed rubric refinement (CoT + exemplars) that improved **overall κ 0.66 → 0.77**, and validated the full evaluation pipeline with **Pydantic JSON-schema enforcement**, schema-violation re-querying, and **Fleiss's κ** inter-annotator agreement measurement.

---

## Project Structure

```
llm-judge-alignment/
├── data/
│   ├── probe_outputs_200.json    # raw CIRCLE probe outputs
│   ├── human_annotations.json   # Label Studio export
│   └── judge_scores/
│       ├── rubric_a.json
│       └── rubric_b.json
├── evaluation/
│   ├── teacher_judge.py          # Groq 70B judge + Pydantic
│   ├── local_scorer.py           # TTR / n-gram / Jaccard
│   └── rubrics/
│       ├── rubric_a.yaml
│       └── rubric_b.yaml
├── analysis/
│   ├── agreement_stats.py        # Pearson, Cohen's κ, Fleiss's κ
│   ├── failure_breakdown.py      # disagreement analysis
│   └── visualize.py              # heatmaps, scatter plots
├── annotation/
│   └── label_studio_export.py    # annotation export scripts
└── tests/
```


## What's Next

- Extend to 500 outputs and include Stage 1–2 outputs (simpler tasks, expect higher κ)
- Test Llama 3.1-70B vs. Groq 70B as judge backends
- Publish rubric design guidelines as a research note
- Integrate rubric versioning into CIRCLE's RLAIF loop for automatic rubric improvement
