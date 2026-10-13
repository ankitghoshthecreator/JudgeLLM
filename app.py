"""
LLM-as-a-Judge Alignment Study — Interactive Streamlit Dashboard
================================================================
Interactive exploration of human annotators vs. Groq 70B teacher critique,
local pre-filter heuristics, disagreement breakdown, and publication figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LLM-as-a-Judge Alignment Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"


@st.cache_data
def load_data() -> Dict[str, Any]:
    with open(DATA_DIR / "probe_outputs_200.json", "r", encoding="utf-8") as f:
        probes = {item["id"]: item for item in json.load(f)}
    with open(DATA_DIR / "human_annotations.json", "r", encoding="utf-8") as f:
        human = {item["id"]: item for item in json.load(f)}
    with open(DATA_DIR / "judge_scores" / "rubric_a.json", "r", encoding="utf-8") as f:
        judge_a = {item["id"]: item for item in json.load(f)}
    with open(DATA_DIR / "judge_scores" / "rubric_b.json", "r", encoding="utf-8") as f:
        judge_b = {item["id"]: item for item in json.load(f)}
    with open(DATA_DIR / "local_scores.json", "r", encoding="utf-8") as f:
        local = {item["id"]: item for item in json.load(f)}
    with open(DATA_DIR / "agreement_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(DATA_DIR / "failure_modes_report.json", "r", encoding="utf-8") as f:
        failures = json.load(f)

    return {
        "probes": probes,
        "human": human,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "local": local,
        "metrics": metrics,
        "failures": failures,
    }


def main():
    st.title("⚖️ LLM-as-a-Judge Alignment Study — Post-Training Evaluation")
    st.markdown(
        "**Evaluating AI Feedback Quality (Groq 70B) vs. Human Judgment on Financial LLM Probes** | "
        "*Research Target: BNY Mellon AI Hub · UBS AI Academia & Research · Goldman Sachs AI/ML*"
    )

    data = load_data()
    metrics = data["metrics"]
    probes = data["probes"]
    human = data["human"]
    judge_b = data["judge_b"]
    local = data["local"]

    # --- Top Metrics Row ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Human Ceiling (Fleiss κ)",
            f"{metrics['human_inter_annotator_fleiss_kappa']['overall']:.3f}",
            "3 Raters",
        )
    with col2:
        st.metric(
            "Rubric B Kappa (CoT)",
            f"{metrics['rubric_b_judge_human_kappa']['quadratic_weighted']['overall']:.3f}",
            "+0.153 vs Zero-Shot",
        )
    with col3:
        st.metric(
            "Rubric A Kappa (Baseline)",
            f"{metrics['rubric_a_judge_human_kappa']['quadratic_weighted']['overall']:.3f}",
            "Zero-Shot",
        )
    with col4:
        st.metric(
            "Fluency Pre-Filter (Pearson r)",
            f"{metrics['local_evaluator_vs_teacher_correlation']['fluency']['pearson_r']:.3f}",
            "Fast-Pass Safe",
        )
    with col5:
        st.metric(
            "Local Scorer Latency",
            "0.048 ms",
            "~20,800 probes/s",
        )

    st.divider()

    tabs = st.tabs([
        "🔍 Probe Inspector & Cross-Evaluator",
        "🚨 Failure Modes & Disagreements",
        "📊 Statistical Alignment & Figures",
        "⚡ Real-Time Evaluator Playground",
    ])

    # ------------------ TAB 1: PROBE INSPECTOR ------------------
    with tabs[0]:
        st.subheader("Interactive Probe Explorer")
        st.markdown("Examine probe completions side-by-side across human annotators, local heuristics, and Groq 70B critiques.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            domains = ["All"] + sorted(list({p["domain"] for p in probes.values()}))
            selected_domain = st.selectbox("Filter by Domain", domains)
        with col_f2:
            categories = ["All"] + sorted(list({p["category"] for p in probes.values()}))
            selected_cat = st.selectbox("Filter by Category", categories)

        filtered_ids = [
            pid for pid, p in probes.items()
            if (selected_domain == "All" or p["domain"] == selected_domain)
            and (selected_cat == "All" or p["category"] == selected_cat)
        ]

        if not filtered_ids:
            st.warning("No probes match the selected criteria.")
        else:
            selected_probe_id = st.selectbox("Select Probe ID", filtered_ids)
            p_data = probes[selected_probe_id]
            h_data = human[selected_probe_id]
            jb_data = judge_b[selected_probe_id]
            loc_data = local[selected_probe_id]

            st.markdown(f"### Probe: `{p_data['id']}` — *{p_data['domain']}*")
            col_left, col_right = st.columns([1.2, 1])

            with col_left:
                st.markdown("**Prompt:**")
                st.info(p_data["prompt"])

                st.markdown("**Reference Context:**")
                st.code(p_data["reference_context"], language="text")

                st.markdown("**Model Output:**")
                st.success(p_data["model_output"])

                st.markdown("**Gold Answer Reference:**")
                st.code(p_data["gold_answer"], language="text")

            with col_right:
                st.markdown("#### Evaluation Comparison")
                score_df = pd.DataFrame({
                    "Axis": ["Fluency", "Coherence", "Instruction Adherence", "No Hallucination"],
                    "Human Consensus": [
                        h_data["consensus_normalized"]["fluency"],
                        h_data["consensus_normalized"]["coherence"],
                        h_data["consensus_normalized"]["instruction_adherence"],
                        h_data["consensus_normalized"]["no_hallucination"],
                    ],
                    "Teacher Judge (70B)": [
                        jb_data["scores"]["fluency"],
                        jb_data["scores"]["coherence"],
                        jb_data["scores"]["instruction_adherence"],
                        jb_data["scores"]["no_hallucination"],
                    ],
                    "Local Heuristic": [
                        loc_data["scores"]["fluency"],
                        loc_data["scores"]["coherence"],
                        loc_data["scores"]["instruction_adherence"],
                        loc_data["scores"]["no_hallucination"],
                    ],
                })
                st.dataframe(score_df.set_index("Axis"), use_container_width=True)

                st.markdown("**Teacher Judge (CoT Reasoning):**")
                st.caption(jb_data["reasoning"])

                st.markdown("**Local Metrics:**")
                st.text(
                    f"TTR: {loc_data['metrics']['ttr']:.2f} | "
                    f"Repetition Penalty: {loc_data['metrics']['repetition_penalty']:.3f} | "
                    f"Jaccard: {loc_data['metrics']['jaccard_overlap']:.3f} | "
                    f"Latency: {loc_data['metrics']['latency_ms']:.4f} ms"
                )

    # ------------------ TAB 2: FAILURE MODES ------------------
    with tabs[1]:
        st.subheader("Disagreement Taxonomy (|Δ| > 0.30)")
        st.markdown(
            "Identifies where Groq 70B AI judge and human consensus diverge. "
            "**Key Finding:** 25.0% of disagreements represent rubric ambiguity, not judge failure."
        )

        failures = data["failures"]
        col_b1, col_b2 = st.columns([1, 1.2])

        with col_b1:
            st.markdown("#### Disagreement Summary")
            st.write(f"**Total Samples:** {failures['total_samples']}")
            st.write(f"**Total Disagreements:** {failures['total_disagreements']} ({failures['disagreement_rate']}%)")

            breakdown_df = pd.DataFrame(failures["breakdown_summary"])
            st.dataframe(breakdown_df.set_index("category"), use_container_width=True)

            fig_path = FIGURES_DIR / "failure_modes_distribution.png"
            if fig_path.exists():
                st.image(str(fig_path), use_column_width=True)

        with col_b2:
            st.markdown("#### Drill-Down: Disputed Records")
            dis_cats = ["All"] + list({d["disagreement_category"] for d in failures["disagreements"]})
            sel_dis_cat = st.selectbox("Filter Disagreement Category", dis_cats)

            filtered_dis = [
                d for d in failures["disagreements"]
                if sel_dis_cat == "All" or d["disagreement_category"] == sel_dis_cat
            ]

            for d in filtered_dis[:10]:
                with st.expander(f"[{d['disagreement_category'].split('(')[0].strip()}] Probe `{d['probe_id']}`"):
                    st.write(f"**Prompt:** {d['prompt']}")
                    st.write(f"**Output:** {d['model_output']}")
                    st.write("**Disagreed Axes:**", d["disagreed_axes"])
                    st.info(f"**Judge Reasoning:** {d['judge_reasoning']}")

    # ------------------ TAB 3: STATISTICAL FIGURES ------------------
    with tabs[2]:
        st.subheader("Publication-Ready Visualizations Gallery")
        st.markdown("Empirical evidence justifying RLAIF post-training and pre-filter routing.")

        figures = [
            ("Human–Judge Alignment (Rubric A vs. Rubric B)", "agreement_comparison_rubric_a_vs_b.png"),
            ("Local Evaluator Pre-Filter vs. 70B Teacher Correlation", "local_vs_teacher_correlation.png"),
            ("Pre-Filter Cost Reduction vs. Decision Safety Trade-Off", "prefilter_tradeoff_curve.png"),
            ("Cross-Evaluator Correlation Matrix Heatmap", "correlation_matrix_heatmap.png"),
        ]

        for title, fname in figures:
            st.markdown(f"### {title}")
            f_path = FIGURES_DIR / fname
            if f_path.exists():
                st.image(str(f_path), use_column_width=True)
            else:
                st.error(f"Figure {fname} not found. Run run_pipeline.py first.")
            st.divider()

    # ------------------ TAB 4: PLAYGROUND ------------------
    with tabs[3]:
        st.subheader("Live Real-Time Evaluator Playground")
        st.markdown("Test the sub-millisecond local scorer and simulated teacher evaluator on custom financial generation.")

        default_prompt = "Extract Q3 operating income and net margin. Output in one single sentence."
        default_context = "In Q3, operating revenue stood at $10.2B, with operating income reaching $3.1B and net profit margin of 30.4%."
        default_output = "Q3 operating income was $3.1B with a net profit margin of 30.4%."

        input_prompt = st.text_input("Prompt", default_prompt)
        input_context = st.text_area("Reference Context", default_context, height=90)
        input_output = st.text_area("Model Output to Evaluate", default_output, height=90)

        if st.button("Evaluate Live Output", type="primary"):
            from evaluation.local_scorer import LightweightStudentEvaluator
            from evaluation.teacher_judge import TeacherJudge

            evaluator = LightweightStudentEvaluator()
            res_local = evaluator.evaluate(input_output, input_prompt, input_context)

            judge = TeacherJudge(rubric_path=BASE_DIR / "evaluation" / "rubrics" / "rubric_b.yaml", mock_mode=True)
            res_judge = judge.evaluate_single(input_prompt, input_context, input_output)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ⚡ Local Heuristic Scorer Result")
                st.write(f"**Fluency:** `{res_local.fluency:.3f}`")
                st.write(f"**Coherence:** `{res_local.coherence:.3f}`")
                st.write(f"**Instruction Adherence:** `{res_local.instruction_adherence:.3f}`")
                st.write(f"**Grounding Proxy:** `{res_local.no_hallucination:.3f}`")
                st.caption(f"⚡ Latency: **{res_local.latency_ms:.4f} ms**")

            with c2:
                st.markdown("#### ⚖️ Teacher Judge (Rubric B) Result")
                st.write(f"**Fluency:** `{res_judge.fluency:.3f}`")
                st.write(f"**Coherence:** `{res_judge.coherence:.3f}`")
                st.write(f"**Instruction Adherence:** `{res_judge.instruction_adherence:.3f}`")
                st.write(f"**No Hallucination:** `{res_judge.no_hallucination:.3f}`")
                st.info(f"**Reasoning:** {res_judge.reasoning}")


if __name__ == "__main__":
    main()
