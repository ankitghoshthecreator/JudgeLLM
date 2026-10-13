"""
Test Suite for Phase 1: Data, Evaluators & Annotation Pipeline
==============================================================
"""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from evaluation.local_scorer import LightweightStudentEvaluator, LocalScoreResult
from evaluation.teacher_judge import (
    JudgeEvaluation,
    RubricConfig,
    TeacherJudge,
    load_rubric,
)
from annotation.label_studio_export import export_probes_to_label_studio, parse_label_studio_export


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def test_probe_dataset_structure():
    """Verify probe dataset contains exactly 200 valid financial QA records."""
    probe_file = DATA_DIR / "probe_outputs_200.json"
    assert probe_file.exists(), "probe_outputs_200.json must exist"

    with open(probe_file, "r", encoding="utf-8") as f:
        probes = json.load(f)

    assert len(probes) == 200, f"Expected 200 probes, got {len(probes)}"
    sample = probes[0]
    required_keys = {"id", "category", "domain", "prompt", "reference_context", "gold_answer", "model_output", "ground_truth_quality"}
    assert required_keys.issubset(sample.keys()), f"Missing keys in probe item: {required_keys - sample.keys()}"

    categories = {p["category"] for p in probes}
    assert "high_quality" in categories
    assert "hallucination" in categories
    assert "instruction_violation" in categories
    assert "fluency_degraded" in categories
    assert "ambiguous" in categories

    for axis in ["fluency", "coherence", "instruction_adherence", "no_hallucination"]:
        assert axis in sample["ground_truth_quality"]
        score = sample["ground_truth_quality"][axis]
        assert 0.0 <= score <= 1.0


def test_human_annotations_structure():
    """Verify 3-annotator ground truth ratings and consensus scores."""
    human_file = DATA_DIR / "human_annotations.json"
    assert human_file.exists(), "human_annotations.json must exist"

    with open(human_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    assert len(annotations) == 200, f"Expected 200 annotations, got {len(annotations)}"
    sample = annotations[0]
    assert "ratings" in sample
    assert "consensus_normalized" in sample

    ratings = sample["ratings"]
    assert len(ratings) == 3, f"Expected 3 annotators, got {len(ratings)}"
    for ann in ratings.values():
        for axis, val in ann.items():
            assert 0 <= val <= 3, f"Likert score out of range: {val}"

    for axis, val in sample["consensus_normalized"].items():
        assert 0.0 <= val <= 1.0, f"Normalized consensus out of range: {val}"


def test_judge_scores_files_exist_and_valid():
    """Verify Rubric A and Rubric B judge score files are generated and valid."""
    rubric_a_file = DATA_DIR / "judge_scores" / "rubric_a.json"
    rubric_b_file = DATA_DIR / "judge_scores" / "rubric_b.json"

    assert rubric_a_file.exists(), "rubric_a.json must exist"
    assert rubric_b_file.exists(), "rubric_b.json must exist"

    with open(rubric_a_file, "r", encoding="utf-8") as f:
        scores_a = json.load(f)
    with open(rubric_b_file, "r", encoding="utf-8") as f:
        scores_b = json.load(f)

    assert len(scores_a) == 200
    assert len(scores_b) == 200

    for item in scores_a:
        assert "id" in item
        assert "scores" in item
        for axis in ["fluency", "coherence", "instruction_adherence", "no_hallucination"]:
            assert 0.0 <= item["scores"][axis] <= 1.0

    for item in scores_b:
        assert item["rubric_version"] == "rubric_b"
        assert "[cot evaluation" in item["reasoning"].lower()


def test_local_scores_file_valid():
    """Verify local scorer cache file structure and metrics."""
    local_file = DATA_DIR / "local_scores.json"
    assert local_file.exists(), "local_scores.json must exist"

    with open(local_file, "r", encoding="utf-8") as f:
        local_data = json.load(f)

    assert len(local_data) == 200
    first = local_data[0]
    assert "scores" in first
    assert "metrics" in first
    assert "ttr" in first["metrics"]
    assert "repetition_penalty" in first["metrics"]
    assert "jaccard_overlap" in first["metrics"]
    assert "latency_ms" in first["metrics"]


def test_rubric_parsing():
    """Verify YAML rubrics parse correctly into Pydantic models."""
    rubric_a_path = BASE_DIR / "evaluation" / "rubrics" / "rubric_a.yaml"
    rubric_b_path = BASE_DIR / "evaluation" / "rubrics" / "rubric_b.yaml"

    cfg_a = load_rubric(rubric_a_path)
    assert cfg_a.name == "rubric_a"
    assert "fluency" in cfg_a.prompt_template.lower()

    cfg_b = load_rubric(rubric_b_path)
    assert cfg_b.name == "rubric_b"
    assert "exemplar" in cfg_b.prompt_template.lower()


def test_pydantic_schema_validation():
    """Test Pydantic v2 validation, value clamping, and error rejection."""
    valid_payload = {
        "fluency": 0.85,
        "coherence": 0.90,
        "instruction_adherence": 0.95,
        "no_hallucination": 0.80,
        "reasoning": "Reasonably articulated financial statement summary.",
        "rubric_version": "rubric_b",
    }
    eval_obj = JudgeEvaluation(**valid_payload)
    assert eval_obj.fluency == 0.85
    assert eval_obj.reasoning.startswith("Reasonably")

    # Clamping tests: scale 0-3 normalization
    scale3_payload = {
        "fluency": 3.0,
        "coherence": 2.0,
        "instruction_adherence": 1.5,
        "no_hallucination": 3.0,
        "reasoning": "Likert scaled feedback.",
    }
    eval_obj_clamped = JudgeEvaluation(**scale3_payload)
    assert eval_obj_clamped.fluency == 1.0
    assert abs(eval_obj_clamped.coherence - 0.667) < 0.01

    # Percentage scale (0-100) normalization
    scale100_payload = {
        "fluency": 95.0,
        "coherence": 85.0,
        "instruction_adherence": 100.0,
        "no_hallucination": 70.0,
        "reasoning": "Percentage scored feedback.",
    }
    eval_obj_100 = JudgeEvaluation(**scale100_payload)
    assert eval_obj_100.fluency == 0.95
    assert eval_obj_100.no_hallucination == 0.70

    # Invalid: empty reasoning
    with pytest.raises(ValidationError):
        JudgeEvaluation(fluency=0.8, coherence=0.8, instruction_adherence=0.8, no_hallucination=0.8, reasoning="")


def test_teacher_judge_json_extraction():
    """Test JSON parser handling markdown code blocks, whitespace, and raw JSON."""
    rubric_b_path = BASE_DIR / "evaluation" / "rubrics" / "rubric_b.yaml"
    judge = TeacherJudge(rubric_path=rubric_b_path, mock_mode=True)

    # Markdown fence
    md_text = '```json\n{"fluency": 0.9, "coherence": 0.85, "instruction_adherence": 0.95, "no_hallucination": 1.0, "reasoning": "Looks solid"}\n```'
    parsed = judge._extract_json(md_text)
    assert parsed["fluency"] == 0.9

    # Free text with embedded JSON
    embedded_text = 'Here is my rating:\n{"fluency": 0.75, "coherence": 0.70, "instruction_adherence": 0.80, "no_hallucination": 0.90, "reasoning": "Slight issue"}\nThank you!'
    parsed_emb = judge._extract_json(embedded_text)
    assert parsed_emb["fluency"] == 0.75


def test_lightweight_student_evaluator():
    """Verify local evaluator heuristics, TTR, Jaccard, and repetition penalties."""
    evaluator = LightweightStudentEvaluator()

    # Empty text case
    res_empty = evaluator.evaluate("")
    assert res_empty.fluency == 0.0
    assert res_empty.repetition_penalty == 1.0

    # High quality financial text
    text_good = "In the third quarter, operating revenue expanded 12% to $4.2 billion, supported by net asset inflows."
    context = "Operating revenue expanded 12% to $4.2 billion in Q3."
    res_good = evaluator.evaluate(text_good, prompt="Summarize revenue", reference_context=context)

    assert isinstance(res_good, LocalScoreResult)
    assert res_good.fluency > 0.6
    assert res_good.latency_ms < 5.0

    # Repetitive looping text
    text_bad = "revenue revenue revenue revenue grew grew grew grew grew grew grew grew grew."
    res_bad = evaluator.evaluate(text_bad, prompt="Summarize revenue", reference_context=context)
    assert res_bad.fluency < res_good.fluency
    assert res_bad.repetition_penalty > 0.3


def test_evaluator_submillisecond_speed():
    """Benchmark local evaluator speed across 100 consecutive runs to ensure sub-millisecond execution."""
    evaluator = LightweightStudentEvaluator()
    text = "The commercial real estate portfolio experienced downward rating migrations primarily driven by office vacancy rates exceeding 22%."
    context = "Commercial real estate office vacancy exceeded 22% in the reporting period."

    latencies = []
    for _ in range(100):
        res = evaluator.evaluate(text, prompt="List CRE risks", reference_context=context)
        latencies.append(res.latency_ms)

    avg_ms = sum(latencies) / len(latencies)
    assert avg_ms < 1.0, f"Average latency too high: {avg_ms:.4f} ms (expected < 1.0 ms)"


def test_teacher_judge_mock():
    """Verify TeacherJudge mock evaluation produces valid scores and CoT reasoning."""
    rubric_b_path = BASE_DIR / "evaluation" / "rubrics" / "rubric_b.yaml"
    judge = TeacherJudge(rubric_path=rubric_b_path, mock_mode=True)

    result = judge.evaluate_single(
        prompt="List key credit risks",
        reference_context="Office vacancies increased to 22% in the CRE portfolio.",
        model_output="- Office vacancies rose above 20%",
    )

    assert 0.0 <= result.fluency <= 1.0
    assert 0.0 <= result.coherence <= 1.0
    assert 0.0 <= result.instruction_adherence <= 1.0
    assert 0.0 <= result.no_hallucination <= 1.0
    assert len(result.reasoning) > 10
    assert result.rubric_version == "rubric_b"


def test_label_studio_task_export_and_parse(tmp_path):
    """Test Label Studio roundtrip export and simulated annotation parsing."""
    probe_file = DATA_DIR / "probe_outputs_200.json"
    temp_tasks = tmp_path / "test_tasks.json"

    tasks = export_probes_to_label_studio(probe_file, temp_tasks)
    assert len(tasks) == 200
    assert temp_tasks.exists()
    assert "data" in tasks[0]
    assert "prompt" in tasks[0]["data"]

    # Test parsing simulated completed Label Studio payload
    simulated_export = [
        {
            "id": "probe_001",
            "annotations": [
                {
                    "completed_by": {"email": "ann1@bank.com"},
                    "result": [
                        {"from_name": "fluency", "value": {"rating": 3}},
                        {"from_name": "coherence", "value": {"rating": 3}},
                        {"from_name": "instruction_adherence", "value": {"rating": 2}},
                        {"from_name": "no_hallucination", "value": {"rating": 3}},
                    ],
                },
                {
                    "completed_by": {"email": "ann2@bank.com"},
                    "result": [
                        {"from_name": "fluency", "value": {"rating": 3}},
                        {"from_name": "coherence", "value": {"rating": 2}},
                        {"from_name": "instruction_adherence", "value": {"rating": 2}},
                        {"from_name": "no_hallucination", "value": {"rating": 3}},
                    ],
                },
            ],
        }
    ]
    sim_file = tmp_path / "simulated_ls_export.json"
    parsed_out = tmp_path / "parsed_annotations.json"
    with open(sim_file, "w", encoding="utf-8") as f:
        json.dump(simulated_export, f)

    parsed = parse_label_studio_export(sim_file, parsed_out)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "probe_001"
    assert parsed[0]["consensus_normalized"]["fluency"] == 1.0
    assert abs(parsed[0]["consensus_normalized"]["coherence"] - 0.833) < 0.01
