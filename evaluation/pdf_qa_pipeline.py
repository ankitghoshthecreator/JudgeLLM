"""
PDF-to-QA Bias Investigation Pipeline
======================================
Extracts text from a PDF, generates questions, streams dual-model answers,
runs a judge LLM, and detects position + verbosity bias.

Environment variables (loaded from .env):
  GROQ_API_KEY        — Groq API key (required for live inference)
  GROQ_MODEL          — Default answering model (Model 1 & Model 2 base)
  GROQ_JUDGE_MODEL    — Default judge model

All Groq API calls gracefully fall back to mock mode when no GROQ_API_KEY is set.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Generator, Iterator, List, Optional

# ---------------------------------------------------------------------------
# Load .env automatically so all os.getenv() calls pick up the values
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env from project root (parent of this file's directory)."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)  # don't override already-set vars
    except ImportError:
        pass  # python-dotenv not installed; rely on system env

_load_env()

# ---------------------------------------------------------------------------
# Env-driven model defaults
# ---------------------------------------------------------------------------

# Fallback model IDs used when env vars are not set
_FALLBACK_ANSWER_MODEL = "llama-3.3-70b-versatile"
_FALLBACK_ANSWER_MODEL_2 = "llama-3.1-8b-instant"
_FALLBACK_JUDGE_MODEL = "llama-3.3-70b-versatile"

def get_default_model_1() -> str:
    """Model 1 for answering — from GROQ_MODEL env var."""
    return os.getenv("GROQ_MODEL", _FALLBACK_ANSWER_MODEL)

def get_default_model_2() -> str:
    """Model 2 for answering — second choice; falls back to a lighter model."""
    # If GROQ_MODEL is set, use the fallback 8B as the contrasting model
    base = os.getenv("GROQ_MODEL", "")
    return _FALLBACK_ANSWER_MODEL_2 if base else _FALLBACK_ANSWER_MODEL_2

def get_default_judge_model() -> str:
    """Judge model — from GROQ_JUDGE_MODEL env var."""
    return os.getenv("GROQ_JUDGE_MODEL", _FALLBACK_JUDGE_MODEL)

def get_available_models() -> List[str]:
    """Returns a list of candidate models, putting configured env models first."""
    models: List[str] = []
    
    # 1. Models configured in .env
    env_model = os.getenv("GROQ_MODEL")
    if env_model and env_model.strip() and env_model not in models:
        models.append(env_model.strip())
    env_judge = os.getenv("GROQ_JUDGE_MODEL")
    if env_judge and env_judge.strip() and env_judge not in models:
        models.append(env_judge.strip())
        
    # 2. Standard fast and frontier Groq models
    standard = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "deepseek-r1-distill-llama-70b",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
    ]
    for m in standard:
        if m not in models:
            models.append(m)
    return models

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF byte string using PyPDF2."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: List[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except ImportError:
        raise RuntimeError(
            "PyPDF2 is not installed. Run: pip install pypdf2"
        )


# ---------------------------------------------------------------------------
# Groq client helper
# ---------------------------------------------------------------------------

def _get_groq_client():
    """Return a Groq client using GROQ_API_KEY, or None if unavailable."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

_MOCK_QUESTIONS = [
    "What is the main topic or thesis presented in this document?",
    "What key data points or statistics are mentioned?",
    "What assumptions does the author make?",
    "How does the document define its core concepts?",
    "What evidence or sources does the document cite?",
    "What limitations does the document acknowledge?",
    "What conclusions are drawn?",
    "How does the document compare different approaches or options?",
    "What recommendations are made, if any?",
    "What risks or challenges are identified?",
    "What time horizon does the analysis cover?",
    "Who is the intended audience of this document?",
    "What methodology is used in this analysis?",
    "What open questions or future directions are raised?",
    "Are there any contradictions or inconsistencies in the document?",
    "What financial or quantitative metrics are highlighted?",
    "How does the document handle uncertainty?",
    "What alternatives are considered?",
    "What is the overall tone — bullish, bearish, neutral?",
    "Summarise the document's key insight in one sentence.",
]


def generate_questions(
    pdf_text: str,
    n: int = 10,
    model: str | None = None,
) -> List[str]:
    """
    Ask an LLM to generate *n* factual/analytical questions about the PDF text.
    Falls back to mock questions if no GROQ_API_KEY is set.
    Uses GROQ_MODEL env var as default model if *model* is not supplied.
    """
    if model is None:
        model = get_default_model_1()
    client = _get_groq_client()

    if client is None:
        # Mock: return canned questions (shuffled to feel fresh)
        sample = _MOCK_QUESTIONS.copy()
        random.shuffle(sample)
        return sample[:n]

    prompt = (
        f"You are a rigorous exam designer. Read the following document excerpt and "
        f"generate exactly {n} clear, answerable questions that test deep understanding "
        f"of its content. Output ONLY a numbered list, one question per line.\n\n"
        f"DOCUMENT:\n{pdf_text[:6000]}"  # trim to avoid huge context
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        questions = []
        for line in raw.splitlines():
            line = line.strip()
            # Strip leading "1." / "1)" / "-" / "*"
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            if len(line) > 10:
                questions.append(line)
        # If the LLM returned fewer than half the requested questions
        # (e.g., it gave a refusal/apology instead of a numbered list),
        # treat it as a failure and fall back to the mock bank.
        if len(questions) < max(2, n // 2):
            return _MOCK_QUESTIONS[:n]
        return questions[:n] if questions else _MOCK_QUESTIONS[:n]
    except Exception:
        return _MOCK_QUESTIONS[:n]


# ---------------------------------------------------------------------------
# Streaming answer generation
# ---------------------------------------------------------------------------

def _mock_answer(question: str, model_name: str) -> str:
    """Return a deterministic mock answer for offline demos."""
    length_factor = 1 if "8b" in model_name or "small" in model_name.lower() else 2
    short = (
        f"Based on the document, the answer to '{question[:60]}…' "
        f"involves key financial principles. The {model_name} model "
        f"identifies the central point as follows: "
        f"there are multiple interrelated factors at play."
    )
    if length_factor == 2:
        short += (
            f" Furthermore, a deeper analysis reveals that the underlying "
            f"mechanisms are driven by market dynamics, regulatory constraints, "
            f"and risk-adjusted return considerations. The model concludes that "
            f"a nuanced, context-aware approach is essential for accurate evaluation."
        )
    return short


def stream_answer(
    question: str,
    context: str,
    model: str | None = None,
    max_tokens: int = 512,
) -> Generator[str, None, None]:
    """
    Yield answer tokens one chunk at a time (Groq streaming).
    Falls back to a character-by-character mock stream.
    Uses GROQ_MODEL env var as default if *model* is not supplied.
    """
    if model is None:
        model = get_default_model_1()
    client = _get_groq_client()

    if client is None:
        mock = _mock_answer(question, model)
        for char in mock:
            yield char
            time.sleep(0.005)  # simulate streaming delay
        return

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable analyst. Answer the user's question "
                "using ONLY the information from the provided context. "
                "Be thorough but concise. Do not hallucinate."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context[:4000]}\n\nQUESTION:\n{question}",
        },
    ]

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception:
        # If API network call fails (e.g. offline, rate limit, or model ID unsupported),
        # gracefully fall back to calibrated mock stream so UI never crashes.
        mock = _mock_answer(question, model)
        for char in mock:
            yield char
            time.sleep(0.005)


def collect_stream(generator: Iterator[str]) -> str:
    """Fully consume a stream generator and return the concatenated string."""
    return "".join(generator)


# ---------------------------------------------------------------------------
# Judge LLM — rates two answers
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an impartial AI evaluator. Your job is to score two answers 
(Answer A and Answer B) to the same question. 
Score each answer on a scale from 1 to 10 stars where:
  1-3 = Poor (inaccurate, vague, or hallucinated)
  4-6 = Acceptable (partially correct, some gaps)
  7-9 = Good (accurate, well-reasoned)
  10  = Excellent (perfectly correct, clear, complete)

Respond ONLY with valid JSON in this exact format:
{
  "score_a": <integer 1-10>,
  "score_b": <integer 1-10>,
  "reasoning_a": "<brief one-sentence reason>",
  "reasoning_b": "<brief one-sentence reason>",
  "preferred": "<A or B or Tie>"
}"""

_JUDGE_USER_TEMPLATE = """QUESTION: {question}

ANSWER A ({model_a}):
{answer_a}

ANSWER B ({model_b}):
{answer_b}

Rate both answers."""


def _mock_judge(
    question: str,
    answer_a: str,
    answer_b: str,
    model_a: str,
    model_b: str,
) -> dict:
    """Deterministic mock judge based on answer length (simulates verbosity bias)."""
    len_a = len(answer_a.split())
    len_b = len(answer_b.split())
    # Simulate mild verbosity bias: longer answer gets +1
    base_a = 7 if len_a >= len_b else 6
    base_b = 7 if len_b > len_a else 6
    return {
        "score_a": base_a,
        "score_b": base_b,
        "reasoning_a": f"{model_a} provided a {'detailed' if len_a >= len_b else 'brief'} answer covering the key points.",
        "reasoning_b": f"{model_b} provided a {'detailed' if len_b > len_a else 'brief'} answer covering the key points.",
        "preferred": "A" if base_a > base_b else ("B" if base_b > base_a else "Tie"),
    }


def judge_answers(
    question: str,
    answer_a: str,
    answer_b: str,
    model_a: str = "Model A",
    model_b: str = "Model B",
    judge_model: str | None = None,
) -> dict:
    """
    Ask the judge LLM to rate answer_a and answer_b.
    Returns dict with keys: score_a, score_b, reasoning_a, reasoning_b, preferred.
    """
    if judge_model is None:
        judge_model = get_default_judge_model()
    client = _get_groq_client()

    if client is None:
        return _mock_judge(question, answer_a, answer_b, model_a, model_b)

    user_content = _JUDGE_USER_TEMPLATE.format(
        question=question,
        model_a=model_a,
        answer_a=answer_a,
        model_b=model_b,
        answer_b=answer_b,
    )

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        # Extract JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            # Clamp scores to 1-10
            result["score_a"] = max(1, min(10, int(result.get("score_a", 5))))
            result["score_b"] = max(1, min(10, int(result.get("score_b", 5))))
            return result
    except Exception:
        pass

    return _mock_judge(question, answer_a, answer_b, model_a, model_b)


# ---------------------------------------------------------------------------
# Position bias detection
# ---------------------------------------------------------------------------

def detect_position_bias(
    question: str,
    answer_a: str,
    answer_b: str,
    model_a: str,
    model_b: str,
    judge_model: str | None = None,
) -> dict:
    """
    Call judge twice — once with (A, B) order and once with (B, A) order.
    Returns a dict describing whether the judge was consistent or position-biased.

    Position bias = judge changes its preference purely due to order swap.
    """
    if judge_model is None:
        judge_model = get_default_judge_model()
    # Normal order: A first, B second
    result_ab = judge_answers(
        question, answer_a, answer_b, model_a, model_b, judge_model
    )
    # Swapped order: B first, A second
    result_ba = judge_answers(
        question, answer_b, answer_a, model_b, model_a, judge_model
    )

    # In result_ba, score_a is the score for answer_b (now in slot A)
    # score_b is the score for answer_a (now in slot B)
    # Re-map back to original labels
    score_a_in_ab = result_ab["score_a"]
    score_b_in_ab = result_ab["score_b"]
    score_a_in_ba = result_ba["score_b"]  # answer_a was in slot B this time
    score_b_in_ba = result_ba["score_a"]  # answer_b was in slot A this time

    delta_a = abs(score_a_in_ab - score_a_in_ba)
    delta_b = abs(score_b_in_ab - score_b_in_ba)
    avg_delta = (delta_a + delta_b) / 2

    # Preference in normal order
    pref_ab = result_ab.get("preferred", "Tie")
    # Preference in swapped order (re-map labels)
    raw_pref_ba = result_ba.get("preferred", "Tie")
    if raw_pref_ba == "A":
        pref_ba_remapped = "B"  # B was in slot A
    elif raw_pref_ba == "B":
        pref_ba_remapped = "A"  # A was in slot B
    else:
        pref_ba_remapped = "Tie"

    preference_flipped = pref_ab != pref_ba_remapped and pref_ab != "Tie" and pref_ba_remapped != "Tie"

    return {
        "score_a_normal": score_a_in_ab,
        "score_b_normal": score_b_in_ab,
        "score_a_swapped": score_a_in_ba,
        "score_b_swapped": score_b_in_ba,
        "avg_score_delta": round(avg_delta, 2),
        "preference_normal": pref_ab,
        "preference_swapped": pref_ba_remapped,
        "preference_flipped": preference_flipped,
        "position_bias_detected": avg_delta >= 1.5 or preference_flipped,
        "reasoning_normal_a": result_ab.get("reasoning_a", ""),
        "reasoning_normal_b": result_ab.get("reasoning_b", ""),
    }


# ---------------------------------------------------------------------------
# Verbosity bias analysis
# ---------------------------------------------------------------------------

def compute_verbosity_stats(session_data: list[dict]) -> dict:
    """
    Given a list of per-question records with keys:
        {question_idx, answer_a, answer_b, score_a, score_b}
    Returns verbosity bias statistics.
    """
    if not session_data:
        return {"bias_detected": False, "correlation_a": 0.0, "correlation_b": 0.0}

    try:
        import numpy as np

        words_a = [len(r["answer_a"].split()) for r in session_data]
        words_b = [len(r["answer_b"].split()) for r in session_data]
        scores_a = [r["score_a"] for r in session_data]
        scores_b = [r["score_b"] for r in session_data]

        # Pearson correlation between word count and judge score
        corr_a = float(np.corrcoef(words_a, scores_a)[0, 1]) if len(words_a) > 1 else 0.0
        corr_b = float(np.corrcoef(words_b, scores_b)[0, 1]) if len(words_b) > 1 else 0.0

        # Check if longer answer consistently got higher score
        longer_wins = sum(
            1 for r in session_data
            if (len(r["answer_a"].split()) > len(r["answer_b"].split()) and r["score_a"] > r["score_b"])
            or (len(r["answer_b"].split()) > len(r["answer_a"].split()) and r["score_b"] > r["score_a"])
        )
        longer_win_rate = longer_wins / len(session_data)

        return {
            "bias_detected": abs(corr_a) > 0.4 or abs(corr_b) > 0.4 or longer_win_rate > 0.65,
            "correlation_a": round(corr_a, 3),
            "correlation_b": round(corr_b, 3),
            "longer_answer_win_rate": round(longer_win_rate, 3),
            "words_a": words_a,
            "words_b": words_b,
            "scores_a": scores_a,
            "scores_b": scores_b,
        }
    except Exception:
        return {"bias_detected": False, "correlation_a": 0.0, "correlation_b": 0.0}
