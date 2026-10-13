"""
Lightweight Local Scorer Engine
================================
High-throughput heuristic evaluator (~0.013ms / probe) ported and adapted from
CIRCLE's `distilled_eval.py` (`LightweightStudentEvaluator`).

Calculates:
- Lexical diversity: Type-Token Ratio (TTR), Hapax Legomena Ratio
- Repetition penalties: 2-gram, 3-gram, 4-gram repetition fractions
- Context grounding & Lexical overlap: Jaccard similarity against context/reference
- Adherence heuristics: Structural length, regex constraints, keyword coverage
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Set, Tuple


@dataclass
class LocalScoreResult:
    fluency: float
    coherence: float
    instruction_adherence: float
    no_hallucination: float
    ttr: float
    repetition_penalty: float
    jaccard_overlap: float
    latency_ms: float
    raw_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LightweightStudentEvaluator:
    """
    Sub-millisecond local scorer serving as a high-throughput pre-filter
    before invoking expensive 70B teacher calls.
    """

    # Common English stopwords to prevent artificially inflated Jaccard overlaps
    STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
        "does", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
        "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
        "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

    TOKEN_PATTERN = re.compile(r"\b[A-Za-z0-9_$%.-]+\b")

    def __init__(self, target_max_words: int = 500):
        self.target_max_words = target_max_words

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [tok.lower() for tok in self.TOKEN_PATTERN.findall(text)]

    def compute_ttr(self, tokens: List[str]) -> Tuple[float, float]:
        """Calculates Type-Token Ratio and Hapax Legomena Ratio."""
        if not tokens:
            return 0.0, 0.0
        n_tokens = len(tokens)
        freqs: Dict[str, int] = {}
        for tok in tokens:
            freqs[tok] = freqs.get(tok, 0) + 1

        ttr = len(freqs) / n_tokens
        hapax = sum(1 for c in freqs.values() if c == 1) / n_tokens
        return round(ttr, 4), round(hapax, 4)

    def compute_ngram_repetitions(self, tokens: List[str], n: int = 3) -> float:
        """Computes fraction of duplicate n-grams in the token sequence."""
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        dup_ratio = (total - unique) / total if total > 0 else 0.0
        return round(dup_ratio, 4)

    def compute_jaccard(self, tokens_a: List[str], tokens_b: List[str], filter_stops: bool = True) -> float:
        """Computes Jaccard similarity index between token sets."""
        if filter_stops:
            set_a = {t for t in tokens_a if t not in self.STOPWORDS}
            set_b = {t for t in tokens_b if t not in self.STOPWORDS}
        else:
            set_a = set(tokens_a)
            set_b = set(tokens_b)

        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return round(intersection / union, 4) if union > 0 else 0.0

    def evaluate(
        self,
        model_output: str,
        prompt: str = "",
        reference_context: str = "",
        gold_answer: str = "",
    ) -> LocalScoreResult:
        """
        Fast evaluation of small-LLM output.
        Returns normalized 0.0 to 1.0 scores for each axis and latency in ms.
        """
        start_time = time.perf_counter()

        cleaned_output = model_output.strip() if model_output else ""
        if not cleaned_output:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return LocalScoreResult(
                fluency=0.0,
                coherence=0.0,
                instruction_adherence=0.0,
                no_hallucination=0.0,
                ttr=0.0,
                repetition_penalty=1.0,
                jaccard_overlap=0.0,
                latency_ms=round(elapsed_ms, 4),
                raw_metrics={"empty_output": True},
            )

        output_tokens = self.tokenize(cleaned_output)
        prompt_tokens = self.tokenize(prompt)
        context_tokens = self.tokenize(reference_context)
        gold_tokens = self.tokenize(gold_answer)

        num_words = len(output_tokens)
        ttr, hapax = self.compute_ttr(output_tokens)
        rep2 = self.compute_ngram_repetitions(output_tokens, n=2)
        rep3 = self.compute_ngram_repetitions(output_tokens, n=3)
        rep4 = self.compute_ngram_repetitions(output_tokens, n=4)

        rep_penalty = (rep2 * 0.2) + (rep3 * 0.4) + (rep4 * 0.4)

        jaccard_context = self.compute_jaccard(output_tokens, context_tokens) if context_tokens else 0.5
        jaccard_gold = self.compute_jaccard(output_tokens, gold_tokens) if gold_tokens else 0.5
        jaccard_prompt = self.compute_jaccard(output_tokens, prompt_tokens) if prompt_tokens else 0.5

        # --- Axis 1: Fluency ---
        # Highly influenced by lexical diversity and lack of degenerative loops
        fluency = min(1.0, max(0.0, (ttr * 0.40) + ((1.0 - min(1.0, rep_penalty * 2.0)) * 0.60)))
        if num_words < 5:
            fluency *= 0.5

        # --- Axis 2: Coherence ---
        # Sentence termination, lack of 3-gram repetition, structural cohesion
        coherence = min(1.0, max(0.0, 0.95 - (rep3 * 1.8) - (rep4 * 1.5)))
        has_terminal_punct = cleaned_output[-1] in {".", "!", "?", '"', "'"} if cleaned_output else False
        if not has_terminal_punct and num_words > 10:
            coherence *= 0.85
        if rep_penalty > 0.3:
            coherence = min(coherence, 0.35)

        # --- Axis 3: Instruction Adherence ---
        # Checks format constraints (bullets, single sentence, json)
        prompt_lower = prompt.lower()
        if "bullet" in prompt_lower or "list" in prompt_lower:
            has_bullets = any(line.strip().startswith(("-", "*", "•")) or re.match(r"^\d+\.", line.strip()) for line in cleaned_output.splitlines())
            adherence = 0.92 if has_bullets else 0.25
        elif "one sentence" in prompt_lower or "single sentence" in prompt_lower:
            sentences = [s for s in re.split(r"[.!?]+", cleaned_output) if s.strip()]
            adherence = 0.95 if len(sentences) == 1 else 0.30
        elif "json" in prompt_lower:
            adherence = 0.95 if ("{" in cleaned_output and "}" in cleaned_output) else 0.20
        else:
            adherence = min(1.0, 0.4 + (jaccard_prompt * 0.6))

        # --- Axis 4: No Hallucination (Factual Grounding Proxy) ---
        # Heuristic token overlap with context/gold (r~0.48 correlation ceiling)
        if context_tokens:
            no_hallucination = min(1.0, max(0.0, (jaccard_context * 0.65) + (jaccard_gold * 0.35)))
            no_hallucination = min(0.92, max(0.20, no_hallucination * 1.5))
        else:
            no_hallucination = 0.60


        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return LocalScoreResult(
            fluency=round(fluency, 3),
            coherence=round(coherence, 3),
            instruction_adherence=round(adherence, 3),
            no_hallucination=round(no_hallucination, 3),
            ttr=ttr,
            repetition_penalty=round(rep_penalty, 4),
            jaccard_overlap=round(jaccard_context, 4),
            latency_ms=round(elapsed_ms, 4),
            raw_metrics={
                "num_words": num_words,
                "hapax_ratio": hapax,
                "rep2": rep2,
                "rep3": rep3,
                "rep4": rep4,
                "jaccard_context": jaccard_context,
                "jaccard_gold": jaccard_gold,
                "jaccard_prompt": jaccard_prompt,
            },
        )


def run_local_evaluation(inputs_path: str, output_path: str) -> None:
    import json
    from pathlib import Path

    evaluator = LightweightStudentEvaluator()
    with open(inputs_path, "r", encoding="utf-8") as f:
        probes = json.load(f)

    results = []
    latencies = []
    for item in probes:
        probe_id = item.get("id")
        res = evaluator.evaluate(
            model_output=item.get("model_output", ""),
            prompt=item.get("prompt", ""),
            reference_context=item.get("reference_context", ""),
            gold_answer=item.get("gold_answer", ""),
        )
        latencies.append(res.latency_ms)
        results.append({
            "id": probe_id,
            "scores": {
                "fluency": res.fluency,
                "coherence": res.coherence,
                "instruction_adherence": res.instruction_adherence,
                "no_hallucination": res.no_hallucination,
            },
            "metrics": {
                "ttr": res.ttr,
                "repetition_penalty": res.repetition_penalty,
                "jaccard_overlap": res.jaccard_overlap,
                "latency_ms": res.latency_ms,
            },
        })

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Evaluated {len(results)} probes with LightweightStudentEvaluator.")
    print(f"[BENCHMARK] Average latency per probe: {avg_latency:.4f} ms")
    print(f"[SAVED] Results written to {out_p}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LightweightStudentEvaluator on probe inputs")
    parser.add_argument("--inputs", default="data/probe_outputs_200.json", help="Path to input probes")
    parser.add_argument("--output", default="data/local_scores.json", help="Path to destination file")
    args = parser.parse_args()

    run_local_evaluation(args.inputs, args.output)

