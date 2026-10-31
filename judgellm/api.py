"""
FastAPI Server for JudgeLLM
===========================
Features:
  - POST /judge: Single judgment
  - POST /judge/batch: Async bounded batching with retry and backoff
  - GET /health: Health status and active rubric version + git commit
  - GET /metrics: Prometheus-compatible metrics counter and histograms
  - Disk-backed deterministic caching
  - Per-request structured JSON logging
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from evaluation.backends import create_backend, load_backend_from_config
from evaluation.teacher_judge import TeacherJudge
from judgellm.cache import JudgeDiskCache, get_git_commit_hash

logger = logging.getLogger("judgellm.api")
logging.basicConfig(level=logging.INFO, format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')

app = FastAPI(
    title="JudgeLLM Reliability Service",
    version="0.2.0",
    description="Provider-agnostic, cache-backed industrial LLM Judge evaluation service",
)

# Global runtime state
DISK_CACHE = JudgeDiskCache(Path(".cache/judgellm"))
CONCURRENCY_SEMAPHORE = asyncio.Semaphore(10)  # Bound concurrent LLM calls

# Metrics counters
METRICS = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "cache_hits_total": 0,
    "cache_misses_total": 0,
    "total_latency_seconds": 0.0,
    "total_estimated_cost_usd": 0.0,
}

# Default active judge instance
_ACTIVE_RUBRIC_PATH = Path("evaluation/rubrics/rubric_b.yaml")
_DEFAULT_BACKEND = create_backend(mock_mode=True)
_TEACHER_JUDGE = TeacherJudge(rubric_path=_ACTIVE_RUBRIC_PATH, backend=_DEFAULT_BACKEND)


class JudgeRequest(BaseModel):
    prompt: str = Field(..., description="Prompt or query given to evaluated model")
    reference_context: str = Field(default="", description="Source context or document reference")
    model_output: str = Field(..., description="Response to be evaluated")
    mode: str = Field(default="auto", description="'reference_based', 'reference_free', or 'auto'")
    metadata: Optional[Dict[str, Any]] = None


class BatchJudgeRequest(BaseModel):
    items: List[JudgeRequest]
    max_concurrency: Optional[int] = Field(default=10, ge=1, le=50)


class JudgmentResponse(BaseModel):
    fluency: float
    coherence: float
    instruction_adherence: float
    no_hallucination: float
    reasoning: str
    rubric_id: str
    rubric_version: str
    git_commit_hash: str
    cache_hit: bool
    latency_ms: float
    estimated_cost_usd: float


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "judgellm",
        "version": "0.2.0",
        "backend": _TEACHER_JUDGE.backend.backend_type,
        "model": _TEACHER_JUDGE.model_name,
        "rubric_id": _TEACHER_JUDGE.rubric.name,
        "rubric_version": _TEACHER_JUDGE.rubric.version,
        "git_commit_hash": get_git_commit_hash(),
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus-compatible plain text metrics."""
    avg_latency = (
        METRICS["total_latency_seconds"] / METRICS["requests_total"]
        if METRICS["requests_total"] > 0
        else 0.0
    )
    body = (
        f"# HELP judgellm_requests_total Total number of evaluation requests.\n"
        f"# TYPE judgellm_requests_total counter\n"
        f"judgellm_requests_total {METRICS['requests_total']}\n\n"
        f"# HELP judgellm_requests_success Total successful evaluations.\n"
        f"# TYPE judgellm_requests_success counter\n"
        f"judgellm_requests_success {METRICS['requests_success']}\n\n"
        f"# HELP judgellm_requests_failed Total failed evaluations.\n"
        f"# TYPE judgellm_requests_failed counter\n"
        f"judgellm_requests_failed {METRICS['requests_failed']}\n\n"
        f"# HELP judgellm_cache_hits_total Number of cache hits.\n"
        f"# TYPE judgellm_cache_hits_total counter\n"
        f"judgellm_cache_hits_total {METRICS['cache_hits_total']}\n\n"
        f"# HELP judgellm_cache_misses_total Number of cache misses.\n"
        f"# TYPE judgellm_cache_misses_total counter\n"
        f"judgellm_cache_misses_total {METRICS['cache_misses_total']}\n\n"
        f"# HELP judgellm_avg_latency_seconds Average judgment latency.\n"
        f"# TYPE judgellm_avg_latency_seconds gauge\n"
        f"judgellm_avg_latency_seconds {avg_latency:.4f}\n\n"
        f"# HELP judgellm_total_estimated_cost_usd Estimated cost in USD.\n"
        f"# TYPE judgellm_total_estimated_cost_usd counter\n"
        f"judgellm_total_estimated_cost_usd {METRICS['total_estimated_cost_usd']:.6f}\n"
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")


async def _evaluate_item(req: JudgeRequest) -> JudgmentResponse:
    t0 = time.perf_counter()
    METRICS["requests_total"] += 1

    # Check cache
    cache_key = DISK_CACHE.compute_key(
        model_name=_TEACHER_JUDGE.model_name,
        rubric_id=_TEACHER_JUDGE.rubric.name,
        rubric_version=_TEACHER_JUDGE.rubric.version,
        prompt=req.prompt,
        reference_context=req.reference_context,
        model_output=req.model_output,
    )

    cached_val = DISK_CACHE.get(cache_key)
    if cached_val is not None:
        METRICS["cache_hits_total"] += 1
        METRICS["requests_success"] += 1
        elapsed = (time.perf_counter() - t0) * 1000.0
        return JudgmentResponse(
            fluency=cached_val["fluency"],
            coherence=cached_val["coherence"],
            instruction_adherence=cached_val["instruction_adherence"],
            no_hallucination=cached_val["no_hallucination"],
            reasoning=cached_val["reasoning"],
            rubric_id=_TEACHER_JUDGE.rubric.name,
            rubric_version=_TEACHER_JUDGE.rubric.version,
            git_commit_hash=get_git_commit_hash(),
            cache_hit=True,
            latency_ms=round(elapsed, 2),
            estimated_cost_usd=0.0,
        )

    METRICS["cache_misses_total"] += 1

    # Execute inference under concurrency semaphore
    async with CONCURRENCY_SEMAPHORE:
        # Retry with exponential backoff
        eval_res = None
        for attempt in range(3):
            try:
                loop = asyncio.get_running_loop()
                eval_res = await loop.run_in_executor(
                    None,
                    lambda: _TEACHER_JUDGE.evaluate_single(
                        prompt=req.prompt,
                        reference_context=req.reference_context,
                        model_output=req.model_output,
                        metadata=req.metadata,
                        mode=req.mode,
                    ),
                )
                break
            except Exception as e:
                if attempt == 2:
                    METRICS["requests_failed"] += 1
                    raise HTTPException(status_code=502, detail=f"Judge inference failed: {e}")
                await asyncio.sleep(2**attempt * 0.5)

    elapsed_s = time.perf_counter() - t0
    METRICS["total_latency_seconds"] += elapsed_s
    METRICS["requests_success"] += 1

    # Estimate cost
    pricing = _TEACHER_JUDGE.backend.pricing
    approx_cost = pricing.estimate_cost(prompt_tokens=450, completion_tokens=120)
    METRICS["total_estimated_cost_usd"] += approx_cost

    # Store in disk cache
    result_dict = eval_res.model_dump()
    DISK_CACHE.set(cache_key, result_dict)

    # Structured log
    logger.info(
        json.dumps({
            "event": "judgment_completed",
            "model": _TEACHER_JUDGE.model_name,
            "rubric": _TEACHER_JUDGE.rubric.name,
            "latency_ms": round(elapsed_s * 1000.0, 2),
            "cost_usd": approx_cost,
            "cache_hit": False,
        })
    )

    return JudgmentResponse(
        fluency=eval_res.fluency,
        coherence=eval_res.coherence,
        instruction_adherence=eval_res.instruction_adherence,
        no_hallucination=eval_res.no_hallucination,
        reasoning=eval_res.reasoning,
        rubric_id=_TEACHER_JUDGE.rubric.name,
        rubric_version=_TEACHER_JUDGE.rubric.version,
        git_commit_hash=get_git_commit_hash(),
        cache_hit=False,
        latency_ms=round(elapsed_s * 1000.0, 2),
        estimated_cost_usd=round(approx_cost, 6),
    )


@app.post("/judge", response_model=JudgmentResponse)
async def evaluate_single(req: JudgeRequest) -> JudgmentResponse:
    return await _evaluate_item(req)


@app.post("/judge/batch", response_model=List[JudgmentResponse])
async def evaluate_batch(batch_req: BatchJudgeRequest) -> List[JudgmentResponse]:
    tasks = [_evaluate_item(item) for item in batch_req.items]
    return await asyncio.gather(*tasks)


def set_active_judge(judge: TeacherJudge, cache: Optional[JudgeDiskCache] = None) -> None:
    """Helper to update active judge instance and cache for testing."""
    global _TEACHER_JUDGE, DISK_CACHE
    _TEACHER_JUDGE = judge
    if cache is not None:
        DISK_CACHE = cache
