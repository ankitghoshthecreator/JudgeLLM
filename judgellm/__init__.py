"""
JudgeLLM - Industrial-grade LLM Judge Reliability Toolkit
"""

from evaluation.backends import (
    BackendPricing,
    GroqJudgeBackend,
    JudgeBackend,
    MockJudgeBackend,
    OpenAICompatibleBackend,
    create_backend,
    load_backend_from_config,
)
from evaluation.teacher_judge import JudgeEvaluation, RubricConfig, TeacherJudge
from judgellm.audit import AuditReport, AxisAuditResult, audit_judge
from judgellm.cache import JudgeDiskCache, get_git_commit_hash
from judgellm.public_api import detect_bias, route

__version__ = "0.2.0"

__all__ = [
    "audit_judge",
    "route",
    "detect_bias",
    "AuditReport",
    "AxisAuditResult",
    "TeacherJudge",
    "JudgeEvaluation",
    "RubricConfig",
    "JudgeBackend",
    "BackendPricing",
    "GroqJudgeBackend",
    "OpenAICompatibleBackend",
    "MockJudgeBackend",
    "create_backend",
    "load_backend_from_config",
    "JudgeDiskCache",
    "get_git_commit_hash",
]
