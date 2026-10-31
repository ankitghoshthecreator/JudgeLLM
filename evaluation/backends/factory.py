"""
Factory and configuration parsing for JudgeLLM backends.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from evaluation.backends.base import BackendPricing, JudgeBackend
from evaluation.backends.groq_backend import GroqJudgeBackend
from evaluation.backends.mock_backend import MockJudgeBackend
from evaluation.backends.openai_compatible import OpenAICompatibleBackend

logger = logging.getLogger(__name__)


def create_backend(
    backend_type: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    pricing: Optional[Dict[str, float]] = None,
    mock_mode: bool = False,
    **kwargs: Any,
) -> JudgeBackend:
    """
    Factory function to initialize a JudgeBackend based on explicit parameters or environment variables.
    """
    pricing_obj = BackendPricing(**pricing) if pricing else None

    # Determine backend type
    btype = (backend_type or os.getenv("JUDGE_BACKEND", "")).lower().strip()

    if mock_mode or btype == "mock":
        return MockJudgeBackend(model_name=model_name or "mock-judge", pricing=pricing_obj, **kwargs)

    # If backend_type is explicitly groq, or no backend is specified but GROQ_API_KEY is present
    if btype in ("groq", "") and (api_key or os.getenv("GROQ_API_KEY")) and not base_url:
        try:
            return GroqJudgeBackend(
                model_name=model_name,
                api_key=api_key,
                pricing=pricing_obj,
                **kwargs,
            )
        except Exception as e:
            logger.warning("Failed to initialize GroqJudgeBackend (%s). Falling back to Mock.", e)
            return MockJudgeBackend(model_name=model_name or "mock-judge", pricing=pricing_obj, **kwargs)

    # If openai_compatible, vllm, ollama, or base_url is specified
    if btype in ("openai", "openai_compatible", "vllm", "ollama", "local") or base_url:
        return OpenAICompatibleBackend(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            pricing=pricing_obj,
            **kwargs,
        )

    # Default fallback
    logger.info("No external backend API key or URL configured. Instantiating MockJudgeBackend.")
    return MockJudgeBackend(model_name=model_name or "mock-judge", pricing=pricing_obj, **kwargs)


def load_backend_from_config(config_path_or_dict: Union[str, Path, Dict[str, Any]]) -> JudgeBackend:
    """Loads a JudgeBackend from a YAML config file or dictionary."""
    if isinstance(config_path_or_dict, (str, Path)):
        p = Path(config_path_or_dict)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config_path_or_dict

    backend_cfg = cfg.get("judge_backend", cfg)
    return create_backend(
        backend_type=backend_cfg.get("type"),
        model_name=backend_cfg.get("model_name"),
        api_key=backend_cfg.get("api_key"),
        base_url=backend_cfg.get("base_url"),
        pricing=backend_cfg.get("pricing"),
        mock_mode=backend_cfg.get("mock_mode", False),
    )
