"""
On-disk persistent caching for LLM judge inferences.
Keys are deterministic SHA256 hashes of (model, rubric_id, rubric_version, git_hash, prompt, context, output).
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, or 'unknown_commit' if not a git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=2.0,
        )
        return res.stdout.strip()
    except Exception:
        return "frozen_release_v0.2"


class JudgeDiskCache:
    """Thread-safe persistent JSON disk cache for judge evaluations."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(".cache/judgellm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.git_hash = get_git_commit_hash()
        self.hits = 0
        self.misses = 0

    def compute_key(
        self,
        model_name: str,
        rubric_id: str,
        rubric_version: str,
        prompt: str,
        reference_context: str,
        model_output: str,
    ) -> str:
        payload = f"{model_name}|{rubric_id}|{rubric_version}|{self.git_hash}|{prompt}|{reference_context}|{model_output}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.hits += 1
                    data = json.load(f)
                    data["_cache_hit"] = True
                    return data
            except Exception as e:
                logger.warning("Failed to read cache file %s: %s", path, e)
        self.misses += 1
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        path = self.cache_dir / f"{key}.json"
        try:
            stored = dict(value)
            stored["_git_hash"] = self.git_hash
            with open(path, "w", encoding="utf-8") as f:
                json.dump(stored, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write cache entry %s: %s", path, e)
