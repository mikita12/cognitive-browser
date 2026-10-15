"""
Configuration settings for Cognitive Browser.
Supports environment variables and programmatic overrides.
"""

import os
from typing import Optional, List


def get_confidence_threshold(default: float = 0.50) -> float:
    """Returns the active Reflex Core confidence threshold."""
    env_val = os.getenv("REFLEX_CONFIDENCE_THRESHOLD")
    if env_val:
        try:
            val = float(env_val)
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
    return default


def get_ollama_url() -> str:
    """Returns the active Ollama API generation endpoint."""
    return os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


def get_ollama_model() -> str:
    """Returns the default model identifier for Ollama escalation."""
    return os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
