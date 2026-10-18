"""
Configuration settings for Cognitive Browser.
Supports environment variables and programmatic overrides.
"""

import os


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


def get_reflex_model(default: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> str:
    """Returns the model identifier for Reflex Core (Multilingual Reranker / ModernBERT / DeBERTa / Jina)."""
    return os.getenv("COGNITIVE_BROWSER_MODEL", default)


def get_rerank_api_url() -> str:
    """Returns the remote Reranker API endpoint (defaults to Jina AI /v1/rerank if JINA_API_KEY is set)."""
    url = os.getenv("RERANK_API_URL", "").strip()
    if url:
        return url
    if os.getenv("JINA_API_KEY", "").strip():
        return "https://api.jina.ai/v1/rerank"
    return ""


def get_rerank_api_key() -> str:
    """Returns the API key for remote Reranker API (RERANK_API_KEY or JINA_API_KEY)."""
    return (os.getenv("RERANK_API_KEY") or os.getenv("JINA_API_KEY") or "").strip()


def get_ollama_url() -> str:
    """Returns the active Ollama API generation endpoint."""
    return os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


def get_ollama_model() -> str:
    """Returns the default model identifier for Ollama escalation."""
    return os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")


def get_capsolver_api_key() -> str:
    """Returns CapSolver API key from env or app .env."""
    key = os.getenv("CAPSOLVER_API_KEY", "")
    if key:
        return key
    app_env = os.path.expanduser("~/.app/.env")
    if os.path.exists(app_env):
        try:
            with open(app_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("CAPSOLVER_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""
