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


BM25_CALIBRATED_THRESHOLD: float = 0.02


def get_bm25_threshold(default: float = BM25_CALIBRATED_THRESHOLD) -> float:
    """Returns the calibrated BM25 element scoring threshold for Arm C."""
    env_val = os.getenv("BM25_CONFIDENCE_THRESHOLD")
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


_CLI_COOKIES_ENABLED: bool = False


def set_cookies_enabled(enabled: bool = True) -> None:
    """Enables or disables system cookie extraction (activated via --enable-cookies CLI flag)."""
    global _CLI_COOKIES_ENABLED
    _CLI_COOKIES_ENABLED = enabled


def is_cookie_extraction_enabled() -> bool:
    """Returns True ONLY if the server was started with --enable-cookies or COGNITIVE_BROWSER_ENABLE_COOKIES=1."""
    if _CLI_COOKIES_ENABLED:
        return True
    env_val = (
        (os.getenv("COGNITIVE_BROWSER_ENABLE_COOKIES") or os.getenv("ENABLE_COOKIE_EXTRACTION") or "").strip().lower()
    )
    return env_val in ("1", "true", "yes", "on")


def get_capsolver_api_key() -> str:
    """Returns CapSolver API key from environment."""
    return os.getenv("CAPSOLVER_API_KEY", "").strip()
