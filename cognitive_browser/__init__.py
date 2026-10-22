"""
Cognitive Browser (v0.2.0): High-speed, token-efficient browser automation MCP server.
Combines local zero-token Cross-Encoder / NLI (Reflex Core) with RouteCache and orchestrating LLM escalation.
"""

from .browser import BrowserEngine
from .cache import RouteCache
from .captcha import CaptchaResolver
from .config import get_capsolver_api_key, get_confidence_threshold, get_reflex_model
from .cookies import extract_firefox_cookies, find_firefox_profiles, import_system_cookies
from .models import ActionCommand, CandidateElement, PageState, StepResult
from .orchestrator import BrowserOrchestrator
from .parser import AriaElementParser
from .payload import PayloadBinder
from .reflex import ReflexCore
from .security import (
    SecurityError,
    async_validate_url,
    is_exempt_origin,
    mask_sensitive_value,
    setup_ssrf_route_guard,
    validate_url,
)

# Backward-compatibility aliases
DebertaDecisionRouter = ReflexCore
AgentCore = BrowserEngine
ApiDecisionRouter = ReflexCore


def get_decision_router(device=None):
    return ReflexCore(device=device)


__version__ = "0.2.0"
__all__ = [
    "ReflexCore",
    "BrowserEngine",
    "RouteCache",
    "CaptchaResolver",
    "AriaElementParser",
    "PayloadBinder",
    "BrowserOrchestrator",
    "CandidateElement",
    "ActionCommand",
    "PageState",
    "StepResult",
    "DebertaDecisionRouter",
    "ApiDecisionRouter",
    "get_decision_router",
    "AgentCore",
    "validate_url",
    "async_validate_url",
    "setup_ssrf_route_guard",
    "is_exempt_origin",
    "SecurityError",
    "mask_sensitive_value",
    "get_confidence_threshold",
    "get_reflex_model",
    "get_capsolver_api_key",
    "extract_firefox_cookies",
    "find_firefox_profiles",
    "import_system_cookies",
]
