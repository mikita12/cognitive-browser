"""
Cognitive Browser (v0.2.0): High-speed, token-efficient browser automation MCP server.
Combines local zero-token NLI (Reflex Core) with orchestrating LLM escalation.
"""

from .browser import BrowserEngine
from .config import get_confidence_threshold
from .reflex import AriaElementParser, ReflexCore
from .security import (
    SecurityError,
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
    "AriaElementParser",
    "DebertaDecisionRouter",
    "ApiDecisionRouter",
    "get_decision_router",
    "AgentCore",
    "validate_url",
    "setup_ssrf_route_guard",
    "is_exempt_origin",
    "SecurityError",
    "mask_sensitive_value",
    "get_confidence_threshold",
]
