"""
Cognitive Browser: Autonomous Web Agent with Reflex Core (DeBERTa NLI) and Cortex Engine Escalation.
"""

from .agent_core import AgentCore, AriaElementParser
from .agent_orchestrator import AgentOrchestrator
from .api_router import ApiDecisionRouter
from .config import get_confidence_threshold
from .deberta_router import DebertaDecisionRouter
from .router_factory import get_decision_router
from .security import (
    SecurityError,
    is_exempt_origin,
    mask_sensitive_value,
    setup_ssrf_route_guard,
    validate_url,
)

__version__ = "0.1.0"
__all__ = [
    "DebertaDecisionRouter",
    "ApiDecisionRouter",
    "get_decision_router",
    "AgentCore",
    "AriaElementParser",
    "AgentOrchestrator",
    "validate_url",
    "setup_ssrf_route_guard",
    "is_exempt_origin",
    "SecurityError",
    "mask_sensitive_value",
    "get_confidence_threshold",
]
