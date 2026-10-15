"""
Cognitive Browser: Autonomous Web Agent with Reflex Core (DeBERTa NLI) and Cortex Engine Escalation.
"""

from .deberta_router import DebertaDecisionRouter
from .api_router import ApiDecisionRouter
from .router_factory import get_decision_router
from .agent_core import AgentCore, AriaElementParser
from .agent_orchestrator import AgentOrchestrator
from .security import validate_url, SecurityError, mask_sensitive_value
from .config import get_confidence_threshold

__version__ = "0.1.0"
__all__ = [
    "DebertaDecisionRouter",
    "ApiDecisionRouter",
    "get_decision_router",
    "AgentCore",
    "AriaElementParser",
    "AgentOrchestrator",
    "validate_url",
    "SecurityError",
    "mask_sensitive_value",
    "get_confidence_threshold",
]
