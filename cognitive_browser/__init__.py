"""
Cognitive Browser: Autonomous Web Agent with System 1 (DeBERTa NLI) and System 2 Escalation.
"""

from .deberta_router import DebertaDecisionRouter
from .api_router import ApiDecisionRouter
from .router_factory import get_decision_router
from .agent_core import AgentCore, AriaElementParser
from .agent_orchestrator import AgentOrchestrator

__version__ = "0.1.0"
__all__ = [
    "DebertaDecisionRouter",
    "ApiDecisionRouter",
    "get_decision_router",
    "AgentCore",
    "AriaElementParser",
    "AgentOrchestrator",
]
