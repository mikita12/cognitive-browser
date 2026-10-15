"""
RouterFactory: Decision Router Factory (Local DeBERTa vs Cloud API).
Dynamically selects the optimal router based on user hardware and configuration.
"""

import os
from typing import Optional, Union

from .api_router import ApiDecisionRouter
from .deberta_router import DebertaDecisionRouter


def get_decision_router(
    mode: Optional[str] = None, device: Optional[str] = None, api_key: Optional[str] = None
) -> Union[DebertaDecisionRouter, ApiDecisionRouter]:
    """
    Returns the appropriate decision router.

    Selection priority:
    1. Explicit 'mode' parameter ('deberta' or 'api')
    2. Environment variable ROUTER_MODE ('deberta' or 'api')
    3. If GROQ_API_KEY / OPENAI_API_KEY / HF_TOKEN is present and ROUTER_MODE=api -> ApiDecisionRouter
    4. Default: DebertaDecisionRouter (100% offline, free, 0 LLM tokens)
    """
    raw_mode = mode or os.getenv("ROUTER_MODE") or "deberta"
    selected_mode = raw_mode.lower()

    if selected_mode == "api":
        return ApiDecisionRouter(api_key=api_key)

    # Default: free local mode with DeBERTa-v3
    try:
        return DebertaDecisionRouter(device=device)
    except Exception as e:
        print(f"[RouterFactory] Warning: Could not load local DeBERTa ({e}).")
        # If DeBERTa fails and an API key is available -> automatic graceful fallback to cloud API
        api_keys = ["HF_TOKEN", "COHERE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"]
        if any(os.getenv(k) for k in api_keys) or api_key:
            print("[RouterFactory] Switching to emergency API fallback (ApiDecisionRouter)...")
            return ApiDecisionRouter(api_key=api_key)
        raise e
