"""
RouterFactory: Fabryka routerów decyzyjnych (DeBERTa Local Free vs API Cloud).
Automatycznie dobiera optymalny router w zależności od konfiguracji i sprzętu użytkownika.
"""

import os
from typing import Optional, Union

try:
    from .deberta_router import DebertaDecisionRouter
    from .api_router import ApiDecisionRouter
except ImportError:
    from deberta_router import DebertaDecisionRouter
    from api_router import ApiDecisionRouter


def get_decision_router(
    mode: Optional[str] = None,
    device: Optional[str] = None,
    api_key: Optional[str] = None
) -> Union[DebertaDecisionRouter, ApiDecisionRouter]:
    """
    Zwraca odpowiedni router decyzyjny.
    
    Priorytet wyboru:
    1. Jawny parametr mode ('deberta' lub 'api')
    2. Zmienna środowiskowa ROUTER_MODE ('deberta' lub 'api')
    3. Jeśli obecny jest GROQ_API_KEY lub OPENAI_API_KEY i użytkownik ustawił ROUTER_MODE=api -> ApiDecisionRouter
    4. Domyślnie: DebertaDecisionRouter (100% lokalny, w 100% darmowy, 0 tokenów)
    """
    selected_mode = (mode or os.getenv("ROUTER_MODE", "deberta")).lower()

    if selected_mode == "api":
        return ApiDecisionRouter(api_key=api_key)

    # Domyślny, darmowy tryb lokalny z DeBERTa-v3
    try:
        return DebertaDecisionRouter(device=device)
    except Exception as e:
        print(f"[RouterFactory] Ostrzeżenie: Nie udało się załadować lokalnej DeBERTy ({e}).")
        # Jeśli DeBERTa zawiodła, a mamy klucz API -> automatyczny bezpieczny fallback na API
        api_keys = ["HF_TOKEN", "COHERE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"]
        if any(os.getenv(k) for k in api_keys) or api_key:
            print("[RouterFactory] Przełączam na tryb awaryjny API (ApiDecisionRouter)...")
            return ApiDecisionRouter(api_key=api_key)
        raise e
