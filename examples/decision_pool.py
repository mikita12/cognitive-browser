"""
Parallel Decision Worker Pool (System 1 Ensemble / Parallel Evaluator).
Executes multiple concurrent decision questions across local fast models.
"""

import asyncio
import time
import httpx
from typing import List, Dict, Any, Optional

class DecisionWorker:
    """
    Pojedynczy pracownik decyzyjny odpytujący lokalną Ollamę.
    Wymusza schemat wyjścia (constrained decoding / JSON schema).
    """
    def __init__(self, model: str = "llama3.2:1b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.endpoint = f"{host}/api/chat"

    async def decide_action(
        self,
        client: httpx.AsyncClient,
        intent: str,
        candidates_text: str,
        valid_refs: List[str]
    ) -> Dict[str, Any]:
        """
        Ocenia stan i mapuje intencję użytkownika na konkretny element i narzędzie Playwrighta.
        """
        start_time = time.perf_counter()

        system_prompt = (
            "You are a low-latency Web Navigation Decision Router (System 1).\n"
            "Your task is to analyze the user's intent and select the single best interactive element ref to interact with.\n"
            "If no element fits or there is high ambiguity, set should_escalate to true."
        )

        user_content = (
            f"USER INTENT: {intent}\n\n"
            f"AVAILABLE INTERACTIVE ELEMENTS:\n{candidates_text}\n\n"
            f"Select the exact target ref and tool action."
        )

        # JSON schema z twardym enumem dostępnych refów!
        # Dzięki temu model fizycznie nie może wymyślić nieistniejącego elementu (zero halucynacji selektora)
        allowed_refs = valid_refs + ["NONE"]

        json_schema = {
            "type": "object",
            "properties": {
                "target_ref": {
                    "type": "string",
                    "enum": allowed_refs
                },
                "tool_action": {
                    "type": "string",
                    "enum": ["click", "fill", "select", "wait", "escalate"]
                },
                "fill_value": {
                    "type": "string"
                },
                "confidence": {
                    "type": "number"
                },
                "should_escalate": {
                    "type": "boolean"
                }
            },
            "required": ["target_ref", "tool_action", "confidence", "should_escalate"]
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "format": json_schema,
            "options": {
                "temperature": 0.0,
                "num_predict": 100
            },
            "stream": False
        }

        try:
            res = await client.post(self.endpoint, json=payload, timeout=30.0)
            res.raise_for_status()
            data = res.json()
            latency = (time.perf_counter() - start_time) * 1000

            import json
            parsed = json.loads(data["message"]["content"])
            return {
                "success": True,
                "model": self.model,
                "latency_ms": round(latency, 2),
                "eval_count": data.get("eval_count", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "decision": parsed
            }
        except Exception as e:
            return {
                "success": False,
                "model": self.model,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "error": str(e)
            }


class ParallelDecisionPool:
    """
    Pula modeli decyzyjnych działających równolegle (Consensus / Voting / Multi-Question).
    Pozwala na jednoczesne odpalenie kilku pytań lub kilku modeli w celu weryfikacji.
    """
    def __init__(self, models: Optional[List[str]] = None):
        if not models:
            self.models = ["llama3.2:1b", "qwen2.5:1.5b"]
        else:
            self.models = models

    async def evaluate_intent(
        self,
        intent: str,
        candidates_text: str,
        valid_refs: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Odpytuje wszystkie modele w puli równolegle przez asyncio.gather.
        """
        async with httpx.AsyncClient() as client:
            tasks = [
                DecisionWorker(model=m).decide_action(client, intent, candidates_text, valid_refs)
                for m in self.models
            ]
            results = await asyncio.gather(*tasks)
            return results

    async def evaluate_multiple_intents(
        self,
        intents: List[str],
        candidates_text: str,
        valid_refs: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Odpytuje jedno zadanie per intencję (np. sprawdzanie kliknięcia, cookies, powiadomień) równolegle.
        """
        async with httpx.AsyncClient() as client:
            worker = DecisionWorker(model=self.models[0])
            tasks = [
                worker.decide_action(client, intent, candidates_text, valid_refs)
                for intent in intents
            ]
            results = await asyncio.gather(*tasks)
            return results
