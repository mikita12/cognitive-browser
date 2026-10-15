"""
ApiDecisionRouter: Chmurowa alternatywa dla lokalnej DeBERTy (Zero-GPU Decision Router).
Obsługuje prawdziwe modele decyzyjne/dyskryminacyjne przez API:
1. Hugging Face Inference API (dokładnie nasza DeBERTa-v3 NLI w chmurze przez HF_TOKEN)
2. Cohere Rerank API (profesjonalny cross-encoder przez COHERE_API_KEY)
3. Fallback na szybkie LLMy (OpenAI / Groq / OpenRouter)
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Optional


class ApiDecisionRouter:
    """
    Router decyzyjny oparty o API chmurowe.
    Wspiera zarówno natywne cross-encodery (HuggingFace DeBERTa, Cohere),
    jak i fallback na modele generatywne.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        # 1. Priorytet: Prawdziwy model DeBERTa hostowany na Hugging Face (darmowy token HF)
        if os.getenv("HF_TOKEN") or provider == "huggingface":
            self.provider = "huggingface"
            self.api_key = api_key or os.getenv("HF_TOKEN")
            self.model = model or "cross-encoder/nli-deberta-v3-small"
            self.base_url = f"https://api-inference.huggingface.co/models/{self.model}"
            print(f"[ApiDecisionRouter] Aktywny tryb: Prawdziwa DeBERTa-v3 NLI na Hugging Face API ({self.model})")

        # 2. Priorytet: Dedykowany Cross-Encoder / Reranker (Cohere)
        elif os.getenv("COHERE_API_KEY") or provider == "cohere":
            self.provider = "cohere"
            self.api_key = api_key or os.getenv("COHERE_API_KEY")
            self.model = model or "rerank-v3.5"
            self.base_url = "https://api.cohere.com/v2/rerank"
            print(f"[ApiDecisionRouter] Aktywny tryb: Dedykowany Cross-Encoder Cohere Rerank ({self.model})")

        # 3. Fallback: Groq (szybki LLM)
        elif os.getenv("GROQ_API_KEY") or provider == "groq":
            self.provider = "groq"
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model = model or "llama-3.3-70b-versatile"
            print(f"[ApiDecisionRouter] Aktywny tryb: Groq Fast LLM ({self.model})")

        # 4. Fallback: OpenAI
        elif os.getenv("OPENAI_API_KEY") or provider == "openai":
            self.provider = "openai"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.model = model or "gpt-4o-mini"
            print(f"[ApiDecisionRouter] Aktywny tryb: OpenAI ({self.model})")

        else:
            self.provider = "openai"
            self.api_key = api_key or os.getenv("ROUTER_API_KEY", "")
            self.base_url = os.getenv("ROUTER_BASE_URL", "https://api.openai.com/v1/chat/completions")
            self.model = model or os.getenv("ROUTER_MODEL", "gpt-4o-mini")
            print(f"[ApiDecisionRouter] Aktywny tryb domyślny: {self.provider} ({self.model})")

    def evaluate_candidates(
        self,
        intent: str,
        candidates: List[Dict[str, Any]],
        confidence_threshold: float = 0.50
    ) -> Dict[str, Any]:
        if not candidates:
            return {
                "target_ref": "NONE",
                "tool_action": "escalate",
                "confidence": 0.0,
                "should_escalate": True,
                "latency_ms": 0.0,
                "ranked_candidates": []
            }

        if self.provider == "huggingface":
            return self._evaluate_huggingface_nli(intent, candidates, confidence_threshold)
        elif self.provider == "cohere":
            return self._evaluate_cohere_rerank(intent, candidates, confidence_threshold)
        else:
            return self._evaluate_generic_llm(intent, candidates, confidence_threshold)

    def _evaluate_huggingface_nli(self, intent: str, candidates: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
        """
        Natywne wnioskowanie NLI na serwerach Hugging Face (Dokładnie ta sama DeBERTa-v3).
        """
        t0 = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Przygotowanie par Premisa -> Hipoteza
        hypothesis = f"This element represents, opens, or is used to {intent}."
        pairs = []
        for c in candidates[:40]:
            premise = f"Interactive web element: role: {c.get('role')}, name: '{c.get('name')}', text: '{c.get('text')}'."
            pairs.append({"text": premise, "text_pair": hypothesis})

        try:
            resp = requests.post(self.base_url, headers=headers, json={"inputs": pairs}, timeout=8)
            data = resp.json()

            # Hugging Face NLI zwraca listę etykiet ze score dla każdej pary
            scored = []
            for idx, res_item in enumerate(data):
                cand = candidates[idx]
                # Znajdź prawdopodobieństwo entailment
                entail_score = next((x["score"] for x in res_item if "entail" in x.get("label", "").lower()), 0.0)
                scored.append({"cand": cand, "score": entail_score})

            scored.sort(key=lambda x: x["score"], reverse=True)
            best = scored[0]["cand"]
            confidence = round(scored[0]["score"], 4)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            action = "fill" if best["role"] in ["textbox", "combobox", "searchbox"] else "click"
            should_escalate = confidence < threshold

            return {
                "target_ref": best["ref"] if not should_escalate else "NONE",
                "tool_action": action if not should_escalate else "escalate",
                "confidence": confidence,
                "should_escalate": should_escalate,
                "latency_ms": latency_ms,
                "ranked_candidates": [s["cand"] for s in scored]
            }
        except Exception as e:
            return self._fallback_error(candidates, str(e), time.perf_counter() - t0)

    def _evaluate_cohere_rerank(self, intent: str, candidates: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
        """
        Wnioskowanie dyskryminacyjne przez Cohere Rerank API (Cross-Encoder).
        """
        t0 = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        docs = [f"role: {c['role']}, name: {c['name']}, info: {c.get('text', '')}" for c in candidates[:50]]
        
        payload = {
            "model": self.model,
            "query": f"Find button or input to {intent}",
            "documents": docs,
            "top_n": len(docs)
        }

        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=6)
            data = resp.json()
            results = data.get("results", [])

            if not results:
                return self._fallback_error(candidates, "Brak wyników z Cohere", time.perf_counter() - t0)

            best_idx = results[0]["index"]
            confidence = round(results[0]["relevance_score"], 4)
            best = candidates[best_idx]
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            action = "fill" if best["role"] in ["textbox", "combobox", "searchbox"] else "click"
            should_escalate = confidence < threshold

            ranked = [candidates[r["index"]] for r in results]

            return {
                "target_ref": best["ref"] if not should_escalate else "NONE",
                "tool_action": action if not should_escalate else "escalate",
                "confidence": confidence,
                "should_escalate": should_escalate,
                "latency_ms": latency_ms,
                "ranked_candidates": ranked
            }
        except Exception as e:
            return self._fallback_error(candidates, str(e), time.perf_counter() - t0)

    def _evaluate_generic_llm(self, intent: str, candidates: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
        """
        Awaryjny fallback na szybkie zapytanie JSON do OpenAI / Groq.
        """
        t0 = time.perf_counter()
        cands_snippet = "\n".join([f"- {c['ref']}: [{c['role']}] \"{c['name']}\"" for c in candidates[:40]])

        prompt = f"""You are a fast reflex decision router for a browser agent.
Goal: "{intent}".
Candidates:
{cands_snippet}

Output JSON ONLY: {{"chosen_ref": "aria-X", "confidence": 0.95}}
"""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=6)
            res = json.loads(resp.json()["choices"][0]["message"]["content"])
            chosen_ref = res.get("chosen_ref", "NONE")
            confidence = float(res.get("confidence", 0.0))
            matched = [c for c in candidates if c["ref"] == chosen_ref]
            best = matched[0] if matched else candidates[0]
            action = "fill" if best["role"] in ["textbox", "combobox", "searchbox"] else "click"

            return {
                "target_ref": chosen_ref,
                "tool_action": action,
                "confidence": confidence,
                "should_escalate": confidence < threshold or chosen_ref == "NONE",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "ranked_candidates": [best] + [c for c in candidates if c["ref"] != chosen_ref]
            }
        except Exception as e:
            return self._fallback_error(candidates, str(e), time.perf_counter() - t0)

    def _fallback_error(self, candidates: List[Dict[str, Any]], error_msg: str, elapsed_s: float) -> Dict[str, Any]:
        return {
            "target_ref": "NONE",
            "tool_action": "escalate",
            "confidence": 0.0,
            "should_escalate": True,
            "latency_ms": round(elapsed_s * 1000, 2),
            "ranked_candidates": candidates,
            "error": error_msg
        }
