"""
DeBERTa Decision Router (Prawdziwy Dyskryminacyjny Model System 1).
Wykorzystuje Natural Language Inference (NLI) na modelu DeBERTa-v3 na GPU.
Zero generowania tokenów tekstowych. Wnioskowanie w kilkadziesiąt milisekund na RTX 4070 Ti.
"""

import time
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DebertaDecisionRouter:
    """
    Dyskryminacyjny router decyzyjny oparty o DeBERTa-v3 NLI.
    Oblicza bezpośrednio prawdopodobieństwo wynikania (entailment)
    pomiędzy intencją użytkownika a elementami DOM.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[DebertaRouter] Ładowanie modelu '{model_name}' na urządzenie {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        self.entail_idx = next(k for k, v in id2label.items() if "entail" in v.lower())
        self.contra_idx = next(k for k, v in id2label.items() if "contra" in v.lower())

        self._warmup()

    def _warmup(self):
        with torch.no_grad():
            dummy = self.tokenizer(["Web element: button"], ["This element is used to submit."], return_tensors="pt").to(self.device)
            _ = self.model(**dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def evaluate_candidates(
        self,
        intent: str,
        candidates: List[Dict[str, Any]],
        confidence_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Ocenia równolegle wszystkich kandydatów przy użyciu NLI (entailment vs contradiction).
        Zwraca najlepszy pasujący element, wyliczoną akcję i czas wnioskowania.
        """
        if not candidates:
            return {
                "target_ref": "NONE",
                "tool_action": "escalate",
                "confidence": 0.0,
                "should_escalate": True,
                "latency_ms": 0.0,
                "ranked_candidates": []
            }

        start_time = time.perf_counter()

        # Budowanie par (Premisa: element DOM, Hipoteza: cel użytkownika)
        pairs = []
        for c in candidates:
            parts = [f"tag <{c.get('tag')}>", f"role: {c.get('role')}"]
            if c.get("text"):
                parts.append(f"text: '{c['text']}'")
            if c.get("type"):
                parts.append(f"type: {c['type']}")
            if c.get("name"):
                parts.append(f"name: {c['name']}")
            if c.get("id"):
                parts.append(f"id: {c['id']}")

            premise = f"Interactive web element: {', '.join(parts)}."
            hypothesis = f"This element represents, opens, or is used to {intent}."
            pairs.append((premise, hypothesis))

        # Wektorowe wnioskowanie równoległe (Batch Inference na GPU)
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Standardowy wzór Zero-Shot NLI (Yin et al. / Hugging Face):
            # Softmax pomiędzy logitem Entailment a Contradiction
            binary_logits = logits[:, [self.contra_idx, self.entail_idx]]
            binary_probs = torch.softmax(binary_logits, dim=-1)[:, 1]

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start_time) * 1000

        scored_candidates = []
        for i, c in enumerate(candidates):
            score = binary_probs[i].item()
            scored_candidates.append({
                "ref": c.get("ref"),
                "tag": c.get("tag"),
                "text": c.get("text"),
                "role": c.get("role"),
                "type": c.get("type"),
                "name": c.get("name"),
                "confidence_score": round(score, 4)
            })

        scored_candidates.sort(key=lambda x: x["confidence_score"], reverse=True)
        best = scored_candidates[0]
        best_score = best["confidence_score"]

        # Wyznaczenie akcji Playwright
        if best_score < confidence_threshold:
            action = "escalate"
            should_escalate = True
            target_ref = "NONE"
        else:
            should_escalate = False
            target_ref = best["ref"]
            if best["tag"] in ["input", "textarea"] and best.get("type") not in ["submit", "button", "checkbox", "radio"]:
                action = "fill"
            elif best.get("type") == "checkbox":
                action = "check"
            elif best["tag"] == "select":
                action = "select"
            else:
                action = "click"

        return {
            "target_ref": target_ref,
            "tool_action": action,
            "confidence": best_score,
            "should_escalate": should_escalate,
            "latency_ms": round(latency_ms, 2),
            "ranked_candidates": scored_candidates
        }
