"""
DeBERTa Decision Router (Reflex Core Model).
Utilizes Natural Language Inference (NLI) with DeBERTa-v3 on GPU/MPS/CPU.
Zero generative LLM tokens consumed. Sub-50ms inference on modern hardware.
"""

import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import get_confidence_threshold


class DebertaDecisionRouter:
    """
    Discriminative decision router based on DeBERTa-v3 NLI.
    Directly computes entailment probabilities between user intent
    and candidates in the DOM / Accessibility tree.
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

        print(f"[DebertaRouter] Loading model '{model_name}' on device {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        self.entail_idx = next(k for k, v in id2label.items() if "entail" in v.lower())
        self.contra_idx = next(k for k, v in id2label.items() if "contra" in v.lower())

        self._warmup()

    def _warmup(self):
        with torch.no_grad():
            dummy = self.tokenizer(
                ["Web element: button"], ["This element is used to submit."], return_tensors="pt"
            ).to(self.device)
            _ = self.model(**dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def evaluate_candidates(
        self, intent: str, candidates: List[Dict[str, Any]], confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates all candidate elements in parallel using NLI (entailment vs contradiction).
        Returns the top matching element, inferred action, and inference latency.
        """
        if confidence_threshold is None:
            confidence_threshold = get_confidence_threshold()

        if not candidates:
            return {
                "target_ref": "NONE",
                "tool_action": "escalate",
                "confidence": 0.0,
                "should_escalate": True,
                "latency_ms": 0.0,
                "ranked_candidates": [],
            }

        start_time = time.perf_counter()

        # Build premise-hypothesis pairs (Premise: DOM element, Hypothesis: user goal)
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

        # Vectorized batch inference on target device
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Zero-Shot NLI formulation (Yin et al. / Hugging Face):
            # Softmax between Entailment and Contradiction logits
            binary_logits = logits[:, [self.contra_idx, self.entail_idx]]
            binary_probs = torch.softmax(binary_logits, dim=-1)[:, 1]

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start_time) * 1000

        scored_candidates: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            score = float(binary_probs[i].item())
            scored_candidates.append(
                {
                    "ref": c.get("ref", "NONE"),
                    "tag": c.get("tag"),
                    "text": c.get("text"),
                    "role": c.get("role"),
                    "type": c.get("type"),
                    "name": c.get("name"),
                    "confidence_score": float(round(score, 4)),
                }
            )

        scored_candidates.sort(key=lambda x: float(x.get("confidence_score") or 0.0), reverse=True)
        best = scored_candidates[0]
        best_score = float(best["confidence_score"])

        # Determine Playwright action
        if best_score < confidence_threshold:
            action = "escalate"
            should_escalate = True
            target_ref = "NONE"
        else:
            should_escalate = False
            target_ref = str(best["ref"])
            if best["tag"] in ["input", "textarea"] and best.get("type") not in [
                "submit",
                "button",
                "checkbox",
                "radio",
            ]:
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
            "ranked_candidates": scored_candidates,
        }
