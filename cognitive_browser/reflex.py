"""
Reflex Core: Fast local Natural Language Inference (NLI) on Accessibility Trees.
Evaluates candidate DOM elements using cross-encoder/nli-deberta-v3-small.
Consumes 0 LLM tokens, executes in sub-50ms on GPU/MPS/CPU.
"""

import re
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import get_confidence_threshold


class AriaElementParser:
    """Parses official page.aria_snapshot() YAML into structured candidate objects."""

    INTERACTIVE_ROLES = {
        "button",
        "link",
        "combobox",
        "textbox",
        "searchbox",
        "checkbox",
        "radio",
        "tab",
        "menuitem",
        "dialog",
        "alertdialog",
    }

    PATTERN = re.compile(r'^\s*-\s*([a-zA-Z0-9_-]+)(?:\s+"([^"]*)")?(?::\s*(.*))?$')

    @classmethod
    def parse_snapshot(
        cls, snapshot_yaml: str, max_candidates: int = 60, require_input_only: bool = False
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        lines = snapshot_yaml.splitlines()
        for line in lines:
            clean = line.strip()
            if not clean.startswith("-"):
                continue

            # Trailing URL metadata for preceding link candidate
            if clean.startswith("- /url:") and candidates:
                u = clean.replace("- /url:", "").strip().strip('"').strip("'")
                candidates[-1]["url"] = u
                continue

            # Trailing placeholder metadata for preceding input candidate
            if clean.startswith("- /placeholder:") and candidates:
                ph = clean.replace("- /placeholder:", "").strip().strip('"').strip("'")
                candidates[-1]["placeholder"] = ph
                candidates[-1]["text"] = f"{candidates[-1]['text']} (placeholder: {ph})"
                continue

            # Unwrapping single-quoted YAML keys (e.g. - 'link "name: with colon"':)
            clean = re.sub(r"^-\s*'(.*)'\s*:?$", r"- \1", clean)

            m = cls.PATTERN.match(clean)
            if not m:
                continue

            role = m.group(1).lower()
            name = (m.group(2) or "").strip()
            text_val = (m.group(3) or "").strip()
            label = name if name else text_val

            # Strip Private Use Area (PUA) font icons (both raw unicode and literal \\u escapes)
            clean_label = re.sub(r"[\ue000-\uf8ff\uf000-\uffff]|\\u[e-fE-F][0-9a-fA-F]{3}", "", label).strip()
            if clean_label:
                label = clean_label

            if require_input_only and role not in ["textbox", "combobox", "searchbox", "spinbutton"]:
                continue

            if role in cls.INTERACTIVE_ROLES and label:
                candidates.append(
                    {
                        "ref": f"aria-{len(candidates)}",
                        "role": role,
                        "tag": role,
                        "name": label[:120],
                        "text": label[:120],
                        "raw": clean,
                    }
                )

            if len(candidates) >= max_candidates:
                break

        return candidates


class ReflexCore:
    """
    Discriminative NLI Decision Engine.
    Computes entailment probabilities between candidate elements and natural language instructions.
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

        print(f"[ReflexCore] Loading model '{model_name}' on device {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        self.entail_idx = next(k for k, v in id2label.items() if "entail" in v.lower())
        self.contra_idx = next(k for k, v in id2label.items() if "contra" in v.lower())

        self._warmup()

    def _warmup(self) -> None:
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
        Evaluates candidate elements in parallel using NLI.
        Returns the top matching element, recommended action, and confidence score.
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

        # Build premise-hypothesis pairs
        pairs = []
        for c in candidates:
            parts = [f"role: {c.get('role')}"]
            if c.get("name"):
                parts.append(f"name: '{c['name']}'")
            if c.get("text") and c.get("text") != c.get("name"):
                parts.append(f"text: '{c['text']}'")
            if c.get("url"):
                parts.append(f"url: '{c['url']}'")

            premise = f"Interactive web element: {', '.join(parts)}."
            hypothesis = f"This element represents, opens, or is used to {intent}."
            pairs.append((premise, hypothesis))

        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits

            binary_logits = logits[:, [self.contra_idx, self.entail_idx]]
            binary_probs = torch.softmax(binary_logits, dim=-1)[:, 1]

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Lexical stem and role-matching boost
        def _compute_lexical_boost(intent_str: str, cand_role: str, cand_name: str) -> float:
            intent_lower = intent_str.lower()
            name_lower = (cand_name or "").lower()
            role_lower = (cand_role or "").lower()

            boost = 0.0

            # 1. Role keyword matching
            wants_button = any(w in intent_lower for w in ["przycisk", "guzik", "button", "btn", "submit"])
            wants_link = any(w in intent_lower for w in ["link", "odnośnik", "href"])
            wants_input = any(w in intent_lower for w in ["pole", "wpisz", "tekst", "textbox", "input", "wyszukiwark", "searchbox"])

            if wants_button:
                if role_lower == "button":
                    boost += 0.08
                else:
                    boost -= 0.15
            elif wants_link:
                if role_lower == "link":
                    boost += 0.08
                else:
                    boost -= 0.15
            elif wants_input:
                if role_lower in ["textbox", "searchbox", "combobox"]:
                    boost += 0.08
                else:
                    boost -= 0.15

            # 2. Token / stem matching between intent and element name
            intent_words = set(re.findall(r"\w{3,}", intent_lower))
            name_words = set(re.findall(r"\w{3,}", name_lower))

            stop_words = {"lub", "oraz", "dla", "sie", "się", "jest", "albo", "przez", "jego", "pod", "nad", "the", "and", "for", "with"}
            meaningful_intent = {w for w in intent_words if w not in stop_words}
            meaningful_name = {w for w in name_words if w not in stop_words}

            def check_stems(w1: str, w2: str) -> bool:
                min_len = min(len(w1), len(w2))
                if min_len >= 4:
                    return w1[:4] == w2[:4] or (w1[:min_len - 1] == w2[:min_len - 1])
                return w1 in w2 or w2 in w1

            stem_matches = 0
            for iw in meaningful_intent:
                if any(check_stems(iw, nw) for nw in meaningful_name):
                    stem_matches += 1

            if stem_matches > 0:
                boost += min(0.20, stem_matches * 0.10)
            elif len(meaningful_intent) > 0 and len(meaningful_name) > 0:
                boost -= 0.05

            return boost

        scored_candidates: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            raw_score = float(binary_probs[i].item())
            boost = _compute_lexical_boost(intent, c.get("role") or "", c.get("name") or "")
            unclamped_score = raw_score + boost
            scored_candidates.append(
                {
                    "ref": c.get("ref", "NONE"),
                    "role": c.get("role"),
                    "name": c.get("name"),
                    "url": c.get("url"),
                    "_sort_score": unclamped_score,
                    "confidence_score": float(round(max(0.0, min(1.0, unclamped_score)), 4)),
                }
            )

        scored_candidates.sort(key=lambda x: float(x.get("_sort_score") or 0.0), reverse=True)
        best = scored_candidates[0]
        best_score = float(best["confidence_score"])

        should_escalate = best_score < confidence_threshold
        target_ref = str(best["ref"]) if not should_escalate else "NONE"

        # Determine Playwright action
        role = best.get("role", "")
        if role in ["textbox", "searchbox", "combobox"]:
            action = "fill"
        elif role == "checkbox":
            action = "check"
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
