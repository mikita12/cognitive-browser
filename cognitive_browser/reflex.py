"""
Reflex Core: Local Cross-Encoder / NLI neural scorer and Autonomous Action Planner.
Centralizes GPU mini-batch inference in `score_pairs()` and maps natural language intents
to standardized `ActionCommand` contracts with 0 LLM tokens.
"""

import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .captcha import CaptchaResolver
from .config import get_confidence_threshold, get_reflex_model
from .models import ActionCommand, ActionType, CandidateElement
from .parser import AriaElementParser as AriaElementParser
from .payload import PayloadBinder, _lexical_overlap_score


class ReflexCore:
    """
    Discriminative Decision Engine & Autonomous Action Planner.
    1. `score_pairs()`: Single centralized PyTorch GPU mini-batch inference method.
    2. `evaluate_candidates()`: Ranks CandidateElements against an instruction and produces an ActionCommand.
    3. `evaluate_page_satisfaction()`: Evaluates page text chunks against an informational goal.
    """

    # Universal legal/navigation boilerplate links skipped during autonomous goal navigation unless explicitly requested
    NOISE_LINK_PATTERNS = (
        "polityka prywatności",
        "privacy policy",
        "regulamin",
        "terms of service",
        "terms and conditions",
        "cookie policy",
        "skip to content",
        "przejdź do treści",
        "przejdź do stopki",
    )

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        if model_name is None:
            model_name = get_reflex_model()
        self.model_name = model_name

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[ReflexCore] Loading model '{model_name}' on device {self.device}...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        self.is_reranker = getattr(self.model.config, "num_labels", 1) == 1
        if not self.is_reranker:
            id2label = self.model.config.id2label
            self.entail_idx = next((k for k, v in id2label.items() if "entail" in v.lower()), 0)
            self.contra_idx = next((k for k, v in id2label.items() if "contra" in v.lower()), 2)

        self._warmup()

    def _warmup(self) -> None:
        self.score_pairs([["Submit form", "Web element: button 'Submit'"]])

    def score_pairs(self, pairs: List[List[str]], batch_size: int = 64) -> List[float]:
        """
        Single centralized GPU inference method.
        Takes a list of [text_a, text_b] pairs, sorts by length to minimize padding overhead,
        evaluates in mini-batches, and returns calibrated probabilities in [0.0, 1.0].
        """
        if not pairs:
            return []

        indexed_pairs = list(enumerate(pairs))
        indexed_pairs.sort(key=lambda x: len(x[1][0]) + len(x[1][1]))
        probs_ordered = [0.0] * len(pairs)

        with torch.no_grad():
            for b_start in range(0, len(indexed_pairs), batch_size):
                b_slice = indexed_pairs[b_start : b_start + batch_size]
                b_pairs = [item[1] for item in b_slice]
                b_indices = [item[0] for item in b_slice]

                inputs = self.tokenizer(
                    b_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits

                if self.is_reranker:
                    squeezed = logits.squeeze(-1)
                    if "mmarco" in self.model_name.lower():
                        b_probs = torch.sigmoid((squeezed + 4.5) / 2.0)
                    else:
                        b_probs = torch.sigmoid(squeezed)
                    if b_probs.ndim == 0:
                        b_probs = b_probs.unsqueeze(0)
                else:
                    binary_logits = logits[:, [self.contra_idx, self.entail_idx]]
                    b_probs = torch.softmax(binary_logits, dim=-1)[:, 1]

                for orig_idx, prob_val in zip(b_indices, b_probs, strict=False):
                    probs_ordered[orig_idx] = float(prob_val.item())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

        return probs_ordered

    @classmethod
    def is_noise_candidate(cls, candidate: Union[CandidateElement, Dict[str, Any]], intent: str) -> bool:
        """Returns True if the candidate is generic footer/legal boilerplate not requested by intent."""
        el = CandidateElement.from_dict(candidate)
        name_lower = el.name.lower()
        intent_lower = intent.lower()

        if not name_lower:
            return False

        if el.is_footer and not any(
            w in intent_lower for w in ("regulamin", "kontakt", "polityk", "stopk", "terms", "privacy")
        ):
            return True

        for pat in cls.NOISE_LINK_PATTERNS:
            if pat in name_lower and pat not in intent_lower:
                return True

        if el.url == "/" and "strona główna" not in intent_lower and "home" not in intent_lower:
            return True

        return False

    @staticmethod
    def extract_fill_value(intent: str) -> Optional[str]:
        """Extracts a search query or input value from a natural language instruction if present."""
        intent_clean = intent.strip()
        m_colon = re.search(
            r"(?:wpisz|wprowadź|wyszukaj|szukaj|type|fill|enter|search)[^:]{0,35}:\s*(.+?)(?:\s+i\s+(?:kliknij|naciśnij|wciśnij|zatwierdź)|$)",
            intent_clean,
            re.I,
        )
        if m_colon:
            val = m_colon.group(1).strip().strip("'\"")
            if val:
                return val

        m_quote = re.search(
            r"(?:wpisz|wprowadź|type|fill|enter|wybierz|select|choose)\s+['\"„”]([^'\"„”]{2,80})['\"„”]",
            intent_clean,
            re.I,
        )
        if m_quote:
            return m_quote.group(1).strip()

        return None

    @classmethod
    def infer_action_command(
        cls,
        intent: str,
        best_candidate: Union[CandidateElement, Dict[str, Any]],
        explicit_fill_value: Optional[str] = None,
        confidence: float = 1.0,
    ) -> ActionCommand:
        """Builds a strongly-typed ActionCommand from an instruction and the selected CandidateElement."""
        el = CandidateElement.from_dict(best_candidate)
        intent_lower = intent.lower()
        fill_val = explicit_fill_value if explicit_fill_value is not None else cls.extract_fill_value(intent)

        if any(w in intent_lower for w in ("przewiń", "scroll", "na dół", "w dół")):
            direction = "up" if ("gór" in intent_lower or "up" in intent_lower) else "down"
            return ActionCommand(
                action="scroll", target=el, direction=direction, value=direction, confidence=confidence
            )

        wants_select = any(w in intent_lower for w in ("wybierz", "select", "choose", "z listy", "dropdown", "opcję"))
        if el.role in ("combobox", "option") and (
            wants_select or (fill_val and not any(w in intent_lower for w in ("wpisz", "type", "fill")))
        ):
            return ActionCommand(action="select", target=el, value=fill_val, confidence=confidence)

        if el.is_input and fill_val:
            wants_submit = el.role == "searchbox" or any(
                w in intent_lower for w in ("szukaj", "wyszukaj", "search", "enter", "zatwierdź")
            )
            act: ActionType = "fill_and_submit" if wants_submit else "fill"
            return ActionCommand(action=act, target=el, value=fill_val, confidence=confidence)

        if el.is_boolean_control:
            uncheck = any(w in intent_lower for w in ("odznacz", "wyłącz", "uncheck", "disable"))
            return ActionCommand(
                action="uncheck" if uncheck else "check",
                target=el,
                confidence=confidence,
            )

        if any(w in intent_lower for w in ("najedź", "hover", "rozwiń menu")):
            return ActionCommand(action="hover", target=el, confidence=confidence)

        return ActionCommand(action="click", target=el, confidence=confidence)

    @classmethod
    def infer_action_plan(
        cls,
        intent: str,
        best_candidate: Union[CandidateElement, Dict[str, Any]],
        explicit_fill_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper returning ActionCommand as a dictionary."""
        return cls.infer_action_command(intent, best_candidate, explicit_fill_value=explicit_fill_value).to_dict()

    @staticmethod
    def is_goal_satisfied_on_page(goal: str, snapshot_yaml: str, page_title: str, page_url: str) -> bool:
        """
        Universal structural check for whether the current page already satisfies an informational goal.
        """
        if not snapshot_yaml or len(snapshot_yaml) < 200:
            return False

        snap_lower = snapshot_yaml.lower()
        goal_lower = goal.lower()
        url_lower = page_url.lower()

        # 1. Stop immediately on HTTP error / anti-bot block pages
        if any(
            err in snap_lower
            for err in (
                "403 error",
                "request blocked",
                "unusual traffic from your computer network",
                "access denied",
            )
        ):
            return True

        # 2. Respect explicit read-only instructions
        if any(
            ro in goal_lower
            for ro in (
                "nie klikaj",
                "pozostań na stronie",
                "przeczytaj",
                "odczytaj",
                "do not click",
                "stay on page",
            )
        ):
            return True

        # 3. Generic listing check: if goal seeks prices/offers/dimensions and page already displays multiple data points
        price_matches = len(re.findall(r"\b\d[\d\s.,]{0,6}\s*(?:zł|pln|eur|usd|\$|€)\b", snap_lower))
        unit_matches = len(re.findall(r"\b\d+\s*(?:mm|cm|kg|gb|tb|mhz)\b", snap_lower))
        seeks_data = any(
            w in goal_lower
            for w in ("cen", "ofert", "znajdź", "sprawdź", "porównaj", "wymiar", "gabaryt", "price", "offer", "find")
        )
        if seeks_data and (price_matches >= 3 or (unit_matches >= 2 and price_matches >= 1)):
            return True

        is_search_url = any(p in url_lower for p in ("szukaj", "search", "?q=", "&q="))
        if is_search_url and seeks_data and len(snapshot_yaml) > 1200:
            return True

        return False

    def evaluate_page_satisfaction(
        self,
        goal: str,
        snapshot_yaml: str,
        page_title: str = "",
        page_url: str = "",
        max_chunks: int = 16,
    ) -> Dict[str, Any]:
        """
        Evaluates page content sections in parallel on GPU via `score_pairs()` to determine
        whether the current page already satisfies `goal` and ranks the most relevant sections.
        """
        t0 = time.perf_counter()
        structural_hit = self.is_goal_satisfied_on_page(goal, snapshot_yaml, page_title, page_url)

        clean_lines = [
            line
            for line in snapshot_yaml.splitlines()
            if line.strip() and line.strip() not in ("- img", "- separator", "- img:", '- text: ""', "- group")
        ]

        if not clean_lines:
            return {
                "goal_satisfaction_rate": 0.0,
                "is_satisfied": False,
                "latency_ms": 0.0,
                "ranked_chunks": [],
            }

        window_size = 25
        stride = 20
        chunks: List[Dict[str, Any]] = []
        for idx, start in enumerate(range(0, len(clean_lines), stride)):
            if len(chunks) >= max_chunks:
                break
            window = clean_lines[start : start + window_size]
            text_block = "\n".join(window)
            chunks.append(
                {
                    "chunk_index": idx,
                    "start_line": start,
                    "text": text_block[:900],
                    "lines": window,
                }
            )

        pairs: List[List[str]] = []
        for c in chunks:
            doc = f"Page '{page_title}': {c['text']}"
            if self.is_reranker:
                pairs.append([goal, doc])
            else:
                pairs.append([doc, f"This page section answers or satisfies: {goal}."])

        probs = self.score_pairs(pairs)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        for i, c in enumerate(chunks):
            c["score"] = round(probs[i], 4)

        ranked_chunks = sorted(chunks, key=lambda x: float(x["score"]), reverse=True)
        top_neural = float(ranked_chunks[0]["score"]) if ranked_chunks else 0.0

        satisfaction_rate = round(max(top_neural, 0.82 if structural_hit else 0.0), 4)
        is_satisfied = structural_hit or (satisfaction_rate >= 0.72)

        return {
            "goal_satisfaction_rate": satisfaction_rate,
            "is_satisfied": is_satisfied,
            "latency_ms": latency_ms,
            "ranked_chunks": ranked_chunks,
        }

    def evaluate_captcha_presence(
        self,
        candidates: List[Dict[str, Any]],
        page_title: str = "",
        page_url: str = "",
        snapshot_yaml: str = "",
    ) -> Dict[str, Any]:
        """Delegates CAPTCHA presence evaluation to CaptchaResolver."""
        return CaptchaResolver.evaluate_presence(
            candidates=candidates,
            page_title=page_title,
            page_url=page_url,
            snapshot_yaml=snapshot_yaml,
            evaluator=self,
        )

    @staticmethod
    def _compute_role_and_lexical_adjustment(intent: str, el: CandidateElement) -> float:
        """
        Computes lightweight structural role compatibility and normalized lexical stem overlap.
        """
        intent_lower = intent.lower()
        boost = 0.0

        wants_select = any(
            w in intent_lower for w in ("z listy", "dropdown", "select from", "combobox", "opcję z", "wybierz z")
        )
        wants_input = (
            any(
                w in intent_lower
                for w in ("pole", "wpisz", "tekst", "textbox", "input", "wyszukiwark", "searchbox", "fill", "type")
            )
            and not wants_select
        )
        wants_click = (
            any(w in intent_lower for w in ("kliknij", "otwórz", "przejdź", "naciśnij", "click", "open"))
            and not wants_select
            and not wants_input
        )

        if wants_select:
            boost += 0.30 if el.role in ("combobox", "option") else -0.25
        elif wants_input:
            boost += 0.25 if el.is_input else -0.30
        elif wants_click:
            if el.is_input:
                boost -= 0.35
            elif el.role in ("button", "link", "tab", "menuitem"):
                boost += 0.12

        lex_overlap = _lexical_overlap_score(intent, f"{el.name} {el.context}")
        if lex_overlap > 0:
            boost += min(0.28, lex_overlap * 0.6)
        else:
            boost -= 0.12

        return boost

    def evaluate_candidates(
        self,
        intent: str,
        candidates: Sequence[Union[CandidateElement, Dict[str, Any]]],
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates candidate elements in parallel on GPU via `score_pairs()`.
        Returns the top matching element, standardized ActionCommand plan, and confidence score.
        """
        if confidence_threshold is None:
            confidence_threshold = get_confidence_threshold()

        if not candidates:
            return {
                "target_ref": "NONE",
                "tool_action": "none",
                "action_plan": None,
                "confidence": 0.0,
                "should_escalate": True,
                "ambiguity_detected": False,
                "ambiguity_margin": None,
                "latency_ms": 0.0,
                "ranked_candidates": [],
            }

        start_time = time.perf_counter()
        elements = [CandidateElement.from_dict(c) for c in candidates]

        pairs: List[List[str]] = []
        for el in elements:
            elem_desc = el.format_for_model()
            if self.is_reranker:
                pairs.append([intent, elem_desc])
            else:
                pairs.append([elem_desc, f"This element represents, opens, or is used to {intent}."])

        raw_probs = self.score_pairs(pairs)
        latency_ms = (time.perf_counter() - start_time) * 1000

        scored_candidates: List[Dict[str, Any]] = []
        for i, el in enumerate(elements):
            raw_score = raw_probs[i]
            boost = self._compute_role_and_lexical_adjustment(intent, el)

            # Dampen positive lexical boost if neural model scores the element as irrelevant (<0.25)
            if boost > 0 and raw_score < 0.25:
                boost = boost * (raw_score / 0.25)
            elif boost < 0 and raw_score < 0.25:
                boost = boost * 1.25

            unclamped = raw_score + boost
            scored_candidates.append(
                {
                    "ref": el.ref,
                    "role": el.role,
                    "name": el.name,
                    "context": el.context,
                    "nth": el.nth,
                    "url": el.url,
                    "_sort_score": unclamped,
                    "confidence_score": float(round(max(0.0, min(1.0, unclamped)), 4)),
                }
            )

        scored_candidates.sort(key=lambda x: float(x.get("_sort_score") or 0.0), reverse=True)
        best = scored_candidates[0]
        best_score = float(best["confidence_score"])

        should_escalate = best_score < confidence_threshold
        ambiguity_detected = False
        score_diff = 1.0

        if len(scored_candidates) > 1:
            second = scored_candidates[1]
            second_score = float(second["confidence_score"])
            score_diff = best_score - second_score

            is_distinct = (best.get("role") or "").lower() != (second.get("role") or "").lower() or (
                best.get("name") or ""
            ).strip().lower() != (second.get("name") or "").strip().lower()
            if is_distinct and score_diff < 0.04 and best_score < 0.75:
                ambiguity_detected = True
                should_escalate = True

        target_ref = str(best["ref"]) if not should_escalate else "NONE"
        command = self.infer_action_command(intent, best, confidence=best_score)

        return {
            "target_ref": target_ref,
            "tool_action": command.action,
            "action_plan": command.to_dict(),
            "command": command,
            "confidence": best_score,
            "should_escalate": should_escalate,
            "ambiguity_detected": ambiguity_detected,
            "ambiguity_margin": round(score_diff, 4) if len(scored_candidates) > 1 else None,
            "latency_ms": round(latency_ms, 2),
            "ranked_candidates": scored_candidates,
        }

    def bind_payload_to_elements(
        self,
        payload: Dict[str, Any],
        candidates: Sequence[Union[CandidateElement, Dict[str, Any]]],
        min_confidence: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """Delegates form payload binding to PayloadBinder."""
        return PayloadBinder(self).bind(payload, candidates, min_confidence=min_confidence)
