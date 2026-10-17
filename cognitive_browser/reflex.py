"""
Reflex Core: Fast local Cross-Encoder / NLI evaluation and Autonomous Action Planner on Accessibility Trees.
Supports Multilingual Cross-Encoders (mmarco-mMiniLMv2), ModernBERT Reranker, and DeBERTa-v3 NLI models.
Consumes 0 LLM tokens, executes in sub-30ms on GPU/MPS/CPU.
"""

import re
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import get_confidence_threshold, get_reflex_model


from .captcha import CaptchaResolver
from .parser import AriaElementParser
from .payload import PayloadBinder


class ReflexCore:
    """
    Discriminative Decision Engine & Autonomous Action Planner.
    Computes match probabilities between candidate elements and natural language instructions,
    infers the optimal DOM interaction (click, fill_and_submit, select, check, scroll, hover),
    and detects when a page already satisfies an informational goal.
    """

    # Boilerplate links that autonomous goal navigation should never click unless explicitly asked
    NOISE_LINK_PATTERNS = (
        "polityka prywatności",
        "privacy policy",
        "regulamin",
        "terms of service",
        "terms and conditions",
        "ustawienia",
        "settings",
        "pomoc",
        "kontakt",
        "zaloguj",
        "log in",
        "sign in",
        "rejestracja konta",
        "załóż konto",
        "koszyk",
        "twój koszyk",
        "listy zakupowe",
        "ulubione",
        "schowek",
        "mój proline",
        "panel klienta",
        "dodaj ogłoszenie",
        "twoje konto",
        "obserwowane",
        "powiadomienia",
        "czat",
        "dla biznesu",
        "aplikacje mobilne",
        "pobierz aplikację",
        "znajdź swoją paczkę",
        "newsletter",
        "kariera",
        "o nas",
        "reklama",
        "tryb ai",
        "wideo",
        "wiadomości",
        "grafika",
        "mapy",
        "why did this happen",
        "learn more",
        "email us",
        "przejdź do treści",
        "przejdź do głównej",
        "przejdź do stopki",
        "skip to content",
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
        with torch.no_grad():
            dummy = self.tokenizer(
                [["Submit form", "Web element: button 'Submit'"]],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            _ = self.model(**dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    @classmethod
    def is_noise_candidate(cls, candidate: Dict[str, Any], intent: str) -> bool:
        """Returns True if the candidate is a generic footer/header/account distraction not requested by intent."""
        name_lower = (candidate.get("name") or "").strip().lower()
        ctx_lower = (candidate.get("context") or "").strip().lower()
        intent_lower = intent.lower()

        if not name_lower:
            return False

        # Never click footer links unless intent mentions footer/regulamin/kontakt
        if "contentinfo" in ctx_lower and not any(w in intent_lower for w in ["regulamin", "kontakt", "polityk", "stopk"]):
            return True

        for pat in cls.NOISE_LINK_PATTERNS:
            if pat in name_lower and pat not in intent_lower:
                return True

        # Avoid clicking the site logo to go back to "/"
        url = str(candidate.get("url") or "").strip()
        if url == "/" and "strona główna" not in intent_lower and "home" not in intent_lower:
            return True

        return False

    @staticmethod
    def extract_fill_value(intent: str) -> Optional[str]:
        """
        Extracts a search query or input value from a natural language instruction/goal if present.
        Examples:
        - 'Wpisz w pole wyszukiwania: site:olx.pl "rtx 4070 ti" wrocław i kliknij szukaj' -> 'site:olx.pl "rtx 4070 ti" wrocław'
        - 'Wpisz "RTX 4070 Ti" w wyszukiwarkę' -> 'RTX 4070 Ti'
        """
        intent_clean = intent.strip()
        # 1. Explicit colon pattern after wpisz/wyszukaj/type/search
        m_colon = re.search(
            r"(?:wpisz|wprowadź|wyszukaj|szukaj|type|fill|enter|search)[^:]{0,35}:\s*(.+?)(?:\s+i\s+(?:kliknij|naciśnij|wciśnij|zatwierdź)|$)",
            intent_clean,
            re.I,
        )
        if m_colon:
            val = m_colon.group(1).strip().strip("'\"")
            if val:
                return val

        # 2. Quoted value after input or selection verb
        m_quote = re.search(
            r"(?:wpisz|wprowadź|type|fill|enter|wybierz|select|choose)\s+['\"„”]([^'\"„”]{2,80})['\"„”]",
            intent_clean,
            re.I,
        )
        if m_quote:
            return m_quote.group(1).strip()

        return None

    @classmethod
    def infer_action_plan(
        cls,
        intent: str,
        best_candidate: Dict[str, Any],
        explicit_fill_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds a complete executable action plan for BrowserEngine.execute_reflex_action()
        covering fill, fill_and_submit, select, check, uncheck, hover, scroll, press, goto, or click.
        """
        role = (best_candidate.get("role") or "button").lower()
        name = best_candidate.get("name") or ""
        nth = int(best_candidate.get("nth", 0))
        url = best_candidate.get("url")
        intent_lower = intent.lower()

        fill_val = explicit_fill_value if explicit_fill_value is not None else cls.extract_fill_value(intent)

        # Scroll intent check
        if any(w in intent_lower for w in ["przewiń", "scroll", "na dół", "w dół"]):
            direction = "up" if ("gór" in intent_lower or "up" in intent_lower) else "down"
            return {"action": "scroll", "role": role, "name": name, "nth": nth, "value": direction}

        # Select / Dropdown option
        wants_select = any(w in intent_lower for w in ["wybierz", "select", "choose", "z listy", "dropdown", "opcję"])
        if role in ("combobox", "option") and (wants_select or (fill_val and not any(w in intent_lower for w in ["wpisz", "type", "fill"]))):
            return {"action": "select", "role": role, "name": name, "nth": nth, "value": fill_val}

        # Input / Searchbox fill
        if role in ("textbox", "searchbox", "spinbutton") or (role == "combobox" and fill_val):
            if fill_val:
                wants_submit = (
                    role == "searchbox"
                    or any(w in intent_lower for w in ["szukaj", "wyszukaj", "search", "enter", "zatwierdź"])
                )
                return {
                    "action": "fill_and_submit" if wants_submit else "fill",
                    "role": role,
                    "name": name,
                    "nth": nth,
                    "value": fill_val,
                }

        # Checkbox / Radio / Switch
        if role in ("checkbox", "radio", "switch"):
            uncheck = any(w in intent_lower for w in ["odznacz", "wyłącz", "uncheck", "disable"])
            return {
                "action": "uncheck" if uncheck else "check",
                "role": role,
                "name": name,
                "nth": nth,
            }

        # Hover intent
        if any(w in intent_lower for w in ["najedź", "hover", "rozwiń menu"]):
            return {"action": "hover", "role": role, "name": name, "nth": nth}

        # Direct URL navigation or click
        return {
            "action": "click",
            "role": role,
            "name": name,
            "nth": nth,
            "url": url,
        }

    @staticmethod
    def is_goal_satisfied_on_page(
        goal: str, snapshot_yaml: str, page_title: str, page_url: str
    ) -> bool:
        """
        Determines whether the current page ALREADY contains the target data requested by `goal`
        (e.g., search results list with prices, parcel dimensions table, or target article),
        preventing the autonomous loop from clicking away from a page that already has the answer.
        """
        if not snapshot_yaml or len(snapshot_yaml) < 200:
            return False

        snap_lower = snapshot_yaml.lower()
        goal_lower = goal.lower()
        url_lower = page_url.lower()

        # 0. Block/error/captcha detection: stop immediately so we don't click random error links
        if any(
            err in snap_lower
            for err in [
                "403 error",
                "request blocked",
                "unusual traffic from your computer network",
                "captcha",
                "access denied",
            ]
        ):
            return True

        # 1. Read-only / inspection instructions: never click anything if the user says "przeczytaj", "odczytaj", "nie klikaj"
        if any(
            ro in goal_lower
            for ro in [
                "nie klikaj",
                "pozostań na stronie",
                "przeczytaj",
                "odczytaj",
                "treść ogłoszenia",
                "opis ogłoszenia",
                "zweryfikuj z treści",
            ]
        ):
            return True

        # 2. Direct product / offer detail page (e.g. OLX /d/oferta/..., Ceneo /146875949, Allegro /oferta/...)
        if any(
            pat in url_lower
            for pat in ["/d/oferta/", "/oferta/", "/produkt/", "/item/", "ceneo.pl/1"]
        ) or re.search(r"ceneo\.pl/\d{6,}", url_lower):
            return True

        # 3. If the user explicitly asked to type into a search box on a homepage, don't stop before typing
        if any(w in goal_lower for w in ["wpisz w pole", "wpisz ", "type in"]) and not any(
            q in url_lower for q in ["?q=", "&q=", "?qh=", "szukaj", "search", "filter"]
        ):
            return False

        # 2. Check if URL is already a search/listing/filter results page or specific product/pricing page
        is_results_url = any(
            pat in url_lower
            for pat in ["?q=", "&q=", "?qh=", "szukaj", "search", "string=", "cennik", "szybkienadania", "kategoria"]
        )

        # Count price occurrences (e.g. "2 850 zł", "16.49 zł", "3999,00 zł", "PLN")
        price_matches = len(re.findall(r"\b\d[\d\s.,]{0,6}\s*(?:zł|pln|eur|usd)\b", snap_lower))
        # Count dimension occurrences (e.g. "mm", "cm", "kg", "gabaryt")
        dim_matches = len(re.findall(r"\b(?:gabaryt|rozmiar:\s*(?:mała|średnia|duża)|\d+\s*x\s*\d+|\d+\s*kg)\b", snap_lower))

        wants_prices_or_offers = any(
            w in goal_lower for w in ["cen", "ofert", "znajdź", "sprawdź", "porównaj", "odczytaj", "pokaż", "zobacz", "wymiar", "gabaryt"]
        )

        # If goal asks for prices/offers and the current page already lists >= 3 prices!
        if wants_prices_or_offers and price_matches >= 3:
            return True

        # If goal asks for dimensions/gabaryty and the current page already lists dimensions/prices
        if any(w in goal_lower for w in ["gabaryt", "wymiar", "paczkomat", "inpost"]) and (
            dim_matches >= 1 or price_matches >= 1
        ):
            return True

        # If we navigated directly via query URL (like x-kom.pl/szukaj?q=...) and the page has substantial content
        if is_results_url and wants_prices_or_offers and len(snapshot_yaml) > 1200:
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
        Evaluates the current page state on GPU using the Cross-Encoder to compute:
        1. `goal_satisfaction_rate` (0.0 - 1.0): probability that the current page ALREADY satisfies `goal`.
        2. `is_satisfied` (bool): whether the router should allocate state to STAY_ON_PAGE instead of clicking further.
        3. `ranked_chunks`: page content sections ranked by GPU relevance to `goal` (so deep sections like OLX 'OPIS' at line 380 are never lost to truncation).
        """
        t0 = time.perf_counter()
        structural_hit = self.is_goal_satisfied_on_page(goal, snapshot_yaml, page_title, page_url)

        # Filter decorative noise lines first
        clean_lines = [
            line
            for line in snapshot_yaml.splitlines()
            if line.strip()
            and line.strip()
            not in (
                "- img",
                "- separator",
                "- img:",
                '- text: ""',
                "- group",
                '- button "Wyczyść"',
            )
        ]

        if not clean_lines:
            return {
                "goal_satisfaction_rate": 0.0,
                "is_satisfied": False,
                "latency_ms": 0.0,
                "best_chunks": [],
            }

        # Group lines into semantic windows of ~25 lines
        chunk_size = 25
        chunks: List[Dict[str, Any]] = []
        for idx in range(0, len(clean_lines), chunk_size):
            window = clean_lines[idx : idx + chunk_size]
            text_block = "\n".join(window)
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "start_line": idx,
                    "lines": window,
                    "text": text_block,
                }
            )
            if len(chunks) >= max_chunks:
                break

        pairs = [
            [goal, f"Page '{page_title}' content section:\n{c['text'][:700]}"]
            for c in chunks
        ]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=384, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits

            if self.is_reranker:
                squeezed = logits.squeeze(-1)
                if "mmarco" in self.model_name.lower():
                    probs = torch.sigmoid((squeezed + 4.5) / 2.0)
                else:
                    probs = torch.sigmoid(squeezed)
                if probs.ndim == 0:
                    probs = probs.unsqueeze(0)
            else:
                binary_logits = logits[:, [self.contra_idx, self.entail_idx]]
                probs = torch.softmax(binary_logits, dim=-1)[:, 1]

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        for i, c in enumerate(chunks):
            score = float(probs[i].item())
            # Boost chunks that contain offer descriptions only if baseline neural relevance is present (>0.35)
            txt_low = c["text"].lower()
            if score > 0.35 and any(k in txt_low for k in ["heading \"opis\"", "gwarancj", "faktur", "paragon", "gabaryt"]):
                score = min(1.0, score + 0.15)
            c["score"] = round(score, 4)

        ranked_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        top_neural_score = float(ranked_chunks[0]["score"]) if ranked_chunks else 0.0

        # Combine neural chunk satisfaction with structural page satisfaction
        if structural_hit:
            satisfaction_rate = round(max(0.85, top_neural_score), 4)
            is_satisfied = True
        else:
            total_nav_links = sum(1 for ln in clean_lines if ln.strip().startswith(("- link", "- button")))
            paragraph_lines = sum(1 for ln in clean_lines if ln.strip().startswith(("- paragraph:", "- text:", "- code")))

            # Navigation portal discount: high ratio of menu links with very few content paragraphs
            is_navigation_portal = total_nav_links >= 6 and paragraph_lines < 3
            density_factor = 0.65 if is_navigation_portal else 1.0
            satisfaction_rate = round(top_neural_score * density_factor, 4)
            is_satisfied = satisfaction_rate >= 0.78 and top_neural_score >= 0.70

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
        """
        Evaluates the probability and type of CAPTCHA, Turnstile, or security challenge on the page.
        Delegates to CaptchaResolver.evaluate_presence with self as neural evaluator.
        """
        return CaptchaResolver.evaluate_presence(
            candidates=candidates,
            page_title=page_title,
            page_url=page_url,
            snapshot_yaml=snapshot_yaml,
            evaluator=self,
        )

    def evaluate_candidates(
        self, intent: str, candidates: List[Dict[str, Any]], confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates candidate elements in parallel using Cross-Encoder / NLI on GPU.
        Returns the top matching element, recommended action plan, and confidence score.
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

        pairs = []
        for c in candidates:
            parts = [f"role: {c.get('role')}"]
            if c.get("name"):
                parts.append(f"name: '{c['name']}'")
            if c.get("text") and c.get("text") != c.get("name"):
                parts.append(f"text: '{c['text']}'")
            if c.get("context"):
                parts.append(f"section: ({c['context']})")
            if c.get("url"):
                parts.append(f"url: '{c['url']}'")

            elem_desc = f"Interactive web element: {', '.join(parts)}."
            if self.is_reranker:
                pairs.append([intent, elem_desc])
            else:
                hypothesis = f"This element represents, opens, or is used to {intent}."
                pairs.append([elem_desc, hypothesis])

        indexed_pairs = list(enumerate(pairs))
        # Sort pairs by total character length to minimize padding tokens within each GPU mini-batch
        indexed_pairs.sort(key=lambda x: len(x[1][0]) + len(x[1][1]))

        batch_size = 64
        raw_probs_ordered = [0.0] * len(pairs)

        with torch.no_grad():
            for b_start in range(0, len(indexed_pairs), batch_size):
                b_slice = indexed_pairs[b_start : b_start + batch_size]
                b_pairs = [item[1] for item in b_slice]
                b_indices = [item[0] for item in b_slice]

                inputs = self.tokenizer(
                    b_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits

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
                    raw_probs_ordered[orig_idx] = float(prob_val.item())

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start_time) * 1000

        def _compute_lexical_boost(intent_str: str, cand_role: str, cand_name: str, cand_ctx: str) -> float:
            intent_lower = intent_str.lower()
            name_lower = f"{cand_name or ''} {cand_ctx or ''}".lower()
            role_lower = (cand_role or "").lower()

            boost = 0.0

            # 1. Action type & role matching
            wants_select = any(
                w in intent_lower
                for w in ["z listy", "dropdown", "select from", "combobox", "opcję z", "wybierz z", "z listy miasto", "z listy kraj"]
            )
            wants_input = any(
                w in intent_lower
                for w in ["pole", "wpisz", "tekst", "textbox", "input", "wyszukiwark", "searchbox", "fill", "type"]
            ) and not wants_select
            wants_click = any(
                w in intent_lower
                for w in ["kliknij", "otwórz", "przejdź", "naciśnij", "click", "open"]
            ) and not wants_select and not wants_input
            wants_button = any(w in intent_lower for w in ["przycisk", "guzik", "button", "btn", "submit"])
            wants_link = any(w in intent_lower for w in ["link", "odnośnik", "href"])

            if wants_select:
                if role_lower in ["combobox", "option"]:
                    boost += 0.35
                else:
                    boost -= 0.30
            elif wants_input:
                if role_lower in ["textbox", "searchbox", "combobox", "spinbutton"]:
                    boost += 0.25
                else:
                    boost -= 0.35
            elif wants_click:
                if role_lower in ["textbox", "searchbox", "combobox"]:
                    boost -= 0.40
                elif role_lower in ["button", "link", "tab", "menuitem"]:
                    boost += 0.15

            if wants_button:
                if role_lower == "button":
                    boost += 0.15
                else:
                    boost -= 0.20
            elif wants_link:
                if role_lower == "link":
                    boost += 0.15
                else:
                    boost -= 0.20

            # 2. Token / stem matching between intent and element name + section context
            intent_words = set(re.findall(r"\w{3,}", intent_lower))
            name_words = set(re.findall(r"\w{3,}", name_lower))

            stop_words = {
                "lub",
                "oraz",
                "dla",
                "sie",
                "się",
                "jest",
                "albo",
                "przez",
                "jego",
                "pod",
                "nad",
                "the",
                "and",
                "for",
                "with",
                "kliknij",
                "otwórz",
                "przejdź",
                "wpisz",
                "znajdź",
                "sprawdź",
                "zobacz",
                "pokaż",
                "heading",
                "main",
                "listitem",
                "region",
                "navigation",
            }
            meaningful_intent = {w for w in intent_words if w not in stop_words}
            meaningful_name = {w for w in name_words if w not in stop_words}

            def check_stems(w1: str, w2: str) -> bool:
                min_len = min(len(w1), len(w2))
                if min_len >= 4:
                    return w1[:4] == w2[:4] or (w1[: min_len - 1] == w2[: min_len - 1])
                return w1 in w2 or w2 in w1

            exact_matches = sum(1 for iw in meaningful_intent if iw in meaningful_name)
            stem_matches = sum(
                1
                for iw in meaningful_intent
                if iw not in meaningful_name and any(check_stems(iw, nw) for nw in meaningful_name)
            )

            if exact_matches > 0:
                boost += min(0.28, exact_matches * 0.14)
            elif stem_matches > 0:
                boost += min(0.18, stem_matches * 0.09)
            elif len(meaningful_intent) > 0:
                boost -= 0.16

            return boost

        scored_candidates: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            raw_score = raw_probs_ordered[i]
            boost = _compute_lexical_boost(
                intent, c.get("role") or "", c.get("name") or "", c.get("context") or ""
            )
            # Calibration: dampen positive boost if neural model considers element irrelevant (<0.25)
            # to eliminate False Positives on adversarial or impossible goals
            if boost > 0 and raw_score < 0.25:
                boost = boost * (raw_score / 0.25)
            elif boost < 0 and raw_score < 0.25:
                boost = boost * 1.25

            unclamped_score = raw_score + boost
            scored_candidates.append(
                {
                    "ref": c.get("ref", "NONE"),
                    "role": c.get("role"),
                    "name": c.get("name"),
                    "context": c.get("context", ""),
                    "nth": c.get("nth", 0),
                    "url": c.get("url"),
                    "_sort_score": unclamped_score,
                    "confidence_score": float(round(max(0.0, min(1.0, unclamped_score)), 4)),
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

            is_distinct = (
                (best.get("role") or "").lower() != (second.get("role") or "").lower()
                or (best.get("name") or "").strip().lower() != (second.get("name") or "").strip().lower()
            )
            # Ambiguity Margin: if top two distinct candidates are within 0.04 and confidence < 0.75, escalate
            if is_distinct and score_diff < 0.04 and best_score < 0.75:
                ambiguity_detected = True
                should_escalate = True

        target_ref = str(best["ref"]) if not should_escalate else "NONE"

        action_plan = self.infer_action_plan(intent, best)

        return {
            "target_ref": target_ref,
            "tool_action": action_plan["action"],
            "action_plan": action_plan,
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
        candidates: List[Dict[str, Any]],
        min_confidence: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """
        Universal neural semantic bipartite matching between arbitrary user payload and form elements.
        Delegates to PayloadBinder engine.
        """
        return PayloadBinder(self).bind(payload, candidates, min_confidence=min_confidence)
