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


class AriaElementParser:
    """Parses official page.aria_snapshot() YAML into context-aware candidate objects."""

    INTERACTIVE_ROLES = {
        "button",
        "link",
        "combobox",
        "textbox",
        "searchbox",
        "spinbutton",
        "checkbox",
        "radio",
        "switch",
        "option",
        "tab",
        "menuitem",
        "dialog",
        "alertdialog",
    }

    STRUCTURAL_ROLES = {
        "main",
        "navigation",
        "region",
        "form",
        "article",
        "listitem",
        "row",
        "dialog",
        "banner",
        "contentinfo",
        "heading",
    }

    # Matches lines like: - button "Szukaj" [expanded]: text
    PATTERN = re.compile(
        r'^\s*-\s*([a-zA-Z0-9_-]+)(?:\s+"([^"]*)")?(?:\s+\[([^\]]+)\])*(?::\s*(.*))?$'
    )

    @classmethod
    def parse_snapshot(
        cls, snapshot_yaml: str, max_candidates: int = 150, require_input_only: bool = False
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        context_stack: List[tuple[int, str]] = []
        role_name_counts: Dict[tuple[str, str], int] = {}

        lines = snapshot_yaml.splitlines()
        for line in lines:
            clean = line.strip()
            if not clean.startswith("-"):
                continue

            indent = len(line) - len(line.lstrip(" "))

            # Trailing URL metadata for preceding link candidate
            if clean.startswith("- /url:") and candidates:
                u = clean.replace("- /url:", "").strip().strip('"').strip("'")
                candidates[-1]["url"] = u
                continue

            # Trailing placeholder metadata for preceding input candidate
            if clean.startswith("- /placeholder:") and candidates:
                ph = clean.replace("- /placeholder:", "").strip().strip('"').strip("'")
                candidates[-1]["placeholder"] = ph
                if not candidates[-1]["name"]:
                    candidates[-1]["name"] = ph
                candidates[-1]["text"] = f"{candidates[-1]['text']} (placeholder: {ph})".strip()
                continue

            # Unwrapping single-quoted YAML keys (e.g. - 'link "name: with colon"':)
            clean = re.sub(r"^-\s*'(.*)'\s*:?$", r"- \1", clean)

            m = cls.PATTERN.match(clean)
            if not m:
                continue

            role = m.group(1).lower()
            name = (m.group(2) or "").strip()
            attr_str = (m.group(3) or "").strip().lower()
            text_val = (m.group(4) or "").strip()
            label = name if name else text_val

            # Strip Private Use Area (PUA) font icons (both raw unicode and literal \\u escapes)
            clean_label = re.sub(r"[\ue000-\uf8ff\uf000-\uffff]|\\u[e-fE-F][0-9a-fA-F]{3}", "", label).strip()
            if clean_label:
                label = clean_label

            # Maintain hierarchical section context stack using YAML indentation
            while context_stack and context_stack[-1][0] >= indent:
                if context_stack[-1][0] == indent and context_stack[-1][1].startswith("heading:"):
                    break
                context_stack.pop()

            if role in cls.STRUCTURAL_ROLES:
                if role == "heading" and context_stack and context_stack[-1][0] == indent:
                    context_stack.pop()
                ctx_label = f"{role}: '{label[:50]}'" if label else role
                context_stack.append((indent, ctx_label))

            if require_input_only and role not in ["textbox", "combobox", "searchbox", "spinbutton"]:
                continue

            # Allow unnamed inputs (textbox/searchbox/combobox) so placeholder or role-based fill works
            is_unnamed_input = role in ("textbox", "searchbox", "combobox", "spinbutton") and not label
            if role in cls.INTERACTIVE_ROLES and (label or is_unnamed_input):
                effective_label = label or role
                key = (role, effective_label.lower())
                nth_index = role_name_counts.get(key, 0)
                role_name_counts[key] = nth_index + 1

                parent_context = " > ".join(
                    c[1] for c in context_stack[-3:] if c[0] < indent or c[1].startswith("heading:")
                )

                cand_obj: Dict[str, Any] = {
                    "ref": f"aria-{len(candidates)}",
                    "role": role,
                    "tag": role,
                    "name": label[:120],
                    "text": effective_label[:120],
                    "context": parent_context,
                    "nth": nth_index,
                    "raw": clean,
                }
                if attr_str:
                    cand_obj["state"] = attr_str
                candidates.append(cand_obj)

            if len(candidates) >= max_candidates:
                break

        return candidates


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
        Distinguishes active interstitial challenges from passive footer badges (e.g. 'Protected by reCAPTCHA').
        Returns standardized state:
        {"is_captcha": bool, "captcha_type": str, "confidence": float, "heuristic_trigger": bool,
         "neural_score": float, "latency_ms": float, "matched_candidate": Optional[Dict]}
        """
        t0 = time.perf_counter()
        lower_title = (page_title or "").lower()
        lower_url = (page_url or "").lower()
        lower_snap = (snapshot_yaml or "").lower()

        # 1. Passive badge indicators (Google reCAPTCHA badge in footers / login forms)
        passive_badge_indicators = [
            "protected by recaptcha",
            "chronione przez recaptcha",
            "privacy policy and terms of service apply",
            "privacy - terms",
            "polityka prywatności i warunki korzystania",
            "google privacy policy",
            "terms of service apply",
            "ta strona jest chroniona przez recaptcha",
        ]
        has_passive_badge = any(k in lower_snap for k in passive_badge_indicators)
        if not has_passive_badge:
            for c in candidates:
                c_comb = f"{c.get('name', '')} {c.get('context', '')}".lower()
                if any(k in c_comb for k in passive_badge_indicators):
                    has_passive_badge = True
                    break

        # 2. Active challenge indicators
        # Cloudflare interstitial / blocking page
        cf_interstitial_indicators = [
            "just a moment",
            "attention required! | cloudflare",
            "security check",
            "checking your browser",
            "sprawdza, czy nie jesteś botem",
            "enable javascript and cookies to continue",
        ]
        is_cf_interstitial = any(k in lower_title for k in cf_interstitial_indicators) or (
            "ray id" in lower_snap and any(k in lower_snap for k in ["cloudflare", "verify you are human", "moment"])
        )

        # Turnstile challenge (Cloudflare interactive widget)
        turnstile_indicators = [
            "challenges.cloudflare.com",
            "cf-turnstile",
            "turnstile",
            "verify you are human",
            "potwierdź, że jesteś człowiekiem",
        ]
        is_turnstile = any(k in lower_snap for k in turnstile_indicators) or any(
            k in lower_url for k in ["challenges.cloudflare.com", "turnstile"]
        )

        # Active reCAPTCHA challenge (interactive checkbox "I'm not a robot", challenge iframe)
        recaptcha_active_indicators = [
            "i'm not a robot",
            "nie jestem robotem",
            "recaptcha challenge",
            "select all images",
            "wybierz wszystkie kwadraty",
            "google.com/recaptcha/api2/bframe",
            "google.com/recaptcha/api2/anchor",
        ]
        is_active_recaptcha = any(k in lower_snap for k in recaptcha_active_indicators)

        # Active hCaptcha
        hcaptcha_indicators = [
            "hcaptcha.com",
            "i am human",
            "jestem człowiekiem",
            "hcaptcha challenge",
        ]
        is_hcaptcha = any(k in lower_snap for k in hcaptcha_indicators)

        # Candidate element scan for active challenge controls
        matched_cand: Optional[Dict[str, Any]] = None
        for c in candidates:
            c_name = (c.get("name") or "").lower()
            c_all = f"{c.get('role', '')} {c_name} {c.get('context', '')}".lower()

            if any(k in c_name for k in ["i'm not a robot", "nie jestem robotem"]):
                is_active_recaptcha = True
                matched_cand = c
                break
            if any(k in c_all for k in ["turnstile", "verify you are human", "potwierdź, że jesteś człowiekiem"]):
                is_turnstile = True
                matched_cand = c
                break
            if any(k in c_all for k in ["i am human", "jestem człowiekiem"]):
                is_hcaptcha = True
                matched_cand = c
                break

        # Passive badge suppression: If passive badge is present without any active challenge indicators
        if has_passive_badge and not (is_cf_interstitial or is_turnstile or is_active_recaptcha or is_hcaptcha):
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "is_captcha": False,
                "captcha_type": "none",
                "confidence": 0.08,
                "heuristic_trigger": False,
                "neural_score": 0.05,
                "latency_ms": round(latency_ms, 2),
                "matched_candidate": None,
            }

        heuristic_trigger = False
        captcha_type = "none"

        if is_cf_interstitial:
            heuristic_trigger = True
            captcha_type = "cloudflare_interstitial"
        elif is_turnstile:
            heuristic_trigger = True
            captcha_type = "turnstile"
        elif is_active_recaptcha:
            heuristic_trigger = True
            captcha_type = "recaptcha"
        elif is_hcaptcha:
            heuristic_trigger = True
            captcha_type = "hcaptcha"

        # 3. Neural scoring on security verification intent
        intent = "security verification challenge cloudflare turnstile recaptcha hcaptcha verify human weryfikacja zabezpieczeń"
        relevant_candidates: List[Dict[str, Any]] = []
        for c in candidates:
            c_text = f"{c.get('role', '')} {c.get('name', '')} {c.get('context', '')}".lower()
            if any(k in c_text for k in ["turnstile", "captcha", "człowiek", "human", "security", "weryfikuj", "zabezpiecz", "cloudflare", "robot"]):
                if not any(pb in c_text for pb in passive_badge_indicators):
                    relevant_candidates.append(c)

        neural_score = 0.0
        best_cand = matched_cand
        if relevant_candidates or (not heuristic_trigger and candidates):
            eval_cands = relevant_candidates if relevant_candidates else candidates[:12]
            eval_res = self.evaluate_candidates(intent, eval_cands)
            neural_score = float(eval_res.get("confidence", 0.0))
            if not best_cand and eval_res.get("ranked_candidates"):
                cand_candidate = eval_res["ranked_candidates"][0]
                if cand_candidate.get("confidence_score", 0.0) >= 0.50:
                    best_cand = cand_candidate

        if heuristic_trigger:
            confidence = max(0.92, neural_score)
            is_captcha = True
        elif neural_score >= 0.70 and not has_passive_badge:
            confidence = neural_score
            is_captcha = True
            captcha_type = "recaptcha" if "recaptcha" in str(best_cand).lower() else "turnstile"
        else:
            confidence = min(0.35, neural_score)
            is_captcha = False
            captcha_type = "none"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "is_captcha": is_captcha,
            "captcha_type": captcha_type,
            "confidence": round(confidence, 4),
            "heuristic_trigger": heuristic_trigger,
            "neural_score": round(neural_score, 4),
            "latency_ms": round(latency_ms, 2),
            "matched_candidate": best_cand,
        }

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
        Resolves form fields without hardcoded schemas:
          - Automatically identifies file uploads (single or multi-file lists) and triggers upload_file.
          - Maps text inputs, selects, checkboxes, and buttons with optimal 1-to-1 matching.
          - Preserves DOM order for deterministic form filling from top to bottom.
        """
        if not payload or not candidates:
            return []

        has_comboboxes = any((c.get("role") or "").lower() == "combobox" for c in candidates)
        form_roles = {
            "textbox",
            "searchbox",
            "combobox",
            "spinbutton",
            "checkbox",
            "radio",
            "switch",
            "button",
        }
        if not has_comboboxes:
            form_roles.add("option")

        valid_cands = [c for c in candidates if (c.get("role") or "").lower() in form_roles]
        if not valid_cands:
            return []

        common_file_exts = (
            ".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".webp",
            ".txt", ".csv", ".zip", ".xlsx", ".xls", ".json", ".rtf",
            ".odt", ".svg", ".gif", ".mp4", ".mov"
        )
        file_keywords = {
            "cv", "resume", "plik", "file", "zalacznik", "załącznik",
            "attachment", "upload", "dokument", "photo", "foto", "zdjecie",
            "zdjęcie", "files", "pliki", "zdjęcia"
        }
        semantic_expansions = {
            "first_name": "imię imie first name",
            "imie": "imię imie first name",
            "last_name": "nazwisko last name surname family name",
            "nazwisko": "nazwisko last name surname family name",
            "full_name": "imię i nazwisko full name",
            "imie_i_nazwisko": "imię i nazwisko full name",
            "email": "adres email e-mail mail poczta elektroniczna",
            "mail": "adres email e-mail mail poczta elektroniczna",
            "phone": "numer telefonu telefon phone tel telephone mobile",
            "telefon": "numer telefonu telefon phone tel komórkowy",
            "nr_telefonu": "numer telefonu telefon phone tel",
            "city": "miasto city miejscowość",
            "miasto": "miasto city miejscowość",
            "address": "adres ulica address street",
            "adres": "adres ulica address street",
            "postal_code": "kod pocztowy postal code zip",
            "kod_pocztowy": "kod pocztowy postal code zip",
            "country": "kraj country państwo",
            "kraj": "kraj country państwo",
            "cv": "załącznik cv życiorys resume upload dodaj plik dokument browse",
            "resume": "załącznik cv życiorys resume upload dodaj plik dokument browse",
            "cover_letter": "list motywacyjny cover letter dokument załącznik upload browse",
            "list_motywacyjny": "list motywacyjny cover letter dokument załącznik upload browse",
            "portfolio": "portfolio link do portfolio projekty github",
            "linkedin": "profil linkedin url profil zawodowy",
            "github": "profil github profil programisty",
            "experience": "doświadczenie zawodowe lata pracy experience",
            "doswiadczenie": "doświadczenie zawodowe lata pracy experience",
            "education": "wykształcenie edukacja szkoła uczelnia education",
            "wyksztalcenie": "wykształcenie edukacja szkoła uczelnia education",
            "salary": "oczekiwane wynagrodzenie stawka pensja salary expectation",
            "wynagrodzenie": "oczekiwane wynagrodzenie stawka pensja salary expectation",
            "stawka": "stawka wynagrodzenie pensja salary",
            "brand": "marka pojazdu producent car brand make",
            "marka": "marka pojazdu producent car brand make",
            "model": "model pojazdu seria model",
            "year": "rok produkcji year of production",
            "rok": "rok produkcji year of production",
            "rok_od": "rok produkcji od minimalny rok year from",
            "rok_do": "rok produkcji do maksymalny rok year to",
            "price": "cena koszt kwota price",
            "cena": "cena koszt kwota price",
            "cena_od": "cena od minimalna kwota price from",
            "cena_do": "cena do maksymalna kwota price to",
            "mileage": "przebieg w kilometrach mileage km",
            "przebieg": "przebieg w kilometrach mileage km",
            "przebieg_od": "przebieg od mileage from",
            "przebieg_do": "przebieg do mileage to",
            "fuel": "rodzaj paliwa napęd fuel type diesel benzyna hybryda lpg elektryczny",
            "paliwo": "rodzaj paliwa napęd fuel type diesel benzyna hybryda lpg elektryczny",
            "transmission": "skrzynia biegów przekładnia automatyczna manualna gearbox transmission",
            "skrzynia": "skrzynia biegów przekładnia automatyczna manualna gearbox transmission",
            "body": "typ nadwozia sedan kombi hatchback suv coupe",
            "nadwozie": "typ nadwozia sedan kombi hatchback suv coupe",
            "consent": "zgoda oświadczenie rodo akceptuję regulamin consent agree privacy",
            "zgoda": "zgoda oświadczenie rodo akceptuję regulamin consent agree privacy",
            "rodo": "zgoda oświadczenie rodo przetwarzanie danych osobowych privacy consent",
            "regulamin": "akceptuję regulamin warunki serwisu terms conditions accept",
        }

        # Analyze each payload item
        parsed_items = []
        for key, raw_val in payload.items():
            if raw_val is None:
                continue

            k_lower = str(key).lower().strip()
            clean_k = k_lower.replace("_", " ").replace("-", " ")

            is_file = False
            file_paths: Optional[List[str]] = None
            if isinstance(raw_val, (list, tuple)):
                if any(any(str(x).lower().endswith(ext) for ext in common_file_exts) for x in raw_val) or any(
                    kw in k_lower for kw in file_keywords
                ):
                    is_file = True
                    file_paths = [str(x) for x in raw_val]
            elif isinstance(raw_val, str):
                s_val = raw_val.lower().strip()
                if any(s_val.endswith(ext) for ext in common_file_exts) or any(kw in k_lower for kw in file_keywords):
                    is_file = True
                    file_paths = [raw_val]

            is_bool = False
            bool_val = False
            if isinstance(raw_val, bool):
                is_bool = True
                bool_val = raw_val
            elif isinstance(raw_val, str) and raw_val.lower() in ("true", "false", "tak", "nie"):
                is_bool = True
                bool_val = raw_val.lower() in ("true", "tak")

            if k_lower in semantic_expansions:
                query = semantic_expansions[k_lower]
            elif is_file:
                query = f"załącz plik dodaj dokument upload {clean_k}"
            else:
                query = f"{clean_k} {raw_val}"

            parsed_items.append(
                {
                    "key": key,
                    "k_lower": k_lower,
                    "clean_k": clean_k,
                    "raw_val": raw_val,
                    "query": query,
                    "is_file": is_file,
                    "file_paths": file_paths,
                    "is_bool": is_bool,
                    "bool_val": bool_val,
                }
            )

        if not parsed_items:
            return []

        # Construct pairs for neural evaluation: (query, candidate_description)
        pairs: List[List[str]] = []
        pair_meta: List[tuple[int, int]] = []  # (item_idx, cand_idx)

        for p_idx, p_item in enumerate(parsed_items):
            for c_idx, cand in enumerate(valid_cands):
                role = cand.get("role") or ""
                name = cand.get("name") or ""
                ctx = cand.get("context") or ""
                desc = f"[{role}] {name} (context: {ctx})"
                if self.is_reranker:
                    pairs.append([p_item["query"], desc])
                else:
                    hypothesis = f"This element represents, opens, or is used to {p_item['query']}."
                    pairs.append([desc, hypothesis])
                pair_meta.append((p_idx, c_idx))

        raw_probs = [0.0] * len(pairs)
        if pairs:
            indexed_pairs = list(enumerate(pairs))
            indexed_pairs.sort(key=lambda x: len(x[1][0]) + len(x[1][1]))
            batch_size = 64

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
                        raw_probs[orig_idx] = float(prob_val.item())

            if self.device == "cuda":
                torch.cuda.synchronize()

        # Compute boosts and collect scores
        match_candidates = []
        for pair_i, (p_idx, c_idx) in enumerate(pair_meta):
            p_item = parsed_items[p_idx]
            cand = valid_cands[c_idx]

            raw_prob = raw_probs[pair_i]
            c_role = (cand.get("role") or "").lower()
            c_name = f"{cand.get('name') or ''} {cand.get('context') or ''}".lower()

            boost = 0.0

            # Role & Category affinity
            if p_item["is_file"]:
                has_file_kw = any(kw in c_name for kw in file_keywords) or any(
                    w in c_name for w in ["wybierz plik", "dodaj plik", "browse", "załącz", "załaduj"]
                )
                if has_file_kw or c_role == "button":
                    boost += 0.45
                if c_role in ("textbox", "searchbox") and not has_file_kw:
                    boost -= 0.50
            else:
                has_file_kw = any(kw in c_name for kw in file_keywords) or "wybierz plik" in c_name
                if has_file_kw:
                    boost -= 0.50

                if p_item["is_bool"]:
                    if c_role in ("checkbox", "radio", "switch"):
                        boost += 0.35
                    elif c_role in ("textbox", "searchbox"):
                        boost -= 0.40
                else:
                    if c_role in ("textbox", "searchbox", "combobox", "spinbutton", "option"):
                        boost += 0.20
                    elif c_role in ("checkbox", "switch", "radio"):
                        boost -= 0.45

            # Keyword & stem lexical matching against candidate name
            query_words = set(re.findall(r"\w{3,}", p_item["query"].lower()))
            c_words = set(re.findall(r"\w{3,}", c_name))

            def _check_stem(w1: str, w2: str) -> bool:
                min_l = min(len(w1), len(w2))
                if min_l >= 4:
                    return w1[:4] == w2[:4] or (w1[: min_l - 1] == w2[: min_l - 1])
                return w1 in w2 or w2 in w1

            exact_matches = sum(1 for qw in query_words if qw in c_words)
            stem_matches = sum(
                1 for qw in query_words if qw not in c_words and any(_check_stem(qw, cw) for cw in c_words)
            )

            if exact_matches > 0:
                boost += min(0.40, exact_matches * 0.20)
            elif stem_matches > 0:
                boost += min(0.25, stem_matches * 0.12)

            final_score = raw_prob + boost
            final_score = float(max(0.0, min(1.0, final_score)))

            # Determine optimal action
            if p_item["is_file"]:
                action = "upload_file"
                action_val = None
                action_files = p_item["file_paths"]
            elif p_item["is_bool"]:
                action = "check" if p_item["bool_val"] else "uncheck"
                action_val = None
                action_files = None
            elif c_role in ("combobox", "option"):
                action = "select"
                action_val = str(p_item["raw_val"])
                action_files = None
            elif c_role == "button":
                action = "click"
                action_val = None
                action_files = None
            else:
                action = "fill"
                action_val = str(p_item["raw_val"])
                action_files = None

            match_candidates.append(
                {
                    "score": final_score,
                    "p_idx": p_idx,
                    "c_idx": c_idx,
                    "key": p_item["key"],
                    "cand": cand,
                    "action": action,
                    "value": action_val,
                    "file_paths": action_files,
                }
            )

        # Bipartite Greedy Matching
        match_candidates.sort(key=lambda x: x["score"], reverse=True)
        used_payload_keys = set()
        used_cands = set()
        matched_actions = []

        for m in match_candidates:
            if m["score"] < min_confidence:
                continue
            if m["key"] in used_payload_keys or m["c_idx"] in used_cands:
                continue

            used_payload_keys.add(m["key"])
            used_cands.add(m["c_idx"])

            cand = m["cand"]
            matched_actions.append(
                {
                    "dom_index": m["c_idx"],
                    "payload_key": m["key"],
                    "action": m["action"],
                    "role": cand.get("role"),
                    "name": cand.get("name"),
                    "nth": cand.get("nth", 0),
                    "value": m["value"],
                    "file_paths": m["file_paths"],
                    "confidence": round(m["score"], 4),
                }
            )

        # Sort actions according to original DOM order (from top of the page downwards)
        matched_actions.sort(key=lambda x: x["dom_index"])
        for act in matched_actions:
            del act["dom_index"]

        return matched_actions

