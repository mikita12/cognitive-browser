"""
Semantic payload and form parameter binding engine for Cognitive Browser.
Performs universal bipartite matching between user-provided data dictionaries
(e.g., job application details, search filters, uploaded documents) and DOM interactive controls.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


class PayloadBinder:
    """Binds arbitrary payload dictionary parameters to web form controls using neural semantic ranking."""

    COMMON_FILE_EXTENSIONS: Tuple[str, ...] = (
        ".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".webp",
        ".txt", ".csv", ".zip", ".xlsx", ".xls", ".json", ".rtf",
        ".odt", ".svg", ".gif", ".mp4", ".mov"
    )

    FILE_KEYWORDS: Set[str] = {
        "cv", "resume", "plik", "file", "zalacznik", "załącznik",
        "attachment", "upload", "dokument", "photo", "foto", "zdjecie",
        "zdjęcie", "files", "pliki", "zdjęcia"
    }

    SEMANTIC_EXPANSIONS: Dict[str, str] = {
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
        "zip": "kod pocztowy postal code zip",
        "country": "kraj państwo country",
        "kraj": "kraj państwo country",
        "salary": "wynagrodzenie stawka pensja oczekiwania finansowe salary expected rate",
        "wynagrodzenie": "wynagrodzenie stawka pensja oczekiwania finansowe salary",
        "experience": "doświadczenie experience lata staż pracy years",
        "doswiadczenie": "doświadczenie experience lata staż pracy years",
        "linkedin": "profil linkedin url link",
        "github": "profil github portfolio strona www",
        "portfolio": "portfolio github strona projekty cv",
        "agreement": "zgoda rodo przetwarzanie danych regulamin oświadczenie consent terms",
        "consent": "zgoda rodo przetwarzanie danych regulamin oświadczenie consent terms",
        "rodo": "zgoda na przetwarzanie danych osobowych rodo consent privacy",
        "summary": "podsumowanie o mnie list motywacyjny cover letter summary about me",
        "message": "wiadomość treść list motywacyjny cover letter message note",
        "query": "wyszukaj szukaj search find query",
        "search": "wyszukaj szukaj search find query",
    }

    def __init__(self, router: Any):
        """Initializes PayloadBinder with a neural model/router instance."""
        self.router = router

    def bind(
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

        # Parse and expand payload keys
        parsed_items = []
        for key, raw_val in payload.items():
            if raw_val is None:
                continue

            k_lower = str(key).lower().strip()
            clean_k = k_lower.replace("_", " ").replace("-", " ")

            is_file = False
            file_paths: Optional[List[str]] = None
            if isinstance(raw_val, (list, tuple)):
                if any(any(str(x).lower().endswith(ext) for ext in self.COMMON_FILE_EXTENSIONS) for x in raw_val) or any(
                    kw in k_lower for kw in self.FILE_KEYWORDS
                ):
                    is_file = True
                    file_paths = [str(x) for x in raw_val]
            elif isinstance(raw_val, str):
                s_val = raw_val.lower().strip()
                if any(s_val.endswith(ext) for ext in self.COMMON_FILE_EXTENSIONS) or any(kw in k_lower for kw in self.FILE_KEYWORDS):
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

            if k_lower in self.SEMANTIC_EXPANSIONS:
                query = self.SEMANTIC_EXPANSIONS[k_lower]
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
        pair_meta: List[Tuple[int, int]] = []  # (item_idx, cand_idx)

        for p_idx, p_item in enumerate(parsed_items):
            for c_idx, cand in enumerate(valid_cands):
                role = cand.get("role") or ""
                name = cand.get("name") or ""
                ctx = cand.get("context") or ""
                desc = f"[{role}] {name} (context: {ctx})"
                if getattr(self.router, "is_reranker", False):
                    pairs.append([p_item["query"], desc])
                else:
                    hypothesis = f"This element represents, opens, or is used to {p_item['query']}."
                    pairs.append([desc, hypothesis])
                pair_meta.append((p_idx, c_idx))

        raw_probs = [0.0] * len(pairs)
        if pairs and hasattr(self.router, "tokenizer") and hasattr(self.router, "model"):
            indexed_pairs = list(enumerate(pairs))
            indexed_pairs.sort(key=lambda x: len(x[1][0]) + len(x[1][1]))
            batch_size = 64

            with torch.no_grad():
                for b_start in range(0, len(indexed_pairs), batch_size):
                    b_slice = indexed_pairs[b_start : b_start + batch_size]
                    b_pairs = [item[1] for item in b_slice]
                    b_indices = [item[0] for item in b_slice]

                    inputs = self.router.tokenizer(
                        b_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                    ).to(self.router.device)
                    outputs = self.router.model(**inputs)
                    logits = outputs.logits

                    if getattr(self.router, "is_reranker", False):
                        squeezed = logits.squeeze(-1)
                        if "mmarco" in getattr(self.router, "model_name", "").lower():
                            b_probs = torch.sigmoid((squeezed + 4.5) / 2.0)
                        else:
                            b_probs = torch.sigmoid(squeezed)
                        if b_probs.ndim == 0:
                            b_probs = b_probs.unsqueeze(0)
                    else:
                        binary_logits = logits[:, [self.router.contra_idx, self.router.entail_idx]]
                        b_probs = torch.softmax(binary_logits, dim=-1)[:, 1]

                    for orig_idx, prob_val in zip(b_indices, b_probs, strict=False):
                        raw_probs[orig_idx] = float(prob_val.item())

            if getattr(self.router, "device", "") == "cuda":
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
                has_file_kw = any(kw in c_name for kw in self.FILE_KEYWORDS) or any(
                    w in c_name for w in ["wybierz plik", "dodaj plik", "browse", "załącz", "załaduj"]
                )
                if has_file_kw or c_role == "button":
                    boost += 0.45
                if c_role in ("textbox", "searchbox") and not has_file_kw:
                    boost -= 0.50
            else:
                has_file_kw = any(kw in c_name for kw in self.FILE_KEYWORDS) or "wybierz plik" in c_name
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

            if final_score >= min_confidence:
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

        # Greedy maximum-weight bipartite matching
        match_candidates.sort(key=lambda x: x["score"], reverse=True)

        used_payload_keys: Set[str] = set()
        used_cands: Set[int] = set()
        matched_actions: List[Dict[str, Any]] = []

        for m in match_candidates:
            if m["key"] in used_payload_keys:
                continue
            if m["c_idx"] in used_cands:
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
