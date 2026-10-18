"""
Semantic payload and form parameter binding engine for Cognitive Browser.
Performs universal bipartite matching between user-provided data dictionaries
and DOM interactive controls using structural role partitioning and neural scoring.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .models import ActionCommand, ActionType, CandidateElement


def _normalize_ascii(text: str) -> str:
    """Strips diacritics, hyphens, and punctuation for universal lexical comparison."""
    text = text.lower().replace("ł", "l").replace("-", "").replace("_", " ")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", ascii_str).strip()


def _lexical_overlap_score(key_str: str, label_str: str) -> float:
    """Computes normalized token/stem overlap between a payload key and a control label."""
    k_norm = _normalize_ascii(key_str)
    l_norm = _normalize_ascii(label_str)
    if not k_norm or not l_norm:
        return 0.0
    if k_norm == l_norm:
        return 0.65
    # Condensed check for compound words like "e-mail" vs "email"
    if k_norm.replace(" ", "") in l_norm.replace(" ", ""):
        return 0.50

    k_tokens = [t for t in k_norm.split() if len(t) >= 2]
    l_tokens = [t for t in l_norm.split() if len(t) >= 2]
    if not k_tokens or not l_tokens:
        return 0.0

    matches = 0
    for kt in k_tokens:
        stem = kt[:4] if len(kt) >= 5 else kt
        if any(lt == kt or lt.startswith(stem) or kt.startswith(lt[:4]) for lt in l_tokens if len(lt) >= 2):
            matches += 1
    return 0.45 * (matches / len(k_tokens))


class PayloadBinder:
    """
    Binds arbitrary payload dictionary parameters to web form controls.
    1. Partitions payload values by structural type (file upload, boolean toggle, text/select input).
    2. Filters DOM controls to compatible ARIA roles for each structural type.
    3. Scores (key, control_label) pairs via `router.score_pairs()` without hardcoded synonym dictionaries.
    4. Solves 1-to-1 greedy bipartite matching and orders resulting ActionCommands top-to-bottom by DOM index.
    """

    FILE_EXTENSIONS: Tuple[str, ...] = (
        ".pdf",
        ".docx",
        ".doc",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".txt",
        ".csv",
        ".zip",
        ".xlsx",
        ".xls",
        ".json",
        ".rtf",
        ".odt",
        ".svg",
        ".gif",
        ".mp4",
        ".mov",
    )

    FORM_ROLES: Set[str] = {
        "textbox",
        "searchbox",
        "combobox",
        "spinbutton",
        "checkbox",
        "radio",
        "switch",
        "button",
        "option",
    }

    def __init__(self, router: Any):
        """Initializes PayloadBinder with a neural scorer instance exposing `score_pairs()`."""
        self.router = router

    @classmethod
    def _is_file_value(cls, raw_val: Any) -> Tuple[bool, Optional[List[str]]]:
        """Detects whether a payload value represents one or more file paths based on file extensions."""
        if isinstance(raw_val, (list, tuple)) and raw_val:
            strs = [str(x).strip() for x in raw_val if x is not None]
            if strs and any(s.lower().endswith(cls.FILE_EXTENSIONS) or "/" in s for s in strs):
                return True, strs
        elif isinstance(raw_val, str):
            s_val = raw_val.strip()
            if s_val.lower().endswith(cls.FILE_EXTENSIONS):
                return True, [s_val]
        return False, None

    @staticmethod
    def _is_bool_value(raw_val: Any) -> Tuple[bool, bool]:
        """Detects whether a payload value is a boolean flag."""
        if isinstance(raw_val, bool):
            return True, raw_val
        if isinstance(raw_val, str) and raw_val.strip().lower() in ("true", "false"):
            return True, raw_val.strip().lower() == "true"
        return False, False

    def bind_commands(
        self,
        payload: Dict[str, Any],
        candidates: Sequence[Union[CandidateElement, Dict[str, Any]]],
        min_confidence: float = 0.20,
    ) -> List[ActionCommand]:
        """
        Matches payload entries 1-to-1 against form controls and returns ordered ActionCommand objects.
        """
        if not payload or not candidates:
            return []

        elements = [CandidateElement.from_dict(c) for c in candidates]
        has_comboboxes = any(el.role == "combobox" for el in elements)

        valid_elements: List[CandidateElement] = []
        for el in elements:
            if el.role not in self.FORM_ROLES:
                continue
            if el.role == "option" and has_comboboxes:
                continue
            valid_elements.append(el)

        if not valid_elements:
            return []

        # 1. Classify payload items structurally
        parsed_items: List[Dict[str, Any]] = []
        for key, raw_val in payload.items():
            if raw_val is None:
                continue
            clean_key = str(key).replace("_", " ").replace("-", " ").strip()
            is_file, file_paths = self._is_file_value(raw_val)
            is_bool, bool_val = (False, False) if is_file else self._is_bool_value(raw_val)

            parsed_items.append(
                {
                    "key": str(key),
                    "clean_key": clean_key,
                    "raw_val": raw_val,
                    "is_file": is_file,
                    "file_paths": file_paths,
                    "is_bool": is_bool,
                    "bool_val": bool_val,
                }
            )

        if not parsed_items:
            return []

        # 2. Build compatible (payload_item, candidate_element) pairs for neural scoring
        pairs: List[List[str]] = []
        pair_meta: List[Tuple[int, int]] = []

        for p_idx, p_item in enumerate(parsed_items):
            for c_idx, el in enumerate(valid_elements):
                # Strict structural role compatibility filter
                if p_item["is_bool"]:
                    if not el.is_boolean_control:
                        continue
                elif p_item["is_file"]:
                    if el.role not in ("button", "textbox"):
                        continue
                else:
                    if el.role not in ("textbox", "searchbox", "combobox", "spinbutton", "option"):
                        continue

                label = el.name or el.text or el.role
                query_text = p_item["clean_key"]
                if not p_item["is_file"] and not p_item["is_bool"] and el.role in ("combobox", "option"):
                    query_text = f"{p_item['clean_key']}: {p_item['raw_val']}"

                if getattr(self.router, "is_reranker", True):
                    pairs.append([query_text, label])
                else:
                    pairs.append([label, f"This form field is for {query_text}."])
                pair_meta.append((p_idx, c_idx))

        # 3. Score all compatible pairs in a single call to router.score_pairs()
        neural_scores = (
            self.router.score_pairs(pairs) if pairs and hasattr(self.router, "score_pairs") else [0.5] * len(pairs)
        )

        # 4. Combine neural probability with normalized lexical token overlap
        match_candidates: List[Dict[str, Any]] = []
        for idx, (p_idx, c_idx) in enumerate(pair_meta):
            p_item = parsed_items[p_idx]
            el = valid_elements[c_idx]
            base_prob = neural_scores[idx]
            lex_boost = _lexical_overlap_score(p_item["clean_key"], f"{el.name} {el.context}")

            # Prefer buttons over textboxes for file uploads
            role_bonus = 0.15 if (p_item["is_file"] and el.role == "button") else 0.0
            final_score = min(0.99, base_prob + lex_boost + role_bonus)

            if final_score < min_confidence:
                continue

            if p_item["is_file"]:
                action: ActionType = "upload_file"
                val: Optional[str] = None
                files = p_item["file_paths"]
            elif p_item["is_bool"]:
                action = "check" if p_item["bool_val"] else "uncheck"
                val = None
                files = None
            elif el.role in ("combobox", "option"):
                action = "select"
                val = str(p_item["raw_val"])
                files = None
            else:
                action = "fill"
                val = str(p_item["raw_val"])
                files = None

            match_candidates.append(
                {
                    "score": final_score,
                    "p_idx": p_idx,
                    "c_idx": c_idx,
                    "key": p_item["key"],
                    "element": el,
                    "action": action,
                    "value": val,
                    "files": files,
                }
            )

        # 5. Greedy maximum-weight 1-to-1 bipartite matching
        match_candidates.sort(key=lambda x: x["score"], reverse=True)
        used_keys: Set[str] = set()
        used_cands: Set[int] = set()
        matched_pairs: List[Tuple[int, ActionCommand]] = []

        for m in match_candidates:
            if m["key"] in used_keys or m["c_idx"] in used_cands:
                continue
            used_keys.add(m["key"])
            used_cands.add(m["c_idx"])

            cmd = ActionCommand(
                action=m["action"],
                target=m["element"],
                value=m["value"],
                files=m["files"],
                confidence=round(m["score"], 4),
                source="payload_binding",
                payload_key=m["key"],
            )
            matched_pairs.append((m["c_idx"], cmd))

        # 6. Sort matched commands top-to-bottom by DOM order
        matched_pairs.sort(key=lambda item: item[0])
        return [cmd for _, cmd in matched_pairs]

    def bind(
        self,
        payload: Dict[str, Any],
        candidates: Sequence[Union[CandidateElement, Dict[str, Any]]],
        min_confidence: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible wrapper returning a list of action dictionaries.
        """
        commands = self.bind_commands(payload, candidates, min_confidence=min_confidence)
        return [
            {
                "payload_key": cmd.payload_key,
                "action": cmd.action,
                "role": cmd.role,
                "name": cmd.name,
                "nth": cmd.nth,
                "value": cmd.value,
                "file_paths": cmd.files,
                "confidence": cmd.confidence,
            }
            for cmd in commands
        ]
