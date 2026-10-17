"""
Semantic ARIA accessibility tree parser for Cognitive Browser.
Extracts interactive and structural elements from Playwright aria_snapshot() YAML
into enriched, context-aware candidate objects for model evaluation and DOM execution.
"""

import re
from typing import Any, Dict, List, Optional


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

    # Matches lines like: - button "Szukaj" [expanded]: text or - button 'Szukaj'
    PATTERN = re.compile(
        r'^\s*-\s*([a-zA-Z0-9_-]+)(?:\s+["\']([^"\']*)["\'])?(?:\s+\[([^\]]+)\])*(?::\s*(.*))?$'
    )

    @classmethod
    def parse_snapshot(
        cls,
        snapshot_yaml: str,
        max_candidates: Optional[int] = 1000,
        require_input_only: bool = False,
        skip_boilerplate: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Parses YAML output of page.aria_snapshot() into structured candidate objects.

        Args:
            snapshot_yaml: Raw YAML string returned by page.aria_snapshot().
            max_candidates: Safety limit for candidate count (default: 1000, None for unlimited).
            require_input_only: If True, only text/search/combo inputs are retained.
            skip_boilerplate: If True, skips common footer/contentinfo noise candidates.

        Returns:
            List of candidate dictionaries enriched with role, name, context, nth index, and section flags.
        """
        candidates: List[Dict[str, Any]] = []
        context_stack: List[tuple[int, str]] = []
        role_name_counts: Dict[tuple[str, str], int] = {}

        lines = snapshot_yaml.splitlines()
        for line in lines:
            clean = line.strip()
            if not clean.startswith("-"):
                continue

            indent = len(line) - len(line.lstrip(" "))

            # Trailing URL metadata for preceding link candidate (Playwright snapshot extension)
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

            # Strip Private Use Area (PUA) font icons (both raw unicode and literal \u escapes)
            clean_label = re.sub(
                r"[\ue000-\uf8ff\uf000-\uffff]|\\u[e-fE-F][0-9a-fA-F]{3}", "", label
            ).strip()
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

            # Filter for text input operations when required
            if require_input_only and role not in ["textbox", "combobox", "searchbox", "spinbutton"]:
                continue

            # Check section context breadcrumbs
            context_str = " > ".join(
                c[1] for c in context_stack[-3:] if c[0] < indent or c[1].startswith("heading:")
            )
            context_lower = context_str.lower()

            # Optional early skip for footer / contentinfo noise
            if skip_boilerplate and ("contentinfo" in context_lower or "footer" in context_lower):
                continue

            # Allow unnamed inputs (textbox/searchbox/combobox) so placeholder or role-based fill works
            is_unnamed_input = role in ("textbox", "searchbox", "combobox", "spinbutton") and not label
            if role in cls.INTERACTIVE_ROLES and (label or is_unnamed_input):
                effective_label = label or role
                key = (role, effective_label.lower())
                nth_index = role_name_counts.get(key, 0)
                role_name_counts[key] = nth_index + 1

                cand_obj: Dict[str, Any] = {
                    "ref": f"aria-{len(candidates)}",
                    "role": role,
                    "tag": role,
                    "name": label[:120],
                    "text": effective_label[:120],
                    "context": context_str,
                    "nth": nth_index,
                    "raw": clean,
                    "is_main": "main" in context_lower,
                    "is_footer": "contentinfo" in context_lower or "footer" in context_lower,
                    "is_header": "banner" in context_lower or "header" in context_lower,
                }
                if attr_str:
                    cand_obj["state"] = attr_str
                candidates.append(cand_obj)

            if max_candidates is not None and len(candidates) >= max_candidates:
                break

        return candidates
