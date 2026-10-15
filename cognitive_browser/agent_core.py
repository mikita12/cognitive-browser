"""
AgentCore - Universal Autonomous Browser Agent (ARIA + Reflex Core).
Navigates web applications using official Accessibility Trees (ARIA snapshots)
and DeBERTa-v3 discriminative decision routing in sub-50ms with 0 LLM tokens.
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional
from playwright.async_api import Page, async_playwright
try:
    from .deberta_router import DebertaDecisionRouter
    from .security import mask_sensitive_value
    from .config import get_confidence_threshold
except ImportError:
    from deberta_router import DebertaDecisionRouter
    from security import mask_sensitive_value
    from config import get_confidence_threshold


class AriaElementParser:
    """Parses official page.aria_snapshot() YAML format into semantic candidate objects."""

    INTERACTIVE_ROLES = {
        "button", "link", "combobox", "textbox", "searchbox",
        "checkbox", "radio", "tab", "menuitem", "dialog", "alertdialog"
    }

    PATTERN = re.compile(r'^\s*-\s*([a-zA-Z0-9_-]+)(?:\s+"([^"]*)")?(?::\s*(.*))?$')

    @classmethod
    def parse_snapshot(
        cls,
        snapshot_yaml: str,
        max_candidates: int = 60,
        require_input_only: bool = False
    ) -> List[Dict[str, Any]]:
        candidates = []
        lines = snapshot_yaml.splitlines()
        for idx, line in enumerate(lines):
            clean = line.strip()
            if not clean.startswith("-"):
                continue

            # If line is URL for the preceding link candidate
            if clean.startswith("- /url:") and candidates:
                u = clean.replace("- /url:", "").strip().strip('"').strip("'")
                candidates[-1]["url"] = u
                continue

            # If line is placeholder for the preceding input candidate
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

            if require_input_only and role not in ["textbox", "combobox", "searchbox", "spinbutton"]:
                continue

            if role in cls.INTERACTIVE_ROLES and label:
                candidates.append({
                    "ref": f"aria-{len(candidates)}",
                    "role": role,
                    "tag": role,
                    "name": label[:120],
                    "text": label[:120],
                    "raw": clean
                })

            if len(candidates) >= max_candidates:
                break

        return candidates


class AgentCore:
    """
    Core cognitive browser agent engine (Reflex Core + Playwright).
    Eliminates fragile CSS selectors by leveraging native ARIA locators.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cuda",
        router: Optional[Any] = None,
        confidence_threshold: Optional[float] = None
    ):
        if router is not None:
            self.router = router
        else:
            try:
                from .router_factory import get_decision_router
            except ImportError:
                from router_factory import get_decision_router
            self.router = get_decision_router(device=device)
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else get_confidence_threshold()
        self.execution_log = []

    async def auto_dismiss_overlays(self, page: Page) -> bool:
        """
        Universal detection and dismissal of cookie / GDPR / consent modals.
        Executes on any page in ~30-50 ms.
        """
        consent_selectors = [
            "#onetrust-accept-btn-handler",
            "#klaro-cookies-modal button",
            ".cookie-notice button",
            "button:has-text('Accept all')",
            "button:has-text('Accept')",
            "button:has-text('Agree')",
            "button:has-text('Allow all')",
            "button:has-text('I agree')",
            "button:has-text('Zezwalam')",
            "button:has-text('Zgadzam się')",
            "button:has-text('Akceptuję')",
            "button:has-text('Zaakceptuj')",
            "button[id*='accept']",
            "button[id*='consent']",
            "button[data-testid*='accept']"
        ]

        for sel in consent_selectors:
            loc = page.locator(sel).first
            try:
                if await loc.is_visible(timeout=500):
                    print(f"  [Auto-Dismiss] Consent modal detected ({sel})! Clicking accept...")
                    await loc.click(timeout=2000)
                    await page.wait_for_timeout(500)
                    return True
            except Exception:
                pass

        # Fallback via ARIA Snapshot
        snapshot = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(snapshot, max_candidates=25)
        res = self.router.evaluate_candidates("accept all cookies or close modal dialog", candidates)
        if res["confidence"] > 0.60 and res["target_ref"] != "NONE":
            best = res["ranked_candidates"][0]
            text_lower = best["text"].lower()
            if any(k in text_lower for k in ["cookie", "accept", "agree", "allow", "zgadz", "akcept"]):
                print(f"  [Auto-Dismiss] Cookie button detected via ARIA: [{best['role']}] \"{best['text']}\"")
                await page.get_by_role(best["role"], name=best["name"]).first.click()
                await page.wait_for_timeout(1000)
                return True

        return False

    async def execute_step(
        self,
        page: Page,
        instruction: str,
        fill_value: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a single autonomous step based on natural language instruction.
        """
        t0 = time.perf_counter()

        # 1. Retrieve condensed ARIA tree
        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(
            snapshot_yaml,
            max_candidates=50,
            require_input_only=bool(fill_value)
        )

        # Fallback if require_input_only did not match any element
        if not candidates and fill_value:
            candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=50, require_input_only=False)

        # 2. Fast DeBERTa-v3 evaluation on GPU
        decision = self.router.evaluate_candidates(
            intent=instruction,
            candidates=candidates,
            confidence_threshold=self.confidence_threshold
        )
        latency_ms = decision["latency_ms"]

        step_info = {
            "instruction": instruction,
            "latency_ms": latency_ms,
            "confidence": decision["confidence"],
            "target_ref": decision["target_ref"],
            "action": decision["tool_action"],
            "executed": False,
            "error": None
        }

        if decision["should_escalate"] or decision["target_ref"] == "NONE":
            step_info["error"] = "Low confidence element match (escalation to Cortex required)"
            self.execution_log.append(step_info)
            return step_info

        best_cand = decision["ranked_candidates"][0]
        role = best_cand["role"]
        name = best_cand["name"]

        print(f"  -> Selected: [{role}] \"{name}\" (Confidence: {decision['confidence']*100:.1f}%, Latency: {latency_ms:.1f} ms)")

        # 3. Execute action via native Playwright locator with safety timeout
        try:
            clean_name = re.sub(r'[\ue000-\uf8ff]', '', name).strip()
            name_pattern = re.compile(re.escape(clean_name), re.I) if clean_name else name

            if fill_value:
                if role in ["combobox", "textbox", "searchbox", "spinbutton"]:
                    locator = page.get_by_role(role, name=name_pattern).first
                    await locator.fill(fill_value, timeout=5000)
                    masked_val = mask_sensitive_value(fill_value, instruction=instruction)
                    step_info["action_taken"] = f"fill('{masked_val}')"
                else:
                    masked_val = mask_sensitive_value(fill_value, instruction=instruction)
                    step_info["error"] = f"Selected element of role [{role}], which does not accept text input fill_value='{masked_val}'"
                    self.execution_log.append(step_info)
                    return step_info
            else:
                # Click action
                locator = page.get_by_role(role, name=name_pattern).first
                try:
                    await locator.click(timeout=3500)
                except Exception:
                    await locator.click(timeout=2000, force=True)
                step_info["action_taken"] = "click()"

            step_info["executed"] = True
            await page.wait_for_timeout(1000)
        except Exception as e:
            step_info["error"] = str(e)

        step_info["total_step_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self.execution_log.append(step_info)
        return step_info
