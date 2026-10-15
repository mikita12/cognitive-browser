"""
AgentCore - Uniwersalny Autonomiczny Agent Przeglądarkowy (ARIA + DeBERTa System 1).
Działa na dowolnej stronie internetowej, wykorzystując oficjalne Drzewo Dostępności (ARIA)
oraz model dyskryminacyjny DeBERTa-v3 do podejmowania decyzji w ułamku sekundy.
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional
from playwright.async_api import Page, async_playwright
try:
    from .deberta_router import DebertaDecisionRouter
except ImportError:
    from deberta_router import DebertaDecisionRouter


class AriaElementParser:
    """Parsuje oficjalny format page.aria_snapshot() na obiekty semantyczne."""

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

            # Jeśli linia to url dla poprzedniego linku
            if clean.startswith("- /url:") and candidates:
                u = clean.replace("- /url:", "").strip().strip('"').strip("'")
                candidates[-1]["url"] = u
                continue

            # Jeśli linia to placeholder dla poprzedniego elementu
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
    Główna pętla agenta kognitywnego (System 1 + Playwright).
    Zero halucynacji selektorów CSS, natywne lokatory ARIA.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cuda",
        router: Optional[Any] = None
    ):
        if router is not None:
            self.router = router
        else:
            try:
                from .router_factory import get_decision_router
            except ImportError:
                from router_factory import get_decision_router
            self.router = get_decision_router(device=device)
        self.execution_log = []

    async def auto_dismiss_overlays(self, page: Page) -> bool:
        """
        Uniwersalne wykrywanie i zamykanie banerów cookies / zgód / modali RODO.
        Działa na dowolnej stronie w ~30-50 ms.
        """
        # Sprawdzamy najpopularniejsze identyfikatory OneTrust/Didomi/Klaro/Cookies lub przyciski zgód
        consent_selectors = [
            "#onetrust-accept-btn-handler",
            "#klaro-cookies-modal button",
            ".cookie-notice button",
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
                    print(f"  [Auto-Dismiss] Wykryto modal zgód ({sel})! Klikam akceptację...")
                    await loc.click(timeout=2000)
                    await page.wait_for_timeout(500)
                    return True
            except Exception:
                pass

        # Fallback przez ARIA Snapshot
        snapshot = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(snapshot, max_candidates=25)
        res = self.router.evaluate_candidates("accept all cookies or close modal dialog", candidates)
        if res["confidence"] > 0.60 and res["target_ref"] != "NONE":
            best = res["ranked_candidates"][0]
            if "cookie" in best["text"].lower() or "zgadz" in best["text"].lower() or "akcept" in best["text"].lower():
                print(f"  [Auto-Dismiss] Wykryto przycisk cookies przez ARIA: [{best['role']}] \"{best['text']}\"")
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
        Wykonuje pojedynczy krok autonomiczny na podstawie instrukcji w języku naturalnym.
        """
        t0 = time.perf_counter()

        # 1. Pobranie czystego drzewa ARIA
        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(
            snapshot_yaml,
            max_candidates=50,
            require_input_only=bool(fill_value)
        )

        # Fallback jeśli require_input_only nie znalazło żadnego elementu
        if not candidates and fill_value:
            candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=50, require_input_only=False)

        # 2. Błyskawiczna ewaluacja DeBERTa-v3 na GPU
        decision = self.router.evaluate_candidates(intent=instruction, candidates=candidates)
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
            step_info["error"] = "Niska pewność dopasowania elementu (wymagana eskalacja do Systemu 2)"
            self.execution_log.append(step_info)
            return step_info

        best_cand = decision["ranked_candidates"][0]
        role = best_cand["role"]
        name = best_cand["name"]

        print(f"  -> Wybrano: [{role}] \"{name}\" (Pewność: {decision['confidence']*100:.1f}%, Czas: {latency_ms:.1f} ms)")

        # 3. Wykonanie akcji przez natywny lokator Playwrighta z bezpiecznym timeoutem 5s
        try:
            clean_name = re.sub(r'[\ue000-\uf8ff]', '', name).strip()
            name_pattern = re.compile(re.escape(clean_name), re.I) if clean_name else name

            if fill_value:
                if role in ["combobox", "textbox", "searchbox", "spinbutton"]:
                    locator = page.get_by_role(role, name=name_pattern).first
                    await locator.fill(fill_value, timeout=5000)
                    step_info["action_taken"] = f"fill('{fill_value}')"
                else:
                    # Wybrany element nie jest polem tekstowym a chciano wpisać tekst -> eskalacja
                    step_info["error"] = f"Wybrano element typu [{role}], który nie przyjmuje tekstu fill_value='{fill_value}'"
                    self.execution_log.append(step_info)
                    return step_info
            else:
                # Akcja kliknięcia
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
