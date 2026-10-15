"""
Browser execution engine with native security protections.
Handles Playwright lifecycle, overlay auto-dismissal, and exact-match element interactions.
"""

import re
from typing import Any, Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from .reflex import AriaElementParser, ReflexCore
from .security import setup_ssrf_route_guard


class BrowserEngine:
    """Manages secure Playwright sessions and executes element interactions."""

    def __init__(self, reflex: Optional[ReflexCore] = None):
        self.reflex = reflex or ReflexCore()
        self._playwright: Optional[Any] = None
        self._browser: Optional[Browser] = None
        self._active_context: Optional[BrowserContext] = None
        self._active_page: Optional[Page] = None

    async def get_page(self) -> Page:
        """Retrieves or creates a secure page instance."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu-rasterization",
                ],
            )
            self._active_context = await self._browser.new_context(
                service_workers="block",
                viewport={"width": 1280, "height": 900},
            )
            await setup_ssrf_route_guard(self._active_context)
            self._active_page = await self._active_context.new_page()

        if self._active_page is None or self._active_page.is_closed():
            assert self._active_context is not None
            self._active_page = await self._active_context.new_page()

        return self._active_page

    async def close(self) -> None:
        """Closes browser instances and terminates Playwright."""
        if self._active_context:
            await self._active_context.close()
            self._active_context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._active_page = None

    async def auto_dismiss_overlays(self, page: Page) -> bool:
        """Quickly checks for and dismisses consent banners and cookie dialogs."""
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
        ]

        for sel in consent_selectors:
            loc = page.locator(sel).first
            try:
                if await loc.is_visible(timeout=300):
                    await loc.click(timeout=1500)
                    await page.wait_for_timeout(300)
                    return True
            except Exception:
                pass

        # NLI Fallback on ARIA snapshot
        try:
            snapshot = await page.aria_snapshot()
            candidates = AriaElementParser.parse_snapshot(snapshot, max_candidates=25)
            res = self.reflex.evaluate_candidates("accept all cookies or close modal dialog", candidates)
            if res["confidence"] > 0.60 and res["target_ref"] != "NONE":
                best = res["ranked_candidates"][0]
                text_lower = (best.get("name") or "").lower()
                if any(k in text_lower for k in ["cookie", "accept", "agree", "allow", "zgadz", "akcept"]):
                    await self.click_element(page, role=best["role"], name=best["name"])
                    await page.wait_for_timeout(500)
                    return True
        except Exception:
            pass

        return False

    async def click_element(self, page: Page, role: str, name: str) -> bool:
        """Clicks an element using exact-match prioritizing to avoid substring collisions."""
        clean_name = re.sub(r"[\ue000-\uf8ff]", "", name).strip()
        locator = None

        if clean_name:
            exact_pattern = re.compile(r"^\s*" + re.escape(clean_name) + r"\s*$", re.I)
            exact_loc = page.get_by_role(role, name=exact_pattern).first  # type: ignore[arg-type]
            try:
                if await exact_loc.count() > 0:
                    locator = exact_loc
            except Exception:
                pass

        if locator is None:
            name_pattern = re.compile(re.escape(clean_name), re.I) if clean_name else name
            locator = page.get_by_role(role, name=name_pattern).first  # type: ignore[arg-type]

        try:
            await locator.click(timeout=3500)
        except Exception:
            await locator.click(timeout=2000, force=True)

        return True

    async def fill_element(self, page: Page, role: str, name: str, value: str) -> bool:
        """Types text into an input element using exact-match prioritizing."""
        clean_name = re.sub(r"[\ue000-\uf8ff]", "", name).strip()
        locator = None

        if clean_name:
            exact_pattern = re.compile(r"^\s*" + re.escape(clean_name) + r"\s*$", re.I)
            exact_loc = page.get_by_role(role, name=exact_pattern).first  # type: ignore[arg-type]
            try:
                if await exact_loc.count() > 0:
                    locator = exact_loc
            except Exception:
                pass

        if locator is None:
            name_pattern = re.compile(re.escape(clean_name), re.I) if clean_name else name
            locator = page.get_by_role(role, name=name_pattern).first  # type: ignore[arg-type]

        await locator.fill(value, timeout=4000)
        return True
