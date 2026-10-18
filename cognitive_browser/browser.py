"""
Browser execution engine with Stealth anti-bot protections, native security guards, and RouteCache.
Provides a complete suite of autonomous DOM interaction and page iteration methods for ReflexCore:
click, fill (+ auto-submit), select_option, toggle_checkbox, hover, press_key, scroll_page, and auto_dismiss_overlays.
"""

import os
import re
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from .cache import RouteCache
from .captcha import TURNSTILE_INTERCEPT_SCRIPT, CaptchaResolver
from .models import ActionCommand, PageState
from .parser import AriaElementParser
from .security import setup_ssrf_route_guard

STEALTH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
window.chrome = { runtime: {}, loadTimes: function() {}, csi: function() {}, app: {} };
Object.defineProperty(navigator, 'languages', { get: () => ['pl-PL', 'pl', 'en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
"""


class BrowserEngine:
    """Pure Playwright execution driver with Stealth anti-bot protections, SSRF guards, and ActionCommand execution."""

    def __init__(
        self,
        reflex: Optional[Any] = None,
        route_cache: Optional[RouteCache] = None,
        captcha_resolver: Optional[CaptchaResolver] = None,
    ):
        self.reflex = reflex
        self.route_cache = route_cache or RouteCache()
        self.captcha_resolver = captcha_resolver or CaptchaResolver()
        self._playwright: Optional[Any] = None
        self._browser: Optional[Browser] = None
        self._active_context: Optional[BrowserContext] = None
        self._active_page: Optional[Page] = None

    async def get_page(self) -> Page:
        """Retrieves or creates a stealth, SSRF-guarded page instance."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=False,
                args=[
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu-rasterization",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--window-size=1366,900",
                ],
            )
            self._active_context = await self._browser.new_context(
                service_workers="allow",
                viewport={"width": 1366, "height": 900},
                user_agent=STEALTH_USER_AGENT,
                locale="pl-PL",
                timezone_id="Europe/Warsaw",
                extra_http_headers={
                    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                    "Sec-Ch-Ua-Mobile": "?0",
                    "Sec-Ch-Ua-Platform": '"Linux"',
                },
            )
            await self._active_context.add_init_script(STEALTH_INIT_SCRIPT)
            await self._active_context.add_init_script(TURNSTILE_INTERCEPT_SCRIPT)
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
            "button[data-role='accept-consent']",
            "button[data-testid*='accept']",
            "button:has-text('Accept all')",
            "button:has-text('Accept')",
            "button:has-text('Agree')",
            "button:has-text('Allow all')",
            "button:has-text('I agree')",
            "button:has-text('Zezwalam')",
            "button:has-text('Zgadzam się')",
            "button:has-text('Akceptuję')",
            "button:has-text('Zaakceptuj')",
            "button:has-text('Przejdź dalej')",
            "button:has-text('W porządku')",
            "button[id*='accept']",
            "button[id*='consent']",
        ]

        combined_selector = ", ".join(consent_selectors)
        try:
            fast_loc = page.locator(combined_selector).first
            if await fast_loc.is_visible(timeout=250):
                await fast_loc.click(timeout=1200)
                await page.wait_for_timeout(200)
                return True
        except Exception:
            pass

        return False

    async def settle_dom(self, page: Page, timeout_ms: int = 1200) -> None:
        """Waits briefly for DOM and SPA transitions to stabilize after an action."""
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        except Exception:
            pass
        try:
            await page.wait_for_timeout(200)
        except Exception:
            pass

    async def get_page_state(self, page: Page, max_candidates: int = 1000) -> PageState:
        """Captures current PageState (URL, title, ARIA YAML, and parsed CandidateElement list)."""
        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_elements(snapshot_yaml, max_candidates=max_candidates)
        return PageState(
            url=page.url,
            title=await page.title(),
            snapshot_yaml=snapshot_yaml,
            candidates=candidates,
        )

    async def auto_resolve_captcha_if_present(self, page: Page, timeout: float = 25.0) -> Dict[str, Any]:
        """
        Inspects page for CAPTCHA challenges (Cloudflare Turnstile, reCAPTCHA, hCaptcha)
        and automatically resolves them via CapSolver API.
        """
        return await self.captcha_resolver.auto_detect_and_solve(page, timeout=timeout)

    async def _resolve_locator(self, page: Page, role: str, name: str, nth: int = 0) -> Any:
        """Resolves a Playwright locator with exact-match priority, role fallback, and text fallback."""
        clean_name = re.sub(r"[\ue000-\uf8ff\uf000-\uffff]|\\u[e-fE-F][0-9a-fA-F]{3}", "", name or "").strip()
        locator = None

        if clean_name and role:
            escaped_name = re.escape(clean_name).replace("/", r"\/")
            exact_pattern = re.compile(r"^\s*" + escaped_name + r"\s*$", re.I)
            try:
                base_loc = page.get_by_role(role, name=exact_pattern)  # type: ignore[arg-type]
                exact_loc = base_loc.nth(nth) if nth > 0 else base_loc.first
                if await exact_loc.count() > 0:
                    locator = exact_loc
            except Exception:
                pass

        if locator is None and clean_name and role:
            try:
                escaped_name = re.escape(clean_name).replace("/", r"\/")
                name_pattern = re.compile(escaped_name, re.I)
                base_loc = page.get_by_role(role, name=name_pattern)  # type: ignore[arg-type]
                sub_loc = base_loc.nth(nth) if nth > 0 else base_loc.first
                if await sub_loc.count() > 0:
                    locator = sub_loc
            except Exception:
                pass

        if locator is None and clean_name and role:
            try:
                base_loc = page.get_by_role(role, name=clean_name, exact=True)  # type: ignore[arg-type]
                sub_loc = base_loc.nth(nth) if nth > 0 else base_loc.first
                if await sub_loc.count() > 0:
                    locator = sub_loc
            except Exception:
                pass

        if locator is None and role and not clean_name:
            try:
                base_loc = page.get_by_role(role)  # type: ignore[arg-type]
                role_loc = base_loc.nth(nth) if nth > 0 else base_loc.first
                if await role_loc.count() > 0:
                    locator = role_loc
            except Exception:
                pass

        if locator is None and clean_name:
            alt_roles = [
                "button",
                "link",
                "tab",
                "menuitem",
                "option",
                "checkbox",
                "radio",
                "textbox",
                "searchbox",
                "combobox",
            ]
            for alt_role in alt_roles:
                if alt_role == role:
                    continue
                try:
                    alt_pattern = re.compile(re.escape(clean_name), re.I)
                    alt_loc = page.get_by_role(alt_role, name=alt_pattern).first  # type: ignore[arg-type]
                    if await alt_loc.count() > 0:
                        locator = alt_loc
                        break
                except Exception:
                    pass

        if locator is None and clean_name:
            try:
                txt_loc = page.get_by_text(clean_name).first
                if await txt_loc.count() > 0:
                    locator = txt_loc
            except Exception:
                pass

        if locator is None:
            locator = page.get_by_text(name).first

        return locator

    async def click_element(self, page: Page, role: str, name: str, nth: int = 0) -> bool:
        """Clicks an element using exact-match prioritizing and optional nth index."""
        locator = await self._resolve_locator(page, role=role, name=name, nth=nth)
        try:
            await locator.scroll_into_view_if_needed(timeout=1500)
        except Exception:
            pass
        try:
            await locator.click(timeout=3500)
        except Exception:
            await locator.click(timeout=2000, force=True)
        return True

    async def fill_element(
        self, page: Page, role: str, name: str, value: str, nth: int = 0, submit: bool = False
    ) -> bool:
        """Types text into an input element using exact-match prioritizing and optional Enter submission."""
        clean_name = re.sub(r"[\ue000-\uf8ff]", "", name or "").strip()
        locator = None

        if clean_name and role:
            exact_pattern = re.compile(r"^\s*" + re.escape(clean_name) + r"\s*$", re.I)
            base_loc = page.get_by_role(role, name=exact_pattern)  # type: ignore[arg-type]
            exact_loc = base_loc.nth(nth) if nth > 0 else base_loc.first
            try:
                if await exact_loc.count() > 0:
                    locator = exact_loc
            except Exception:
                pass

        if locator is None and role:
            try:
                name_pattern = re.compile(re.escape(clean_name), re.I) if clean_name else name
                base_loc = page.get_by_role(role, name=name_pattern) if clean_name else page.get_by_role(role)  # type: ignore[arg-type]
                cand_loc = base_loc.nth(nth) if nth > 0 else base_loc.first
                if await cand_loc.count() > 0:
                    locator = cand_loc
            except Exception:
                pass

        # Fallback to placeholder or generic visible search/text input
        if locator is None and clean_name:
            try:
                ph_loc = page.get_by_placeholder(re.compile(re.escape(clean_name), re.I)).first
                if await ph_loc.count() > 0:
                    locator = ph_loc
            except Exception:
                pass

        if locator is None:
            try:
                fallback_input = page.locator(
                    "input[type='search'], input[type='text'], input:not([type]), textarea"
                ).first
                if await fallback_input.count() > 0:
                    locator = fallback_input
            except Exception:
                pass

        if locator is None:
            name_pattern = re.compile(re.escape(clean_name), re.I) if clean_name else name
            base_loc = page.get_by_role(role, name=name_pattern)  # type: ignore[arg-type]
            locator = base_loc.nth(nth) if nth > 0 else base_loc.first

        await locator.fill(value, timeout=4000)
        if submit:
            try:
                await locator.press("Enter", timeout=2000)
            except Exception:
                await page.keyboard.press("Enter")
            await page.wait_for_timeout(450)
        return True

    async def select_option(self, page: Page, role: str, name: str, value: str, nth: int = 0) -> bool:
        """Selects an option in a <select>, combobox, or listbox."""
        # 1. If role is option, prioritize finding the parent combobox or select
        target_role = "combobox" if role == "option" else (role or "combobox")
        locator = await self._resolve_locator(page, role=target_role, name=name, nth=nth)

        # 2. Check if element is an <option> and get its parent <select>
        try:
            tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
            if tag_name == "option":
                parent_select = locator.locator("xpath=ancestor::select[1]")
                if await parent_select.count() > 0:
                    await parent_select.select_option(label=value, timeout=2500)
                    return True
        except Exception:
            pass

        try:
            await locator.select_option(label=value, timeout=2500)
            return True
        except Exception:
            try:
                await locator.select_option(value=value, timeout=2500)
                return True
            except Exception:
                pass

        # 3. Custom dropdown fallback: click combobox, then click option matching value
        try:
            await locator.click(timeout=1500)
            await page.wait_for_timeout(150)
            await self.click_element(page, role="option", name=value)
            return True
        except Exception:
            pass

        # 4. Fallback: page-level select containing the option
        try:
            page_select = page.locator(f"select:has(option:text-is('{value}'))").first
            if await page_select.count() > 0:
                await page_select.select_option(label=value, timeout=2000)
                return True
        except Exception:
            pass

        return False

    async def toggle_checkbox(self, page: Page, role: str, name: str, checked: bool = True, nth: int = 0) -> bool:
        """Checks or unchecks a checkbox, radio button, or switch."""
        locator = await self._resolve_locator(page, role=role or "checkbox", name=name, nth=nth)
        try:
            if checked:
                await locator.check(timeout=2500)
            else:
                await locator.uncheck(timeout=2500)
        except Exception:
            await locator.click(timeout=2500, force=True)
        return True

    async def hover_element(self, page: Page, role: str, name: str, nth: int = 0) -> bool:
        """Hovers over an element to expand dropdown menus or tooltips."""
        locator = await self._resolve_locator(page, role=role, name=name, nth=nth)
        await locator.hover(timeout=2500)
        await page.wait_for_timeout(200)
        return True

    async def press_key(self, page: Page, key: str = "Enter") -> bool:
        """Presses a keyboard key on the active page (e.g., Enter, Escape, PageDown, ArrowDown)."""
        await page.keyboard.press(key)
        await page.wait_for_timeout(200)
        return True

    async def scroll_page(self, page: Page, direction: str = "down", amount: int = 750) -> bool:
        """Scrolls the page up, down, top, or bottom to reveal lazy-loaded content."""
        dir_lower = direction.lower()
        if dir_lower == "top":
            await page.evaluate("window.scrollTo(0, 0)")
        elif dir_lower == "bottom":
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif dir_lower == "up":
            await page.evaluate(f"window.scrollBy(0, -{abs(amount)})")
        else:
            await page.evaluate(f"window.scrollBy(0, {abs(amount)})")
        await page.wait_for_timeout(250)
        return True

    async def upload_file(
        self,
        page: Page,
        role: str = "",
        name: str = "",
        file_paths: Union[str, List[str]] = "",
        nth: int = 0,
    ) -> bool:
        """
        Uploads one or multiple files via direct <input type='file'> or modern dropzone/button triggers.
        Supports:
          1. Direct set_input_files on resolved locator if it's an <input type='file'>.
          2. page.expect_file_chooser() triggered by clicking the custom button / dropzone.
          3. Sibling/child <input type='file'> inside custom container.
          4. Page-wide fallback to first/nth <input type='file'>.
        """
        if not file_paths:
            return False

        files = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        files = [os.path.abspath(os.path.expanduser(f)) if os.path.exists(os.path.expanduser(f)) else f for f in files]

        # 1. Direct input locator if element is or wraps <input type='file'>
        locator = None
        if role or name:
            try:
                locator = await self._resolve_locator(page, role=role or "button", name=name, nth=nth)
            except Exception:
                locator = None

        if locator is not None:
            try:
                tag_type = await locator.evaluate("el => (el.tagName + ':' + (el.type || '')).toLowerCase()")
                if "input:file" in tag_type:
                    await locator.set_input_files(files, timeout=3000)
                    return True
            except Exception:
                pass

        # 2. Modern React/Vue file chooser event triggered by clicking the button / dropzone
        try:
            async with page.expect_file_chooser(timeout=2500) as fc_info:
                if locator is not None:
                    try:
                        await locator.click(timeout=1500)
                    except Exception:
                        await locator.click(timeout=1000, force=True)
            raw_val: Any = fc_info.value
            file_chooser: Any = await raw_val if hasattr(raw_val, "__await__") else raw_val
            await file_chooser.set_files(files)
            return True
        except Exception:
            pass

        # 3. Check for nested <input type='file'> inside resolved container
        if locator is not None:
            try:
                nested = locator.locator("input[type='file']").first
                if await nested.count() > 0:
                    await nested.set_input_files(files, timeout=2500)
                    return True
            except Exception:
                pass

        # 4. Fallback: page-level <input type='file'>
        try:
            page_inputs = page.locator("input[type='file']")
            cnt = await page_inputs.count()
            if cnt > 0:
                idx = min(nth, cnt - 1)
                await page_inputs.nth(idx).set_input_files(files, timeout=2500)
                return True
        except Exception:
            pass

        return False

    async def import_cookies_for_domain(
        self,
        page: Page,
        domain: str,
        browser_type: str = "firefox",
        profile_path: Optional[str] = None,
    ) -> int:
        """Imports cookies from system browsers (Firefox, Chrome) or cookie files into the active session."""
        from .cookies import import_system_cookies

        return await import_system_cookies(
            page.context, domain=domain, browser_type=browser_type, profile_path=profile_path
        )

    async def execute(self, page: Page, command: Union[ActionCommand, Dict[str, Any]]) -> str:
        """
        Executes an ActionCommand on the given Playwright page and stabilizes the DOM.
        Returns a concise string describing the executed interaction.
        """
        cmd = ActionCommand.from_dict(command)
        action_type = cmd.action
        role = cmd.role
        name = cmd.name
        nth = cmd.nth
        value = cmd.value
        cand_url = cmd.url

        if action_type == "scroll":
            direction = str(value or cmd.direction or "down")
            await self.scroll_page(page, direction=direction, amount=cmd.amount)
            return f"scroll('{direction}')"

        if action_type == "press":
            key = str(value or cmd.key or "Enter")
            await self.press_key(page, key=key)
            return f"press('{key}')"

        if action_type == "hover":
            await self.hover_element(page, role=role, name=name, nth=nth)
            return "hover()"

        if action_type == "upload_file":
            files = cmd.files or ([str(value)] if value else None)
            if files:
                await self.upload_file(page, role=role, name=name, file_paths=files, nth=nth)
                return f"upload_file('{files}')"

        if action_type in ("fill", "fill_and_submit") and value is not None:
            submit = action_type == "fill_and_submit"
            if nth > 0:
                await self.fill_element(page, role=role, name=name, value=str(value), nth=nth, submit=submit)
            else:
                await self.fill_element(page, role=role, name=name, value=str(value), submit=submit)
            return f"fill_and_submit('{value}')" if submit else f"fill('{value}')"

        if action_type == "select" and value is not None:
            await self.select_option(page, role=role, name=name, value=str(value), nth=nth)
            return f"select('{value}')"

        if action_type in ("check", "uncheck"):
            await self.toggle_checkbox(page, role=role, name=name, checked=(action_type == "check"), nth=nth)
            return f"{action_type}()"

        if cand_url and str(cand_url).startswith("http"):
            from .security import validate_url

            valid_target = validate_url(str(cand_url))
            await page.goto(valid_target, wait_until="domcontentloaded", timeout=15000)
            return f"goto('{valid_target}')"

        if nth > 0:
            await self.click_element(page, role=role, name=name, nth=nth)
        else:
            await self.click_element(page, role=role, name=name)
        return "click()"

    async def execute_reflex_action(self, page: Page, action_plan: Union[ActionCommand, Dict[str, Any]]) -> str:
        """Backward-compatible alias for BrowserEngine.execute()."""
        return await self.execute(page, action_plan)
