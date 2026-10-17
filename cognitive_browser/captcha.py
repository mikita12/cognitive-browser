"""
Automated CAPTCHA detection and solving module with CapSolver API integration.
Supports Cloudflare Turnstile, Google reCAPTCHA (v2/v3), and hCaptcha with zero LLM tokens.
"""

import asyncio
import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, Optional

from playwright.async_api import Page

from .config import get_capsolver_api_key

TURNSTILE_INTERCEPT_SCRIPT = """
(function() {
    window._turnstileOptions = null;
    window._turnstileCallbacks = [];
    window._lastTurnstileCallback = null;
    window._turnstileSolvedToken = null;

    function wrapRender() {
        if (window.turnstile && typeof window.turnstile.render === 'function') {
            if (window.turnstile.render._isWrapped) return true;
            const origRender = window.turnstile.render;
            const wrapped = function(container, options) {
                window._turnstileOptions = options;
                if (options && typeof options.callback === 'function') {
                    window._lastTurnstileCallback = options.callback;
                    window._turnstileCallbacks.push(options.callback);
                    const origCb = options.callback;
                    options.callback = function(tok) {
                        window._turnstileSolvedToken = tok;
                        return origCb.apply(this, arguments);
                    };
                }
                return origRender.apply(this, arguments);
            };
            wrapped._isWrapped = true;
            window.turnstile.render = wrapped;
            return true;
        }
        return false;
    }

    if (!wrapRender()) {
        const pollTurnstile = setInterval(() => {
            if (wrapRender()) {
                clearInterval(pollTurnstile);
            }
        }, 10);
    }
})();
"""


def _sync_http_post(url: str, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class CaptchaResolver:
    """Detects, solves, and injects tokens for Cloudflare Turnstile, reCAPTCHA, and hCaptcha."""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or get_capsolver_api_key()
        self.api_url = api_url or os.getenv("CAPSOLVER_API_URL", "https://api.capsolver.com")

    async def detect_captcha(self, page: Page) -> Optional[Dict[str, Any]]:
        """
        Inspects the page DOM, frames, and JavaScript globals for active CAPTCHA challenges.
        Returns a dictionary describing the captcha type, sitekey, container, and action, or None.
        """
        # 1. Check Cloudflare Turnstile
        turnstile_info = await self._detect_turnstile(page)
        if turnstile_info:
            return turnstile_info

        # 2. Check Google reCAPTCHA
        recaptcha_info = await self._detect_recaptcha(page)
        if recaptcha_info:
            return recaptcha_info

        # 3. Check hCaptcha
        hcaptcha_info = await self._detect_hcaptcha(page)
        if hcaptcha_info:
            return hcaptcha_info

        return None

    async def _detect_turnstile(self, page: Page) -> Optional[Dict[str, Any]]:
        """Detects Cloudflare Turnstile presence and extracts parameters."""
        info = await page.evaluate("""() => {
            let sitekey = null;
            let action = null;
            let containerSelector = null;

            // Check intercepted options
            if (window._turnstileOptions) {
                sitekey = window._turnstileOptions.sitekey || null;
                action = window._turnstileOptions.action || null;
            }

            // Check DOM attributes
            const tsEl = document.querySelector('.cf-turnstile, [data-turnstile-sitekey], [data-sitekey]');
            if (tsEl) {
                sitekey = sitekey || tsEl.getAttribute('data-sitekey') || tsEl.getAttribute('data-turnstile-sitekey');
                action = action || tsEl.getAttribute('data-action');
                containerSelector = tsEl.id ? '#' + tsEl.id : (tsEl.className ? '.' + tsEl.className.split(' ')[0] : null);
            }

            // Check iframes
            const iframes = Array.from(document.querySelectorAll('iframe'));
            const tsIframe = iframes.find(f => (f.src && (f.src.includes('challenges.cloudflare.com') || f.src.includes('turnstile'))));
            if (tsIframe) {
                const src = tsIframe.src;
                const m = src.match(/sitekey=([^&]+)/) || src.match(/k=([^&]+)/);
                if (m) sitekey = sitekey || decodeURIComponent(m[1]);
            }

            const hasActiveChallenge = !!(tsEl || tsIframe || window._turnstileOptions || window._lastTurnstileCallback);
            return {
                has_turnstile: hasActiveChallenge,
                sitekey: sitekey,
                action: action,
                container: containerSelector
            };
        }""")

        if info and info.get("has_turnstile"):
            sitekey = info.get("sitekey")
            # If sitekey not in DOM yet, attempt regex on page scripts
            if not sitekey:
                content = await page.content()
                m = re.search(r'0x4[A-Za-z0-9_-]{18,24}', content)
                if m:
                    sitekey = m.group(0)

            return {
                "type": "turnstile",
                "sitekey": sitekey or "",
                "url": page.url,
                "action": info.get("action") or "application-form-submit",
                "container": info.get("container"),
            }

        return None

    async def _detect_recaptcha(self, page: Page) -> Optional[Dict[str, Any]]:
        """Detects Google reCAPTCHA v2 / v3."""
        info = await page.evaluate("""() => {
            let sitekey = null;
            const rcEl = document.querySelector('.g-recaptcha, [data-sitekey]');
            if (rcEl && !rcEl.classList.contains('cf-turnstile')) {
                sitekey = rcEl.getAttribute('data-sitekey');
            }
            const iframes = Array.from(document.querySelectorAll('iframe'));
            const rcIframe = iframes.find(f => (f.src && f.src.includes('google.com/recaptcha')));
            if (rcIframe && !sitekey) {
                const m = rcIframe.src.match(/k=([^&]+)/);
                if (m) sitekey = decodeURIComponent(m[1]);
            }
            const isV3 = !!(document.querySelector('script[src*="render="]') || (rcIframe && rcIframe.src.includes('invisible')));
            return {
                has_recaptcha: !!(rcEl || rcIframe || window.grecaptcha),
                sitekey: sitekey,
                is_v3: isV3
            };
        }""")

        if info and info.get("has_recaptcha") and info.get("sitekey"):
            return {
                "type": "recaptcha_v3" if info.get("is_v3") else "recaptcha_v2",
                "sitekey": info.get("sitekey"),
                "url": page.url,
            }
        return None

    async def _detect_hcaptcha(self, page: Page) -> Optional[Dict[str, Any]]:
        """Detects hCaptcha presence."""
        info = await page.evaluate("""() => {
            const hEl = document.querySelector('.h-captcha, [data-hcaptcha-sitekey]');
            let sitekey = hEl ? (hEl.getAttribute('data-sitekey') || hEl.getAttribute('data-hcaptcha-sitekey')) : null;
            const iframes = Array.from(document.querySelectorAll('iframe'));
            const hIframe = iframes.find(f => f.src && f.src.includes('hcaptcha.com'));
            if (hIframe && !sitekey) {
                const m = hIframe.src.match(/sitekey=([^&]+)/);
                if (m) sitekey = decodeURIComponent(m[1]);
            }
            return {
                has_hcaptcha: !!(hEl || hIframe),
                sitekey: sitekey
            };
        }""")

        if info and info.get("has_hcaptcha") and info.get("sitekey"):
            return {
                "type": "hcaptcha",
                "sitekey": info.get("sitekey"),
                "url": page.url,
            }
        return None

    async def solve(self, captcha_info: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """
        Requests token solution from CapSolver API.
        Returns a dictionary with status, token, and latency.
        """
        if not self.api_key:
            return {"status": "error", "error": "CapSolver API key not found in environment or .env"}

        c_type = captcha_info.get("type", "turnstile")
        sitekey = captcha_info.get("sitekey", "")
        url = captcha_info.get("url", "")
        action = captcha_info.get("action")

        if not sitekey:
            return {"status": "error", "error": "Missing sitekey for CAPTCHA challenge"}

        # Build task payload according to CapSolver schema
        task_payload: Dict[str, Any] = {"websiteURL": url, "websiteKey": sitekey}
        if c_type == "turnstile":
            task_type = "AntiTurnstileTaskProxyLess"
            if action:
                task_payload["action"] = action
        elif c_type == "recaptcha_v3":
            task_type = "ReCaptchaV3TaskProxyLess"
            task_payload["pageAction"] = action or "verify"
        elif c_type == "recaptcha_v2":
            task_type = "ReCaptchaV2TaskProxyLess"
        elif c_type == "hcaptcha":
            task_type = "HCaptchaTaskProxyLess"
        else:
            task_type = "AntiTurnstileTaskProxyLess"

        t0 = time.perf_counter()
        req_data = {
            "clientKey": self.api_key,
            "task": {"type": task_type, **task_payload},
        }

        try:
            create_res = await asyncio.to_thread(
                _sync_http_post,
                f"{self.api_url}/createTask",
                req_data,
                10.0,
            )

            task_id = create_res.get("taskId")
            if not task_id:
                return {
                    "status": "error",
                    "error": f"CapSolver createTask failed: {create_res.get('errorDescription', 'No taskId returned')}",
                }

            # Poll for solution
            poll_start = time.perf_counter()
            while time.perf_counter() - poll_start < timeout:
                await asyncio.sleep(1.5)
                poll_payload = {"clientKey": self.api_key, "taskId": task_id}
                p_res = await asyncio.to_thread(
                    _sync_http_post,
                    f"{self.api_url}/getTaskResult",
                    poll_payload,
                    10.0,
                )

                if p_res.get("status") == "ready":
                    sol = p_res.get("solution", {})
                    token = sol.get("token") or sol.get("gRecaptchaResponse")
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return {
                        "status": "ready",
                        "token": token,
                        "latency_ms": elapsed_ms,
                        "type": c_type,
                        "sitekey": sitekey,
                    }
                elif p_res.get("status") == "failed":
                    return {
                        "status": "error",
                        "error": f"CapSolver task failed: {p_res.get('errorDescription', 'Unknown failure')}",
                    }

            return {"status": "error", "error": f"CapSolver timed out after {timeout}s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def inject_token(self, page: Page, captcha_info: Dict[str, Any], token: str) -> bool:
        """Injects solution token into DOM inputs and triggers registered client callbacks."""
        c_type = captcha_info.get("type", "turnstile")

        if c_type == "turnstile":
            res = await page.evaluate("""(t) => {
                let called = false;
                if (typeof window._lastTurnstileCallback === 'function') {
                    try { window._lastTurnstileCallback(t); called = true; } catch (e) {}
                }
                if (typeof window._turnstileCb === 'function' && !called) {
                    try { window._turnstileCb(t); called = true; } catch (e) {}
                }
                const inputs = document.querySelectorAll('input[name*="turnstile"], [name="cf-turnstile-response"]');
                inputs.forEach(el => {
                    el.value = t;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                });
                return { called, inputs: inputs.length };
            }""", token)
            return bool(res.get("called") or res.get("inputs", 0) > 0)

        elif "recaptcha" in c_type:
            res = await page.evaluate("""(t) => {
                const inputs = document.querySelectorAll('textarea[name="g-recaptcha-response"], input[name="g-recaptcha-response"]');
                inputs.forEach(el => {
                    el.value = t;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                });
                try {
                    if (window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients) {
                        for (const cid in window.___grecaptcha_cfg.clients) {
                            const client = window.___grecaptcha_cfg.clients[cid];
                            for (const k in client) {
                                if (client[k] && typeof client[k].callback === 'function') {
                                    client[k].callback(t);
                                }
                            }
                        }
                    }
                } catch(e) {}
                return { inputs: inputs.length };
            }""", token)
            return bool(res.get("inputs", 0) > 0)

        elif c_type == "hcaptcha":
            res = await page.evaluate("""(t) => {
                const inputs = document.querySelectorAll('textarea[name="h-captcha-response"], [name="g-recaptcha-response"]');
                inputs.forEach(el => {
                    el.value = t;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                });
                return { inputs: inputs.length };
            }""", token)
            return bool(res.get("inputs", 0) > 0)

        return False

    async def auto_detect_and_solve(self, page: Page, timeout: float = 25.0) -> Dict[str, Any]:
        """
        Unified method:
        1. Checks for CAPTCHA challenges on page.
        2. If found, requests solution from CapSolver.
        3. Dispatches native click to interactive checkbox if present.
        4. Injects token into client callback and hidden inputs.
        """
        t0 = time.perf_counter()
        info = await self.detect_captcha(page)
        if not info:
            return {"detected": False, "solved": False, "latency_ms": 0.0}

        # Attempt native click on checkbox element or bounding box for realism
        if info.get("type") == "turnstile":
            try:
                cf_iframe = page.locator("iframe[src*='challenges.cloudflare.com'], iframe[src*='turnstile']").first
                if await cf_iframe.count() > 0 and await cf_iframe.is_visible():
                    box = await cf_iframe.bounding_box()
                    if box:
                        await page.mouse.click(box["x"] + 30, box["y"] + (box["height"] / 2))
            except Exception:
                pass

        # Solve via CapSolver API
        solve_res = await self.solve(info, timeout=timeout)
        if solve_res.get("status") == "ready" and solve_res.get("token"):
            token = str(solve_res["token"])
            injected = await self.inject_token(page, info, token)
            total_latency = (time.perf_counter() - t0) * 1000.0
            return {
                "detected": True,
                "type": info.get("type"),
                "sitekey": info.get("sitekey"),
                "solved": True,
                "injected": injected,
                "token": token,
                "latency_ms": total_latency,
            }

        return {
            "detected": True,
            "type": info.get("type"),
            "sitekey": info.get("sitekey"),
            "solved": False,
            "error": solve_res.get("error"),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
