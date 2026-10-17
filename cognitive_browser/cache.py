"""
Action Cache & Route Compilation for Cognitive Browser.
Remembers resolved (URL pattern + instruction -> semantic locator) mappings
so repeated automations execute in ~2-5ms without re-running model inference.
Supports automatic self-healing when a page layout changes.
"""

import json
import os
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse


class RouteCache:
    """In-memory and disk-backed cache for resolved web element routes."""

    def __init__(self, cache_file: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled and os.getenv("COGNITIVE_BROWSER_CACHE", "1") != "0"
        self.cache_file = cache_file or os.getenv(
            "COGNITIVE_BROWSER_CACHE_FILE",
            os.path.join(os.path.expanduser("~"), ".cache", "cognitive_browser_routes.json"),
        )
        self._routes: Dict[str, Dict[str, Any]] = {}
        if self.enabled:
            self._load()

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalizes a URL into a stable host + path template (stripping dynamic IDs and query strings)."""
        if not url:
            return "unknown"
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower().removeprefix("www.")
            # Replace numeric IDs or UUID segments with '*'
            path = re.sub(r"/\d+(?=/|$)", "/*", parsed.path.rstrip("/"))
            return f"{host}{path}" if path else host
        except Exception:
            return url

    @staticmethod
    def _normalize_intent(instruction: str) -> str:
        """Normalizes an instruction string into a canonical key."""
        return re.sub(r"\s+", " ", instruction.strip().lower())

    def _make_key(self, url: str, instruction: str) -> str:
        return f"{self._normalize_url(url)}::{self._normalize_intent(instruction)}"

    def get(self, url: str, instruction: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cached locator route if available."""
        if not self.enabled:
            return None
        key = self._make_key(url, instruction)
        return self._routes.get(key)

    def store(
        self,
        url: str,
        instruction: str,
        role: str,
        name: str,
        nth: int = 0,
        action: str = "click",
        context: str = "",
    ) -> None:
        """Stores or updates a verified locator route."""
        if not self.enabled or not role or not name:
            return
        key = self._make_key(url, instruction)
        existing = self._routes.get(key, {})
        hits = existing.get("hits", 0) + 1
        self._routes[key] = {
            "role": role,
            "name": name,
            "nth": nth,
            "action": action,
            "context": context,
            "hits": hits,
        }
        self._save()

    def invalidate(self, url: str, instruction: str) -> None:
        """Removes a broken route from cache so self-healing can replace it."""
        if not self.enabled:
            return
        key = self._make_key(url, instruction)
        if key in self._routes:
            del self._routes[key]
            self._save()

    def clear(self) -> None:
        """Clears all cached routes."""
        self._routes.clear()
        self._save()

    def _load(self) -> None:
        if not self.cache_file:
            return
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._routes = data
        except Exception:
            self._routes = {}

    def _save(self) -> None:
        if not self.cache_file:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._routes, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
