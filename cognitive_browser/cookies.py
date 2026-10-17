"""
System and browser cookie management for Cognitive Browser.
Extracts existing authenticated session cookies from local browsers (e.g., Firefox, Chrome)
or exported JSON cookie files to bypass Cloudflare, DataDome, and login walls without re-authentication.
"""

import glob
import json
import os
import shutil
import sqlite3
import tempfile
import urllib.parse
from typing import Any, Dict, List, Optional


def find_firefox_profiles() -> List[str]:
    """Finds all available Firefox profile directories containing cookies.sqlite."""
    base_dirs = [
        os.path.expanduser("~/.mozilla/firefox"),
        os.path.expanduser("~/.var/app/org.mozilla.firefox/.mozilla/firefox"),  # Flatpak
        os.path.expanduser("~/snap/firefox/common/.mozilla/firefox"),  # Snap
    ]
    profiles = []
    for base in base_dirs:
        if os.path.exists(base):
            for path in glob.glob(os.path.join(base, "*")):
                if os.path.isdir(path) and os.path.exists(os.path.join(path, "cookies.sqlite")):
                    profiles.append(path)

    # Sort profiles prioritizing default / default-release
    profiles.sort(key=lambda p: (0 if "default-release" in p else (1 if "default" in p else 2), p))
    return profiles


def extract_firefox_cookies(
    domain: str, profile_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extracts cookies for a specific domain from Firefox's cookies.sqlite.
    Safely copies the SQLite database to a temporary file to avoid locking issues.
    Converts cookies to Playwright-compatible dictionary format.
    """
    if profile_path and os.path.exists(os.path.join(profile_path, "cookies.sqlite")):
        db_path = os.path.join(profile_path, "cookies.sqlite")
    else:
        profiles = find_firefox_profiles()
        if not profiles:
            return []
        db_path = os.path.join(profiles[0], "cookies.sqlite")

    clean_domain = domain.lower()
    if clean_domain.startswith("http://") or clean_domain.startswith("https://"):
        parsed = urllib.parse.urlparse(clean_domain)
        clean_domain = parsed.hostname or clean_domain
    clean_domain = clean_domain.lstrip(".").split(":")[0]
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]

    cookies: List[Dict[str, Any]] = []

    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp_file:
            tmp_db = tmp_file.name
        shutil.copy2(db_path, tmp_db)

        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        query = (
            "SELECT host, name, value, path, expiry, isSecure, isHttpOnly, sameSite "
            "FROM moz_cookies "
            "WHERE host LIKE ? OR ? LIKE ('%' || TRIM(host, '.'))"
        )
        cur.execute(query, (f"%{clean_domain}%", clean_domain))
        rows = cur.fetchall()
        conn.close()

        for host, name, value, path, expiry, is_secure, is_http_only, same_site in rows:
            # Playwright requires sameSite to be Strict, Lax, or None
            s_site = "None"
            if same_site == 1:
                s_site = "Lax"
            elif same_site == 2:
                s_site = "Strict"

            # Firefox expiry is sometimes milliseconds (13 digits) or seconds
            expires_val: float = -1.0
            if expiry:
                if expiry > 1e11:
                    expires_val = float(expiry / 1000.0)
                else:
                    expires_val = float(expiry)

            cookie_dict: Dict[str, Any] = {
                "name": str(name),
                "value": str(value),
                "domain": str(host),
                "path": str(path or "/"),
                "expires": expires_val,
                "httpOnly": bool(is_http_only),
                "secure": bool(is_secure),
                "sameSite": s_site,
            }
            cookies.append(cookie_dict)

    except Exception:
        pass
    finally:
        if "tmp_db" in locals() and os.path.exists(tmp_db):
            try:
                os.remove(tmp_db)
            except Exception:
                pass

    return cookies


def load_cookies_from_file(file_path: str, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads cookies from a JSON file (exported by browser extensions or Playwright)."""
    expanded = os.path.expanduser(file_path)
    if not os.path.exists(expanded):
        return []

    try:
        with open(expanded, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                if domain:
                    clean = domain.lower().lstrip(".").split(":")[0]
                    return [c for c in data if clean in str(c.get("domain", "")).lower()]
                return data
    except Exception:
        pass
    return []


async def import_system_cookies(
    context_or_page: Any,
    domain: str,
    browser_type: str = "firefox",
    profile_path: Optional[str] = None,
) -> int:
    """
    Imports system cookies into a Playwright BrowserContext or Page instance.
    Returns the count of cookies successfully added.
    """
    if hasattr(context_or_page, "add_cookies") and callable(context_or_page.add_cookies):
        context = context_or_page
    else:
        context = getattr(context_or_page, "context", context_or_page)

    if not hasattr(context, "add_cookies"):
        return 0

    cookies: List[Dict[str, Any]] = []

    if browser_type.endswith(".json") or os.path.exists(os.path.expanduser(browser_type)):
        cookies = load_cookies_from_file(browser_type, domain=domain)
    else:
        # Default to Firefox profiles
        cookies = extract_firefox_cookies(domain=domain, profile_path=profile_path)

    if not cookies:
        return 0

    try:
        await context.add_cookies(cookies)
        return len(cookies)
    except Exception:
        # Retry with sanitized cookies if any field had schema mismatches
        sanitized = []
        for c in cookies:
            clean_c = {
                "name": c.get("name"),
                "value": c.get("value"),
                "domain": c.get("domain"),
                "path": c.get("path", "/"),
            }
            if c.get("secure") is not None:
                clean_c["secure"] = c["secure"]
            if c.get("httpOnly") is not None:
                clean_c["httpOnly"] = c["httpOnly"]
            if c.get("sameSite") in ("Strict", "Lax", "None"):
                clean_c["sameSite"] = c["sameSite"]
            sanitized.append(clean_c)
        try:
            await context.add_cookies(sanitized)
            return len(sanitized)
        except Exception:
            return 0
