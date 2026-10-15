"""
Security utilities for Cognitive Browser:
- SSRF prevention (blocking loopback, private networks, cloud metadata endpoints).
- URL validation and domain allowlist enforcement.
- Sensitive value masking for input values and action telemetry.
- Indirect prompt injection sanitization.
"""

import ipaddress
import os
import re
import socket
import urllib.parse
from typing import Any, List, Optional, Set


class SecurityError(ValueError):
    """Raised when an operation violates security policies."""

    pass


# Cloud metadata endpoints and well-known internal hostnames
METADATA_HOSTS: Set[str] = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.internal",
    "instance-data",
    "100.100.100.200",  # Alibaba Cloud metadata
}

SENSITIVE_KEYWORDS: Set[str] = {
    "password",
    "haslo",
    "hasło",
    "secret",
    "token",
    "apikey",
    "api_key",
    "pin",
    "cvv",
    "cvc",
    "card",
    "credit_card",
    "ssn",
    "auth",
    "bearer",
}


def is_private_ip(ip_str: str) -> bool:
    """Checks whether an IP address belongs to loopback, private, link-local, or reserved ranges."""
    try:
        clean_ip = ip_str.strip("[]")
        ip = ipaddress.ip_address(clean_ip)

        # Check IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            mapped = ip.ipv4_mapped
            return (
                mapped.is_private
                or mapped.is_loopback
                or mapped.is_link_local
                or mapped.is_reserved
                or mapped.is_multicast
                or mapped.is_unspecified
            )

        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        return False


def validate_url(url: str, allow_private: Optional[bool] = None, allowed_domains: Optional[List[str]] = None) -> str:
    """
    Validates a URL before navigation to prevent SSRF and unsafe protocols.

    Rules:
    1. Scheme must be strictly 'http' or 'https'.
    2. Private/loopback IP addresses (IPv4 & IPv6) and metadata endpoints are rejected by default.
    3. If ALLOWED_DOMAINS is configured, only matching domains are allowed.

    Raises:
        SecurityError: If the URL fails security validation.
    """
    if not url or not isinstance(url, str):
        raise SecurityError("Target URL must be a non-empty string.")

    cleaned_url = url.strip()
    parsed = urllib.parse.urlsplit(cleaned_url)

    # 1. Scheme check
    if parsed.scheme.lower() not in ("http", "https"):
        raise SecurityError(f"Blocked scheme '{parsed.scheme}'. Only 'http' and 'https' protocols are permitted.")

    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        raise SecurityError("URL does not contain a valid hostname.")

    clean_host = hostname.strip("[]")

    # Determine whether private network access is allowed
    if allow_private is None:
        allow_private = os.getenv("ALLOW_PRIVATE_NETWORKS", "false").lower() in ("true", "1", "yes")

    # 2. SSRF check
    if not allow_private:
        # Check known metadata hosts
        if clean_host in METADATA_HOSTS or hostname in METADATA_HOSTS:
            raise SecurityError(f"Access to cloud metadata endpoint '{hostname}' is strictly blocked.")

        # Check loopback hostnames
        if clean_host in ("localhost", "localhost.localdomain", "ip6-localhost"):
            raise SecurityError(f"Access to loopback host '{hostname}' is blocked.")

        # Check if hostname is direct IP literal (IPv4 or IPv6)
        if is_private_ip(clean_host):
            raise SecurityError(f"Access to private/local IP address '{hostname}' is prohibited (SSRF prevention).")

        # Resolve hostname to IP to verify it does not resolve to a private network
        try:
            addr_info = socket.getaddrinfo(clean_host, None)
            for entry in addr_info:
                ip_str = str(entry[4][0])
                if is_private_ip(ip_str):
                    raise SecurityError(f"Hostname '{hostname}' resolves to private IP '{ip_str}' (SSRF blocked).")
        except socket.gaierror:
            # If DNS resolution fails, allow standard browser navigation to handle NXDOMAIN
            pass

    # 3. Optional domain allowlist enforcement
    if allowed_domains is None:
        env_domains = os.getenv("ALLOWED_DOMAINS", "").strip()
        if env_domains:
            allowed_domains = [d.strip().lower() for d in env_domains.split(",") if d.strip()]

    if allowed_domains:
        domain_match = False
        for domain in allowed_domains:
            clean_dom = domain.lstrip(".")
            if clean_host == clean_dom or clean_host.endswith("." + clean_dom):
                domain_match = True
                break

        if not domain_match:
            raise SecurityError(f"Host '{hostname}' is not in the allowed domains list: {allowed_domains}")

    return cleaned_url


def mask_sensitive_value(value: Optional[str], instruction: str = "", force_mask: Optional[bool] = None) -> str:
    """
    Masks values (passwords, tokens, sensitive inputs) in execution logs and telemetry.
    Masking is ON by default for safety. Users can opt-out by setting DISABLE_FILL_MASKING=true.
    """
    if not value:
        return ""

    if force_mask is None:
        disabled = os.getenv("DISABLE_FILL_MASKING", "false").lower() in ("true", "1", "yes")
        if disabled or os.getenv("MASK_ALL_FILL_VALUES", "true").lower() in ("false", "0", "no"):
            force_mask = False
        else:
            force_mask = True

    if force_mask:
        return "***"

    # Even when global masking is opted out, mask sensitive keywords
    inst_lower = instruction.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in inst_lower:
            return "***"

    return value


def sanitize_untrusted_text(text: str, max_length: int = 150) -> str:
    """
    Sanitizes untrusted text extracted from web pages before feeding into prompts.
    Strips dangerous control characters and caps string length.
    """
    if not text:
        return ""
    # Strip non-printable / control characters except standard space
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    # Strip potential prompt injection delimiter markers
    clean = clean.replace("```", "'''")
    return clean[:max_length].strip()


async def setup_ssrf_route_guard(
    target: Any,
    allow_private: Optional[bool] = None,
    allowed_domains: Optional[List[str]] = None,
    exempt_origins: Optional[Set[str]] = None,
) -> None:
    """
    Registers a route handler on a Playwright BrowserContext or Page to intercept
    and validate every outgoing network request (main document, subresources, XHR,
    fetch, and HTTP redirects) before dispatch.
    Requests to private networks, loopback addresses, or cloud metadata endpoints are aborted.
    """
    context = getattr(target, "context", target)
    if not hasattr(context, "route"):
        return

    async def _ssrf_route_interceptor(route: Any) -> None:
        req = route.request
        url = req.url

        # Check if URL belongs to an explicitly exempt origin (used in test fixtures with local servers)
        is_exempt = exempt_origins is not None and any(url.startswith(origin) for origin in exempt_origins)

        if not is_exempt:
            try:
                validate_url(url, allow_private=allow_private, allowed_domains=allowed_domains)
            except SecurityError:
                await route.abort("blockedbyclient")
                return

        try:
            # Intercept redirect responses before Playwright/Chromium follows them
            response = await route.fetch(max_redirects=0)
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if location:
                    target_redirect_url = urllib.parse.urljoin(url, location)
                    is_redirect_exempt = exempt_origins is not None and any(
                        target_redirect_url.startswith(origin) for origin in exempt_origins
                    )
                    if not is_redirect_exempt:
                        try:
                            validate_url(
                                target_redirect_url,
                                allow_private=allow_private,
                                allowed_domains=allowed_domains,
                            )
                        except SecurityError:
                            await route.abort("blockedbyclient")
                            return

            await route.fulfill(response=response)
        except Exception:
            await route.abort("blockedbyclient")

    await context.route("**/*", _ssrf_route_interceptor)
