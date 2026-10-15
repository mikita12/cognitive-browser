"""
Unit tests for security controls: URL validation, SSRF prevention, and sensitive value masking.
"""

import unittest

from cognitive_browser.security import SecurityError, mask_sensitive_value, sanitize_untrusted_text, validate_url


class TestSecurityControls(unittest.TestCase):
    def test_valid_public_urls(self):
        self.assertEqual(validate_url("https://example.com"), "https://example.com")
        self.assertEqual(
            validate_url("http://en.wikipedia.org/wiki/Main_Page"), "http://en.wikipedia.org/wiki/Main_Page"
        )
        self.assertEqual(validate_url("https://github.com/login"), "https://github.com/login")

    def test_invalid_schemes_blocked(self):
        invalid_schemes = [
            "file:///etc/passwd",
            "ftp://ftp.example.com/file.txt",
            "javascript:alert(1)",
            "data:text/html,<h1>Hello</h1>",
            "gopher://gopher.example.com",
        ]
        for url in invalid_schemes:
            with self.assertRaises(SecurityError, msg=f"Should block: {url}"):
                validate_url(url)

    def test_ssrf_loopback_and_private_blocked(self):
        blocked_hosts = [
            "http://localhost",
            "http://localhost:8080/admin",
            "http://127.0.0.1",
            "http://127.0.0.1:5000",
            "http://[::1]/",
            "http://[::1]:8080/",
            "http://[fc00::1]/internal",
            "http://[fe80::1]/config",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://192.168.1.1/router",
            "http://10.0.0.1/internal",
            "http://172.16.0.1/config",
            "http://169.254.169.254/latest/meta-data/",
        ]
        for url in blocked_hosts:
            with self.assertRaises(SecurityError, msg=f"Should block private target: {url}"):
                validate_url(url)

    def test_valid_public_ipv6(self):
        # Cloudflare public DNS IPv6
        url = "https://[2606:4700:4700::1111]"
        self.assertEqual(validate_url(url), url)

    def test_private_network_override(self):
        # When explicitly allowed, private IP validation succeeds
        url = "http://127.0.0.1:8080/test"
        res = validate_url(url, allow_private=True)
        self.assertEqual(res, url)

    def test_domain_allowlist(self):
        allowed = ["wikipedia.org", "github.com"]
        # Exact match and subdomain match should pass
        self.assertEqual(validate_url("https://wikipedia.org", allowed_domains=allowed), "https://wikipedia.org")
        self.assertEqual(
            validate_url("https://en.wikipedia.org/wiki/AI", allowed_domains=allowed),
            "https://en.wikipedia.org/wiki/AI",
        )

        # Unlisted domain should fail
        with self.assertRaises(SecurityError):
            validate_url("https://malicious-site.com", allowed_domains=allowed)

    def test_mask_sensitive_value_default_on(self):
        # Default behavior: all fill values are masked for privacy/telemetry safety
        self.assertEqual(mask_sensitive_value("Artificial intelligence", instruction="search topic"), "***")
        self.assertEqual(mask_sensitive_value("my_super_secret_password", instruction="enter user password"), "***")
        self.assertEqual(mask_sensitive_value("sk-1234567890", instruction="fill api_key into input"), "***")

    def test_mask_sensitive_value_opt_out(self):
        import os

        # Opt-out via environment variable DISABLE_FILL_MASKING
        os.environ["DISABLE_FILL_MASKING"] = "true"
        try:
            # Non-sensitive text is preserved
            self.assertEqual(
                mask_sensitive_value("Artificial intelligence", instruction="search article topic"),
                "Artificial intelligence",
            )
            # Sensitive keywords remain masked even when global masking is opted out
            self.assertEqual(
                mask_sensitive_value("my_super_secret_password", instruction="enter user password"),
                "***",
            )
            self.assertEqual(mask_sensitive_value("sk-1234567890", instruction="fill api_key into input"), "***")
        finally:
            del os.environ["DISABLE_FILL_MASKING"]

    def test_mask_all_fill_values_explicit_flag(self):
        # Explicit force_mask=False preserves non-sensitive text
        self.assertEqual(mask_sensitive_value("normal term", instruction="search", force_mask=False), "normal term")
        self.assertEqual(mask_sensitive_value("normal term", instruction="search", force_mask=True), "***")

    def test_sanitize_untrusted_text(self):
        dirty_input = "Hello\x00\x08World! ```prompt injection```"
        cleaned = sanitize_untrusted_text(dirty_input, max_length=50)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("```", cleaned)
        self.assertIn("'''", cleaned)
        self.assertLessEqual(len(cleaned), 50)


if __name__ == "__main__":
    unittest.main()
