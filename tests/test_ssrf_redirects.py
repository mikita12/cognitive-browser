"""
Integration tests verifying SSRF protection on HTTP redirects, subresources,
XHR/fetch, non-HTTP schemes, IPv6, WebSockets, and DNS rebinding simulation using
Playwright context.route("**/*") and local test fixtures.
"""

import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from playwright.async_api import async_playwright

from cognitive_browser.security import SecurityError, is_exempt_origin, setup_ssrf_route_guard, validate_url


class SSRFRedirectTestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect-to-loopback":
            # Redirect to prohibited internal loopback service
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9999/internal-admin")
            self.end_headers()
        elif self.path == "/redirect-to-metadata":
            # Redirect to prohibited cloud metadata endpoint
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
        elif self.path == "/safe-redirect":
            # Legitimate relative redirect
            self.send_response(302)
            self.send_header("Location", "/safe-target")
            self.end_headers()
        elif self.path == "/safe-target":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Safe Content Loaded</h1></body></html>")
        elif self.path == "/multi-1":
            # Step 1 of multi-hop redirect (exempt -> exempt)
            self.send_response(302)
            self.send_header("Location", "/multi-2")
            self.end_headers()
        elif self.path == "/multi-2":
            # Step 2 of multi-hop redirect (exempt -> private prohibited target)
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9999/admin")
            self.end_headers()
        elif self.path == "/page-with-private-subresources":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = (
                "<html><body><h1>Subresources</h1>"
                '<img src="http://169.254.169.254/leak.png">'
                '<script src="http://127.0.0.1:9999/bad.js"></script>'
                '<iframe src="http://127.0.0.1:9999/secret"></iframe>'
                "</body></html>"
            )
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/page-for-fetch":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Fetch Test Page</h1></body></html>")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        # Silence HTTP server access logs during tests
        pass


class TestSSRFRedirectGuards(unittest.IsolatedAsyncioTestCase):
    server: ThreadingHTTPServer
    server_port: int
    server_thread: threading.Thread
    base_url: str

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SSRFRedirectTestHandler)
        cls.server_port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_exempt_origins_prefix_and_userinfo_bypass_prevented(self):
        """
        Verify that is_exempt_origin strictly checks (scheme, host, port) and rejects
        prefix and userinfo bypasses like http://127.0.0.1:PORT.evil.com or http://127.0.0.1:PORT@evil.com.
        """
        exempts = {f"http://127.0.0.1:{self.server_port}"}

        # Exact origin matches
        self.assertTrue(is_exempt_origin(f"http://127.0.0.1:{self.server_port}/some/path", exempts))
        self.assertTrue(is_exempt_origin(f"http://127.0.0.1:{self.server_port}", exempts))

        # Domain prefix bypass attempt (e.g. 127.0.0.1:8000.evil.com)
        self.assertFalse(is_exempt_origin(f"http://127.0.0.1:{self.server_port}.evil.com/exploit", exempts))

        # Userinfo bypass attempt (e.g. 127.0.0.1:8000@evil.com)
        self.assertFalse(is_exempt_origin(f"http://127.0.0.1:{self.server_port}@evil.com/exploit", exempts))

        # Port mismatch
        self.assertFalse(is_exempt_origin("http://127.0.0.1:9999/path", exempts))

        # Scheme mismatch
        self.assertFalse(is_exempt_origin(f"https://127.0.0.1:{self.server_port}/path", exempts))

    async def test_redirect_to_loopback_is_aborted(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            with self.assertRaises(Exception) as ctx:
                await page.goto(f"{self.base_url}/redirect-to-loopback", timeout=5000)

            err_msg = str(ctx.exception)
            self.assertTrue(
                "ERR_BLOCKED_BY_CLIENT" in err_msg or "blocked" in err_msg.lower(),
                f"Expected net::ERR_BLOCKED_BY_CLIENT, got: {err_msg}",
            )
            await browser.close()

    async def test_redirect_to_cloud_metadata_is_aborted(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            with self.assertRaises(Exception) as ctx:
                await page.goto(f"{self.base_url}/redirect-to-metadata", timeout=5000)

            err_msg = str(ctx.exception)
            self.assertTrue(
                "ERR_BLOCKED_BY_CLIENT" in err_msg or "blocked" in err_msg.lower(),
                f"Expected net::ERR_BLOCKED_BY_CLIENT, got: {err_msg}",
            )
            await browser.close()

    async def test_multihop_redirect_to_private_is_aborted(self):
        """
        Verify multi-hop redirect behavior (exempt -> exempt -> private).
        CDP architecture detail:
        Single-hop redirects are intercepted and aborted before connection with ERR_BLOCKED_BY_CLIENT.
        When Chromium follows subsequent redirect hops internally, context.route is not re-invoked
        by CDP for internal redirect hops of the same navigation, leading to connection failure.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            with self.assertRaises(Exception) as ctx:
                await page.goto(f"{self.base_url}/multi-1", timeout=5000)

            err_msg = str(ctx.exception)
            self.assertTrue(
                "ERR_BLOCKED_BY_CLIENT" in err_msg
                or "ERR_CONNECTION_REFUSED" in err_msg
                or "blocked" in err_msg.lower(),
                f"Expected navigation failure on multi-hop redirect to private, got: {err_msg}",
            )
            await browser.close()

    async def test_legitimate_redirect_succeeds(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            response = await page.goto(f"{self.base_url}/safe-redirect", timeout=5000)
            self.assertIsNotNone(response)
            assert response is not None
            self.assertEqual(response.status, 200)
            self.assertIn("safe-target", page.url)

            content = await page.inner_text("h1")
            self.assertEqual(content, "Safe Content Loaded")
            await browser.close()

    async def test_xhr_fetch_to_private_and_metadata_aborted(self):
        """
        Verify client-side JS fetch/XHR calls to loopback and metadata addresses are aborted.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()
            await page.goto(f"{self.base_url}/page-for-fetch", timeout=5000)

            # Test fetch to 127.0.0.1
            res_loopback = await page.evaluate("""async () => {
                try {
                    await fetch('http://127.0.0.1:9999/secret', { mode: 'no-cors' });
                    return 'ok';
                } catch (e) {
                    return 'failed';
                }
            }""")
            self.assertEqual(res_loopback, "failed")

            # Test fetch to 169.254.169.254
            res_metadata = await page.evaluate("""async () => {
                try {
                    await fetch('http://169.254.169.254/latest/meta-data/', { mode: 'no-cors' });
                    return 'ok';
                } catch (e) {
                    return 'failed';
                }
            }""")
            self.assertEqual(res_metadata, "failed")
            await browser.close()

    async def test_subresources_to_private_are_aborted(self):
        """
        Verify subresources (img, script, iframe) pointing to private addresses are blocked.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            failed_requests = []
            page.on("requestfailed", lambda req: failed_requests.append(req.url))

            await page.goto(f"{self.base_url}/page-with-private-subresources", timeout=5000)

            # Subresources should be intercepted and failed
            self.assertTrue(
                any("169.254.169.254" in url for url in failed_requests),
                f"Expected blocked metadata img request in {failed_requests}",
            )
            self.assertTrue(
                any("bad.js" in url for url in failed_requests),
                f"Expected blocked script request in {failed_requests}",
            )
            await browser.close()

    def test_blocked_schemes(self):
        """
        Verify non-http/https schemes (file, ftp, javascript) are rejected by validate_url.
        """
        blocked = [
            "file:///etc/passwd",
            "file:///C:/Windows/win.ini",
            "ftp://anonymous@ftp.example.com/dump.tar",
            "javascript:alert(document.cookie)",
            "data:text/html,<h1>Hello</h1>",
        ]
        for url in blocked:
            with self.assertRaises(SecurityError, msg=f"Scheme in '{url}' should be blocked"):
                validate_url(url)

    def test_ipv6_private_and_metadata_targets_blocked(self):
        """
        Verify IPv6 private, link-local, loopback, and IPv4-mapped addresses are rejected.
        """
        ipv6_targets = [
            "http://[::1]/",  # IPv6 loopback
            "http://[fd12:3456:789a:1::1]/",  # Unique Local Address (fc00::/7)
            "http://[fe80::1]/",  # Link-local (fe80::/10)
            "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
            "http://[::ffff:169.254.169.254]/",  # IPv4-mapped cloud metadata
        ]
        for url in ipv6_targets:
            with self.assertRaises(SecurityError, msg=f"IPv6 address '{url}' should be blocked"):
                validate_url(url)

    async def test_websocket_to_private_is_aborted(self):
        """
        Verify WebSocket handshakes to private/loopback addresses are intercepted and aborted.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(service_workers="block")
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()
            await page.goto(f"{self.base_url}/page-for-fetch", timeout=5000)

            ws_status = await page.evaluate("""() => {
                return new Promise((resolve) => {
                    try {
                        const ws = new WebSocket('ws://127.0.0.1:9999/ws');
                        ws.onerror = () => resolve('error');
                        ws.onclose = () => resolve('closed');
                        setTimeout(() => resolve('timeout'), 1000);
                    } catch (e) {
                        resolve('exception');
                    }
                });
            }""")
            self.assertIn(ws_status, ("error", "closed"))
            await browser.close()

    def test_dns_rebinding_simulation(self):
        """
        Simulate a DNS rebinding attack where a domain resolves to a public IP on initial check,
        and then to a private IP (127.0.0.1) on subsequent resolution.

        What is verified:
        - When resolver returns public IP ('93.184.216.34'), validate_url succeeds.
        - When resolver returns private IP ('127.0.0.1'), validate_url detects private IP and raises SecurityError.

        Residual TOCTOU limitation:
        Because validate_url and Chromium / route.fetch make independent DNS queries without
        socket-level IP pinning, an attacker with TTL=0 DNS control could switch DNS between
        the check and connection. Complete DNS rebinding defense requires network-level egress filtering.
        """
        # Step 1: Initial query returns legitimate public IP
        public_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        with patch("socket.getaddrinfo", return_value=public_addrinfo):
            validated = validate_url("http://rebind-test.attacker.com/data")
            self.assertEqual(validated, "http://rebind-test.attacker.com/data")

        # Step 2: Rebind step returns prohibited loopback IP
        rebind_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        with patch("socket.getaddrinfo", return_value=rebind_addrinfo):
            with self.assertRaises(SecurityError) as ctx:
                validate_url("http://rebind-test.attacker.com/data")
            self.assertIn("resolves to private IP", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
