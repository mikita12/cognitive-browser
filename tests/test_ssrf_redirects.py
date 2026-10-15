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


class InternalSecretHandler(BaseHTTPRequestHandler):
    """Local server acting as an internal protected service with a hit counter."""

    hits: int = 0

    def do_GET(self):
        InternalSecretHandler.hits += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"CONFIDENTIAL_INTERNAL_DATA")

    def log_message(self, *args):
        pass


class SSRFRedirectTestHandler(BaseHTTPRequestHandler):
    server_b_port: int = 0

    def do_GET(self):
        if self.path == "/redirect-to-loopback":
            # Single-hop redirect to prohibited internal loopback service
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server_b_port}/secret")
            self.end_headers()
        elif self.path == "/redirect-to-metadata":
            # Single-hop redirect to prohibited cloud metadata endpoint
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
        elif self.path == "/multi-hop-loopback-1":
            # Hop 1 of multi-hop redirect (exempt -> exempt)
            self.send_response(302)
            self.send_header("Location", "/multi-hop-loopback-2")
            self.end_headers()
        elif self.path == "/multi-hop-loopback-2":
            # Hop 2 of multi-hop redirect (exempt -> private prohibited target)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server_b_port}/secret")
            self.end_headers()
        elif self.path == "/multi-hop-metadata-1":
            # Hop 1 of multi-hop redirect to metadata
            self.send_response(302)
            self.send_header("Location", "/multi-hop-metadata-2")
            self.end_headers()
        elif self.path == "/multi-hop-metadata-2":
            # Hop 2 of multi-hop redirect to metadata mock host
            self.send_response(302)
            self.send_header("Location", f"http://metadata.mock:{self.server_b_port}/latest/meta-data/")
            self.end_headers()
        elif self.path == "/loop-a":
            self.send_response(302)
            self.send_header("Location", "/loop-b")
            self.end_headers()
        elif self.path == "/loop-b":
            self.send_response(302)
            self.send_header("Location", "/loop-a")
            self.end_headers()
        elif self.path.startswith("/chain-"):
            # Hop limit test: chain-1 -> chain-2 -> ... -> chain-25
            n = int(self.path.split("-")[1])
            if n < 25:
                self.send_response(302)
                self.send_header("Location", f"/chain-{n + 1}")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"end of chain")
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

    server_b: ThreadingHTTPServer
    server_b_port: int
    server_b_thread: threading.Thread
    server_b_url: str

    @classmethod
    def setUpClass(cls):
        # Server B: Internal protected service with hit counter
        cls.server_b = ThreadingHTTPServer(("127.0.0.1", 0), InternalSecretHandler)
        cls.server_b_port = cls.server_b.server_address[1]
        cls.server_b_thread = threading.Thread(target=cls.server_b.serve_forever, daemon=True)
        cls.server_b_thread.start()
        cls.server_b_url = f"http://127.0.0.1:{cls.server_b_port}"

        # Server A: External website fixture
        SSRFRedirectTestHandler.server_b_port = cls.server_b_port
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SSRFRedirectTestHandler)
        cls.server_port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_b.shutdown()
        cls.server_b.server_close()

    def setUp(self):
        InternalSecretHandler.hits = 0

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

    async def test_single_hop_redirect_to_loopback_is_aborted(self):
        """
        Verify single-hop redirect directly to loopback service is aborted before connection.
        Server B hit counter must remain 0.
        """
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
            self.assertEqual(InternalSecretHandler.hits, 0, "Server B must NOT receive any requests")
            await browser.close()

    async def test_single_hop_redirect_to_cloud_metadata_is_aborted(self):
        """
        Verify single-hop redirect directly to cloud metadata endpoint is aborted before connection.
        """
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

    async def test_multihop_redirect_to_loopback_diagnostics_and_behavior(self):
        """
        Empirically test multi-hop redirect behavior (exempt -> exempt -> private Server B)
        using production setup_ssrf_route_guard.

        Known limitation demonstrated by this test:
        1. Hop 1 (/multi-hop-loopback-1): Intercepted by context.route via setup_ssrf_route_guard.
           The route guard inspects the 302 response and finds Location=/multi-hop-loopback-2.
           Because /multi-hop-loopback-2 belongs to exempt_origins (Server A), route.fulfill(302) is called.
        2. Hop 2 (/multi-hop-loopback-2): Dispatched by Chromium following the 302 response.
           In Playwright's CDP implementation (coreBundle.js:35823), when redirectedFrom is set,
           Playwright automatically calls Fetch.continueRequest and DOES NOT create a user RouteImpl.
           Therefore, Hop 2 and the subsequent redirect to Server B completely bypass context.route.
        3. Result: Server B receives the request (hits == 1).
           This test asserts and proves this exact Playwright context.route architectural limitation.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()

            intercepted_urls = []
            original_route = context.route

            async def route_spy(pattern, handler):
                async def wrapped_handler(route):
                    intercepted_urls.append(route.request.url)
                    return await handler(route)

                return await original_route(pattern, wrapped_handler)

            context.route = route_spy

            # Run production setup_ssrf_route_guard
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            # Execute multi-hop navigation
            resp = await page.goto(f"{self.base_url}/multi-hop-loopback-1", timeout=5000)

            # Empirically verify intercepted routes: only Hop 1 is seen by context.route
            self.assertEqual(
                intercepted_urls,
                [f"{self.base_url}/multi-hop-loopback-1"],
                f"context.route was expected to intercept only Hop 1, intercepted: {intercepted_urls}",
            )

            # Empirically verify that Server B received the hit due to CDP's redirectedFrom bypass
            self.assertEqual(
                InternalSecretHandler.hits,
                1,
                "Known limitation: Server B was hit because secondary redirect hops bypass context.route in Playwright CDP",
            )
            self.assertIsNotNone(resp)
            assert resp is not None
            self.assertEqual(resp.status, 200)
            self.assertIn(str(self.server_b_port), page.url)
            await browser.close()

    @unittest.expectedFailure
    async def test_target_multihop_redirect_blocked_by_future_proxy(self):
        """
        Target specification for multi-hop redirect protection (public -> public -> private).
        Desired security behavior: multi-hop redirect chains terminating in a private IP
        must be aborted before connection, resulting in 0 hits on the private server.

        Current status: expectedFailure.
        Under Playwright's context.route architecture, secondary redirect hops bypass route
        interception. This test will only pass once a connection-level egress validation
        mechanism (e.g. local forward proxy via --proxy-server validating IP addresses upon
        connection) is implemented.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            with self.assertRaises(Exception):
                await page.goto(f"{self.base_url}/multi-hop-loopback-1", timeout=5000)

            self.assertEqual(
                InternalSecretHandler.hits,
                0,
                "Desired behavior: Server B must not receive any requests",
            )
            await browser.close()

    async def test_multihop_redirect_to_metadata_mock_behavior(self):
        """
        Verify multi-hop redirect behavior towards cloud metadata domain using --host-resolver-rules.
        Demonstrates that multi-hop redirects bypassing context.route also reach mapped metadata endpoints.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[f"--host-resolver-rules=MAP metadata.mock 127.0.0.1:{self.server_b_port}"],
            )
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            # Execute multi-hop navigation targeting metadata.mock
            resp = await page.goto(f"{self.base_url}/multi-hop-metadata-1", timeout=5000)

            # Server B received the hit via the mapped metadata hostname
            self.assertEqual(
                InternalSecretHandler.hits,
                1,
                "Server B received hit because secondary redirect hop to metadata.mock bypassed context.route",
            )
            self.assertIsNotNone(resp)
            assert resp is not None
            self.assertEqual(resp.status, 200)
            await browser.close()

    async def test_redirect_loop_detected(self):
        """
        Verify that a redirect loop (A -> B -> A) is detected and halted by Chromium
        with net::ERR_TOO_MANY_REDIRECTS without infinite looping.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            with self.assertRaises(Exception) as ctx:
                await page.goto(f"{self.base_url}/loop-a", timeout=5000)

            err_msg = str(ctx.exception)
            self.assertIn("ERR_TOO_MANY_REDIRECTS", err_msg)
            await browser.close()

    async def test_redirect_hop_limit_enforced(self):
        """
        Verify that redirect chains exceeding Chromium's internal redirect hop limit (20 hops)
        are aborted with net::ERR_TOO_MANY_REDIRECTS.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            with self.assertRaises(Exception) as ctx:
                await page.goto(f"{self.base_url}/chain-1", timeout=5000)

            err_msg = str(ctx.exception)
            self.assertIn("ERR_TOO_MANY_REDIRECTS", err_msg)
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
            res_loopback = await page.evaluate(f"""async () => {{
                try {{
                    await fetch('http://127.0.0.1:{self.server_b_port}/secret', {{ mode: 'no-cors' }});
                    return 'ok';
                }} catch (e) {{
                    return 'failed';
                }}
            }}""")
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
        Asserts hits == 0 on Server B so that error/closed cannot be attributed to HTTP handshake rejection.
        """
        InternalSecretHandler.hits = 0
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(service_workers="block")
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()
            await page.goto(f"{self.base_url}/page-for-fetch", timeout=5000)

            ws_status = await page.evaluate(f"""() => {{
                return new Promise((resolve) => {{
                    try {{
                        const ws = new WebSocket('ws://127.0.0.1:{self.server_b_port}/ws');
                        ws.onerror = () => resolve('error');
                        ws.onclose = () => resolve('closed');
                        setTimeout(() => resolve('timeout'), 1000);
                    }} catch (e) {{
                        resolve('exception');
                    }}
                }});
            }}""")
            self.assertIn(ws_status, ("error", "closed"))
            self.assertEqual(
                InternalSecretHandler.hits,
                0,
                "Server B must receive 0 requests; WebSocket handshake aborted by client-side guard",
            )
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
