"""
Integration tests verifying SSRF protection on HTTP redirects using Playwright context.route("**/*")
and a local ThreadingHTTPServer returning 302 redirects to 127.0.0.1 and 169.254.169.254.
"""

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from playwright.async_api import async_playwright

from cognitive_browser.security import setup_ssrf_route_guard


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
        elif self.path == "/page-with-private-subresource":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b'<html><body><h1>Page</h1><img src="http://169.254.169.254/leak.png"></body></html>')
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

    async def test_redirect_to_loopback_is_aborted(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            # Allow origin test server itself so it can respond, but block private redirect targets
            await setup_ssrf_route_guard(context, exempt_origins={self.base_url})
            page = await context.new_page()

            # Attempting to follow a redirect to 127.0.0.1 must be aborted before connecting
            with self.assertRaises(Exception) as ctx:
                await page.goto(f"{self.base_url}/redirect-to-loopback", timeout=5000)

            # Playwright raises Error when route is aborted with blockedbyclient
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

            # Attempting to follow a redirect to 169.254.169.254 must be aborted before connecting
            with self.assertRaises(Exception) as ctx:
                await page.goto(f"{self.base_url}/redirect-to-metadata", timeout=5000)

            err_msg = str(ctx.exception)
            self.assertTrue(
                "ERR_BLOCKED_BY_CLIENT" in err_msg or "blocked" in err_msg.lower(),
                f"Expected net::ERR_BLOCKED_BY_CLIENT, got: {err_msg}",
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


if __name__ == "__main__":
    unittest.main()
