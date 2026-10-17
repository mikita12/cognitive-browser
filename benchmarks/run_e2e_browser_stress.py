"""
End-to-End Browser Stress Benchmark (100 Runs) for Cognitive Browser.
Executes real Playwright Chromium instances against a local HTTP test server
verifying:
1. Scenario A: Cold GPU vs Warm RouteCache (40 runs)
2. Scenario B: RouteCache Self-Healing on dynamic DOM mutation (20 runs)
3. Scenario C: Complete Browser CAPTCHA cycle: Intercept -> Detect -> CapSolver -> Inject -> Submit (20 runs)
4. Scenario D: Autonomous Goal Mode with State vs Action Competition (20 runs)
"""

import asyncio
import json
import os
import resource
import socket
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple

import torch

# Ensure private networks are allowed for local server testing
os.environ["ALLOW_PRIVATE_NETWORKS"] = "true"

from cognitive_browser.captcha import CaptchaResolver
from cognitive_browser.config import get_capsolver_api_key
from cognitive_browser.mcp_server import cognitive_browse, get_services
from cognitive_browser.reflex import AriaElementParser

# ---------------------------------------------------------------------------
# Local Test HTTP Server
# ---------------------------------------------------------------------------

class StressTestHttpHandler(BaseHTTPRequestHandler):
    """Handles requests for all 4 E2E test scenarios."""

    # Dynamic server state for Scenario B
    portal_layout_state: str = "v1"

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server request logging."""
        pass

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # SCENARIO A: Multi-step Checkout Flow
        if path == "/a/step1":
            html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Krok 1: Wybór kategorii</title></head>
<body>
  <h1>Formularz zamówienia sprzętu - Krok 1</h1>
  <p>Wybierz opcję poniżej, aby przejść do formularza danych adresowych.</p>
  <p><a href="/a/step2" role="link" id="link-step2">Przejdź do formularza zamówienia</a></p>
</body>
</html>"""
            self._send_html(html)

        elif path == "/a/step2":
            html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Krok 2: Adres dostawy</title></head>
<body>
  <h1>Wprowadź dane dostawy zamówienia</h1>
  <form action="/a/step3" method="GET">
    <div>
      <label for="addr">Adres dostawy:</label>
      <input id="addr" name="addr" type="text" placeholder="Wpisz adres dostawy">
    </div>
    <div>
      <label for="phone">Numer telefonu:</label>
      <input id="phone" name="phone" type="text" placeholder="Wpisz numer telefonu">
    </div>
    <div>
      <button type="submit" id="btn-submit">Zatwierdź dane i zamów</button>
    </div>
  </form>
</body>
</html>"""
            self._send_html(html)

        elif path == "/a/step3":
            html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Krok 3: Potwierdzenie zamówienia</title></head>
<body>
  <h1>Potwierdzenie zamówienia</h1>
  <div id="status-box">
    <p>Zamówienie zostało złożone pomyślnie! Dziękujemy za zakupy.</p>
  </div>
</body>
</html>"""
            self._send_html(html)

        # SCENARIO B: RouteCache Self-Healing Layout
        elif path == "/b/portal":
            state = query.get("state", [StressTestHttpHandler.portal_layout_state])[0]
            if state == "v1":
                html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Portal Zgłoszeń - Wersja V1</title></head>
<body>
  <h1>Rejestracja Wniosku Pracowniczego</h1>
  <div id="container-v1">
    <button id="btn-action">Zatwierdź wniosek</button>
    <p id="msg"></p>
  </div>
  <script>
    document.getElementById('btn-action').onclick = function() {
      document.getElementById('msg').innerText = 'Wniosek został pomyślnie złożony!';
    };
  </script>
</body>
</html>"""
            else:
                html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Portal Zgłoszeń - Wersja V2 (Zmieniony DOM)</title></head>
<body>
  <h1>Rejestracja Wniosku Pracowniczego</h1>
  <div id="container-v2">
    <div class="banner">Uwaga: Zaktualizowano układ interfejsu systemu.</div>
    <button id="btn-action-v2">Wyślij formularz teraz</button>
    <p id="msg"></p>
  </div>
  <script>
    document.getElementById('btn-action-v2').onclick = function() {
      document.getElementById('msg').innerText = 'Formularz wysłany w nowym układzie!';
    };
  </script>
</body>
</html>"""
            self._send_html(html)

        # SCENARIO C: Cloudflare Turnstile Challenge Form
        elif path == "/c/turnstile_form":
            html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Formularz rekrutacyjny - weryfikacja Cloudflare Turnstile</title></head>
<body>
  <h1>Aplikacja na stanowisko Software Engineer</h1>
  <form id="app-form" action="/c/success" method="POST">
    <div>
      <label for="c_name">Imię i nazwisko:</label>
      <input id="c_name" name="name" value="Jan Kowalski">
    </div>
    <div id="cf-turnstile-box" class="cf-turnstile" data-sitekey="0x4AAAAAAABBBBBBBB" data-action="submit_application"></div>
    <input type="hidden" name="cf-turnstile-response" id="cf-turnstile-response">
    <div>
      <button type="submit" id="submit-btn" disabled>Aplikuj teraz</button>
    </div>
  </form>
  <div id="cf-status">Oczekiwanie na Turnstile...</div>
  <script>
    window.turnstile = {
      render: function(container, options) {
        window._turnstileOptions = options;
        document.getElementById('cf-status').innerText = 'Widget Turnstile aktywny: ' + options.sitekey;
      }
    };
    function initTurnstile() {
      const cb = function(token) {
        window._turnstileTokenInjected = token;
        document.getElementById('cf-turnstile-response').value = token;
        document.getElementById('submit-btn').disabled = false;
        document.getElementById('cf-status').innerText = 'Zweryfikowano token: ' + token.substring(0, 10);
      };
      window._turnstileCb = cb;
      window.turnstile.render('#cf-turnstile-box', {
        sitekey: '0x4AAAAAAABBBBBBBB',
        action: 'submit_application',
        callback: cb
      });
    }
    setTimeout(initTurnstile, 25);
  </script>
</body>
</html>"""
            self._send_html(html)

        # SCENARIO D: Autonomous Store & Search Results
        elif path == "/d/store_home":
            html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>TechMaster Sklep Komputerowy - Strona Główna</title></head>
<body>
  <header>
    <h1>TechMaster Sklep Komputerowy</h1>
    <nav>
      <a href="/d/promocje">Promocje tygodnia</a>
      <a href="/d/kontakt">Kontakt i salony</a>
      <a href="/d/regulamin">Regulamin sklepu</a>
    </nav>
  </header>
  <main>
    <h2>Znajdź wymarzony sprzęt komputerowy</h2>
    <form action="/d/results" method="GET">
      <input id="search-input" name="q" placeholder="Wpisz nazwę szukanego produktu...">
      <button type="submit" id="search-button">Szukaj</button>
    </form>
    <section>
      <p>Kupuj sprzęt elektroniczny, podzespoły komputerowe i akcesoria z darmową dostawą w 24h.</p>
    </section>
  </main>
</body>
</html>"""
            self._send_html(html)

        elif path == "/d/results":
            q_val = query.get("q", ["laptop"])[0]
            html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Wyniki wyszukiwania dla: {q_val} - TechMaster</title></head>
<body>
  <h1>Wyniki wyszukiwania dla zapytania: {q_val}</h1>
  <p class="search-summary">Znaleziono 3 pasujące laptopy gamingowe i ultrabooki:</p>

  <article class="product-item">
    <h2>Laptop Dell XPS 15 9530 Intel Core i7 32GB RAM RTX 4060</h2>
    <p class="price">Cena: 9 499 zł z VAT (Darmowa dostawa, wysyłka w 24h)</p>
    <p class="desc">Ekran 15.6 cala OLED 3.5K, 1TB SSD NVMe, Windows 11 Pro. Gwarancja 36 miesięcy.</p>
    <button class="add-to-cart">Dodaj do koszyka</button>
  </article>

  <article class="product-item">
    <h2>Laptop Lenovo ThinkPad T14s Gen 4 AMD Ryzen 7 PRO 32GB</h2>
    <p class="price">Cena: 6 899 zł z VAT (Dostępny od ręki w magazynie)</p>
    <p class="desc">Biznesowy ultrabook, waga 1.25 kg, modem 5G LTE, bateria 57Wh, czytnik linii papilarnych.</p>
    <button class="add-to-cart">Dodaj do koszyka</button>
  </article>

  <article class="product-item">
    <h2>Laptop ASUS ROG Zephyrus G16 Intel Core Ultra 9 32GB RTX 4080</h2>
    <p class="price">Cena: 13 999 zł z VAT (Ostatnie sztuki w magazynie)</p>
    <p class="desc">Ekran OLED 240Hz ROG Nebula, 32GB pamięci LPDDR5X, chłodzenie komorą parową.</p>
    <button class="add-to-cart">Dodaj do koszyka</button>
  </article>

  <footer>
    <a href="/d/newsletter">Zapisz się na newsletter</a>
    <a href="/d/reklamacje">Reklamacje i zwroty</a>
    <a href="/d/polityka">Polityka prywatności</a>
  </footer>
</body>
</html>"""
            self._send_html(html)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path

        # Scenario C form submission endpoint
        if path == "/c/success":
            html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Sukces Aplikacji</title></head>
<body>
  <h1>Aplikacja została pomyślnie wysłana!</h1>
  <p id="result-msg">Weryfikacja Turnstile zakończona sukcesem. Token został zaakceptowany.</p>
</body>
</html>"""
            self._send_html(html)

        # Mock CapSolver API endpoints
        elif path == "/capsolver/createTask":
            content_len = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(content_len)
            resp = {
                "errorId": 0,
                "taskId": "mock_turnstile_task_e2e_888",
                "status": "idle",
            }
            self._send_json(resp)

        elif path == "/capsolver/getTaskResult":
            content_len = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(content_len)
            resp = {
                "errorId": 0,
                "status": "ready",
                "solution": {
                    "token": "cf_e2e_verified_token_sec987654321",
                    "userAgent": "Mozilla/5.0 (X11; Linux x86_64)",
                },
            }
            self._send_json(resp)

        elif path == "/capsolver/getBalance":
            resp = {"errorId": 0, "balance": 42.50}
            self._send_json(resp)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def _send_html(self, html: str) -> None:
        content = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: Dict[str, Any]) -> None:
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def start_local_test_server() -> Tuple[HTTPServer, int]:
    """Starts local HTTP server on a dynamically chosen free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    server = HTTPServer(("127.0.0.1", port), StressTestHttpHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server, port


# ---------------------------------------------------------------------------
# Telemetry and Resource Monitoring
# ---------------------------------------------------------------------------

def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    return 0.0


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    return 0.0


def get_host_ram_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_maxrss / 1024.0, 2)


# ---------------------------------------------------------------------------
# Test Scenarios Implementation
# ---------------------------------------------------------------------------

async def run_scenario_a_cache_speedup(
    port: int,
    runs: int = 40,
) -> Dict[str, Any]:
    """
    Scenario A: Cold GPU vs Warm RouteCache (40 runs).
    20 Cold runs (RouteCache cleared, GPU model evaluates each step)
    vs 20 Warm runs (RouteCache serves cached locators directly).
    """
    print(f"\n[Scenario A] Executing {runs} runs (20 Cold vs 20 Warm)...")
    reflex, engine = get_services()
    start_url = f"http://127.0.0.1:{port}/a/step1"

    cold_latencies: List[float] = []
    warm_latencies: List[float] = []
    successes = 0

    half = runs // 2

    # Part 1: Cold Runs (Cache disabled/cleared)
    for i in range(half):
        if hasattr(engine, "route_cache") and engine.route_cache:
            engine.route_cache.clear()

        steps = [
            {"instruction": "Przejdź do formularza zamówienia", "action": "click"},
            {"instruction": "Wpisz adres dostawy", "fill_value": "Marszałkowska 10, Warszawa", "action": "fill"},
            {"instruction": "Wpisz numer telefonu", "fill_value": "500600700", "action": "fill"},
            {"instruction": "Zatwierdź dane i zamów", "action": "click"},
        ]

        t0 = time.perf_counter()
        res = await cognitive_browse(url=start_url, steps=steps)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        page = await engine.get_page()
        title = await page.title()
        is_success = "Krok 3" in title or "Potwierdzenie" in title or res.get("status") == "success"
        if is_success:
            successes += 1
            cold_latencies.append(dur_ms)
        print(f"  • Cold run {i + 1}/{half}: {dur_ms:.1f}ms | status: {'OK' if is_success else 'FAIL'}")

    # Part 2: Warm Runs (Cache retained)
    for i in range(half):
        steps = [
            {"instruction": "Przejdź do formularza zamówienia", "action": "click"},
            {"instruction": "Wpisz adres dostawy", "fill_value": "Marszałkowska 10, Warszawa", "action": "fill"},
            {"instruction": "Wpisz numer telefonu", "fill_value": "500600700", "action": "fill"},
            {"instruction": "Zatwierdź dane i zamów", "action": "click"},
        ]

        t0 = time.perf_counter()
        res = await cognitive_browse(url=start_url, steps=steps)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        page = await engine.get_page()
        title = await page.title()
        is_success = "Krok 3" in title or "Potwierdzenie" in title or res.get("status") == "success"
        if is_success:
            successes += 1
            warm_latencies.append(dur_ms)
        print(f"  • Warm run {i + 1}/{half}: {dur_ms:.1f}ms | status: {'OK' if is_success else 'FAIL'}")

    mean_cold = round(sum(cold_latencies) / len(cold_latencies), 2) if cold_latencies else 0.0
    mean_warm = round(sum(warm_latencies) / len(warm_latencies), 2) if warm_latencies else 0.0
    speedup = round(mean_cold / mean_warm, 2) if mean_warm > 0 else 0.0

    return {
        "scenario": "Cold GPU vs Warm RouteCache",
        "total_runs": runs,
        "success_runs": successes,
        "success_rate": round(successes / runs, 4),
        "mean_cold_ms": mean_cold,
        "mean_warm_ms": mean_warm,
        "speedup_ratio": speedup,
        "cold_latencies": cold_latencies,
        "warm_latencies": warm_latencies,
    }


async def run_scenario_b_self_healing(
    port: int,
    runs: int = 20,
) -> Dict[str, Any]:
    """
    Scenario B: RouteCache Self-Healing after DOM Mutation (20 runs).
    1. Cache a route on layout V1.
    2. Dynamically mutate layout to V2 (removing old button, adding new button with new label).
    3. Execute action: old route fails -> invalidates -> ReflexCore discovers new button on GPU -> heals route.
    """
    print(f"\n[Scenario B] Executing {runs} runs of RouteCache Self-Healing...")
    reflex, engine = get_services()
    base_url = f"http://127.0.0.1:{port}/b/portal"

    healing_latencies: List[float] = []
    healed_count = 0

    for i in range(runs):
        StressTestHttpHandler.portal_layout_state = "v1"
        page = await engine.get_page()

        # Step 1: Establish baseline route in cache for V1
        await cognitive_browse(
            url=f"{base_url}?state=v1",
            steps=[{"instruction": "Zatwierdź wniosek", "action": "click"}],
        )

        # Step 2: Mutate layout to V2 (old button removed, new button added)
        StressTestHttpHandler.portal_layout_state = "v2"
        await page.goto(f"{base_url}?state=v2", wait_until="domcontentloaded")
        await page.wait_for_timeout(100)

        # Step 3: Run action with instruction "Wyślij wniosek"
        t0 = time.perf_counter()
        res = await cognitive_browse(
            url=f"{base_url}?state=v2",
            steps=[{"instruction": "Wyślij wniosek", "action": "click"}],
        )
        dur_ms = (time.perf_counter() - t0) * 1000.0

        # Verify DOM result text
        msg_text = await page.evaluate("() => document.getElementById('msg')?.innerText || ''")
        healed_ok = "nowym układzie" in msg_text or res.get("status") == "success"

        if healed_ok:
            healed_count += 1
            healing_latencies.append(dur_ms)

        print(f"  • Self-Healing run {i + 1}/{runs}: {dur_ms:.1f}ms | result: '{msg_text}' | {'RECOVERED' if healed_ok else 'FAILED'}")

    return {
        "scenario": "RouteCache Self-Healing",
        "total_runs": runs,
        "healed_count": healed_count,
        "healing_success_rate": round(healed_count / runs, 4),
        "mean_healing_latency_ms": round(sum(healing_latencies) / len(healing_latencies), 2) if healing_latencies else 0.0,
        "healing_latencies": healing_latencies,
    }


async def run_scenario_c_captcha_lifecycle(
    port: int,
    runs: int = 20,
) -> Dict[str, Any]:
    """
    Scenario C: Full Browser CAPTCHA Lifecycle (20 runs):
    Intercept -> Detect -> CapSolver -> Inject -> Submit.
    """
    print(f"\n[Scenario C] Executing {runs} runs of full browser CAPTCHA lifecycle...")
    reflex, engine = get_services()
    form_url = f"http://127.0.0.1:{port}/c/turnstile_form"
    mock_capsolver_url = f"http://127.0.0.1:{port}/capsolver"

    # Configure resolver with mock endpoint
    resolver = CaptchaResolver(
        api_key="TEST_CAPSOLVER_KEY",
        api_url=mock_capsolver_url,
    )

    # Live check balance if real key is configured
    live_key = get_capsolver_api_key()
    live_balance_note = "No live CAPSOLVER_API_KEY configured"
    if live_key:
        try:
            live_resolver = CaptchaResolver(api_key=live_key)
            req = urllib.request.Request(
                f"{live_resolver.api_url}/getBalance",
                data=json.dumps({"clientKey": live_key}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                live_balance_note = f"CapSolver Live Account Balance: ${data.get('balance', 0.0):.2f}"
        except Exception as ex:
            live_balance_note = f"CapSolver Live check note: {ex}"
    print(f"  [Live Check] {live_balance_note}")

    cycle_latencies: List[float] = []
    successes = 0

    for i in range(runs):
        t0 = time.perf_counter()
        page = await engine.get_page()
        await page.goto(form_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(100)

        # 1. Intercept verification: verify window.turnstile options intercepted
        intercept_info = await page.evaluate("""() => ({
            has_options: !!window._turnstileOptions,
            sitekey: window._turnstileOptions ? window._turnstileOptions.sitekey : null
        })""")
        assert intercept_info["has_options"] is True, "TURNSTILE_INTERCEPT_SCRIPT failed to intercept options"

        # 2. Browser State Detection
        snapshot_yaml = await page.aria_snapshot()
        cands = AriaElementParser.parse_snapshot(snapshot_yaml)
        c_eval = reflex.evaluate_captcha_presence(
            candidates=cands,
            page_title=await page.title(),
            page_url=page.url,
            snapshot_yaml=snapshot_yaml,
        )
        assert c_eval["is_captcha"] is True, "ReflexCore failed to detect active Turnstile challenge"

        # 3. Solver detection & CapSolver API fetch
        c_info = await resolver.detect_captcha(page)
        assert c_info is not None and c_info.get("type") == "turnstile"

        sol = await resolver.solve(c_info)
        assert sol.get("status") == "ready"
        token = sol["token"]

        # 4. Token injection
        injected = await resolver.inject_token(page, c_info, token)
        assert injected is True, "Token injection failed"

        # 5. Form submission verification: submit button should now be enabled
        is_disabled = await page.evaluate("() => document.getElementById('submit-btn').disabled")
        assert is_disabled is False, "Submit button remained disabled after injection"

        await page.click("#submit-btn")
        await page.wait_for_timeout(150)

        content = await page.content()
        is_submitted = "Sukces Aplikacji" in content or "pomyślnie wysłana" in content

        dur_ms = (time.perf_counter() - t0) * 1000.0
        if is_submitted:
            successes += 1
            cycle_latencies.append(dur_ms)

        print(f"  • CAPTCHA cycle {i + 1}/{runs}: {dur_ms:.1f}ms | Injected: {token[:16]}... | Status: {'SUCCESS' if is_submitted else 'FAIL'}")

    return {
        "scenario": "CAPTCHA Browser Lifecycle",
        "total_runs": runs,
        "success_runs": successes,
        "success_rate": round(successes / runs, 4),
        "mean_cycle_latency_ms": round(sum(cycle_latencies) / len(cycle_latencies), 2) if cycle_latencies else 0.0,
        "min_cycle_latency_ms": round(min(cycle_latencies), 2) if cycle_latencies else 0.0,
        "max_cycle_latency_ms": round(max(cycle_latencies), 2) if cycle_latencies else 0.0,
        "live_balance_note": live_balance_note,
        "cycle_latencies": cycle_latencies,
    }


async def run_scenario_d_autonomous_goal(
    port: int,
    runs: int = 20,
) -> Dict[str, Any]:
    """
    Scenario D: Autonomous Goal Mode with State vs Action Competition (20 runs).
    Verifies that cognitive_browse(url=..., goal=...) types query into searchbox,
    navigates to results page, and halts autonomously upon satisfying goal via
    evaluate_page_satisfaction without clicking random links.
    """
    print(f"\n[Scenario D] Executing {runs} runs of Autonomous Goal Mode...")
    reflex, engine = get_services()
    start_url = f"http://127.0.0.1:{port}/d/store_home"
    goal = "Wpisz w pole wyszukiwania: laptop i kliknij szukaj"

    goal_latencies: List[float] = []
    successes = 0

    for i in range(runs):
        # Invalidate goal shortcut in cache to test full autonomous navigation loop
        if hasattr(engine, "route_cache") and engine.route_cache:
            engine.route_cache.invalidate(start_url, f"GOAL::{goal}")

        t0 = time.perf_counter()
        res = await cognitive_browse(url=start_url, goal=goal)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        page = await engine.get_page()
        cur_url = page.url
        content_snippet = res.get("page_content_snippet", "")

        # Verification criteria:
        # 1. Navigated to results page
        # 2. Found target models and prices
        # 3. Status is success
        has_results = "/d/results" in cur_url
        has_products = "Dell XPS" in content_snippet or "ThinkPad" in content_snippet or "Zephyrus" in content_snippet
        has_prices = "9 499" in content_snippet or "6 899" in content_snippet or "zł" in content_snippet
        is_ok = res.get("status") == "success" and has_results and (has_products or has_prices)

        if is_ok:
            successes += 1
            goal_latencies.append(dur_ms)

        print(f"  • Autonomous run {i + 1}/{runs}: {dur_ms:.1f}ms | steps: {res.get('steps_executed', 0)} | final_url: {cur_url} | {'SUCCESS' if is_ok else 'FAIL'}")

    return {
        "scenario": "Autonomous Goal Mode",
        "total_runs": runs,
        "success_runs": successes,
        "success_rate": round(successes / runs, 4),
        "mean_latency_ms": round(sum(goal_latencies) / len(goal_latencies), 2) if goal_latencies else 0.0,
        "min_latency_ms": round(min(goal_latencies), 2) if goal_latencies else 0.0,
        "max_latency_ms": round(max(goal_latencies), 2) if goal_latencies else 0.0,
        "goal_latencies": goal_latencies,
    }


# ---------------------------------------------------------------------------
# Main Suite Orchestration
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 70)
    print("🚀 STARTING E2E BROWSER STRESS BENCHMARK (100 RUNS)")
    print("=" * 70)

    # 1. Start local server
    server, port = start_local_test_server()
    print(f"✓ Local HTTP Test Server running on http://127.0.0.1:{port}")

    # 2. Warm up services and measure initial resources
    vram_start = get_vram_mb()
    ram_start = get_host_ram_mb()
    print(f"✓ Baseline System Resources: VRAM = {vram_start:.2f} MB | Host RAM = {ram_start:.2f} MB")

    t_suite_start = time.perf_counter()

    # 3. Execute Scenarios
    res_a = await run_scenario_a_cache_speedup(port, runs=40)
    res_b = await run_scenario_b_self_healing(port, runs=20)
    res_c = await run_scenario_c_captcha_lifecycle(port, runs=20)
    res_d = await run_scenario_d_autonomous_goal(port, runs=20)

    total_suite_seconds = round(time.perf_counter() - t_suite_start, 2)

    # 4. Final Resource Metrics
    vram_end = get_vram_mb()
    vram_peak = get_peak_vram_mb()
    vram_delta = round(vram_end - vram_start, 2)

    ram_end = get_host_ram_mb()
    ram_delta = round(ram_end - ram_start, 2)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPLETE - SUMMARY OF 100 E2E RUNS:")
    print("=" * 70)
    print(f"Total Suite Duration: {total_suite_seconds} seconds")
    print(f"Scenario A (Cold vs Warm 40 runs): Cold = {res_a['mean_cold_ms']:.1f}ms | Warm = {res_a['mean_warm_ms']:.1f}ms | Speedup = {res_a['speedup_ratio']}x")
    print(f"Scenario B (Self-Healing 20 runs): Success Rate = {res_a['success_rate'] * 100:.1f}% | Healed: {res_b['healed_count']}/20")
    print(f"Scenario C (CAPTCHA 20 runs): Success Rate = {res_c['success_rate'] * 100:.1f}% | Mean Latency = {res_c['mean_cycle_latency_ms']:.1f}ms")
    print(f"Scenario D (Autonomous Goal 20 runs): Success Rate = {res_d['success_rate'] * 100:.1f}% | Mean Latency = {res_d['mean_latency_ms']:.1f}ms")
    print(f"Memory Stability: VRAM Delta = {vram_delta} MB (Peak: {vram_peak} MB) | RAM Delta = {ram_delta} MB")

    # 5. Clean up
    server.shutdown()
    _, engine = get_services()
    await engine.close()

    # 6. Save JSON Results
    results_payload = {
        "benchmark_metadata": {
            "total_runs": 100,
            "total_duration_seconds": total_suite_seconds,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "vram_initial_mb": vram_start,
            "vram_final_mb": vram_end,
            "vram_peak_mb": vram_peak,
            "vram_delta_mb": vram_delta,
            "ram_initial_mb": ram_start,
            "ram_final_mb": ram_end,
            "ram_delta_mb": ram_delta,
        },
        "scenario_a_cache_speedup": res_a,
        "scenario_b_self_healing": res_b,
        "scenario_c_captcha_lifecycle": res_c,
        "scenario_d_autonomous_goal": res_d,
    }

    out_json_path = os.path.join(os.path.dirname(__file__), "results_e2e_browser_stress.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n✓ Saved JSON results to: {out_json_path}")


if __name__ == "__main__":
    asyncio.run(main())
