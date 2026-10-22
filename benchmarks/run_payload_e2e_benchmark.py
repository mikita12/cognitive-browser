"""
End-to-End Benchmark & Demonstration for Universal Payload Parameter Binding,
Intelligent File Upload, and System Cookie Import (Stage 5).
Uses a real local HTTP server with Playwright Chromium to test live browser interactions.
"""

import asyncio
import http.server
import json
import os
import socketserver
import tempfile
import threading
import time
from typing import Any, Dict, List

# Allow local test server loopback
os.environ["ALLOW_PRIVATE_NETWORKS"] = "true"

from cognitive_browser.browser import BrowserEngine
from cognitive_browser.cookies import extract_firefox_cookies
from cognitive_browser.reflex import AriaElementParser, ReflexCore

HTML_FORM_PAGE = """<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Aplikacja o pracę - TechCorp</title>
    <style>
        body { font-family: sans-serif; margin: 30px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="text"], input[type="email"], select { width: 300px; padding: 6px; }
        .custom-upload-btn { padding: 8px 14px; background: #0066cc; color: #fff; border: none; cursor: pointer; border-radius: 4px; }
        #file-chosen { margin-left: 10px; font-size: 13px; color: #555; }
        #status-msg { margin-top: 20px; font-weight: bold; color: green; }
    </style>
</head>
<body>
    <h1>Formularz rekrutacyjny - Senior Python Engineer</h1>
    <form id="recruitment-form">
        <div class="form-group">
            <label for="first_name">Imię</label>
            <input type="text" id="first_name" name="first_name">
        </div>
        <div class="form-group">
            <label for="last_name">Nazwisko</label>
            <input type="text" id="last_name" name="last_name">
        </div>
        <div class="form-group">
            <label for="email_addr">Adres e-mail</label>
            <input type="email" id="email_addr" name="email_addr">
        </div>
        <div class="form-group">
            <label for="phone_num">Numer telefonu</label>
            <input type="text" id="phone_num" name="phone_num">
        </div>
        <div class="form-group">
            <label for="level">Poziom doświadczenia</label>
            <select id="level" name="level">
                <option value="Junior">Junior</option>
                <option value="Mid">Mid</option>
                <option value="Senior">Senior</option>
                <option value="Lead">Lead / Architect</option>
            </select>
        </div>
        <div class="form-group">
            <label>Załącz CV (PDF, DOCX)</label>
            <input type="file" id="cv_input" name="cv_input" style="display:none">
            <button type="button" id="cv_btn" class="custom-upload-btn" onclick="document.getElementById('cv_input').click()">Załącz dokument CV</button>
            <span id="cv-chosen">Brak pliku</span>
        </div>
        <div class="form-group">
            <label>Zdjęcia referencyjne / Certyfikaty</label>
            <input type="file" id="gallery_input" name="gallery_input" multiple style="display:none">
            <button type="button" id="gallery_btn" class="custom-upload-btn" onclick="document.getElementById('gallery_input').click()">Dodaj certyfikaty (wiele plików)</button>
            <span id="gallery-chosen">Brak plików</span>
        </div>
        <div class="form-group">
            <input type="checkbox" id="rodo_consent" name="rodo_consent">
            <label for="rodo_consent" style="display:inline">Zgadzam się na przetwarzanie danych osobowych (RODO)</label>
        </div>
        <button type="submit" id="submit-btn" style="padding: 10px 20px; font-size: 16px;">Wyślij aplikację</button>
        <div id="status-msg"></div>
    </form>

    <script>
        document.getElementById('cv_input').addEventListener('change', function(e) {
            if (this.files && this.files.length > 0) {
                document.getElementById('cv-chosen').textContent = this.files[0].name;
            }
        });
        document.getElementById('gallery_input').addEventListener('change', function(e) {
            if (this.files && this.files.length > 0) {
                document.getElementById('gallery-chosen').textContent = this.files.length + ' pliki załączone';
            }
        });
        document.getElementById('recruitment-form').addEventListener('submit', function(e) {
            e.preventDefault();
            document.getElementById('status-msg').textContent = 'Sukces: Aplikacja została pomyślnie wysłana!';
        });
    </script>
</body>
</html>
"""


class MockHttpHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_FORM_PAGE.encode("utf-8"))

    def log_message(self, format, *args):
        pass


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def run_test_server():
    server = ReusableTCPServer(("127.0.0.1", 0), MockHttpHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server, port


async def run_benchmark():
    print("=" * 75)
    print("  STAGE 5: UNIVERSAL PAYLOAD BINDING & FILE UPLOAD E2E BENCHMARK")
    print("=" * 75)

    server, port = run_test_server()
    test_url = f"http://127.0.0.1:{port}"

    reflex = ReflexCore()
    engine = BrowserEngine(reflex=reflex)
    page = await engine.get_page()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": {},
        "system_cookies": {},
    }

    # 1. System cookie extraction test on local system (count only, no names or domains)
    print("\n[Step 1] System Cookies Verification:")
    firefox_cookies = extract_firefox_cookies("pracuj.pl")
    print(f"  -> Extracted {len(firefox_cookies)} cookies for verification.")
    results["system_cookies"]["count"] = len(firefox_cookies)

    # Create dummy files for upload
    with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as f_cv:
        f_cv.write("Dummy CV Content")
        cv_path = f_cv.name

    with tempfile.NamedTemporaryFile("w", suffix=".png", delete=False) as f_cert1:
        f_cert1.write("Cert 1")
        cert1_path = f_cert1.name

    with tempfile.NamedTemporaryFile("w", suffix=".png", delete=False) as f_cert2:
        f_cert2.write("Cert 2")
        cert2_path = f_cert2.name

    try:
        # Navigate to local test form
        print(f"\n[Step 2] Navigating to test form: {test_url}")
        t_nav = time.perf_counter()
        await page.goto(test_url, wait_until="domcontentloaded")
        nav_ms = round((time.perf_counter() - t_nav) * 1000, 2)
        print(f"  -> Page loaded in {nav_ms} ms")

        # Parse Accessibility Tree
        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=50)
        print(f"  -> Extracted {len(candidates)} interactive candidates from ARIA snapshot.")

        # Test Neural Bipartite Binding
        payload = {
            "first_name": "Antoni",
            "last_name": "Zieliński",
            "email": "antoni.zielinski@example.com",
            "phone": "+48 501 234 567",
            "experience": "Senior",
            "cv": cv_path,
            "certificates": [cert1_path, cert2_path],
            "rodo": True,
        }

        print("\n[Step 3] Running ReflexCore Neural Bipartite Matching on GPU...")
        t_match = time.perf_counter()
        matched_actions = reflex.bind_payload_to_elements(payload, candidates)
        match_ms = round((time.perf_counter() - t_match) * 1000, 2)
        print(f"  -> Matched {len(matched_actions)} fields in {match_ms} ms!")

        results["neural_matching"] = {
            "latency_ms": match_ms,
            "fields_matched": len(matched_actions),
            "actions": matched_actions,
        }

        # Step 4: Execute actions sequentially
        print("\n[Step 4] Executing Planned Actions in Chromium...")
        executed_log = []
        for act in matched_actions:
            t_exec = time.perf_counter()
            exec_desc = await engine.execute_reflex_action(page, act)
            e_ms = round((time.perf_counter() - t_exec) * 1000, 2)
            print(f"  -> Executed: {act['payload_key']} -> {exec_desc} in {e_ms} ms")
            executed_log.append({
                "key": act["payload_key"],
                "action": exec_desc,
                "latency_ms": e_ms,
            })

        results["execution_log"] = executed_log

        # Verify DOM values after execution
        cv_text = await page.locator("#cv-chosen").text_content()
        gallery_text = await page.locator("#gallery-chosen").text_content()
        email_val = await page.locator("#email_addr").input_value()
        rodo_checked = await page.locator("#rodo_consent").is_checked()

        print("\n[Step 5] DOM State Verification:")
        print(f"  -> Email value in input: '{email_val}' (Expected: 'antoni.zielinski@example.com')")
        print(f"  -> CV chosen label: '{cv_text}' (Expected: '{os.path.basename(cv_path)}')")
        print(f"  -> Gallery chosen label: '{gallery_text}' (Expected: '2 pliki załączone')")
        print(f"  -> RODO checkbox state: {rodo_checked} (Expected: True)")

        assert email_val == "antoni.zielinski@example.com"
        assert os.path.basename(cv_path) in (cv_text or "")
        assert "2 pliki" in (gallery_text or "")
        assert rodo_checked is True

        # Click Submit button
        await engine.click_element(page, role="button", name="Wyślij aplikację")
        await page.wait_for_timeout(300)
        status_msg = await page.locator("#status-msg").text_content()
        print(f"  -> Form submission status: '{status_msg}'")
        assert "Sukces" in (status_msg or "")

        results["status"] = "SUCCESS"
        results["dom_verification"] = {
            "email_ok": email_val == "antoni.zielinski@example.com",
            "cv_upload_ok": os.path.basename(cv_path) in (cv_text or ""),
            "multi_file_upload_ok": "2 pliki" in (gallery_text or ""),
            "rodo_ok": rodo_checked is True,
            "submission_ok": "Sukces" in (status_msg or ""),
        }
        print("\n✅ ALL VERIFICATION CHECKS PASSED PERFECTLY!")

    finally:
        await engine.close()
        server.shutdown()
        if os.path.exists(cv_path):
            os.remove(cv_path)
        if os.path.exists(cert1_path):
            os.remove(cert1_path)
        if os.path.exists(cert2_path):
            os.remove(cert2_path)

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/results_payload_e2e.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved benchmark results to benchmarks/results_payload_e2e.json")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
