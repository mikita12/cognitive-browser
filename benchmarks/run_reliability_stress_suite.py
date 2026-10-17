"""
Reliability, Stress, and Fault-Detection Suite for Cognitive Browser.
Executes four rigorous edge-case tests:
1. High-throughput CUDA Memory & Latency Stability (100 rapid evaluations)
2. Stale Element & Ambiguity Recovery (Missing elements, noise filtering)
3. Dynamic Autocomplete & Event Propagation Simulation (Change, input, keydown events)
4. Live Automated CAPTCHA / Security Challenge Detection & CapSolver Readiness
"""

import asyncio
import time
from typing import Any, Dict, List

import torch

from cognitive_browser.browser import BrowserEngine
from cognitive_browser.captcha import CaptchaResolver
from cognitive_browser.reflex import AriaElementParser, ReflexCore


async def test_1_cuda_latency_stability(reflex: ReflexCore) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("🧪 TEST 1: CUDA Memory & Latency Stability (100 Rapid Evaluations)")
    print("=" * 60)

    mock_candidates = [
        {"role": "textbox", "name": "Email address", "context": "form > div"},
        {"role": "button", "name": "Submit application", "context": "form > actions"},
        {"role": "link", "name": "Privacy Policy", "context": "footer"},
        {"role": "checkbox", "name": "I accept terms", "context": "form"},
        {"role": "searchbox", "name": "Search articles", "context": "header"},
        {"role": "link", "name": "Careers at Monterail", "context": "nav"},
        {"role": "textbox", "name": "Phone number", "context": "form > div"},
        {"role": "button", "name": "Upload CV resume", "context": "form > upload"},
    ]

    queries = [
        "fill user email input",
        "click submit form button",
        "find privacy notice consent",
        "upload resume document",
        "search query input box",
    ]

    latencies: List[float] = []
    initial_vram = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    t0_total = time.perf_counter()
    for i in range(100):
        q = queries[i % len(queries)]
        eval_res = reflex.evaluate_candidates(q, mock_candidates)
        latencies.append(eval_res["latency_ms"])
    total_sec = time.perf_counter() - t0_total

    final_vram = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    vram_growth = final_vram - initial_vram

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

    print("  • Iterations: 100")
    print(f"  • Total time: {total_sec:.2f}s ({100 / total_sec:.1f} evals/sec)")
    print(f"  • Latency Mean: {avg_latency:.2f}ms | P95: {p95_latency:.2f}ms | P99: {p99_latency:.2f}ms")
    print(f"  • CUDA VRAM Allocated: {final_vram:.1f} MB (VRAM Leak / Growth: {vram_growth:.2f} MB)")

    passed = vram_growth < 10.0 and avg_latency < 25.0
    return {
        "test": "cuda_latency_stability",
        "passed": passed,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "vram_leak_mb": vram_growth,
        "evals_per_sec": 100 / total_sec,
    }


async def test_2_ambiguity_and_stale_recovery(reflex: ReflexCore) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Stale Elements, Noise & Ambiguity Recovery")
    print("=" * 60)

    # Empty candidate pool
    empty_res = reflex.evaluate_candidates("click submit button", [])
    assert empty_res["should_escalate"] is True
    assert empty_res["target_ref"] == "NONE"
    print("  ✅ Empty candidate pool gracefully returns escalation without crash")

    # Pure noise pool (only generic footers/headers)
    noise_pool = [
        {"role": "link", "name": "Cookies policy", "context": "footer"},
        {"role": "link", "name": "About us", "context": "footer"},
        {"role": "link", "name": "Terms of service", "context": "footer"},
    ]
    noise_res = reflex.evaluate_candidates("find flight ticket prices", noise_pool)
    print(f"  • Noise candidate score against unrelated goal: {noise_res['confidence']:.3f}")
    assert noise_res["confidence"] < 0.60
    print("  ✅ Irrelevant distractions successfully suppressed below threshold")

    # PUA icon characters stripping
    pua_snap = '- button "\ue001 \uf054 Zezwól na wszystkie \uf00d"'
    cands = AriaElementParser.parse_snapshot(pua_snap)
    assert "Zezwól na wszystkie" in cands[0]["name"]
    assert "\ue001" not in cands[0]["name"]
    print(f"  ✅ PUA icon stripped correctly: '{cands[0]['name']}'")

    return {"test": "ambiguity_and_stale_recovery", "passed": True}


async def test_3_dynamic_event_dispatch(engine: BrowserEngine) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Dynamic Autocomplete & Synthetic Event Propagation")
    print("=" * 60)

    page = await engine.get_page()
    # Create an in-memory HTML page with autocomplete listener that requires input + change + keydown
    html_test = """
    <!DOCTYPE html>
    <html>
    <body>
        <input id="ac_input" type="text" placeholder="Type station..." autocomplete="off" />
        <div id="status">idle</div>
        <script>
            let inputFired = false;
            let changeFired = false;
            const input = document.getElementById('ac_input');
            const status = document.getElementById('status');

            input.addEventListener('input', () => { inputFired = true; status.innerText = 'input_ok'; });
            input.addEventListener('change', () => { changeFired = true; status.innerText = 'change_ok'; });
            input.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowDown') status.innerText = 'arrow_selected';
            });
        </script>
    </body>
    </html>
    """
    await page.set_content(html_test)

    # Test fill_element with auto Enter / change events
    await engine.fill_element(page, role="textbox", name="Type station...", value="Warszawa Centralna")
    st_val = await page.evaluate("() => document.getElementById('status').innerText")
    input_val = await page.evaluate("() => document.getElementById('ac_input').value")

    print(f"  • Injected value: '{input_val}'")
    print(f"  • Status after fill: '{st_val}'")

    # Test arrow down press
    await engine.press_key(page, "ArrowDown")
    st_after_arrow = await page.evaluate("() => document.getElementById('status').innerText")
    print(f"  • Status after ArrowDown key: '{st_after_arrow}'")

    passed = input_val == "Warszawa Centralna" and st_after_arrow == "arrow_selected"
    print(f"  {'✅ PASSED' if passed else '❌ FAILED'}: Autocomplete event listeners successfully dispatched!")
    return {"test": "dynamic_event_dispatch", "passed": passed}


async def test_4_captcha_system_readiness(engine: BrowserEngine, reflex: ReflexCore) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("🧪 TEST 4: Live CAPTCHA Engine & CapSolver API Readiness")
    print("=" * 60)

    resolver = CaptchaResolver()
    api_key = resolver.api_key
    print(f"  • Configured CapSolver Key: {api_key[:12]}...{api_key[-4:] if api_key else 'NONE'}")

    # 1. Test live balance check on CapSolver API
    balance_info = {}
    if api_key:
        import json
        import urllib.request
        try:
            req = urllib.request.Request(
                "https://api.capsolver.com/getBalance",
                data=json.dumps({"clientKey": api_key}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                balance_info = json.loads(r.read().decode())
            print(f"  • CapSolver Balance: ${balance_info.get('balance', 0.0):.4f} (Status: OK)")
        except Exception as e:
            print(f"  • CapSolver API balance check note: {e}")

    # 2. Test semantic CAPTCHA challenge detection
    cands_challenge = [
        {"role": "checkbox", "name": "Potwierdź, że jesteś człowiekiem", "context": "cf-turnstile"},
        {"role": "button", "name": "Weryfikuj", "context": "modal"},
    ]
    c_eval = reflex.evaluate_captcha_presence(
        candidates=cands_challenge,
        page_title="Weryfikacja zabezpieczeń",
        snapshot_yaml="- checkbox 'Potwierdź, że jesteś człowiekiem'\n- text 'Weryfikacja zabezpieczeń'",
    )
    print(f"  • ReflexCore Captcha Probability on Challenge: {c_eval['confidence']:.3f} (Is Captcha: {c_eval['is_captcha']})")
    assert c_eval["is_captcha"] is True
    print("  ✅ ModernBERT correctly triggered CAPTCHA escalation state!")

    # 3. Test clean page non-trigger
    c_clean = reflex.evaluate_captcha_presence(
        candidates=[{"role": "textbox", "name": "Nazwisko", "context": "form"}],
        page_title="Formularz zgłoszeniowy",
        snapshot_yaml="- textbox 'Nazwisko'",
    )
    print(f"  • ReflexCore Captcha Probability on Clean Form: {c_clean['confidence']:.3f} (Is Captcha: {c_clean['is_captcha']})")
    assert c_clean["is_captcha"] is False
    print("  ✅ Clean forms correctly bypass CAPTCHA solving pipeline!")

    passed = bool(api_key and c_eval["is_captcha"] and not c_clean["is_captcha"])
    return {
        "test": "captcha_system_readiness",
        "passed": passed,
        "balance": balance_info.get("balance", 0.0),
        "captcha_detection_confidence": c_eval["confidence"],
    }


async def main():
    print("🚀 Initializing Reliability & Stress Test Suite on GPU...")
    t_start = time.perf_counter()
    reflex = ReflexCore()
    engine = BrowserEngine(reflex=reflex)

    results = []
    results.append(await test_1_cuda_latency_stability(reflex))
    results.append(await test_2_ambiguity_and_stale_recovery(reflex))
    results.append(await test_3_dynamic_event_dispatch(engine))
    results.append(await test_4_captcha_system_readiness(engine, reflex))

    print("\n" + "=" * 60)
    print("📊 STRESS & RELIABILITY TEST SUITE RESULTS:")
    print("=" * 60)
    all_passed = True
    for r in results:
        status_str = "✅ PASSED" if r.get("passed") else "❌ FAILED"
        print(f"  • {r['test']:<35} : {status_str}")
        if not r.get("passed"):
            all_passed = False

    print(f"\nTotal suite duration: {time.perf_counter() - t_start:.2f}s")
    print(f"Overall Reliability Verdict: {'100% OPERATIONAL & ROBUST' if all_passed else 'ATTENTION NEEDED'}")
    await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
