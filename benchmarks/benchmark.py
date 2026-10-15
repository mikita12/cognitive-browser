"""
Benchmarking & Comparison Harness.
Compares:
1. Standard Agent Pattern: Full DOM snapshot sent to large cloud LLM prompt.
2. System 1 Decision Router: Pruned candidates sent to local fast model pool.
"""

import asyncio
import time
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples")))

from playwright.async_api import async_playwright
try:
    from extractor import DomCandidateExtractor
    from decision_pool import ParallelDecisionPool, DecisionWorker
except ImportError:
    from examples.extractor import DomCandidateExtractor
    from examples.decision_pool import ParallelDecisionPool, DecisionWorker


async def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Traditional LLM Agent vs System 1 Decision Router")
    print("=" * 70)

    test_url = "https://news.ycombinator.com/login"
    intent = "I want to click the 'login' submit button to submit the form"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print(f"\n[1/4] Navigating to benchmark page: {test_url}")
        await page.goto(test_url)

        # -------------------------------------------------------------
        # Approach 1: Full DOM snapshot (Traditional Agent Approach)
        # -------------------------------------------------------------
        raw_html = await page.content()
        raw_html_len = len(raw_html)
        approx_raw_tokens = raw_html_len // 4

        print(f"\n[2/4] METHOD A (Traditional Cloud LLM Agent):")
        print(f"  -> Raw HTML size: {raw_html_len:,} characters")
        print(f"  -> Approximate prompt tokens: ~{approx_raw_tokens:,} tokens")
        print(f"  -> Estimated response time (Claude/GPT): ~3.0 - 5.5 seconds")
        print(f"  -> Estimated API cost (Claude 3.5 Sonnet): ~${(approx_raw_tokens / 1_000_000) * 3.00:.4f} for this single step")

        # -------------------------------------------------------------
        # Approach 2: System 1 Decision Router
        # -------------------------------------------------------------
        print(f"\n[3/4] METHOD B (Our System 1 Decision Router):")
        t0 = time.perf_counter()
        candidates = await DomCandidateExtractor.extract_interactive_elements(page)
        extract_time = (time.perf_counter() - t0) * 1000

        valid_refs = [c["ref"] for c in candidates]
        prompt_text = DomCandidateExtractor.format_for_prompt(candidates)
        prompt_tokens_est = len(prompt_text) // 4

        print(f"  -> Visible interactive elements found: {len(candidates)}")
        print(f"  -> In-browser extraction latency: {extract_time:.2f} ms")
        print(f"  -> Payload size to router: {len(prompt_text)} characters (~{prompt_tokens_est} tokens)")
        print(f"  -> Data size reduction: {((raw_html_len - len(prompt_text)) / raw_html_len) * 100:.1f}% LESS DATA")

        # -------------------------------------------------------------
        # Step 3: Concurrent decision model pool query
        # -------------------------------------------------------------
        print(f"\n[4/4] Querying parallel decision model pool (Ensemble)...")
        pool = ParallelDecisionPool(models=["llama3.2:1b", "qwen2.5:1.5b"])

        t_pool_start = time.perf_counter()
        results = await pool.evaluate_intent(intent, prompt_text, valid_refs)
        total_pool_time = (time.perf_counter() - t_pool_start) * 1000

        print(f"\n" + "-" * 70)
        print("DECISION MODEL RESULTS (CONCURRENT EXECUTION):")
        print("-" * 70)

        for res in results:
            if res.get("success"):
                dec = res["decision"]
                print(f"Model: {res['model']}")
                print(f"  Latency: {res['latency_ms']} ms")
                print(f"  Input tokens (actual): {res.get('prompt_eval_count')}")
                print(f"  Output tokens: {res.get('eval_count')}")
                print(f"  Target ref: {dec.get('target_ref')}")
                print(f"  Playwright action: {dec.get('tool_action')}")
                print(f"  Confidence: {dec.get('confidence') * 100:.1f}%")
                print(f"  Escalate to primary LLM: {dec.get('should_escalate')}")
                print()
            else:
                print(f"Model: {res['model']} -> ERROR: {res.get('error')}")

        print(f"Total concurrent inference latency: {total_pool_time:.2f} ms")

        # -------------------------------------------------------------
        # Step 4: Playwright action execution
        # -------------------------------------------------------------
        best_decision = results[0].get("decision") if results[0].get("success") else None
        if best_decision and best_decision.get("target_ref") != "NONE":
            chosen_ref = best_decision["target_ref"]
            print(f"\n[PLAYWRIGHT ACTION]: Clicking element with attribute data-s1-ref='{chosen_ref}'...")
            await page.locator(f"[data-s1-ref='{chosen_ref}']").click()
            print("-> Action executed successfully! Zero CSS selector hallucination.")

        await browser.close()
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
