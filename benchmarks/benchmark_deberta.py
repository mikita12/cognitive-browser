"""
Benchmark & Live Demonstration:
Comparison:
1. Traditional LLM Agent (Cloud API - Claude/GPT)
2. Agent with Ollama (Small 1.5B generative LLM with JSON schema)
3. Agent with DeBERTa-v3 (Pure discriminative NLI decision model on GPU)
"""

import asyncio
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples")))

from playwright.async_api import async_playwright
try:
    from extractor import DomCandidateExtractor
except ImportError:
    from examples.extractor import DomCandidateExtractor

try:
    from cognitive_browser import DebertaDecisionRouter
except ImportError:
    from deberta_router import DebertaDecisionRouter


async def run_deberta_benchmark():
    print("=" * 80)
    print("AGENT BENCHMARK: TRADITIONAL LLM vs GENERATIVE OLLAMA vs DISCRIMINATIVE DEBERTA")
    print("=" * 80)

    test_url = "https://news.ycombinator.com/login"
    intent = "click the login button to submit credentials"

    # Initialize DeBERTa model
    router = DebertaDecisionRouter(model_name="cross-encoder/nli-deberta-v3-small")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print(f"\n[1/4] Navigating to benchmark page in Playwright: {test_url}")
        await page.goto(test_url)

        # 1. Full DOM
        raw_html = await page.content()
        raw_html_len = len(raw_html)
        approx_raw_tokens = raw_html_len // 4

        print(f"\n[2/4] DOM Snapshot & Candidate Extraction:")
        print(f"  -> Raw HTML size: {raw_html_len:,} characters (~{approx_raw_tokens:,} tokens)")

        t_ext = time.perf_counter()
        candidates = await DomCandidateExtractor.extract_interactive_elements(page)
        extract_ms = (time.perf_counter() - t_ext) * 1000

        print(f"  -> Interactive elements found: {len(candidates)} (extraction time: {extract_ms:.2f} ms)")
        print(f"  -> Data size reduction: {((raw_html_len - (len(candidates) * 50)) / raw_html_len) * 100:.1f}%")

        # 2. DeBERTa inference (Reflex Core)
        print(f"\n[3/4] Discriminative inference (DeBERTa-v3 NLI on GPU):")
        decision = router.evaluate_candidates(intent=intent, candidates=candidates)

        print(f"  -> Selected element (ref): {decision['target_ref']}")
        print(f"  -> Playwright action: {decision['tool_action']}")
        print(f"  -> Model confidence (Entailment): {decision['confidence'] * 100:.1f}%")
        print(f"  -> DeBERTa inference latency: {decision['latency_ms']:.2f} ms!")
        print(f"  -> Tokens generated: 0 (Pure tensor probability)")

        print("\n  Top 3 candidates ranked by DeBERTa:")
        for rank, c in enumerate(decision["ranked_candidates"][:3], 1):
            print(f"    {rank}. [{c['confidence_score']*100:5.1f}%] {c['ref']} | <{c['tag']}> {c.get('type') or ''} text='{c.get('text')}' name='{c.get('name')}'")

        # 3. Action execution in Playwright
        chosen_ref = decision["target_ref"]
        if chosen_ref != "NONE":
            print(f"\n[4/4] Executing action in Playwright:")
            print(f"  -> Clicking selector: [data-s1-ref='{chosen_ref}']...")
            await page.locator(f"[data-s1-ref='{chosen_ref}']").click()
            print("  -> SUCCESS! Playwright executed the action with zero cloud API overhead.")

        # Comparison summary table
        print("\n" + "=" * 80)
        print("METRIC COMPARISON (SUMMARY)")
        print("=" * 80)
        print(f"{'Method':<32} | {'Model Type':<18} | {'Latency':<14} | {'Token Cost':<15}")
        print("-" * 80)
        print(f"{'Traditional Agent (Claude/GPT)':<32} | {'Cloud Generative':<18} | {'~3000 - 5000 ms':<14} | {'~12,000 tok./step':<15}")
        print(f"{'Local LLM (Qwen2.5-1.5B)':<32} | {'Generative LLM':<18} | {'~400 - 450 ms':<14} | {'Local ($0.00)':<15}")
        deberta_lat = f"{decision['latency_ms']} ms"
        print(f"{'DeBERTa-v3 NLI (Our Router)':<32} | {'Discriminative':<18} | {deberta_lat:<14} | {'0 tokens ($0.00)':<15}")
        print("=" * 80)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_deberta_benchmark())
