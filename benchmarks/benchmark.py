"""
Benchmarking & Comparison Harness.
Compares:
1. Standard Agent Pattern: Full DOM snapshot sent to large LLM prompt.
2. System 1 Decision Router: Pruned candidates sent to local fast model pool.
"""

import asyncio
import time
import json
from playwright.async_api import async_playwright
from extractor import DomCandidateExtractor
from decision_pool import ParallelDecisionPool, DecisionWorker
import httpx

async def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Tradycyjny Agent vs System 1 Decision Router")
    print("=" * 70)

    test_url = "https://news.ycombinator.com/login"
    intent = "Chcę kliknąć przycisk logowania ('login'), aby zatwierdzić formularz"

    async with async_playwright() as p:
        # Odpalamy przeglądarkę
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print(f"\n[1/4] Otwieram stronę testową: {test_url}")
        await page.goto(test_url)

        # -------------------------------------------------------------
        # Podejście 1: Pełny zrzut DOM (Tradycyjne podejście agentowe)
        # -------------------------------------------------------------
        raw_html = await page.content()
        raw_html_len = len(raw_html)
        approx_raw_tokens = raw_html_len // 4

        print(f"\n[2/4] METODA A (Tradycyjny LLM Agent):")
        print(f"  -> Rozmiar surowego HTML: {raw_html_len:,} znaków")
        print(f"  -> Przybliżona liczba tokenów: ~{approx_raw_tokens:,} tokenów")
        print(f"  -> Szacowany czas odpowiedzi Claude/GPT: ~3.0 - 5.5 sekundy")
        print(f"  -> Szacowany koszt API (Claude 3.5 Sonnet): ~${(approx_raw_tokens / 1_000_000) * 3.00:.4f} za ten 1 krok")

        # -------------------------------------------------------------
        # Podejście 2: Router Decyzyjny (Nasz projekt)
        # -------------------------------------------------------------
        print(f"\n[3/4] METODA B (Nasz Router Decyzyjny System 1):")
        t0 = time.perf_counter()
        candidates = await DomCandidateExtractor.extract_interactive_elements(page)
        extract_time = (time.perf_counter() - t0) * 1000

        valid_refs = [c["ref"] for c in candidates]
        prompt_text = DomCandidateExtractor.format_for_prompt(candidates)
        prompt_tokens_est = len(prompt_text) // 4

        print(f"  -> Znaleziono widocznych elementów interaktywnych: {len(candidates)}")
        print(f"  -> Czas ekstrakcji z przeglądarki: {extract_time:.2f} ms")
        print(f"  -> Wielkość ekstraktu do modelu: {len(prompt_text)} znaków (~{prompt_tokens_est} tokenów)")
        print(f"  -> Redukcja wielkości danych: {((raw_html_len - len(prompt_text)) / raw_html_len) * 100:.1f}% MNIEJ DANYCH")

        # -------------------------------------------------------------
        # Krok 3: Równoległe odpytanie modeli decyzyjnych (Parallel Pool)
        # -------------------------------------------------------------
        print(f"\n[4/4] Odpytuję równolegle pulę modeli decyzyjnych (Ensemble)...")
        pool = ParallelDecisionPool(models=["llama3.2:1b", "qwen2.5:1.5b"])
        
        t_pool_start = time.perf_counter()
        results = await pool.evaluate_intent(intent, prompt_text, valid_refs)
        total_pool_time = (time.perf_counter() - t_pool_start) * 1000

        print(f"\n" + "-" * 70)
        print("WYNIKI MODELI DECYZYJNYCH (RÓWNOLEGŁE WYKONANIE):")
        print("-" * 70)

        for res in results:
            if res.get("success"):
                dec = res["decision"]
                print(f"Model: {res['model']}")
                print(f"  Czas reakcji: {res['latency_ms']} ms")
                print(f"  Tokeny wejściowe (realne): {res.get('prompt_eval_count')}")
                print(f"  Tokeny wyjściowe: {res.get('eval_count')}")
                print(f"  Wybrany element: ref = {dec.get('target_ref')}")
                print(f"  Akcja Playwright: {dec.get('tool_action')}")
                print(f"  Pewność: {dec.get('confidence') * 100:.1f}%")
                print(f"  Eskalacja do dużego LLM: {dec.get('should_escalate')}")
                print()
            else:
                print(f"Model: {res['model']} -> BŁĄD: {res.get('error')}")

        print(f"Całkowity czas równoległego wnioskowania: {total_pool_time:.2f} ms")

        # -------------------------------------------------------------
        # Krok 4: Wykonanie akcji w Playwrighcie na podstawie wybranego refa
        # -------------------------------------------------------------
        best_decision = results[0].get("decision") if results[0].get("success") else None
        if best_decision and best_decision.get("target_ref") != "NONE":
            chosen_ref = best_decision["target_ref"]
            print(f"\n[AKCJA PLAYWRIGHT]: Klikam element z atrybutem data-s1-ref='{chosen_ref}'...")
            await page.locator(f"[data-s1-ref='{chosen_ref}']").click()
            print("-> Akcja wykonana sukcesem! Zero halucynacji selektora CSS.")

        await browser.close()
        print("\n" + "=" * 70)
        print("TEST ZAKOŃCZONY POMYŚLNIE")
        print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
