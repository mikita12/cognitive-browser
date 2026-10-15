#!/usr/bin/env python3
"""
Benchmark: Bazowy oficjalny Playwright MCP (Standardowe podejście większości programistów).
Mierzy liczbę roundtripów, rozmiar snapshotów DOM w bajtach/tokenach oraz czas trwania.
"""

import asyncio
import time
import json
from playwright.async_api import async_playwright

async def benchmark_base_playwright():
    print("=" * 80)
    print("🚀 START BENCHMARKU: BAZOWY PLAYWRIGHT MCP (Standard branżowy)")
    print("=" * 80)

    t0 = time.perf_counter()
    tokens_total = 0
    roundtrips = 0
    step_times = []

    async with async_playwright() as p:
        t_launch = time.perf_counter()
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # KROK 1: browser_navigate
        print("\n[Krok 1] browser_navigate('https://koleo.pl')...")
        t_step = time.perf_counter()
        await page.goto("https://koleo.pl", wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        roundtrips += 1
        step_times.append(("browser_navigate", time.perf_counter() - t_step))

        # KROK 2: browser_snapshot (Standardowy Playwright MCP zrzuca stan strony)
        print("[Krok 2] browser_snapshot() strony głównej...")
        t_step = time.perf_counter()
        snapshot_main_str = await page.aria_snapshot()
        html_main_str = await page.content()
        tokens_step_2_aria = int(len(snapshot_main_str) / 3.5)
        tokens_step_2_html = int(len(html_main_str) / 3.5)
        tokens_total += tokens_step_2_aria
        roundtrips += 1
        step_times.append(("browser_snapshot_main", time.perf_counter() - t_step))
        print(f"  -> Rozmiar ARIA YAML: {len(snapshot_main_str)} bajtów (~{tokens_step_2_aria} tokenów)")
        print(f"  -> Rozmiar pełnego HTML DOM: {len(html_main_str)} bajtów (~{tokens_step_2_html} tokenów)")

        # KROK 3: browser_type (Stacja odjazdu)
        print("[Krok 3] browser_type(Stacja odjazdu = 'Wrocław Główny')...")
        t_step = time.perf_counter()
        await page.get_by_role("combobox", name="Stacja odjazdu").fill("Wrocław Główny")
        await page.wait_for_timeout(500)
        roundtrips += 1
        step_times.append(("browser_type_departure", time.perf_counter() - t_step))

        # KROK 4: browser_type (Stacja przyjazdu)
        print("[Krok 4] browser_type(Stacja przyjazdu = 'Warszawa Centralna')...")
        t_step = time.perf_counter()
        await page.get_by_role("combobox", name="Stacja przyjazdu").fill("Warszawa Centralna")
        await page.wait_for_timeout(500)
        roundtrips += 1
        step_times.append(("browser_type_arrival", time.perf_counter() - t_step))

        # KROK 5: browser_click (Znajdź połączenie)
        print("[Krok 5] browser_click('Znajdź połączenie')...")
        t_step = time.perf_counter()
        await page.get_by_role("button", name=" Znajdź połączenie").click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2000)
        roundtrips += 1
        step_times.append(("browser_click_search", time.perf_counter() - t_step))

        # KROK 6: browser_snapshot (Zrzut strony wyników z połączeniami)
        print("[Krok 6] browser_snapshot() strony z wynikami...")
        t_step = time.perf_counter()
        snapshot_results_str = await page.aria_snapshot()
        tokens_step_6 = int(len(snapshot_results_str) / 3.5)
        tokens_total += tokens_step_6
        roundtrips += 1
        step_times.append(("browser_snapshot_results", time.perf_counter() - t_step))
        print(f"  -> Rozmiar drzewa wyników: {len(snapshot_results_str)} bajtów (~{tokens_step_6} tokenów)")

        # Podsumowanie
        total_browser_time = time.perf_counter() - t0
        
        # Obliczenie rzeczywistego czasu z uwzględnieniem narzutu LLM na każdy roundtrip:
        # Standardowy LLM w chmurze (Claude 3.5 Sonnet / GPT-4o) potrzebuje ~3.5 sekundy na przetworzenie każdego zrzutu i decyzję:
        llm_delay_per_roundtrip = 3.5
        realistic_total_time = total_browser_time + (roundtrips * llm_delay_per_roundtrip)

        print("\n" + "=" * 80)
        print("📊 WYNIKI BAZOWEGO PLAYWRIGHT MCP:")
        print("=" * 80)
        print(f"Liczba zapytań do LLM (Roundtrips): {roundtrips}")
        print(f"Czysty czas operacji przeglądarki: {total_browser_time:.2f} s")
        print(f"Szacowany realny czas z narzutem LLM (~3.5 s / krok): {realistic_total_time:.2f} s")
        print(f"Łączny rozmiar przesłanych danych do LLM: {len(snapshot_main_str) + len(snapshot_results_str)} bajtów")
        print(f"Łączne zużycie tokenów wejściowych (Input Context): ~{tokens_total} tokenów")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(benchmark_base_playwright())
