#!/usr/bin/env python3
"""
Benchmark: Base Official Playwright MCP (Standard Industry Agent Pattern).
Measures roundtrips, DOM snapshot sizes in bytes/tokens, and execution latency.
"""

import asyncio
import time

from playwright.async_api import async_playwright


async def benchmark_base_playwright():
    print("=" * 80)
    print("🚀 BENCHMARK: BASE PLAYWRIGHT MCP (Industry Baseline)")
    print("=" * 80)

    t0 = time.perf_counter()
    tokens_total = 0
    roundtrips = 0
    step_times = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # STEP 1: browser_navigate
        print("\n[Step 1] browser_navigate('https://koleo.pl')...")
        t_step = time.perf_counter()
        await page.goto("https://koleo.pl", wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        roundtrips += 1
        step_times.append(("browser_navigate", time.perf_counter() - t_step))

        # STEP 2: browser_snapshot (Standard Playwright MCP dumps full page state)
        print("[Step 2] browser_snapshot() on homepage...")
        t_step = time.perf_counter()
        snapshot_main_str = await page.aria_snapshot()
        html_main_str = await page.content()
        tokens_step_2_aria = int(len(snapshot_main_str) / 3.5)
        tokens_step_2_html = int(len(html_main_str) / 3.5)
        tokens_total += tokens_step_2_aria
        roundtrips += 1
        step_times.append(("browser_snapshot_main", time.perf_counter() - t_step))
        print(f"  -> ARIA YAML size: {len(snapshot_main_str)} bytes (~{tokens_step_2_aria} tokens)")
        print(f"  -> Raw HTML DOM size: {len(html_main_str)} bytes (~{tokens_step_2_html} tokens)")

        # STEP 3: browser_type (Departure station)
        print("[Step 3] browser_type(Departure station = 'Wroclaw Central')...")
        t_step = time.perf_counter()
        await page.get_by_role("combobox", name="Stacja odjazdu").fill("Wrocław Główny")
        await page.wait_for_timeout(500)
        roundtrips += 1
        step_times.append(("browser_type_departure", time.perf_counter() - t_step))

        # STEP 4: browser_type (Arrival station)
        print("[Step 4] browser_type(Arrival station = 'Warszawa Centralna')...")
        t_step = time.perf_counter()
        await page.get_by_role("combobox", name="Stacja przyjazdu").fill("Warszawa Centralna")
        await page.wait_for_timeout(500)
        roundtrips += 1
        step_times.append(("browser_type_arrival", time.perf_counter() - t_step))

        # STEP 5: browser_click (Search connections)
        print("[Step 5] browser_click('Search connections')...")
        t_step = time.perf_counter()
        await page.get_by_role("button", name=" Znajdź połączenie").click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2000)
        roundtrips += 1
        step_times.append(("browser_click_search", time.perf_counter() - t_step))

        # STEP 6: browser_snapshot (Results page snapshot)
        print("[Step 6] browser_snapshot() on search results...")
        t_step = time.perf_counter()
        snapshot_results_str = await page.aria_snapshot()
        tokens_step_6 = int(len(snapshot_results_str) / 3.5)
        tokens_total += tokens_step_6
        roundtrips += 1
        step_times.append(("browser_snapshot_results", time.perf_counter() - t_step))
        print(f"  -> Results ARIA tree size: {len(snapshot_results_str)} bytes (~{tokens_step_6} tokens)")

        # Summary
        total_browser_time = time.perf_counter() - t0

        # Realistic latency calculation including cloud LLM roundtrip overhead:
        # Standard cloud LLM (Claude 3.5 Sonnet / GPT-4o) requires ~3.5s per snapshot reasoning roundtrip:
        llm_delay_per_roundtrip = 3.5
        realistic_total_time = total_browser_time + (roundtrips * llm_delay_per_roundtrip)

        print("\n" + "=" * 80)
        print("📊 BASE PLAYWRIGHT MCP RESULTS:")
        print("=" * 80)
        print(f"LLM Queries (Roundtrips): {roundtrips}")
        print(f"Raw Browser Execution Time: {total_browser_time:.2f} s")
        print(f"Estimated Realistic Duration with LLM Overhead (~3.5 s / step): {realistic_total_time:.2f} s")
        print(f"Total Payload Delivered to LLM: {len(snapshot_main_str) + len(snapshot_results_str)} bytes")
        print(f"Total Input Context Tokens Consumed: ~{tokens_total} tokens")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(benchmark_base_playwright())
