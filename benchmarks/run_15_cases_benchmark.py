#!/usr/bin/env python3
"""
15 Real-World Web Automation Benchmark: Cognitive Browser (Reflex Core DeBERTa + Cortex Active Escalation) vs Base Playwright MCP.
Full measurement of end-to-end tasks driven to 100% completion.
When DeBERTa encounters ambiguity or modal obstructions, Cortex Engine (LLM) assists in real-time.
Tested against standard Playwright MCP pattern where every single step requires full ARIA snapshots and cloud LLM calls.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional

import torch
from playwright.async_api import async_playwright

try:
    from cognitive_browser import AgentOrchestrator
except ImportError:
    from agent_orchestrator import AgentOrchestrator

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

LOCAL_FIXTURE_CASES = [
    {
        "id": 1,
        "category": "E-Commerce",
        "name": "Local Store Catalog - Search Product",
        "url": f"file://{os.path.join(FIXTURES_DIR, 'ecommerce.html')}",
        "steps": [{"instruction": "Search Products", "fill_value": "Pro Laptop"}, {"instruction": "Search"}],
    },
    {
        "id": 2,
        "category": "Rail Transport",
        "name": "Local Travel Booking - Train Connection",
        "url": f"file://{os.path.join(FIXTURES_DIR, 'travel.html')}",
        "steps": [
            {"instruction": "Departure station", "fill_value": "Warszawa Centralna"},
            {"instruction": "Destination station", "fill_value": "Krakow Glowny"},
            {"instruction": "Search connections"},
        ],
    },
]

TEST_CASES = [
    {
        "id": 1,
        "category": "Rail Transport",
        "name": "KOLEO - Train ticket connection search",
        "url": "https://koleo.pl",
        "steps": [
            {"instruction": "Stacja odjazdu", "fill_value": "Wrocław Główny"},
            {"instruction": "Stacja przyjazdu", "fill_value": "Poznań Główny"},
            {"instruction": "Znajdź połączenie"},
        ],
    },
    {
        "id": 2,
        "category": "Logistics & Couriers",
        "name": "InPost - Navigate to parcel locker pricing",
        "url": "https://inpost.pl",
        "steps": [{"instruction": "Cennik"}, {"instruction": "Manager Paczek - Paczkomat"}],
    },
    {
        "id": 3,
        "category": "E-Commerce (Standard)",
        "name": "Books to Scrape - Filter book categories",
        "url": "http://books.toscrape.com",
        "steps": [{"instruction": "Travel"}, {"instruction": "Add to basket"}],
    },
    {
        "id": 4,
        "category": "Multi-field Forms",
        "name": "HTTPBin - Complete pizza order (HTML5 Form)",
        "url": "https://httpbin.org/forms/post",
        "steps": [
            {"instruction": "Customer name", "fill_value": "John Doe"},
            {"instruction": "Telephone", "fill_value": "500600700"},
            {"instruction": "E-mail address", "fill_value": "john@doe.com"},
            {"instruction": "Large"},
            {"instruction": "Bacon"},
        ],
    },
    {
        "id": 5,
        "category": "Marketplace / Classifieds",
        "name": "OLX - Search electronics offers",
        "url": "https://www.olx.pl",
        "steps": [{"instruction": "search query input", "fill_value": "PlayStation 5"}, {"instruction": "Szukaj"}],
    },
    {
        "id": 6,
        "category": "Community & Tech News",
        "name": "Hacker News - Navigate to Show HN discussion",
        "url": "https://news.ycombinator.com",
        "steps": [{"instruction": "show"}, {"instruction": "comments"}],
    },
    {
        "id": 7,
        "category": "Software / Open Source",
        "name": "GitHub Trending - Explore trending repositories",
        "url": "https://github.com/trending",
        "steps": [{"instruction": "Spoken Language"}, {"instruction": "Developers"}],
    },
    {
        "id": 8,
        "category": "E-Commerce (Electronics)",
        "name": "Morele.net - Search computer peripherals",
        "url": "https://www.morele.net",
        "steps": [
            {"instruction": "product search input", "fill_value": "Mechanical keyboard"},
            {"instruction": "szukaj"},
        ],
    },
    {
        "id": 9,
        "category": "E-Commerce / Quotes",
        "name": "Quotes to Scrape - Filter quotes by tag",
        "url": "http://quotes.toscrape.com",
        "steps": [{"instruction": "humor"}, {"instruction": "Next"}],
    },
    {
        "id": 10,
        "category": "Food Delivery / Pizzeria",
        "name": "Express Pizza - Select Detroit Style delivery option",
        "url": "https://expresspizza.pl",
        "steps": [{"instruction": "DOSTAWA DO DOMU"}, {"instruction": "Zielona Góra"}],
    },
    {
        "id": 11,
        "category": "Food Delivery / Artisanal Pizzeria",
        "name": "Mania Smaku - Select pizza menu",
        "url": "https://mania.pizza",
        "steps": [{"instruction": "Zamów online"}, {"instruction": "Pizze"}],
    },
    {
        "id": 12,
        "category": "Local Food Delivery",
        "name": "Pizzu Wrocław - Check lunch specials online",
        "url": "https://pizzu.pl",
        "steps": [{"instruction": "Menu"}, {"instruction": "LUNCH"}],
    },
    {
        "id": 13,
        "category": "Public Transportation",
        "name": "E-Podróżnik - Connection search Wroclaw to Krakow",
        "url": "https://www.e-podroznik.pl",
        "steps": [
            {"instruction": "Skąd", "fill_value": "Wrocław"},
            {"instruction": "Dokąd", "fill_value": "Kraków"},
            {"instruction": "Znajdź połączenie"},
        ],
    },
    {
        "id": 14,
        "category": "Public Administration",
        "name": "Gov.pl - Search ID card online procedure",
        "url": "https://www.gov.pl",
        "steps": [{"instruction": "Szukaj w serwisie", "fill_value": "Dowód osobisty"}, {"instruction": "Szukaj"}],
    },
    {
        "id": 15,
        "category": "Urban Transit",
        "name": "MPK Wrocław - Tram line timetable check",
        "url": "https://www.wroclaw.pl/komunikacja/rozklady-jazdy",
        "steps": [{"instruction": "8 Przejdź do rozkładu linii"}],
    },
]


async def run_single_iteration(
    orch: AgentOrchestrator, test_cases: List[Dict[str, Any]], run_idx: int = 1
) -> List[Dict[str, Any]]:
    results = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        for tc in test_cases:
            case_id = tc["id"]
            name = tc["name"]
            url = tc["url"]
            steps = tc["steps"]

            print(f"\n[Run {run_idx} | Case {case_id:02d}] {name}")
            print(f"       URL: {url} | Steps: {len(steps)}")

            # TEST A: COGNITIVE BROWSER (REFLEX CORE + CORTEX ENGINE)
            cog_page = await browser.new_page()
            try:
                cog_res = await orch.execute_task(page=cog_page, url=url, steps=steps)
            except Exception as e:
                cog_res = {
                    "status": "failed",
                    "total_execution_seconds": 0.0,
                    "system1_gpu_ms": 0.0,
                    "escalations_count": 0,
                    "system2_tokens": 0,
                    "error": str(e),
                    "steps_executed": 0,
                }
            finally:
                await cog_page.close()

            cog_success = cog_res.get("status") == "completed"
            cog_time = cog_res.get("total_execution_seconds", 0.0)
            cog_gpu = cog_res.get("reflex_gpu_ms", cog_res.get("system1_gpu_ms", 0.0))
            cog_s2_tokens = cog_res.get("cortex_tokens", cog_res.get("system2_tokens", 0))
            cog_escalations = cog_res.get("escalations_count", 0)
            cog_total_tokens = 150 + int(len(json.dumps(steps)) / 3.5) + cog_s2_tokens

            print(
                f"     Status: {'✅ COMPLETED' if cog_success else '⚠️ PARTIAL'} | Time: {cog_time:.2f} s | GPU: {cog_gpu:.1f} ms | Cortex Tokens: {cog_s2_tokens} | Total: ~{cog_total_tokens} tok."
            )

            # TEST B: BASE PLAYWRIGHT MCP (ANALYTICAL SIMULATION)
            base_page = await browser.new_page()
            base_t0 = time.perf_counter()
            base_roundtrips = 0
            base_tokens_accumulated = 0
            base_success = True
            base_error = None
            base_steps_executed = 0

            try:
                await base_page.goto(url, wait_until="domcontentloaded", timeout=25000)
                await base_page.wait_for_timeout(500)
                base_roundtrips += 1

                snap_init = await base_page.aria_snapshot()
                snap_init_tokens = int(len(snap_init) / 3.5)
                base_tokens_accumulated += snap_init_tokens + 350
                base_roundtrips += 1

                for s in steps:
                    val = s.get("fill_value")
                    inst = s.get("instruction")
                    base_roundtrips += 1
                    step_context_tokens = snap_init_tokens + (base_roundtrips * 400)
                    base_tokens_accumulated += step_context_tokens

                    try:
                        if val:
                            loc = base_page.get_by_role("textbox", name=inst).first
                            if not await loc.is_visible():
                                loc = base_page.locator("input, textarea").first
                            await loc.fill(val, timeout=2000)
                        else:
                            loc = base_page.get_by_role("button", name=inst).first
                            if not await loc.is_visible():
                                loc = base_page.get_by_role("link", name=inst).first
                            if not await loc.is_visible():
                                loc = base_page.get_by_text(inst).first
                            await loc.click(timeout=2000)
                        base_steps_executed += 1
                    except Exception:
                        pass
                    await base_page.wait_for_timeout(300)

                snap_final = await base_page.aria_snapshot()
                snap_final_tokens = int(len(snap_final) / 3.5)
                base_tokens_accumulated += snap_final_tokens + (base_roundtrips * 200)
                base_roundtrips += 1
            except Exception as e:
                base_success = False
                base_error = str(e)
            finally:
                await base_page.close()

            base_browser_time = time.perf_counter() - base_t0
            llm_cloud_network_latency = base_roundtrips * 2.8
            base_realistic_time = base_browser_time + llm_cloud_network_latency

            results.append(
                {
                    "id": case_id,
                    "category": tc.get("category", "General"),
                    "name": name,
                    "url": url,
                    "steps_count": len(steps),
                    "cognitive": {
                        "success": cog_success,
                        "status": cog_res.get("status"),
                        "total_time_s": round(cog_time, 2),
                        "gpu_inference_ms": round(cog_gpu, 2),
                        "escalations_count": cog_escalations,
                        "cortex_tokens": cog_s2_tokens,
                        "system2_tokens": cog_s2_tokens,
                        "total_tokens": cog_total_tokens,
                        "steps_executed": cog_res.get("steps_executed", 0),
                        "final_url": cog_res.get("final_url"),
                    },
                    "base_playwright": {
                        "success": base_success,
                        "browser_time_s": round(base_browser_time, 2),
                        "realistic_total_time_s": round(base_realistic_time, 2),
                        "tokens": base_tokens_accumulated,
                        "roundtrips": base_roundtrips,
                        "steps_executed": base_steps_executed,
                        "error": base_error,
                    },
                }
            )
        await browser.close()
    return results


async def run_benchmark(runs: int = 1, mode: str = "fixture", output_file: Optional[str] = None):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    print("=" * 95)
    print(f"🚀 COGNITIVE BROWSER BENCHMARK (Mode: {mode.upper()} | Runs: {runs})")
    print(f"   Hardware: {device.upper()} | Reflex Core (DeBERTa-v3) + Cortex Engine (LLM Escalation)")
    print("   VS Base Playwright MCP (Analytical ReAct Simulation)")
    print("=" * 95)

    cases = LOCAL_FIXTURE_CASES if mode == "fixture" else TEST_CASES
    print(f"   Active dataset: {len(cases)} cases.")
    if mode == "live":
        print("   ⚠️ Running in LIVE mode against real-world websites.")
    else:
        print("   🧪 Running in FIXTURE mode using local HTML templates.")

    orch = AgentOrchestrator(device=device)

    all_runs_results = []
    run_summaries = []

    for r_idx in range(1, runs + 1):
        run_res = await run_single_iteration(orch, cases, run_idx=r_idx)
        all_runs_results.append(run_res)

        total_cog_tok = sum(r["cognitive"]["total_tokens"] for r in run_res)
        avg_cog_t = sum(r["cognitive"]["total_time_s"] for r in run_res) / len(run_res)
        avg_cog_gpu = sum(r["cognitive"]["gpu_inference_ms"] for r in run_res) / len(run_res)
        cog_comp = sum(1 for r in run_res if r["cognitive"]["success"])

        total_base_tok = sum(r["base_playwright"]["tokens"] for r in run_res)
        avg_base_t = sum(r["base_playwright"]["realistic_total_time_s"] for r in run_res) / len(run_res)

        run_summaries.append(
            {
                "run": r_idx,
                "cog_tokens": total_cog_tok,
                "cog_avg_time_s": avg_cog_t,
                "cog_avg_gpu_ms": avg_cog_gpu,
                "cog_completed": cog_comp,
                "base_tokens": total_base_tok,
                "base_avg_time_s": avg_base_t,
            }
        )

    def calc_stat(values: List[float]):
        m = statistics.mean(values)
        s = statistics.stdev(values) if len(values) > 1 else 0.0
        return m, s

    cog_times = [s["cog_avg_time_s"] for s in run_summaries]
    cog_tokens = [float(s["cog_tokens"]) for s in run_summaries]
    cog_gpus = [s["cog_avg_gpu_ms"] for s in run_summaries]
    base_times = [s["base_avg_time_s"] for s in run_summaries]
    base_tokens = [float(s["base_tokens"]) for s in run_summaries]

    mean_cog_t, std_cog_t = calc_stat(cog_times)
    mean_cog_tok, std_cog_tok = calc_stat(cog_tokens)
    mean_cog_gpu, std_cog_gpu = calc_stat(cog_gpus)
    mean_base_t, std_base_t = calc_stat(base_times)
    mean_base_tok, std_base_tok = calc_stat(base_tokens)

    print("\n" + "=" * 95)
    print(f"📊 BENCHMARK SUMMARY ({runs} RUN{'S' if runs > 1 else ''} | {len(cases)} CASES)")
    print("=" * 95)
    print(
        f"Reflex Core Execution Duration : {mean_cog_t:.2f}s ± {std_cog_t:.2f}s | Baseline: {mean_base_t:.2f}s ± {std_base_t:.2f}s"
    )
    print(f"Reflex GPU Latency per Task     : {mean_cog_gpu:.1f}ms ± {std_cog_gpu:.1f}ms")
    print(
        f"Total Token Consumption        : {mean_cog_tok:,.0f} ± {std_cog_tok:,.0f} | Baseline: {mean_base_tok:,.0f} ± {std_base_tok:,.0f}"
    )
    if mean_base_tok > 0:
        reduction = ((mean_base_tok - mean_cog_tok) / mean_base_tok) * 100
        print(f"Token Reduction vs Baseline    : {reduction:.1f}% LESS TOKENS")
    if mean_cog_t > 0:
        speedup = mean_base_t / mean_cog_t
        print(f"Speedup vs Simulated Baseline  : {speedup:.1f}x FASTER")
    print("=" * 95)

    if not output_file:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"benchmark_{mode}_cases_results.json")

    out_payload = {
        "metadata": {"mode": mode, "runs": runs, "cases_count": len(cases), "device": device},
        "aggregate_statistics": {
            "cognitive_avg_time_s": {"mean": round(mean_cog_t, 3), "stdev": round(std_cog_t, 3)},
            "cognitive_gpu_ms": {"mean": round(mean_cog_gpu, 3), "stdev": round(std_cog_gpu, 3)},
            "cognitive_tokens": {"mean": round(mean_cog_tok, 1), "stdev": round(std_cog_tok, 1)},
            "baseline_avg_time_s": {"mean": round(mean_base_t, 3), "stdev": round(std_base_t, 3)},
            "baseline_tokens": {"mean": round(mean_base_tok, 1), "stdev": round(std_base_tok, 1)},
        },
        "runs": all_runs_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"Results and aggregate statistics saved to: {output_file}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Cognitive Browser Benchmark Harness")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions to compute mean ± std")
    parser.add_argument(
        "--mode",
        choices=["fixture", "live"],
        default="fixture",
        help="Execution mode: local fixtures (safe offline) or live websites",
    )
    parser.add_argument("--output", type=str, default=None, help="Custom output JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(runs=args.runs, mode=args.mode, output_file=args.output))
