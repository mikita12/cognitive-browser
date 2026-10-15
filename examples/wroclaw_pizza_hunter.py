#!/usr/bin/env python3
"""
Wroclaw Pizza Delivery Hunter - powered by Agentic Decision Router (Reflex Core + ARIA Snapshot).
Rapid verification of local pizzerias with dedicated online delivery systems.
"""

import asyncio
import time
import json
from typing import List, Dict, Any
import os
from playwright.async_api import async_playwright
try:
    from cognitive_browser import AriaElementParser, DebertaDecisionRouter
except ImportError:
    from agent_core import AriaElementParser
    from deberta_router import DebertaDecisionRouter


# Candidate database: pizzerias in Wroclaw with web ordering domains
CANDIDATE_PIZZERIAS = [
    {
        "name": "Express Pizza (Detroit Deep Dish)",
        "url": "https://expresspizza.pl",
        "style": "Detroit Deep Dish (Fluffy American Hi Pie)",
        "districts": "Citywide Wroclaw"
    },
    {
        "name": "Mania Smaku",
        "url": "https://mania.pizza",
        "style": "Artisanal & craft specialty pizza",
        "districts": "Citywide Wroclaw (Krzyki, Grabiszyn, Huby, Śródmieście)"
    },
    {
        "name": "Pizzeria Muzyczna",
        "url": "https://pizzeriamuzyczna.pl",
        "style": "Classic local style / rich toppings",
        "districts": "Citywide Wroclaw"
    },
    {
        "name": "Pizzu",
        "url": "https://pizzu.pl",
        "style": "Traditional Italian / Neapolitan",
        "districts": "Nowy Dwór, Kuźniki, Muchobór, Żerniki, Gądów"
    },
    {
        "name": "Pizza Mobile",
        "url": "https://pizzamobile.pl",
        "style": "Classic / American",
        "districts": "Gądów, Nowy Dwór, Kuźniki, Kozanów, Popowice, Pilczyce"
    },
    {
        "name": "Pizza Station",
        "url": "https://pizzastation.pl",
        "style": "American pan and thin crust",
        "districts": "Wroclaw West / North / Downtown"
    },
    {
        "name": "Pizza Time",
        "url": "https://pizza-time.pl",
        "style": "Italian thin crust",
        "districts": "Krzyki, Borek, Oporów, Gaj, Downtown"
    },
    {
        "name": "Pizzeria Migliori",
        "url": "https://pizzamigliori.pl",
        "style": "Stone-baked Italian",
        "districts": "Kozanów, Pilczyce, Popowice, Maślice"
    },
    {
        "name": "Appetito",
        "url": "https://appe.pl",
        "style": "Wood-fired traditional Italian",
        "districts": "Grabiszyn, Gajowice, Krzyki, City Center"
    },
    {
        "name": "La Finezze",
        "url": "https://lafinezze.pl",
        "style": "Traditional Italian",
        "districts": "Popowice, Szczepin, Nadodrze, Nowy Dwór, Stabłowice"
    },
    {
        "name": "Da Grasso Wroclaw",
        "url": "https://www.dagrasso.pl",
        "style": "Polish classic (famous house sauces)",
        "districts": "Multiple locations across Wroclaw"
    },
    {
        "name": "Pizza Hut Wroclaw",
        "url": "https://pizzahut.pl",
        "style": "American Pan / San Francisco",
        "districts": "All Wroclaw districts"
    },
    {
        "name": "Papa John's Wroclaw",
        "url": "https://papajohns.pl",
        "style": "American style with signature garlic dip",
        "districts": "City Center, Krzyki, Downtown"
    },
    {
        "name": "Pizzeria Bube",
        "url": "https://pizzeriabube.pl",
        "style": "Authentic Neapolitan",
        "districts": "Krzyki, Nadodrze, Downtown"
    },
    {
        "name": "Pizzeria 105",
        "url": "https://105.pl",
        "style": "Classic Polish crust",
        "districts": "Psie Pole, Fabryczna, Krzyki"
    },
    {
        "name": "Pizzeria Bravo",
        "url": "https://bravopizza.pl",
        "style": "Classic budget student pizza",
        "districts": "Downtown, Plac Grunwaldzki, Ołbin"
    },
    {
        "name": "Pizzeria Biesiadowo",
        "url": "https://biesiadowo.pl",
        "style": "Large Polish banquet pizza",
        "districts": "Fabryczna, Nowy Dwór, Psie Pole"
    }
]


async def verify_pizzerias():
    print("=" * 80)
    print("🚀 START: WROCLAW PIZZA DELIVERY HUNTER (Reflex Core + ARIA Snapshot)")
    print("=" * 80)

    router = DebertaDecisionRouter(device="cuda")
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )

        for item in CANDIDATE_PIZZERIAS:
            name = item["name"]
            url = item["url"]
            print(f"\n🔍 Checking: {name} ({url})...")
            start_t = time.perf_counter()

            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=12000)
                await page.wait_for_timeout(1000)

                # Extract condensed ARIA tree
                snapshot = await page.aria_snapshot()
                candidates = AriaElementParser.parse_snapshot(snapshot, max_candidates=40)

                # Query DeBERTa router
                decision = router.evaluate_candidates(
                    intent="order food online for home delivery or add to cart",
                    candidates=candidates
                )

                elapsed_ms = (time.perf_counter() - start_t) * 1000
                conf = decision["confidence"]
                best_cand = decision["ranked_candidates"][0] if decision["ranked_candidates"] else None
                action_text = best_cand["text"] if best_cand else "None"

                is_valid_order_site = conf > 0.45 and best_cand is not None

                print(f"  ⏱ Analysis latency: {elapsed_ms:.1f} ms | NLI Confidence: {conf:.2%}")
                print(f"  🎯 Action trigger found: [{best_cand['role'] if best_cand else ''}] \"{action_text}\"")

                results.append({
                    "name": name,
                    "url": url,
                    "style": item["style"],
                    "districts": item["districts"],
                    "verified_online_order": is_valid_order_site,
                    "confidence": conf,
                    "order_trigger": action_text,
                    "latency_ms": elapsed_ms
                })

            except Exception as e:
                print(f"  ❌ Error loading {url}: {e}")
                results.append({
                    "name": name,
                    "url": url,
                    "style": item["style"],
                    "districts": item["districts"],
                    "verified_online_order": False,
                    "error": str(e)
                })
            finally:
                await page.close()

        await browser.close()

    print("\n" + "=" * 80)
    print("📊 VERIFIED PIZZERIAS SUMMARY")
    print("=" * 80)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wroclaw_pizzerias_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for r in results:
        status = "✅ ONLINE ORDERING" if r.get("verified_online_order") else "⚠️ PHONE ORDER ONLY"
        print(f"{status} | {r['name']} ({r['url']})")
        print(f"   Style: {r['style']}")
        print(f"   Delivery area: {r['districts']}")
        if "order_trigger" in r:
            print(f"   Order button: \"{r['order_trigger']}\" (Confidence: {r['confidence']:.1%})\n")


if __name__ == "__main__":
    asyncio.run(verify_pizzerias())
