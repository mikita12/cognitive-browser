#!/usr/bin/env python3
"""
Wrocław Pizza Delivery Hunter - powered by Agentic Decision Router (System 1 + ARIA Snapshot).
Szybka weryfikacja pizzerii we Wrocławiu posiadających własną stronę WWW oraz bezpośredni system zamówień online z dostawą.
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


# Baza kandydatów: pizzerie we Wrocławiu posiadające własne domeny WWW
CANDIDATE_PIZZERIAS = [
    {
        "name": "Express Pizza (Detroit Deep Dish)",
        "url": "https://expresspizza.pl",
        "style": "Detroit Deep Dish (amerykańska puszysta w stylu Hi Pie)",
        "districts": "Cały Wrocław (lokale: Sokolnicza, Krzycka, Barskich/Żerniki, Śliczna, Nadodrze, Psie Pole)"
    },
    {
        "name": "Mania Smaku",
        "url": "https://mania.pizza",
        "style": "Rzemieślnicza / klasyczna i autorskie kompozycje",
        "districts": "Cały Wrocław (Krzyki, Grabiszyn, Huby, Śródmieście)"
    },
    {
        "name": "Pizzeria Muzyczna",
        "url": "https://pizzeriamuzyczna.pl",
        "style": "Klasyczna wrocławska / bogate składniki",
        "districts": "Cały Wrocław (Jedności Narodowej, Oleska, Bulwar Dedala, Krzyki)"
    },
    {
        "name": "Pizzu",
        "url": "https://pizzu.pl",
        "style": "Tradycyjna włoska / neapolitańska",
        "districts": "Nowy Dwór, Kuźniki, Muchobór, Żerniki, Gądów"
    },
    {
        "name": "Pizza Mobile",
        "url": "https://pizzamobile.pl",
        "style": "Klasyczna / amerykańska",
        "districts": "Gądów, Nowy Dwór, Kuźniki, Kozanów, Popowice, Pilczyce"
    },
    {
        "name": "Pizza Station",
        "url": "https://pizzastation.pl",
        "style": "Amerykańska na grubym i cienkim cieście",
        "districts": "Wrocław Zachód / Północ / Śródmieście"
    },
    {
        "name": "Pizza Time",
        "url": "https://pizza-time.pl",
        "style": "Włoska cienkie ciasto",
        "districts": "Krzyki, Borek, Oporów, Gaj, Śródmieście"
    },
    {
        "name": "Pizzeria Migliori",
        "url": "https://pizzamigliori.pl",
        "style": "Klasyczna / włoska z pieca",
        "districts": "Kozanów, Pilczyce, Popowice, Maślice"
    },
    {
        "name": "Appetito",
        "url": "https://appe.pl",
        "style": "Włoska tradycyjna z pieca",
        "districts": "Grabiszyn, Gajowice, Krzyki, Centrum"
    },
    {
        "name": "La Finezze",
        "url": "https://lafinezze.pl",
        "style": "Tradycyjna z bogatym menu",
        "districts": "Popowice, Szczepin, Nadodrze, Nowy Dwór, Stabłowice"
    },
    {
        "name": "Da Grasso Wrocław",
        "url": "https://www.dagrasso.pl",
        "style": "Polska klasyczna (z kultowymi sosami)",
        "districts": "Wielolokalowa sieć we Wrocławiu (Krzyki, Śródmieście, Fabryczna)"
    },
    {
        "name": "Pizza Hut Wrocław",
        "url": "https://pizzahut.pl",
        "style": "Amerykańska Pan / San Francisco",
        "districts": "Wszystkie dzielnice Wrocławia"
    },
    {
        "name": "Papa John's Wrocław",
        "url": "https://papajohns.pl",
        "style": "Amerykańska z sosem czosnkowym",
        "districts": "Centrum, Krzyki, Śródmieście"
    },
    {
        "name": "Pizzeria Bube",
        "url": "https://pizzeriabube.pl",
        "style": "Autentyczna włoska neapolitańska",
        "districts": "Krzyki, Nadodrze, Śródmieście"
    },
    {
        "name": "Pizzeria 105 (Stopiątka)",
        "url": "https://105.pl",
        "style": "Klasyczna polska / szeroki wybór",
        "districts": "Psie Pole, Fabryczna, Krzyki"
    },
    {
        "name": "Pizzeria Bravo",
        "url": "https://bravopizza.pl",
        "style": "Klasyczna / studencka",
        "districts": "Śródmieście, Plac Grunwaldzki, Ołbin"
    },
    {
        "name": "Pizzeria Biesiadowo",
        "url": "https://biesiadowo.pl",
        "style": "Polska biesiadna (duże rozmiary)",
        "districts": "Fabryczna, Nowy Dwór, Psie Pole"
    }
]


async def verify_pizzerias():
    print("=" * 80)
    print("🚀 START: WROCŁAW PIZZA DELIVERY HUNTER (System 1 + ARIA Snapshot)")
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
            print(f"\n🔍 Sprawdzam: {name} ({url})...")
            start_t = time.perf_counter()

            page = await context.new_page()
            try:
                # Wchodzimy na stronę z timeoutem 12s
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=12000)
                await page.wait_for_timeout(1000)

                # Pobieramy czyste drzewo ARIA
                snapshot = await page.aria_snapshot()
                candidates = AriaElementParser.parse_snapshot(snapshot, max_candidates=40)

                # Pytamy DeBERTę: czy strona posiada przycisk / moduł zamówienia online z dostawą?
                decision = router.evaluate_candidates(
                    intent="zamów online jedzenie z dostawą do domu lub do koszyka",
                    candidates=candidates
                )

                elapsed_ms = (time.perf_counter() - start_t) * 1000
                conf = decision["confidence"]
                best_cand = decision["ranked_candidates"][0] if decision["ranked_candidates"] else None
                action_text = best_cand["text"] if best_cand else "Brak"

                is_valid_order_site = conf > 0.45 and best_cand is not None

                print(f"  ⏱ Czas analizy: {elapsed_ms:.1f} ms | Pewność NLI: {conf:.2%}")
                print(f"  🎯 Znaleziony element akcji: [{best_cand['role'] if best_cand else ''}] \"{action_text}\"")

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
                print(f"  ❌ Błąd ładowania {url}: {e}")
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
    print("📊 PODSUMOWANIE ZWERYFIKOWANYCH PIZZERII WE WROCŁAWIU")
    print("=" * 80)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wroclaw_pizzerias_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for r in results:
        status = "✅ ZAMÓWIENIA ONLINE" if r.get("verified_online_order") else "⚠️ SPRAWDŹ TELEFONICZNIE"
        print(f"{status} | {r['name']} ({r['url']})")
        print(f"   Styl: {r['style']}")
        print(f"   Zasięg dowozu: {r['districts']}")
        if "order_trigger" in r:
            print(f"   Przycisk zamówienia: \"{r['order_trigger']}\" (Pewność: {r['confidence']:.1%})\n")


if __name__ == "__main__":
    asyncio.run(verify_pizzerias())
