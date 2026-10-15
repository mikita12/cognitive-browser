"""
OLX Deal Hunter - Autonomous Bargain Hunter (ARIA + DeBERTa System 1).
Searches offers on OLX, filters accessories and broken items, ranks by price and semantics,
and navigates directly into the best value deal to extract specifications.
"""

import asyncio
import re
import sys
import time
from typing import List, Dict, Any, Optional
from playwright.async_api import async_playwright
try:
    from cognitive_browser import AgentCore
except ImportError:
    from agent_core import AgentCore


class OlxDealHunter:
    """Deal hunter powered by Playwright and local DeBERTa router."""

    def __init__(self):
        self.agent = AgentCore()

    @staticmethod
    def parse_price(price_str: str) -> Optional[float]:
        """Cleans price string, e.g. '6 499 PLN negotiable' -> 6499.0."""
        clean = price_str.replace(" ", "").replace("\xa0", "").lower()
        m = re.search(r"(\d+(?:[.,]\d+)?)", clean)
        if m:
            val = m.group(1).replace(",", ".")
            try:
                return float(val)
            except ValueError:
                return None
        return None

    async def search_and_hunt(
        self,
        query: str,
        min_price: float = 1500.0,
        max_price: float = 15000.0
    ) -> Dict[str, Any]:
        print("=" * 80)
        print(f"🎯 HUNTING FOR DEALS: '{query}'")
        print(f"   Price range: {min_price} PLN - {max_price} PLN")
        print("=" * 80)

        t_start = time.perf_counter()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                locale="pl-PL",
                viewport={"width": 1366, "height": 900}
            )
            page = await context.new_page()

            # 1. Open homepage
            print("\n[1/5] Opening target homepage https://www.olx.pl/...")
            await page.goto("https://www.olx.pl/", wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)

            # 2. Fast Auto-Dismiss cookies
            print("[2/5] Checking and auto-dismissing cookie overlays...")
            await self.agent.auto_dismiss_overlays(page)

            # 3. Search query via ARIA
            print(f"[3/5] Typing '{query}' into search field...")
            search_box = page.get_by_role("combobox", name="Znajdź coś dla siebie")
            if await search_box.is_visible(timeout=3000):
                await search_box.fill(query)
            else:
                await page.locator("input[data-testid='search-input']").fill(query)

            search_btn = page.get_by_role("button", name="Szukaj")
            await search_btn.click()
            print("  -> Search form submitted. Waiting for listings...")
            await page.wait_for_timeout(3000)

            print(f"  -> Current URL: {page.url}")

            # 4. Extract and rank listings from ARIA tree
            print("\n[4/5] Analyzing offer listings from Accessibility Tree (ARIA)...")
            raw_snapshot = await page.aria_snapshot()

            lines = raw_snapshot.splitlines()
            offers = []

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("- link \"") and not any(skip in line.lower() for skip in ["strona główna", "obserwowane", "twoje konto", "dodaj ogłoszenie", "kategorie", "pomoc", "home"]):
                    m_title = re.search(r'^- link "([^"]+)"', line)
                    if m_title:
                        title = m_title.group(1).strip()
                        price_val = None
                        price_text = ""
                        for j in range(1, 5):
                            if i + j < len(lines):
                                next_line = lines[i + j].strip()
                                if "zł" in next_line:
                                    price_text = next_line.replace("- paragraph:", "").replace("- text:", "").strip()
                                    price_val = self.parse_price(price_text)
                                    break

                        # Filter out accessories and broken hardware
                        title_lower = title.lower()
                        is_accessory_or_broken = any(w in title_lower for w in [
                            "etui", "case", "pudełko", "box", "karton", "opakowanie",
                            "chłodzenie", "cooler", "blok", "waterblock", "wentylator", "backplate",
                            "uszkodzon", "na części", "skup", "zablokowan", "artefakty", "spalona"
                        ])

                        if price_val and min_price <= price_val <= max_price and not is_accessory_or_broken:
                            offers.append({
                                "title": title,
                                "price": price_val,
                                "price_raw": price_text
                            })
                i += 1

            # Deduplicate
            unique_offers = {}
            for o in offers:
                if o["title"] not in unique_offers:
                    unique_offers[o["title"]] = o
            offer_list = list(unique_offers.values())

            print(f"  -> Found {len(offer_list)} valid listings (excluding coolers, boxes, and broken units):")
            for idx, o in enumerate(offer_list[:10], 1):
                print(f"     {idx}. {o['price']} PLN | {o['title']}")

            if not offer_list:
                print("⚠️ No matching offers found for given criteria.")
                await browser.close()
                return {"status": "no_offers"}

            # Sort by lowest price
            offer_list.sort(key=lambda x: x["price"])
            best_deal = offer_list[0]

            print(f"\n🔥 TOP VALUE DEAL SELECTED BY SYSTEM 1:")
            print(f"   Title: {best_deal['title']}")
            print(f"   Price: {best_deal['price']} PLN ({best_deal['price_raw']})")

            # 5. Open deal and extract details
            print(f"\n[5/5] Navigating to deal details live in Playwright...")
            target_link = page.get_by_role("link", name=best_deal["title"]).first
            await target_link.click()
            await page.wait_for_timeout(3000)

            offer_url = page.url
            print(f"  -> Opened listing: {offer_url}")

            # Extract specs from ARIA
            offer_snapshot = await page.aria_snapshot()
            description_lines = []
            for l in offer_snapshot.splitlines():
                cl = l.strip()
                if any(k in cl.lower() for k in ["stan:", "gwarancja", "faktura", "kup teraz", "opis", "rtx", "condition", "warranty"]):
                    description_lines.append(cl)

            print("\n📋 Key specifications extracted:")
            for dl in description_lines[:8]:
                print(f"   • {dl}")

            total_elapsed = round(time.perf_counter() - t_start, 2)
            print(f"\n⏱️ Total execution latency from start to deal opening: {total_elapsed} s!")

            await browser.close()

            return {
                "status": "success",
                "deal": best_deal,
                "url": offer_url,
                "elapsed_s": total_elapsed,
                "total_candidates": len(offer_list)
            }


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "RTX 4090"
    min_p = float(sys.argv[2]) if len(sys.argv) > 2 else 2500.0
    max_p = float(sys.argv[3]) if len(sys.argv) > 3 else 12000.0
    hunter = OlxDealHunter()
    await hunter.search_and_hunt(query=query, min_price=min_p, max_price=max_p)


if __name__ == "__main__":
    asyncio.run(main())
