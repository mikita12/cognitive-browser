"""
Test nowego formatu Playwright Aria Snapshot na OLX.pl.
Wykorzystuje oficjalne page.aria_snapshot() z Playwrighta.
"""

import asyncio
import re
from playwright.async_api import async_playwright
try:
    from cognitive_browser import DebertaDecisionRouter
except ImportError:
    from deberta_router import DebertaDecisionRouter


def parse_aria_snapshot_lines(snapshot_text: str, max_items: int = 50):
    """
    Parsuje linie z formatu aria_snapshot (YAML-like):
    np. '- button "Zaakceptuj wszystkie"'
        '- textbox "Znajdź coś dla siebie"'
        '- link "Elektronika"'
    """
    pattern = re.compile(r'^\s*-\s*([a-zA-Z0-9_-]+)\s*(?:"([^"]*)")?(?::\s*(.*))?$')
    elements = []
    
    for line in snapshot_text.splitlines():
        line_clean = line.strip()
        if not line_clean.startswith("-"):
            continue
            
        m = pattern.match(line_clean)
        if m:
            role = m.group(1)
            name = m.group(2) or ""
            text = m.group(3) or ""
            
            # Interesują nas elementy z etykietą ARIA
            if name or text:
                elements.append({
                    "role": role,
                    "name": name if name else text,
                    "text": text,
                    "raw_line": line_clean
                })
        else:
            # Fallback dla innych linii
            elements.append({
                "role": "element",
                "name": line_clean[2:],
                "text": "",
                "raw_line": line_clean
            })
            
        if len(elements) >= max_items:
            break
            
    return elements


async def run_olx_aria_test():
    print("=" * 80)
    print("PLAYWRIGHT ARIA SNAPSHOT + DEBERTA-V3 (TEST NA ŻYWO NA OLX.PL)")
    print("=" * 80)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            locale="pl-PL",
            viewport={"width": 1280, "height": 800}
        )
        page = await context.new_page()

        print("\n[1/3] Wchodzę na stronę https://www.olx.pl/...")
        try:
            await page.goto("https://www.olx.pl/", wait_until="domcontentloaded", timeout=25000)
            await page.wait_for_timeout(2500) # Czas na pojawienie się banera cookies
        except Exception as e:
            print(f"Błąd ładowania: {e}")
            await browser.close()
            return

        # -------------------------------------------------------------
        # POBIERAMY NATIVE ARIA SNAPSHOT
        # -------------------------------------------------------------
        print("\n[2/3] Pobieram oficjalny page.aria_snapshot() z Playwrighta...")
        snapshot_raw = await page.aria_snapshot()

        print("\nPierwsze 30 linii surowego Aria Snapshot:")
        print("-" * 80)
        lines = snapshot_raw.splitlines()
        for l in lines[:30]:
            print(l)
        print(f"... [Łącznie: {len(lines)} linii czystej semantyki ARIA]")
        print("-" * 80)

        # Parsujemy elementy
        parsed_elements = parse_aria_snapshot_lines(snapshot_raw, max_items=40)
        print(f"Wyciągnięto {len(parsed_elements)} kluczowych obiektów ARIA.")

        # -------------------------------------------------------------
        # TESTUJEMY DEBERTĘ NA ELEMENTACH ARIA
        # -------------------------------------------------------------
        print("\n[3/3] Wnioskowanie DeBERTa-v3 na GPU dla wybranych intencji...")
        router = DebertaDecisionRouter(model_name="cross-encoder/nli-deberta-v3-small")

        candidates = []
        for i, el in enumerate(parsed_elements):
            candidates.append({
                "ref": f"aria-{i}",
                "tag": el["role"],
                "role": el["role"],
                "text": el["name"],
                "name": el["name"],
                "raw": el["raw_line"]
            })

        # Test 1: Zaakceptuj cookies
        intent_1 = "accept all cookies and terms to close the consent banner"
        res_1 = router.evaluate_candidates(intent_1, candidates)
        print(f"\nIntencja: '{intent_1}'")
        print(f"  -> Czas reakcji DeBERTy na RTX 4070 Ti: {res_1['latency_ms']} ms")
        print(f"  -> Top 3 dopasowania:")
        for rank, c in enumerate(res_1['ranked_candidates'][:3], 1):
            print(f"     {rank}. [{c['confidence_score']*100:5.1f}%] [{c['role']}] \"{c['text']}\"")

        # Test 2: Pole wyszukiwarki ogłoszeń
        intent_2 = "enter query in search box to find items"
        res_2 = router.evaluate_candidates(intent_2, candidates)
        print(f"\nIntencja: '{intent_2}'")
        print(f"  -> Czas reakcji DeBERTy na RTX 4070 Ti: {res_2['latency_ms']} ms")
        print(f"  -> Top 3 dopasowania:")
        for rank, c in enumerate(res_2['ranked_candidates'][:3], 1):
            print(f"     {rank}. [{c['confidence_score']*100:5.1f}%] [{c['role']}] \"{c['text']}\"")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_olx_aria_test())
