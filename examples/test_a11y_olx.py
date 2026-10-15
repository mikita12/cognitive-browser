"""
Test of Playwright Aria Snapshot format on live portal (OLX.pl).
Leverages official page.aria_snapshot() from Playwright.
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
    Parses lines from aria_snapshot YAML-like format:
    e.g. '- button "Accept all cookies"'
         '- textbox "What are you looking for?"'
         '- link "Electronics"'
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

            if name or text:
                elements.append({
                    "role": role,
                    "name": name if name else text,
                    "text": text,
                    "raw_line": line_clean
                })
        else:
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
    print("PLAYWRIGHT ARIA SNAPSHOT + DEBERTA-V3 (LIVE TEST)")
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

        print("\n[1/3] Navigating to https://www.olx.pl/...")
        try:
            await page.goto("https://www.olx.pl/", wait_until="domcontentloaded", timeout=25000)
            await page.wait_for_timeout(2500)
        except Exception as e:
            print(f"Loading error: {e}")
            await browser.close()
            return

        # Retrieve native ARIA snapshot
        print("\n[2/3] Retrieving official page.aria_snapshot() from Playwright...")
        snapshot_raw = await page.aria_snapshot()

        print("\nFirst 30 lines of raw Aria Snapshot:")
        print("-" * 80)
        lines = snapshot_raw.splitlines()
        for l in lines[:30]:
            print(l)
        print(f"... [Total: {len(lines)} lines of semantic ARIA]")
        print("-" * 80)

        parsed_elements = parse_aria_snapshot_lines(snapshot_raw, max_items=40)
        print(f"Extracted {len(parsed_elements)} semantic ARIA candidates.")

        # Test DeBERTa on candidates
        print("\n[3/3] Evaluating DeBERTa-v3 on GPU for target intents...")
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

        # Test 1: Accept cookies
        intent_1 = "accept all cookies and terms to close the consent banner"
        res_1 = router.evaluate_candidates(intent_1, candidates)
        print(f"\nIntent: '{intent_1}'")
        print(f"  -> DeBERTa latency: {res_1['latency_ms']} ms")
        print("  -> Top 3 matches:")
        for rank, c in enumerate(res_1['ranked_candidates'][:3], 1):
            print(f"     {rank}. [{c['confidence_score']*100:5.1f}%] [{c['role']}] \"{c['text']}\"")

        # Test 2: Search input
        intent_2 = "enter query in search box to find items"
        res_2 = router.evaluate_candidates(intent_2, candidates)
        print(f"\nIntent: '{intent_2}'")
        print(f"  -> DeBERTa latency: {res_2['latency_ms']} ms")
        print("  -> Top 3 matches:")
        for rank, c in enumerate(res_2['ranked_candidates'][:3], 1):
            print(f"     {rank}. [{c['confidence_score']*100:5.1f}%] [{c['role']}] \"{c['text']}\"")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_olx_aria_test())
