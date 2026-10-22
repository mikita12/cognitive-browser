#!/usr/bin/env python3
"""
Probe all URLs in tasks.jsonl using pure Playwright headless (no stealth, no solver).
Verifies:
- HTTP status < 400
- No WAF / Captcha blocks (Cloudflare, DataDome, Captcha, Access Denied, Zablokowano)
- aria_snapshot() length > 500 characters
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from playwright.async_api import async_playwright

BLOCKED_PATTERNS = [
    re.compile(r"captcha", re.I),
    re.compile(r"cloudflare", re.I),
    re.compile(r"datadome", re.I),
    re.compile(r"access\s*denied", re.I),
    re.compile(r"(?:dostęp|ip|użytkownik|adres)\s*zablokowan|zablokowano\s*dostęp", re.I),
    re.compile(r"attention\s*required", re.I),
    re.compile(r"security\s*check", re.I),
    re.compile(r"just\s*a\s*moment", re.I),
    re.compile(r"pardon\s*our\s*interruption", re.I),
    re.compile(r"bot\s*detection", re.I),
    re.compile(r"enable\s*javascript\s*and\s*cookies", re.I),
    re.compile(r"403\s*forbidden", re.I),
]


async def probe_task(page, task: dict) -> dict:
    url = task["url"]
    task_id = task["task_id"]
    res = {
        "task_id": task_id,
        "url": url,
        "status": "OK",
        "http_code": None,
        "title": "",
        "aria_len": 0,
        "reason": "",
    }

    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_timeout(1000)
        http_code = response.status if response else 0
        res["http_code"] = http_code

        if http_code >= 400:
            res["status"] = "BLOCKED"
            res["reason"] = f"HTTP {http_code}"
            return res

        title = await page.title()
        res["title"] = title

        try:
            snapshot = await page.aria_snapshot()
        except Exception as e:
            snapshot = ""
            res["reason"] = f"aria_snapshot_error: {e}"

        res["aria_len"] = len(snapshot)

        text_to_check = f"{title}\n{snapshot}\n{page.url}"
        for pat in BLOCKED_PATTERNS:
            if pat.search(text_to_check):
                res["status"] = "BLOCKED"
                res["reason"] = f"Matched block pattern: '{pat.pattern}'"
                return res

        if len(snapshot) <= 500:
            res["status"] = "BLOCKED"
            res["reason"] = f"Aria snapshot too short ({len(snapshot)} <= 500 chars)"
            return res

    except Exception as e:
        res["status"] = "BLOCKED"
        res["reason"] = f"Navigation error: {type(e).__name__} ({str(e)[:80]})"

    return res


async def main():
    target_arg = sys.argv[1] if len(sys.argv) > 1 else "results/real_world/new_benchmark_80/tasks.jsonl"
    tasks_path = Path(target_arg)
    if not tasks_path.exists():
        print(f"Error: {tasks_path} not found.")
        sys.exit(1)

    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )

        for idx, task in enumerate(tasks, 1):
            page = await context.new_page()
            try:
                r = await probe_task(page, task)
            finally:
                await page.close()
            results.append(r)
            status_flag = "[OK]     " if r["status"] == "OK" else "[BLOCKED]"
            print(f"[{idx:02d}/{len(tasks)}] {status_flag} {task['task_id']} ({task['url']}): {r['status']} | http={r['http_code']} | aria_len={r['aria_len']} | {r['reason']}")

        await browser.close()

    blocked = [r for r in results if r["status"] == "BLOCKED"]
    ok = [r for r in results if r["status"] == "OK"]
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(ok)} OK, {len(blocked)} BLOCKED / FAILED")
    print("=" * 60)
    if blocked:
        print("\nBlocked / Failed tasks:")
        for b in blocked:
            print(f"- {b['task_id']}: {b['url']} -> {b['reason']}")

    report_path = Path("results/real_world/new_benchmark_80/probe_report.json")
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
