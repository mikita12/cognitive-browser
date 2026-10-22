#!/usr/bin/env python3
"""
BM25 Calibration on dev_tasks.jsonl (20 tasks).
1. Runs Arm A (ReflexCore reranker) across all 20 dev tasks and records its escalation rate.
2. Sweeps BM25 thresholds to find the threshold where BM25 escalation rate matches Arm A (+/- 2 percentage points).
3. Saves calibration results to results/real_world/new_benchmark_80/bm25_calibration.json.
"""

import asyncio
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("results/real_world/run2"))

from playwright.async_api import async_playwright
from cognitive_browser.parser import AriaElementParser
from cognitive_browser.reflex import ReflexCore
from run_bm25_offline import BM25Scorer


async def dismiss_cookie_overlays(page) -> None:
    for sel in [
        "button:has-text('Zaakceptuj')",
        "button:has-text('Zgadzam się')",
        "button:has-text('Akceptuję')",
        "button:has-text('Accept all')",
        "button:has-text('Accept')",
        "button:has-text('Alle akzeptieren')",
        "button:has-text('Zezwól na wszystkie')",
        "#onetrust-accept-btn-handler",
    ]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=500):
                await btn.click(timeout=1000)
                await page.wait_for_timeout(300)
                break
        except Exception:
            pass


async def main():
    dev_path = Path("results/real_world/new_benchmark_80/dev_tasks.jsonl")
    if not dev_path.exists():
        print(f"Error: {dev_path} not found.")
        sys.exit(1)

    tasks = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(tasks)} tasks from {dev_path}")

    # Use CPU/CUDA depending on availability
    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") != "" else "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"

    print(f"Initializing ReflexCore (Arm A) on device='{device}'...")
    reflex = ReflexCore(device=device)

    dev_data = []
    arm_a_escalations = 0

    print("\n--- Running Arm A (Reranker) on 20 DEV tasks ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )

        for idx, task in enumerate(tasks, 1):
            url = task["url"]
            goal = task["goal"]
            task_id = task["task_id"]
            page = await context.new_page()

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await dismiss_cookie_overlays(page)
                await page.wait_for_timeout(800)
                snap = await page.aria_snapshot()
                cands = AriaElementParser.parse_snapshot(snap, max_candidates=300)

                dec_a = reflex.evaluate_candidates(intent=goal, candidates=cands)
                esc_a = dec_a.get("should_escalate", False)
                conf_a = dec_a.get("confidence", 0.0)
                target_a = dec_a.get("target_ref", "NONE")

                if esc_a:
                    arm_a_escalations += 1

                dev_data.append({
                    "task_id": task_id,
                    "url": url,
                    "goal": goal,
                    "candidates": cands,
                    "arm_a_escalate": esc_a,
                    "arm_a_confidence": conf_a,
                    "arm_a_target": target_a,
                })

                esc_str = "ESCALATED" if esc_a else "ROUTED"
                print(f"[{idx:02d}/20] {task_id} ({url}): {len(cands)} cands | conf={conf_a:.3f} | {esc_str}")

            except Exception as e:
                print(f"[{idx:02d}/20] {task_id} ({url}): Error {e}")
            finally:
                await page.close()

        await browser.close()

    total_tasks = len(dev_data)
    arm_a_esc_rate = arm_a_escalations / max(1, total_tasks)

    print("\n" + "=" * 60)
    print(f"Arm A (Reranker) DEV Escalation Rate: {arm_a_esc_rate * 100:.2f}% ({arm_a_escalations}/{total_tasks})")
    print("=" * 60)

    # BM25 threshold sweep
    bm25 = BM25Scorer()
    thresholds = [round(0.01 * i, 2) for i in range(1, 81)]  # 0.01 to 0.80

    best_thresh = None
    min_diff = 999.0
    best_bm_escalations = 0
    best_bm_esc_rate = 0.0

    print("\n--- Sweeping BM25 Thresholds (Arm C) ---")
    sweep_results = []
    for th in thresholds:
        esc_c = 0
        for item in dev_data:
            c_dicts = [dict(c) for c in item["candidates"]]
            dec_c = bm25.score_candidates(item["goal"], c_dicts, threshold=th)
            if dec_c.get("should_escalate", False):
                esc_c += 1

        bm_rate = esc_c / max(1, total_tasks)
        diff = abs(bm_rate - arm_a_esc_rate)
        sweep_results.append((th, bm_rate, esc_c, diff))

        if diff < min_diff:
            min_diff = diff
            best_thresh = th
            best_bm_escalations = esc_c
            best_bm_esc_rate = bm_rate

    # Print representative samples
    for th, bm_rate, esc_c, diff in sweep_results:
        if th in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] or th == best_thresh:
            marker = " <-- OPTIMAL MATCH" if th == best_thresh else ""
            print(f"  Threshold {th:.2f}: BM25 Escalation = {bm_rate * 100:.2f}% ({esc_c}/{total_tasks}) [diff={diff * 100:.2f}pp]{marker}")

    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY:")
    print(f"  Arm A (Reranker) Escalation Rate : {arm_a_esc_rate * 100:.2f}% ({arm_a_escalations}/{total_tasks})")
    print(f"  Arm C (BM25) Escalation Rate     : {best_bm_esc_rate * 100:.2f}% ({best_bm_escalations}/{total_tasks})")
    print(f"  Difference                       : {min_diff * 100:.2f} percentage points")
    print(f"  Calibrated BM25 Threshold        : {best_thresh:.2f}")
    print("=" * 60)

    calib_record = {
        "calibrated_bm25_threshold": best_thresh,
        "calibration_split": "dev",
        "dev_tasks_total": total_tasks,
        "arm_a_escalations": arm_a_escalations,
        "arm_a_escalation_rate": round(arm_a_esc_rate, 4),
        "arm_c_escalations": best_bm_escalations,
        "arm_c_escalation_rate": round(best_bm_esc_rate, 4),
        "diff_percentage_points": round(min_diff * 100, 2),
        "threshold_sweep_samples": [
            {"threshold": th, "escalation_rate": round(r, 4), "escalations": e}
            for th, r, e, _ in sweep_results if th in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, best_thresh]
        ]
    }

    out_json = Path("results/real_world/new_benchmark_80/bm25_calibration.json")
    out_json.write_text(json.dumps(calib_record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCalibration data saved to: {out_json}")

    return best_thresh, arm_a_esc_rate, best_bm_esc_rate


if __name__ == "__main__":
    asyncio.run(main())
