"""
Run 6 Local Agent Benchmark Runner (100% Offline / Self-Hosted).
Uses Ornith 1.5 (35B MoE) via SSH Tunnel as the controlling LLM agent.
Evaluates 80 TEST tasks across 3 repetitions (240 episodes per arm, 720 total):
- Arm A: Cognitive Browser (ReflexCore neural reranker)
- Arm B: Standard Playwright MCP (browser_navigate, browser_click, browser_type)
- Arm C: Cognitive Browser (BM25ElementScorer, calibrated threshold = 0.02)

Settings:
- Model: ornith-1.5:35b via http://127.0.0.1:11440/v1
- Temperature: 0.1, max_tokens: 1024, max_steps: 12
- Concurrency: Semaphore(4) with DomainRateLimiter (>= 1.0s per domain)
- Zero reliance on external commercial APIs (OpenAI)
"""

import asyncio
import json
import math
import os
import random
import re
import socket
import sys
import time

socket.setdefaulttimeout(75.0)
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urlparse

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("results/real_world/run2"))

from cognitive_browser.browser import BrowserEngine
from cognitive_browser.orchestrator import BrowserOrchestrator
from cognitive_browser.reflex import ReflexCore
from cognitive_browser.parser import AriaElementParser
from run_bm25_offline import BM25Scorer

TUNNEL_URL = os.getenv("LLM_URL", "http://127.0.0.1:11440/v1/chat/completions")
MODEL_NAME = os.getenv("LLM_MODEL", "ornith-1.5:35b")


class DomainRateLimiter:
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last_access: Dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    async def acquire(self, url: str) -> None:
        try:
            domain = urlparse(url).netloc or url
        except Exception:
            domain = url
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_access[domain]
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_access[domain] = time.time()


class BM25ReflexCore(ReflexCore):
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.bm25_scorer = BM25Scorer()
        self.model = None

    def evaluate_candidates(
        self,
        intent: str,
        candidates: Sequence[Union[Any, Dict[str, Any]]],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        th = threshold if threshold is not None else self.threshold
        cands_dict = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in candidates]
        res = self.bm25_scorer.score_candidates(intent, cands_dict, threshold=th)
        res["selection_source"] = "bm25"
        return res


def call_agent_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    req_body = {
        "model": MODEL_NAME,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    t0 = time.perf_counter()
    req_data = json.dumps(req_body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(5):
        try:
            req = urllib.request.Request(TUNNEL_URL, data=req_data, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                lat_ms = round((time.perf_counter() - t0) * 1000, 2)
                choice = raw["choices"][0]
                usage = raw.get("usage", {})
                return {
                    "message": choice["message"],
                    "finish_reason": choice.get("finish_reason"),
                    "latency_ms": lat_ms,
                    "usage": usage,
                }
        except Exception as e:
            if attempt < 4:
                time.sleep(2.0 + attempt * 2.0)
                continue
            raise


COGNITIVE_BROWSER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cognitive_browse",
            "description": "Autonomously navigates to URL and solves multi-step workflow on GPU with 0 LLM tokens. Returns final page snippet and reached URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Target web page URL"},
                    "goal": {"type": "string", "description": "Natural language goal to accomplish autonomously"}
                },
                "required": ["url", "goal"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "smart_action",
            "description": "Executes a targeted smart action (click, fill) on the currently active page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "Action instruction"},
                    "target_name": {"type": "string", "description": "Name or text of target element"},
                    "fill_value": {"type": "string", "description": "Value to fill if typing"}
                },
                "required": ["instruction"]
            }
        }
    }
]

PLAYWRIGHT_MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigates browser to a URL and returns current URL and ARIA accessibility snapshot of the page.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to navigate to"}},
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Clicks an interactive element on the page identified by visible label, text or CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {"selector_or_text": {"type": "string", "description": "Visible text label or CSS selector"}},
                "required": ["selector_or_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Types text into a form input or search box.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector_or_text": {"type": "string", "description": "Target input element label or placeholder"},
                    "text": {"type": "string", "description": "Text string to enter"}
                },
                "required": ["selector_or_text", "text"]
            }
        }
    }
]

SYSTEM_PROMPT = (
    "You are an expert autonomous web browsing assistant.\n"
    "Solve the web navigation task step by step using the provided tools.\n"
    "Inspect each tool result to verify if the goal is satisfied.\n"
    "When the goal is fully accomplished, reply with your final findings."
)


def detect_blocked_page(page_url: str, title: str, content: str) -> bool:
    low = f"{page_url} {title} {content}".lower()
    blocked_indicators = [
        "access denied", "403 forbidden", "cloudflare", "attention required",
        "cf-browser-verification", "ddos-guard", "datadome", "security challenge",
        "bot check", "verify you are human", "captcha", "imperva", "incapsula"
    ]
    return any(ind in low for ind in blocked_indicators)


async def run_episode(
    sample_id: str,
    task: Dict[str, Any],
    repetition: int,
    arm: str,
    browser: Browser,
    rate_limiter: DomainRateLimiter,
    orchestrator: Optional[BrowserOrchestrator] = None,
) -> Dict[str, Any]:
    task_id = task["task_id"]
    initial_url = task["url"]
    goal = task["goal"]

    t0_ep = time.perf_counter()
    context: BrowserContext = await browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    )
    page: Page = await context.new_page()

    steps_log = []
    total_in_tokens = 0
    total_out_tokens = 0
    status = "success"
    err_msg = ""
    screenshot_path = f"results/real_world/run6_local/screenshots/{sample_id}.png"
    last_stop_reason = ""
    partial_fallback_count = 0
    agent_interventions = 0

    try:
        await rate_limiter.acquire(initial_url)

        if arm in ("cognitive_browser", "cognitive_bm25"):
            tools = COGNITIVE_BROWSER_TOOLS
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {goal}\nStart URL: {initial_url}"}
            ]

            for step_i in range(12):
                llm_resp = await asyncio.to_thread(call_agent_llm, messages, tools)
                usage = llm_resp.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                total_in_tokens += in_tok
                total_out_tokens += out_tok

                msg = llm_resp["message"]
                messages.append(msg)
                tool_calls = msg.get("tool_calls", [])
                if not tool_calls:
                    break

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args_raw = tc["function"]["arguments"]
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    res = {}

                    try:
                        if fn_name == "cognitive_browse":
                            nav_url = fn_args.get("url", initial_url)
                            nav_goal = fn_args.get("goal", goal)
                            assert orchestrator is not None
                            res = await orchestrator.browse(page, url=nav_url, goal=nav_goal, browser_context="ephemeral")
                            last_stop_reason = res.get("stop_reason", "")
                            if res.get("status") in ("needs_fallback", "partial_needs_fallback"):
                                partial_fallback_count += 1

                            tool_output = {
                                "status": res["status"],
                                "final_url": res["final_url"],
                                "steps_executed": res.get("steps_executed", 0),
                                "stop_reason": res.get("stop_reason", ""),
                                "message": res.get("message", ""),
                                "top_candidates": res.get("top_candidates", []),
                                "page_snippet": res.get("page_content_snippet", "")[:1500]
                            }
                            if res.get("status") == "blocked":
                                status = "blocked"

                        elif fn_name == "smart_action":
                            agent_interventions += 1
                            instr = fn_args.get("instruction", "")
                            t_name = fn_args.get("target_name")
                            f_val = fn_args.get("fill_value")
                            assert orchestrator is not None
                            res = await orchestrator.smart_action(page, instruction=instr, target_name=t_name, fill_value=f_val)
                            tool_output = {
                                "status": res.get("status", "success"),
                                "final_url": str(page.url),
                                "message": res.get("message", ""),
                            }
                        else:
                            tool_output = {"status": "error", "message": f"Unknown tool {fn_name}"}
                    except Exception as tool_e:
                        tool_output = {"status": "error", "message": str(tool_e)}

                    res_dict = res if isinstance(res, dict) else {}
                    tool_status = tool_output.get("status", "success") if isinstance(tool_output, dict) else "success"
                    preview_str = json.dumps(tool_output, ensure_ascii=False) if isinstance(tool_output, dict) else str(tool_output)
                    sel_source = res_dict.get("selection_source", "llm_target_name" if fn_args.get("target_name") else ("bm25" if arm == "cognitive_bm25" else "reranker"))
                    top5 = res_dict.get("top_candidates", [])[:5]

                    steps_log.append({
                        "step": step_i + 1,
                        "tool": fn_name,
                        "args": fn_args,
                        "tool_status": tool_status,
                        "stop_reason": res_dict.get("stop_reason", ""),
                        "tool_result_preview": preview_str[:300],
                        "selection_source": sel_source,
                        "top_5_candidates": top5,
                        "llm_latency_ms": llm_resp["latency_ms"],
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{step_i}"),
                        "content": json.dumps(tool_output, ensure_ascii=False)
                    })

        elif arm == "playwright_mcp":
            tools = PLAYWRIGHT_MCP_TOOLS
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {goal}\nStart URL: {initial_url}"}
            ]

            for step_i in range(12):
                llm_resp = await asyncio.to_thread(call_agent_llm, messages, tools)
                usage = llm_resp.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                total_in_tokens += in_tok
                total_out_tokens += out_tok

                msg = llm_resp["message"]
                messages.append(msg)
                tool_calls = msg.get("tool_calls", [])
                if not tool_calls:
                    break

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args_raw = tc["function"]["arguments"]
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw

                    try:
                        if fn_name == "browser_navigate":
                            target_url = fn_args.get("url", initial_url)
                            await page.goto(target_url, wait_until="domcontentloaded", timeout=25000)
                            await page.wait_for_timeout(300)
                            snap = await page.aria_snapshot()
                            tool_output = f"Navigated to {page.url}\nTitle: {await page.title()}\n\nARIA SNAPSHOT:\n{snap[:8000] if snap else ''}"

                        elif fn_name == "browser_click":
                            target = fn_args.get("selector_or_text", "")
                            loc = page.locator(f"text='{target}'").first
                            if await loc.count() == 0:
                                loc = page.locator(target).first
                            await loc.click(timeout=3500)
                            await page.wait_for_timeout(400)
                            snap = await page.aria_snapshot()
                            tool_output = f"Clicked {target}. Current URL: {page.url}\n\nARIA SNAPSHOT:\n{snap[:8000] if snap else ''}"

                        elif fn_name == "browser_type":
                            target = fn_args.get("selector_or_text", "")
                            text = fn_args.get("text", "")
                            loc = page.locator(f"input[placeholder*='{target}']").first
                            if await loc.count() == 0:
                                loc = page.locator("input, textarea").first
                            await loc.fill(text, timeout=3000)
                            try:
                                await loc.press("Enter", timeout=1500)
                            except Exception:
                                pass
                            await page.wait_for_timeout(400)
                            snap = await page.aria_snapshot()
                            tool_output = f"Typed '{text}' into {target}. Current URL: {page.url}\n\nARIA SNAPSHOT:\n{snap[:8000] if snap else ''}"

                        else:
                            tool_output = "Unknown tool"
                    except Exception as tool_e:
                        tool_output = f"Tool action error: {tool_e}"

                    pw_status = "error" if any(err in tool_output.lower() for err in ("error", "exception", "timeout", "failed")) else "success"
                    steps_log.append({
                        "step": step_i + 1,
                        "tool": fn_name,
                        "args": fn_args,
                        "tool_status": pw_status,
                        "stop_reason": "",
                        "tool_result_preview": str(tool_output)[:300],
                        "selection_source": "llm_target_name",
                        "top_5_candidates": [],
                        "llm_latency_ms": llm_resp["latency_ms"],
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{step_i}"),
                        "content": tool_output
                    })

        final_url = page.url
        final_title = await page.title()

        try:
            await page.screenshot(path=screenshot_path, full_page=False, timeout=5000)
        except Exception:
            screenshot_path = ""

        try:
            page_snippet = await page.evaluate("() => document.body ? document.body.innerText : ''")
            page_snippet = page_snippet[:3500]
        except Exception:
            page_snippet = ""

        if status != "blocked" and detect_blocked_page(final_url, final_title, page_snippet):
            status = "blocked"
            err_msg = "Site blocked bot (Cloudflare / 403 / Captcha / Access Denied)"

    except Exception as e:
        status = "infra_error"
        err_msg = str(e)
        final_url = page.url if page else "about:blank"
        final_title = ""
        page_snippet = ""
        try:
            if page:
                await page.screenshot(path=screenshot_path, full_page=False, timeout=3000)
        except Exception:
            screenshot_path = ""
    finally:
        await context.close()

    e2e_lat_ms = round((time.perf_counter() - t0_ep) * 1000, 2)

    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "language": task.get("language", "pl"),
        "min_steps": task.get("min_steps", 1),
        "repetition": repetition,
        "arm": arm,
        "initial_url": initial_url,
        "final_url": final_url,
        "final_title": final_title,
        "status": status,
        "error_message": err_msg,
        "screenshot_path": screenshot_path,
        "steps_count": len(steps_log),
        "steps_log": steps_log,
        "page_snippet": page_snippet,
        "e2e_latency_ms": e2e_lat_ms,
        "agent_input_tokens": total_in_tokens,
        "agent_output_tokens": total_out_tokens,
        "total_agent_tokens": total_in_tokens + total_out_tokens,
        "stop_reason": last_stop_reason,
        "partial_fallback_count": partial_fallback_count,
        "agent_interventions": agent_interventions,
    }


async def main():
    tasks_path = Path("results/real_world/new_benchmark_80/tasks.jsonl")
    if not tasks_path.exists():
        print(f"Error: {tasks_path} not found.")
        sys.exit(1)

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    assert len(tasks) == 80, f"Expected exactly 80 tasks, got {len(tasks)}"

    print("==========================================================================")
    print("🚀 Starting RUN 6 BENCHMARK (Ornith 1.5 via SSH Tunnel, 100% Offline)")
    print("==========================================================================")
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")
    print(f"Endpoint: {TUNNEL_URL} | Model: {MODEL_NAME}")
    print("Settings: max_steps=12, max_tokens=1024, temp=0.1, BM25 threshold=0.02")

    # Devices
    device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
    print(f"Initializing Arm A (ReflexCore) on device='{device}'...")
    reflex_a = ReflexCore(device=device)
    engine_a = BrowserEngine()
    orchestrator_a = BrowserOrchestrator(engine_a, reflex_a)

    print("Initializing Arm C (BM25ReflexCore, threshold=0.02)...")
    reflex_c = BM25ReflexCore(threshold=0.02)
    engine_c = BrowserEngine()
    orchestrator_c = BrowserOrchestrator(engine_c, reflex_c)

    rate_limiter = DomainRateLimiter(min_interval=1.0)

    out_cb = "results/real_world/run6_local/runner_cognitive_browser.jsonl"
    out_pw = "results/real_world/run6_local/baseline_playwright_mcp.jsonl"
    out_bm = "results/real_world/run6_local/baseline_cognitive_bm25.jsonl"

    completed_ids = set()
    for p in [out_cb, out_pw, out_bm]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f_done:
                for line in f_done:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            if d.get("status") != "infra_error":
                                completed_ids.add(d["sample_id"])
                        except Exception:
                            pass

    print(f"Resumption check: {len(completed_ids)} valid episodes already completed on disk.")

    all_jobs = []
    for rep in [1, 2, 3]:
        for t in tasks:
            s_cb = f"cb_{t['task_id']}_r{rep}"
            s_pw = f"pw_{t['task_id']}_r{rep}"
            s_bm = f"bm_{t['task_id']}_r{rep}"
            all_jobs.append((s_cb, t, rep, "cognitive_browser"))
            all_jobs.append((s_pw, t, rep, "playwright_mcp"))
            all_jobs.append((s_bm, t, rep, "cognitive_bm25"))

    random.seed(42)
    random.shuffle(all_jobs)

    remaining_jobs = [j for j in all_jobs if j[0] not in completed_ids]
    print(f"Total jobs: {len(all_jobs)}, Valid completed: {len(completed_ids)}, Remaining to execute: {len(remaining_jobs)}.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )

        sem = asyncio.Semaphore(4)
        consecutive_infra_errors = defaultdict(int)
        halted = False
        write_lock = asyncio.Lock()

        async def worker(job, job_idx, total_count):
            nonlocal halted
            if halted:
                return None

            s_id, t, rep, arm = job
            async with sem:
                orch = orchestrator_a if arm == "cognitive_browser" else (orchestrator_c if arm == "cognitive_bm25" else None)
                try:
                    res = await asyncio.wait_for(
                        run_episode(
                            sample_id=s_id,
                            task=t,
                            repetition=rep,
                            arm=arm,
                            browser=browser,
                            rate_limiter=rate_limiter,
                            orchestrator=orch,
                        ),
                        timeout=420.0
                    )
                except asyncio.TimeoutError:
                    print(f"⚠️ [TIMEOUT] {arm} on {t['task_id']} (r{rep}) exceeded 420s. Continuing.")
                    res = {
                        "sample_id": s_id,
                        "task_id": t["task_id"],
                        "language": t.get("language", "pl"),
                        "min_steps": t.get("min_steps", 1),
                        "repetition": rep,
                        "arm": arm,
                        "initial_url": t.get("url", ""),
                        "final_url": "about:blank",
                        "final_title": "",
                        "status": "infra_error",
                        "error_message": "Episode execution timed out (>420s)",
                        "screenshot_path": "",
                        "steps_count": 0,
                        "steps_log": [],
                        "page_snippet": "",
                        "e2e_latency_ms": 420000.0,
                        "agent_input_tokens": 0,
                        "agent_output_tokens": 0,
                        "total_agent_tokens": 0,
                        "stop_reason": "timeout",
                        "partial_fallback_count": 0,
                        "agent_interventions": 0,
                    }

                out_f = out_cb if arm == "cognitive_browser" else (out_pw if arm == "playwright_mcp" else out_bm)
                async with write_lock:
                    with open(out_f, "a", encoding="utf-8") as f_out:
                        f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

                if res.get("status") == "infra_error":
                    consecutive_infra_errors[arm] += 1
                    print(f"⚠️ [INFRA_ERROR] {arm} on {t['task_id']}: {res.get('error_message')}")
                    if consecutive_infra_errors[arm] >= 5:
                        halted = True
                        print(f"🚨 CRITICAL: 5 consecutive infra_errors in {arm}! HALTING RUN 6.")
                        return None
                else:
                    consecutive_infra_errors[arm] = 0

                print(
                    f"[{job_idx:03d}/{total_count}] [{arm:18s}] {t['task_id']} (r{rep}) -> {res.get('status')} "
                    f"({res.get('e2e_latency_ms', 0)}ms, {res.get('total_agent_tokens', 0)} tok, steps={res.get('steps_count', 0)})"
                )
                return res

        tasks_async = [worker(j, i + 1, len(remaining_jobs)) for i, j in enumerate(remaining_jobs)]
        await asyncio.gather(*tasks_async)
        await browser.close()

    print("\n✅ RUN 6 Benchmark execution completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
