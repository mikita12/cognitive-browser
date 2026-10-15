"""
Cognitive Browser MCP Server (v0.2.0).
Autonomous web navigation combining local zero-token NLI (Reflex Core)
with interactive agent-in-the-loop escalation (orchestrating LLM as Cortex).
"""

import time
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from .browser import BrowserEngine
from .reflex import AriaElementParser, ReflexCore
from .security import mask_sensitive_value, validate_url

mcp = MCPServer(
    "cognitive-browser",
    instructions=(
        "Cognitive Browser is a high-speed, token-efficient browser automation server. "
        "It uses a local NLI model (Reflex Core) to execute routine clicks and form inputs with zero LLM tokens. "
        "When Reflex encounters ambiguous elements or confidence drops below threshold, it returns "
        "'clarification_needed' with top element candidates so you (the orchestrator) can resolve the choice."
    ),
)

# Shared singleton services
_reflex: Optional[ReflexCore] = None
_engine: Optional[BrowserEngine] = None
_last_candidates: List[Dict[str, Any]] = []


def get_services() -> tuple[ReflexCore, BrowserEngine]:
    global _reflex, _engine
    if _reflex is None:
        _reflex = ReflexCore()
    if _engine is None:
        _engine = BrowserEngine(reflex=_reflex)
    return _reflex, _engine


@mcp.tool()
async def cognitive_browse(url: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Executes a multi-step web workflow autonomously.
    Routine steps are resolved locally via Reflex Core (DeBERTa-v3 on GPU) in ~25ms with 0 LLM tokens.
    If an element is ambiguous or confidence falls below threshold, execution pauses and returns
    'clarification_needed' with candidate elements so you can resolve the next action.

    Args:
        url: Target web page URL to open.
        steps: List of action steps, e.g. [{"instruction": "Search query input", "fill_value": "Playwright"}, {"instruction": "Search submit button"}]
    """
    global _last_candidates
    validated_url = validate_url(url)
    reflex, engine = get_services()
    page = await engine.get_page()

    t_start = time.perf_counter()
    await page.goto(validated_url, wait_until="domcontentloaded", timeout=20000)
    await page.wait_for_timeout(1000)
    await engine.auto_dismiss_overlays(page)

    steps_log: List[Dict[str, Any]] = []
    total_reflex_gpu_ms = 0.0

    for idx, s in enumerate(steps):
        instruction = s.get("instruction", "")
        fill_val = s.get("fill_value")

        # 1. Capture current ARIA snapshot
        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(
            snapshot_yaml, max_candidates=50, require_input_only=bool(fill_val)
        )
        if not candidates and fill_val:
            candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=50, require_input_only=False)

        _last_candidates = candidates

        # 2. Fast local NLI evaluation on GPU
        decision = reflex.evaluate_candidates(intent=instruction, candidates=candidates)
        total_reflex_gpu_ms += decision.get("latency_ms", 0.0)

        # 3. Check confidence: if ambiguous -> escalate to orchestrating agent
        if decision.get("should_escalate") or decision.get("target_ref") == "NONE":
            top_candidates = decision.get("ranked_candidates", [])[:5]
            return {
                "status": "clarification_needed",
                "completed_steps": idx,
                "current_url": page.url,
                "current_title": await page.title(),
                "pending_step": instruction,
                "confidence": decision.get("confidence", 0.0),
                "top_candidates": top_candidates,
                "message": (
                    f"Reflex Core confidence ({decision.get('confidence', 0.0):.2f}) is below threshold for step: '{instruction}'. "
                    f"Please inspect 'top_candidates' and call smart_action with the chosen target_ref or target_name to continue."
                ),
                "telemetry": {
                    "execution_seconds": round(time.perf_counter() - t_start, 2),
                    "reflex_gpu_ms": round(total_reflex_gpu_ms, 2),
                    "steps_completed_zero_tokens": idx,
                },
                "steps_log": steps_log,
            }

        # 4. High confidence -> execute action directly
        best = decision["ranked_candidates"][0]
        role = best["role"]
        name = best["name"]

        try:
            if fill_val:
                await engine.fill_element(page, role=role, name=name, value=fill_val)
                masked_val = mask_sensitive_value(fill_val, instruction=instruction)
                action_desc = f"fill('{masked_val}')"
            else:
                await engine.click_element(page, role=role, name=name)
                action_desc = "click()"

            await page.wait_for_timeout(800)
            steps_log.append(
                {
                    "step": idx + 1,
                    "instruction": instruction,
                    "executed_element": f"[{role}] {name}",
                    "action": action_desc,
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "status": "success",
                }
            )
        except Exception as e:
            return {
                "status": "execution_error",
                "failed_step": idx + 1,
                "instruction": instruction,
                "error": str(e),
                "current_url": page.url,
                "steps_log": steps_log,
            }

    t_total = time.perf_counter() - t_start
    return {
        "status": "success",
        "final_url": page.url,
        "final_title": await page.title(),
        "steps_executed": len(steps_log),
        "telemetry": {
            "total_execution_seconds": round(t_total, 2),
            "reflex_gpu_ms": round(total_reflex_gpu_ms, 2),
            "tokens_consumed_by_server": 0,
        },
        "steps_log": steps_log,
    }


@mcp.tool()
async def smart_action(
    instruction: str,
    target_ref: Optional[str] = None,
    target_name: Optional[str] = None,
    fill_value: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a single action on the current active browser page.
    Can either resolve the element via Reflex NLI automatically,
    or execute directly on a target_ref / target_name chosen by the orchestrator.

    Args:
        instruction: Natural language action goal.
        target_ref: Optional element reference (e.g. 'aria-14') from previous candidates list.
        target_name: Optional exact name of the element to interact with.
        fill_value: Optional text to type into an input field.
    """
    global _last_candidates
    reflex, engine = get_services()
    page = await engine.get_page()

    t_start = time.perf_counter()
    await engine.auto_dismiss_overlays(page)

    # 1. Direct execution if orchestrator specified target_ref or target_name
    if target_ref or target_name:
        chosen_candidate = None
        if target_ref and _last_candidates:
            chosen_candidate = next((c for c in _last_candidates if c.get("ref") == target_ref), None)

        if chosen_candidate:
            role = chosen_candidate["role"]
            name = chosen_candidate["name"]
        else:
            role = "textbox" if fill_value else "link"
            name = target_name or instruction

        if fill_value:
            await engine.fill_element(page, role=role, name=name, value=fill_value)
            action_desc = f"fill('{mask_sensitive_value(fill_value)}')"
        else:
            await engine.click_element(page, role=role, name=name)
            action_desc = "click()"

        await page.wait_for_timeout(800)
        return {
            "status": "success",
            "action": action_desc,
            "target": f"[{role}] {name}",
            "current_url": page.url,
            "current_title": await page.title(),
            "execution_seconds": round(time.perf_counter() - t_start, 2),
        }

    # 2. Reflex NLI automated execution
    snapshot_yaml = await page.aria_snapshot()
    candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=50, require_input_only=bool(fill_value))
    _last_candidates = candidates

    decision = reflex.evaluate_candidates(intent=instruction, candidates=candidates)
    if decision.get("should_escalate") or decision.get("target_ref") == "NONE":
        return {
            "status": "clarification_needed",
            "current_url": page.url,
            "pending_instruction": instruction,
            "confidence": decision.get("confidence", 0.0),
            "top_candidates": decision.get("ranked_candidates", [])[:5],
        }

    best = decision["ranked_candidates"][0]
    role = best["role"]
    name = best["name"]

    if fill_value:
        await engine.fill_element(page, role=role, name=name, value=fill_value)
        action_desc = f"fill('{mask_sensitive_value(fill_value)}')"
    else:
        await engine.click_element(page, role=role, name=name)
        action_desc = "click()"

    await page.wait_for_timeout(800)
    return {
        "status": "success",
        "action": action_desc,
        "target": f"[{role}] {name}",
        "confidence": decision["confidence"],
        "latency_ms": decision["latency_ms"],
        "current_url": page.url,
        "current_title": await page.title(),
        "execution_seconds": round(time.perf_counter() - t_start, 2),
    }


@mcp.tool()
async def get_aria_snapshot(max_candidates: int = 40) -> Dict[str, Any]:
    """
    Returns the condensed Accessibility Tree (ARIA snapshot) of the current page.
    98% smaller than raw HTML, capturing strictly interactive candidates.
    """
    global _last_candidates
    _, engine = get_services()
    page = await engine.get_page()

    t_start = time.perf_counter()
    snapshot_yaml = await page.aria_snapshot()
    candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=max_candidates)
    _last_candidates = candidates

    return {
        "current_url": page.url,
        "current_title": await page.title(),
        "candidates_count": len(candidates),
        "candidates": candidates,
        "execution_seconds": round(time.perf_counter() - t_start, 2),
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
