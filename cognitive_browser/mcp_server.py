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
    Executes a multi-step web workflow autonomously in a single turn.

    CRITICAL RULES FOR AGENTS:
    1. NEVER call this tool with empty steps (`"steps": []`).
    2. DO NOT perform manual turn-by-turn ReAct loops by inspecting snapshots after every step.
    3. Formulate the COMPLETE sequence of steps upfront in the `steps` list.
       Example: [{"instruction": "Search input", "fill_value": "news"}, {"instruction": "Search submit button"}, {"instruction": "first result link"}]
    4. Routine steps execute locally on GPU via Reflex Core (~25ms/step, ZERO cloud tokens) in ~3-5 seconds total.
    5. Only if the result status is 'clarification_needed' should you intervene to resolve the ambiguous step with `smart_action`.

    Args:
        url: Target web page URL to open.
        steps: List of action steps with 'instruction' (and optional 'fill_value').
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
    result: Dict[str, Any] = {
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

    # Extract compact semantic ARIA snippet of final page so caller can read content with zero extra tool calls
    try:
        main_loc = page.locator("main")
        if await main_loc.count() == 0:
            main_loc = page.locator("body")
        raw_aria = await main_loc.aria_snapshot()
        lines = raw_aria.splitlines()
        if len(lines) > 80:
            result["page_content_snippet"] = "\n".join(lines[:80]) + f"\n... (and {len(lines) - 80} more lines. Call get_aria_snapshot for full tree)"
        else:
            result["page_content_snippet"] = raw_aria
    except Exception:
        pass

    return result


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
async def get_aria_snapshot(
    max_candidates: int = 40,
    selector: str = "body",
    include_aria_tree: bool = True,
    max_tree_lines: int = 150,
) -> Dict[str, Any]:
    """
    Returns the condensed Accessibility Tree (ARIA snapshot) of the current page.
    Includes readable semantic content (headings, paragraphs, text, lists) and interactive candidates.
    NEVER use read_url_content or fetch raw HTML; all page knowledge is present directly in this ARIA snapshot.

    Args:
        max_candidates: Maximum number of interactive candidates to extract for smart_action.
        selector: CSS selector to scope snapshot (e.g. 'main', 'article', or 'body'). Default 'body'.
        include_aria_tree: Whether to include the readable ARIA YAML tree text. Default True.
        max_tree_lines: Max lines of ARIA YAML tree to return to keep token usage low. Default 150.
    """
    global _last_candidates
    _, engine = get_services()
    page = await engine.get_page()

    t_start = time.perf_counter()
    loc = page.locator(selector)
    if await loc.count() == 0:
        loc = page.locator("body")

    snapshot_yaml = await loc.aria_snapshot()
    candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=max_candidates)
    _last_candidates = candidates

    res: Dict[str, Any] = {
        "current_url": page.url,
        "current_title": await page.title(),
        "candidates_count": len(candidates),
        "candidates": candidates,
        "execution_seconds": round(time.perf_counter() - t_start, 2),
    }

    if include_aria_tree:
        lines = snapshot_yaml.splitlines()
        if len(lines) > max_tree_lines:
            truncated = "\n".join(lines[:max_tree_lines])
            res["aria_tree"] = f"{truncated}\n... (truncated {len(lines) - max_tree_lines} remaining lines. Increase max_tree_lines or use selector e.g. 'main' to view more)"
        else:
            res["aria_tree"] = snapshot_yaml

    return res


def main():
    mcp.run()


if __name__ == "__main__":
    main()
