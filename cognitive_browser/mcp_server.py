"""
Cognitive Browser MCP Server (v0.3.0).
Autonomous web navigation combining local zero-token Cross-Encoder (Reflex Core)
with Stealth BrowserEngine, RouteCache, full DOM action planning, and smart goal termination.
Refactored to delegate execution to BrowserOrchestrator.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

from mcp.server.mcpserver import MCPServer

from .browser import BrowserEngine
from .cache import RouteCache
from .orchestrator import BrowserOrchestrator
from .parser import AriaElementParser
from .reflex import ReflexCore
from .security import mask_sensitive_value, validate_url

SERVER_INSTRUCTIONS = (
    "Cognitive Browser is an ultra-fast, zero-token autonomous browser server powered by a local GPU Cross-Encoder (Reflex Core), Stealth BrowserEngine, and RouteCache.\n\n"
    "## CRITICAL RULE - SINGLE-CALL EXECUTION:\n"
    "- NEVER call `get_aria_snapshot` first to explore a website step-by-step!\n"
    "- NEVER search the filesystem (`ls`/`find`) for MCP config files! Call `mcp_cognitive-browser_cognitive_browse` directly.\n"
    "- For ANY information lookup, search, form interaction, or navigation task, make ONE SINGLE CALL to `cognitive_browse(url=..., goal=...)` (or pass `steps=[...]`).\n"
    "- Inside `cognitive_browse`, the local GPU model autonomously maps the Accessibility Tree, fills search boxes, selects filters, scrolls, clicks through menus/accordions/links across multiple hops (with 0 LLM tokens), and stops automatically as soon as the destination page or search results are visible, returning the clean content in `page_content_snippet`.\n"
    "- Synthesize your final answer immediately from `page_content_snippet` in a single turn!"
)

mcp = MCPServer(
    "cognitive-browser",
    instructions=SERVER_INSTRUCTIONS,
)

# Shared singleton services
_reflex: Optional[ReflexCore] = None
_engine: Optional[BrowserEngine] = None
_orchestrator: Optional[BrowserOrchestrator] = None
_last_candidates: List[Dict[str, Any]] = []

MIN_GOAL_CONFIDENCE = 0.35


def get_services() -> tuple[ReflexCore, BrowserEngine]:
    global _reflex, _engine
    if _reflex is None:
        _reflex = ReflexCore()
    if _engine is None:
        _engine = BrowserEngine(reflex=_reflex)
    return _reflex, _engine


def get_orchestrator() -> BrowserOrchestrator:
    global _orchestrator
    reflex, engine = get_services()
    if _orchestrator is None or _orchestrator.engine is not engine or _orchestrator.reflex is not reflex:
        _orchestrator = BrowserOrchestrator(engine=engine, reflex=reflex)
    return _orchestrator


# Backward-compatibility helper functions
def _compact_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return BrowserOrchestrator.compact_candidates(candidates)


def _clean_aria_lines(raw_aria: str, max_lines: int = 140, max_chars: int = 5800) -> str:
    return BrowserOrchestrator.clean_aria_lines(raw_aria, max_lines=max_lines, max_chars=max_chars)


async def _safe_resolve_captcha(engine: Any, page: Any) -> Dict[str, Any]:
    orch = get_orchestrator()
    return await orch.safe_resolve_captcha(page)


async def _extract_page_content(
    page: Any,
    max_lines: int = 140,
    ranked_chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    orch = get_orchestrator()
    return await orch.extract_page_content(page, max_lines=max_lines, ranked_chunks=ranked_chunks)


async def _dispatch_step_action(
    engine: BrowserEngine,
    page: Any,
    action_plan: Dict[str, Any],
    instruction: str = "",
) -> str:
    orch = get_orchestrator()
    return await orch.dispatch_step_action(page, action_plan, instruction=instruction)


@mcp.tool()
async def cognitive_browse(
    url: Optional[str] = None,
    goal: Optional[str] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    payload: Optional[Dict[str, Any]] = None,
    browser_context: Optional[str] = "ephemeral",
) -> Dict[str, Any]:
    """
    [PRIMARY TOOL - USE THIS IN A SINGLE CALL FOR ALL WEB TASKS]
    Autonomously navigates to a goal or executes a multi-step workflow on GPU with 0 LLM tokens.

    Modes:
    1. Autonomous Goal Mode (FASTEST - 1 Tool Call):
       Pass `url` and `goal` (e.g. `cognitive_browse(url="https://www.gov.pl", goal="założenie firmy...")`).
       The local GPU model autonomously maps the page, fills search inputs, clicks through menus/accordions/links up to 5 hops, stops as soon as the target data is visible, and returns the final content in `page_content_snippet`.
    2. Planned Steps Mode:
       Pass `url` and `steps`: `[{"instruction": "...", "fill_value": "...", "action": "click|fill|fill_and_submit|select|check|scroll|hover|press|upload_file"}, ...]`.
    3. Universal Parameter Binding & File Upload (`payload`):
       Pass `payload: {"key": "val", "file_key": "/path/file.pdf", ...}` to automatically fill arbitrary complex forms, job applications with CVs, filter selectors, or multi-file galleries with neural Cross-Encoder element matching without hardcoded schemas.

    Args:
        url: Target web page URL to open. If omitted, continues on the current active page.
        goal: High-level natural language goal for autonomous multi-hop navigation on GPU (include relevant keywords/synonyms).
        steps: Optional explicit list of action steps with 'instruction' (and optional 'fill_value' / 'action').
        payload: Optional dictionary of form field values, filters, or file paths (str or List[str]) to bind and fill.
        browser_context: Browser context mode: 'ephemeral' (default clean session), 'firefox' or 'system' (import real authenticated cookies to bypass Cloudflare), or path to cookies sqlite/json.
    """
    global _last_candidates
    orchestrator = get_orchestrator()
    page = await orchestrator.engine.get_page()

    res = await orchestrator.browse(
        page,
        url=url,
        goal=goal,
        steps=steps,
        payload=payload,
        browser_context=browser_context,
    )
    _last_candidates = orchestrator.last_candidates
    return res


@mcp.tool()
async def smart_action(
    instruction: str,
    target_ref: Optional[str] = None,
    target_name: Optional[str] = None,
    fill_value: Optional[str] = None,
    file_paths: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    [ESCALATION / FALLBACK ONLY - DO NOT USE FOR NORMAL WORKFLOWS]
    Executes a single action on the current active browser page.
    CRITICAL: DO NOT use this tool in a loop for regular browsing! Use cognitive_browse(url=..., goal=...) instead to execute multi-step workflows with 0 tokens.
    """
    global _last_candidates
    orchestrator = get_orchestrator()
    if _last_candidates and not orchestrator.last_candidates:
        orchestrator.last_candidates = _last_candidates

    page = await orchestrator.engine.get_page()
    res = await orchestrator.smart_action(
        page,
        instruction=instruction,
        target_ref=target_ref,
        target_name=target_name,
        fill_value=fill_value,
        file_paths=file_paths,
    )
    _last_candidates = orchestrator.last_candidates
    return res


@mcp.tool()
async def get_aria_snapshot(
    url: Optional[str] = None,
    max_candidates: int = 80,
    selector: str = "body",
    include_aria_tree: bool = True,
    max_tree_lines: int = 120,
) -> Dict[str, Any]:
    """
    Returns the condensed Accessibility Tree (ARIA snapshot) of the current or specified page.
    CRITICAL: Prefer `cognitive_browse(url=..., goal=...)` instead of manual snapshots!
    """
    global _last_candidates
    orchestrator = get_orchestrator()
    page = await orchestrator.engine.get_page()

    res = await orchestrator.get_aria_snapshot(
        page,
        url=url,
        max_candidates=max_candidates,
        selector=selector,
        include_aria_tree=include_aria_tree,
        max_tree_lines=max_tree_lines,
    )
    _last_candidates = orchestrator.last_candidates
    return res


def main():
    mcp.run()


if __name__ == "__main__":
    main()
