"""
Cognitive Browser MCP Server (v0.3.0).
Autonomous web navigation combining local zero-token Cross-Encoder (Reflex Core)
with Stealth BrowserEngine, RouteCache, full DOM action planning, and smart goal termination.
"""

import asyncio
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Union

from mcp.server.mcpserver import MCPServer

from .browser import BrowserEngine
from .cache import RouteCache
from .reflex import AriaElementParser, ReflexCore
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
_last_candidates: List[Dict[str, Any]] = []

MIN_GOAL_CONFIDENCE = 0.35


def get_services() -> tuple[ReflexCore, BrowserEngine]:
    global _reflex, _engine
    if _reflex is None:
        _reflex = ReflexCore()
    if _engine is None:
        _engine = BrowserEngine(reflex=_reflex)
    return _reflex, _engine


async def _safe_resolve_captcha(engine: Any, page: Any) -> Dict[str, Any]:
    """Safely invokes auto_resolve_captcha_if_present, handling coroutines and mock objects gracefully."""
    resolver = getattr(engine, "auto_resolve_captcha_if_present", None)
    if callable(resolver):
        try:
            res = resolver(page)
            if asyncio.iscoroutine(res):
                return await res
            if isinstance(res, dict):
                return res
        except Exception:
            pass
    return {}


def _compact_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strips redundant fields from candidates to keep MCP JSON payloads small (<8KB)."""
    compact = []
    for c in candidates:
        item: Dict[str, Any] = {
            "ref": c.get("ref"),
            "role": c.get("role"),
            "name": c.get("name"),
        }
        if c.get("context"):
            item["context"] = c["context"]
        if c.get("url"):
            item["url"] = c["url"]
        if c.get("state"):
            item["state"] = c["state"]
        compact.append(item)
    return compact


def _clean_aria_lines(raw_aria: str, max_lines: int = 140, max_chars: int = 5800) -> str:
    """
    Removes decorative noise lines (empty `- img`, `- separator`, repetitive `Wyczyść` filter buttons)
    and collapses long runs of sidebar filter checkboxes so MCP responses stay under ~6.5KB
    (never triggering AGY 10KB file truncation) while keeping all offers and descriptions visible.
    """
    raw_lines = raw_aria.splitlines()
    skip_banner = len(raw_lines) > max_lines
    in_banner = False
    banner_indent = 0
    consecutive_checkboxes = 0

    cleaned: List[str] = []
    total_chars = 0
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped or stripped in (
            "- img",
            "- separator",
            "- img:",
            '- text: ""',
            "- group",
            '- button "Wyczyść"',
            '- grid "Skip links":',
            '- row "Przejdź do głównej części":',
            '- gridcell "Przejdź do głównej części"',
            '- row "Przejdź do stopki":',
            '- gridcell "Przejdź do stopki"',
        ):
            continue

        indent = len(line) - len(line.lstrip(" "))

        # Skip top site-wide header banner if the page is large
        if skip_banner:
            if stripped.startswith("- banner"):
                in_banner = True
                banner_indent = indent
                continue
            elif in_banner:
                if indent > banner_indent:
                    continue
                else:
                    in_banner = False

        # Collapse sidebar filter lists of dozens of checkboxes
        if stripped.startswith("- checkbox"):
            consecutive_checkboxes += 1
            if consecutive_checkboxes > 4:
                continue
        elif not stripped.startswith("- text:"):
            consecutive_checkboxes = 0

        # Trim huge tracking query parameters on /url: lines to save tokens
        if stripped.startswith("- /url:") and len(line) > 130:
            line = line.split("?utm_")[0].split("&utm_")[0][:140]

        if total_chars + len(line) + 1 > max_chars or len(cleaned) >= max_lines:
            remaining = len(raw_lines) - idx
            cleaned.append(f"... (truncated {remaining} remaining lines)")
            break

        cleaned.append(line)
        total_chars += len(line) + 1

    return "\n".join(cleaned)


async def _extract_page_content(
    page: Any,
    max_lines: int = 140,
    ranked_chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Extracts clean, noise-free semantic ARIA content from the destination page.
    If `ranked_chunks` from GPU `evaluate_page_satisfaction` are provided and the page is long,
    combines the top summary lines with the highest-scoring deep chunks (e.g. offer 'OPIS' at line 380).
    """
    try:
        for sel in ["main article", "article", "main", "body"]:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                raw_aria = await loc.aria_snapshot()
                if len(raw_aria.strip()) > 60:
                    if ranked_chunks and len(raw_aria.splitlines()) > max_lines:
                        # Take top 2chunks (e.g. header/price) + top 3 highest-scoring deep chunks in document order
                        selected_indices = {0, 1}
                        for rc in ranked_chunks[:3]:
                            selected_indices.add(int(rc["chunk_index"]))
                        ordered_chunks = sorted(
                            [c for c in ranked_chunks if int(c["chunk_index"]) in selected_indices],
                            key=lambda x: int(x["chunk_index"]),
                        )
                        combined_lines: List[str] = []
                        for ch in ordered_chunks:
                            combined_lines.extend(ch.get("lines", []))
                        return _clean_aria_lines("\n".join(combined_lines), max_lines=max_lines)
                    return _clean_aria_lines(raw_aria, max_lines=max_lines)
    except Exception:
        pass
    return ""


async def _dispatch_step_action(
    engine: BrowserEngine,
    page: Any,
    action_plan: Dict[str, Any],
    instruction: str = "",
) -> str:
    """Executes an inferred or explicit action plan on BrowserEngine while preserving test mocks."""
    act = action_plan.get("action", "click")
    role = action_plan.get("role", "button")
    name = action_plan.get("name", "")
    nth = int(action_plan.get("nth", 0))
    val = action_plan.get("value")
    url = action_plan.get("url")

    if act == "scroll" and hasattr(engine, "scroll_page"):
        direction = str(val or "down")
        await engine.scroll_page(page, direction=direction)
        return f"scroll('{direction}')"

    if act == "press" and hasattr(engine, "press_key"):
        key = str(val or "Enter")
        await engine.press_key(page, key=key)
        return f"press('{key}')"

    if act == "hover" and hasattr(engine, "hover_element"):
        await engine.hover_element(page, role=role, name=name, nth=nth)
        return "hover()"

    if act == "select" and val is not None and hasattr(engine, "select_option"):
        await engine.select_option(page, role=role, name=name, value=str(val), nth=nth)
        return f"select('{val}')"

    if act in ("check", "uncheck") and hasattr(engine, "toggle_checkbox"):
        await engine.toggle_checkbox(page, role=role, name=name, checked=(act == "check"), nth=nth)
        return f"{act}()"

    if act == "upload_file" and hasattr(engine, "upload_file"):
        files = action_plan.get("file_paths") or val
        if files:
            await engine.upload_file(page, role=role, name=name, file_paths=files, nth=nth)
            return f"upload_file('{files}')"

    if act in ("fill", "fill_and_submit") and val is not None:
        submit = act == "fill_and_submit"
        masked_val = mask_sensitive_value(str(val), instruction=instruction)
        if submit:
            try:
                await engine.fill_element(page, role=role, name=name, value=str(val), nth=nth, submit=True)
            except TypeError:
                await engine.fill_element(page, role=role, name=name, value=str(val))
            return f"fill_and_submit('{masked_val}')"
        else:
            if nth > 0:
                await engine.fill_element(page, role=role, name=name, value=str(val), nth=nth)
            else:
                await engine.fill_element(page, role=role, name=name, value=str(val))
            return f"fill('{masked_val}')"

    if url and str(url).startswith("http"):
        valid_target = validate_url(str(url))
        await page.goto(valid_target, wait_until="domcontentloaded", timeout=15000)
        return f"goto('{valid_target}')"

    if nth > 0:
        await engine.click_element(page, role=role, name=name, nth=nth)
    else:
        await engine.click_element(page, role=role, name=name)
    return "click()"


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
    reflex, engine = get_services()
    page = await engine.get_page()

    t_start = time.perf_counter()
    route_cache = getattr(engine, "route_cache", None)
    has_cache = isinstance(route_cache, RouteCache)

    if url:
        validated_url = validate_url(url)
        # Import system cookies if browser_context is enabled
        if browser_context and browser_context != "ephemeral":
            parsed = urllib.parse.urlparse(validated_url)
            target_domain = parsed.hostname or validated_url
            if hasattr(engine, "import_cookies_for_domain"):
                await engine.import_cookies_for_domain(page, domain=target_domain, browser_type=browser_context)

        if str(page.url).rstrip("/") != validated_url.rstrip("/"):
            try:
                await page.goto(validated_url, wait_until="domcontentloaded", timeout=15000)
            except Exception as nav_err:
                if "blockedbyclient" in str(nav_err).lower():
                    raise
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
            await page.wait_for_timeout(300)
            await engine.auto_dismiss_overlays(page)
            await _safe_resolve_captcha(engine, page)

    # PAYLOAD BINDING: If payload is provided, bind fields semantically and execute actions
    payload_steps_log: List[Dict[str, Any]] = []
    if payload:
        snapshot_yaml = await page.aria_snapshot()
        all_cands = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=150)
        _last_candidates = all_cands
        if hasattr(reflex, "bind_payload_to_elements"):
            bound_actions = reflex.bind_payload_to_elements(payload=payload, candidates=all_cands)
            for b_act in bound_actions:
                t_b = time.perf_counter()
                action_desc = await _dispatch_step_action(
                    engine,
                    page,
                    b_act,
                    instruction=f"Fill payload field: {b_act.get('payload_key')}",
                )
                b_ms = round((time.perf_counter() - t_b) * 1000, 2)
                payload_steps_log.append(
                    {
                        "step": len(payload_steps_log) + 1,
                        "instruction": f"Set payload field '{b_act.get('payload_key')}'",
                        "executed_element": f"[{b_act.get('role')}] {b_act.get('name')}",
                        "action": action_desc,
                        "confidence": b_act.get("confidence", 1.0),
                        "latency_ms": b_ms,
                        "source": "payload_binding",
                        "status": "success",
                    }
                )

        if not goal and not steps:
            return {
                "status": "success",
                "final_url": page.url,
                "final_title": await page.title(),
                "steps_executed": len(payload_steps_log),
                "telemetry": {
                    "total_execution_seconds": round(time.perf_counter() - t_start, 2),
                    "reflex_gpu_ms": 0.0,
                    "tokens_consumed_by_server": 0,
                },
                "steps_log": payload_steps_log,
                "page_content_snippet": await _extract_page_content(page),
            }

    # MODE A: Autonomous Multi-Hop Goal Navigation (when goal is provided and steps is empty)
    if goal and not steps:
        start_url = str(page.url)
        steps_log: List[Dict[str, Any]] = list(payload_steps_log)
        total_reflex_gpu_ms = 0.0

        # Check if RouteCache already knows the direct destination URL for this (start_url, goal)
        if has_cache and route_cache is not None:
            cached_dest = route_cache.get(start_url, f"GOAL::{goal}")
            if cached_dest and cached_dest.get("name", "").startswith("http"):
                dest_url = validate_url(cached_dest["name"])
                t_c = time.perf_counter()
                await page.goto(dest_url, wait_until="domcontentloaded", timeout=15000)
                await page.wait_for_timeout(250)
                cache_ms = round((time.perf_counter() - t_c) * 1000, 2)
                steps_log.append(
                    {
                        "step": 1,
                        "instruction": f"Direct cached jump for goal: {goal}",
                        "executed_element": f"[url] {dest_url}",
                        "action": "goto()",
                        "confidence": 1.0,
                        "latency_ms": cache_ms,
                        "source": "route_cache",
                        "status": "success",
                    }
                )
                return {
                    "status": "success",
                    "final_url": page.url,
                    "final_title": await page.title(),
                    "steps_executed": 1,
                    "telemetry": {
                        "total_execution_seconds": round(time.perf_counter() - t_start, 2),
                        "reflex_gpu_ms": 0.0,
                        "tokens_consumed_by_server": 0,
                    },
                    "steps_log": steps_log,
                    "page_content_snippet": await _extract_page_content(page),
                }

        visited: set[tuple[str, str]] = set()
        goal_words = {
            w
            for w in re.findall(r"\w{4,}", goal.lower())
            if w
            not in {
                "oraz",
                "jak",
                "gdzie",
                "kiedy",
                "żeby",
                "zeby",
                "strona",
                "stronie",
                "znajdz",
                "znajdź",
                "sprawdź",
                "sprawdz",
                "zobacz",
                "pokaż",
                "wyszukaj",
                "odczytaj",
                "cookies",
                "zaakceptuj",
            }
        }
        extracted_query = ReflexCore.extract_fill_value(goal)
        did_fill_search = False
        last_satisfaction_rate = 0.0
        last_ranked_chunks: Optional[List[Dict[str, Any]]] = None

        for hop in range(1, 6):
            snapshot_yaml = await page.aria_snapshot()
            current_title = await page.title()

            # Check for CAPTCHA challenges (Cloudflare Turnstile, reCAPTCHA, etc.)
            if hasattr(reflex, "evaluate_captcha_presence"):
                c_eval = reflex.evaluate_captcha_presence(
                    candidates=AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=50),
                    page_title=current_title,
                    page_url=str(page.url),
                    snapshot_yaml=snapshot_yaml,
                )
                if c_eval.get("is_captcha"):
                    c_res = await _safe_resolve_captcha(engine, page)
                    if c_res.get("solved"):
                        await page.wait_for_timeout(1000)
                        snapshot_yaml = await page.aria_snapshot()
                        current_title = await page.title()

            # 1. GPU STATE EVALUATION: Compute goal_satisfaction_rate on current page chunks
            if hasattr(reflex, "evaluate_page_satisfaction") and callable(reflex.evaluate_page_satisfaction):
                sat_res = reflex.evaluate_page_satisfaction(goal, snapshot_yaml, current_title, str(page.url))
                if isinstance(sat_res, dict):
                    total_reflex_gpu_ms += float(sat_res.get("latency_ms", 0.0))
                    last_satisfaction_rate = float(sat_res.get("goal_satisfaction_rate", 0.0))
                    last_ranked_chunks = sat_res.get("ranked_chunks")
                    if sat_res.get("is_satisfied"):
                        break

            # Fallback structural check
            if ReflexCore.is_goal_satisfied_on_page(goal, snapshot_yaml, current_title, str(page.url)):
                last_satisfaction_rate = max(last_satisfaction_rate, 0.90)
                break

            all_cands = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=150)
            _last_candidates = all_cands

            # Filter out visited elements and generic footer/header/account noise
            avail = [
                c
                for c in all_cands
                if (c["role"], c["name"]) not in visited
                and not ReflexCore.is_noise_candidate(c, goal)
                and c["role"]
                in (
                    "link",
                    "button",
                    "tab",
                    "menuitem",
                    "searchbox",
                    "textbox",
                    "combobox",
                    "checkbox",
                    "radio",
                    "option",
                )
            ]
            if not avail:
                break

            # If goal explicitly asks to type/search a query and we haven't typed it yet, prioritize input candidates
            if extracted_query and not did_fill_search:
                input_cands = [c for c in avail if c["role"] in ("searchbox", "textbox", "combobox")]
                if input_cands:
                    avail = input_cands

            decision = reflex.evaluate_candidates(intent=goal, candidates=avail)
            total_reflex_gpu_ms += decision.get("latency_ms", 0.0)
            confidence = float(decision.get("confidence", 0.0))

            # STATE vs ACTION COMPETITION:
            # If current page satisfaction rate beats the best clickable element confidence, STAY on current page!
            if last_satisfaction_rate >= 0.65 and last_satisfaction_rate > confidence + 0.05:
                break

            # Confidence brake: stop immediately if no candidate matches the goal with >= 0.35 confidence
            if confidence < MIN_GOAL_CONFIDENCE and not (extracted_query and not did_fill_search and avail[0]["role"] in ("searchbox", "textbox")):
                break

            best = decision["ranked_candidates"][0]
            role = best["role"]
            name = best["name"]
            visited.add((role, name))

            action_plan = ReflexCore.infer_action_plan(
                goal, best, explicit_fill_value=extracted_query if not did_fill_search else None
            )
            if action_plan["action"] in ("fill", "fill_and_submit"):
                did_fill_search = True

            try:
                action_desc = await _dispatch_step_action(engine, page, action_plan, instruction=goal)
                await page.wait_for_timeout(300)
                steps_log.append(
                    {
                        "step": hop,
                        "instruction": goal,
                        "executed_element": f"[{role}] {name}",
                        "action": action_desc,
                        "confidence": confidence,
                        "goal_satisfaction_before_step": last_satisfaction_rate,
                        "latency_ms": decision["latency_ms"],
                        "source": "reflex_gpu",
                        "status": "success",
                    }
                )

                if action_plan["action"] == "fill_and_submit":
                    await page.wait_for_timeout(400)
                    continue

                title_lower = (await page.title()).lower()
                name_lower = name.lower()
                cand_url = best.get("url")
                matches_in_title = sum(1 for gw in goal_words if gw[:5] in title_lower or gw[:5] in name_lower)
                if (cand_url and str(cand_url).startswith("http") and matches_in_title >= 2) or matches_in_title >= 3:
                    break
            except Exception:
                continue

        # Save direct route shortcut in RouteCache only if we actually navigated to a new valid destination URL
        final_url_str = str(page.url)
        if (
            has_cache
            and route_cache is not None
            and final_url_str != start_url
            and len(steps_log) > 0
            and not any(k in final_url_str.lower() for k in ["captcha", "sorry", "error", "login"])
        ):
            route_cache.store(start_url, f"GOAL::{goal}", role="url", name=final_url_str, action="goto")

        return {
            "status": "success",
            "final_url": page.url,
            "final_title": await page.title(),
            "steps_executed": len(steps_log),
            "goal_satisfaction_rate": last_satisfaction_rate,
            "telemetry": {
                "total_execution_seconds": round(time.perf_counter() - t_start, 2),
                "reflex_gpu_ms": round(total_reflex_gpu_ms, 2),
                "tokens_consumed_by_server": 0,
            },
            "steps_log": steps_log,
            "page_content_snippet": await _extract_page_content(page, ranked_chunks=last_ranked_chunks),
        }

    # If neither goal nor steps provided, return compact initial ARIA state
    if not steps:
        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=60)
        _last_candidates = candidates
        return {
            "status": "ready_for_plan",
            "current_url": page.url,
            "current_title": await page.title(),
            "candidates_count": len(candidates),
            "candidates": _compact_candidates(candidates),
            "aria_tree": _clean_aria_lines(snapshot_yaml, max_lines=90),
            "message": "Page loaded. Pass `goal` or `steps` to cognitive_browse to execute autonomously on GPU.",
            "execution_seconds": round(time.perf_counter() - t_start, 2),
        }

    # MODE B: Explicit Step-by-Step Execution
    steps_log = list(payload_steps_log)
    total_reflex_gpu_ms = 0.0

    for idx, s in enumerate(steps):
        instruction = s.get("instruction", "")
        fill_val = s.get("fill_value")
        explicit_action = s.get("action")
        current_page_url = str(page.url)

        # Page-level actions (scroll / press) that don't require candidate evaluation
        if explicit_action in ("scroll", "press"):
            t_act = time.perf_counter()
            action_desc = await _dispatch_step_action(
                engine,
                page,
                {"action": explicit_action, "value": fill_val or ("down" if explicit_action == "scroll" else "Enter")},
                instruction=instruction,
            )
            act_ms = round((time.perf_counter() - t_act) * 1000, 2)
            steps_log.append(
                {
                    "step": idx + 1,
                    "instruction": instruction,
                    "executed_element": f"[page] {explicit_action}",
                    "action": action_desc,
                    "confidence": 1.0,
                    "latency_ms": act_ms,
                    "source": "reflex_direct",
                    "status": "success",
                }
            )
            continue

        # 0. FAST PATH: Check RouteCache first
        if has_cache and route_cache is not None:
            cached = route_cache.get(current_page_url, instruction)
            if cached:
                c_role = cached["role"]
                c_name = cached["name"]
                c_nth = cached.get("nth", 0)
                c_act = explicit_action or cached.get("action") or ("fill" if fill_val else "click")
                t_cache = time.perf_counter()
                try:
                    action_desc = await _dispatch_step_action(
                        engine,
                        page,
                        {"action": c_act, "role": c_role, "name": c_name, "nth": c_nth, "value": fill_val},
                        instruction=instruction,
                    )
                    cache_ms = round((time.perf_counter() - t_cache) * 1000, 2)
                    await page.wait_for_timeout(250)
                    route_cache.store(
                        current_page_url,
                        instruction,
                        role=c_role,
                        name=c_name,
                        nth=c_nth,
                        action=c_act,
                        context=cached.get("context", ""),
                    )
                    steps_log.append(
                        {
                            "step": idx + 1,
                            "instruction": instruction,
                            "executed_element": f"[{c_role}] {c_name}",
                            "action": action_desc,
                            "confidence": 1.0,
                            "latency_ms": cache_ms,
                            "source": "route_cache",
                            "status": "success",
                        }
                    )
                    continue
                except Exception:
                    route_cache.invalidate(current_page_url, instruction)

        # 1. Capture current ARIA snapshot
        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(
            snapshot_yaml, max_candidates=150, require_input_only=bool(fill_val and explicit_action not in ("select", "check"))
        )
        if not candidates and fill_val:
            candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=150, require_input_only=False)

        _last_candidates = candidates

        # 2. Fast local evaluation on GPU
        decision = reflex.evaluate_candidates(intent=instruction, candidates=candidates)
        total_reflex_gpu_ms += decision.get("latency_ms", 0.0)

        # 3. Check confidence: if ambiguous -> escalate to orchestrating agent
        if decision.get("should_escalate") or decision.get("target_ref") == "NONE":
            top_candidates = _compact_candidates(decision.get("ranked_candidates", [])[:5])
            for tc, raw_tc in zip(top_candidates, decision.get("ranked_candidates", [])[:5], strict=False):
                tc["confidence_score"] = raw_tc.get("confidence_score", 0.0)
            remaining_steps = steps[idx:]
            return {
                "status": "clarification_needed",
                "completed_steps": idx,
                "current_url": page.url,
                "current_title": await page.title(),
                "pending_step": instruction,
                "confidence": decision.get("confidence", 0.0),
                "top_candidates": top_candidates,
                "remaining_steps": remaining_steps,
                "message": (
                    f"Reflex Core confidence ({decision.get('confidence', 0.0):.2f}) is below threshold for step: '{instruction}'. "
                    f"Inspect 'top_candidates' and update the plan: call cognitive_browse(steps=[...]) with updated steps to resume autonomous execution, "
                    f"or call smart_action(target_ref=...) to resolve this single step."
                ),
                "telemetry": {
                    "execution_seconds": round(time.perf_counter() - t_start, 2),
                    "reflex_gpu_ms": round(total_reflex_gpu_ms, 2),
                    "steps_completed_zero_tokens": idx,
                },
                "steps_log": steps_log,
            }

        # 4. High confidence -> execute action via dispatcher and store in RouteCache
        best = decision["ranked_candidates"][0]
        role = best["role"]
        name = best["name"]
        nth = best.get("nth", 0)

        if explicit_action:
            action_plan = {
                "action": explicit_action,
                "role": role,
                "name": name,
                "nth": nth,
                "value": fill_val if fill_val is not None else ReflexCore.extract_fill_value(instruction),
            }
        elif fill_val is not None:
            action_plan = {
                "action": "fill",
                "role": role,
                "name": name,
                "nth": nth,
                "value": fill_val,
            }
        else:
            action_plan = ReflexCore.infer_action_plan(instruction, best)
            # In explicit steps mode, avoid navigating via cand_url so click_element is used consistently
            action_plan.pop("url", None)

        try:
            action_desc = await _dispatch_step_action(engine, page, action_plan, instruction=instruction)

            if has_cache and route_cache is not None:
                route_cache.store(
                    current_page_url,
                    instruction,
                    role=role,
                    name=name,
                    nth=nth,
                    action=action_plan["action"],
                    context=best.get("context", ""),
                )

            await page.wait_for_timeout(300)
            c_sol = await _safe_resolve_captcha(engine, page)
            step_record: Dict[str, Any] = {
                "step": idx + 1,
                "instruction": instruction,
                "executed_element": f"[{role}] {name}",
                "action": action_desc,
                "confidence": decision["confidence"],
                "latency_ms": decision["latency_ms"],
                "source": "reflex_gpu",
                "status": "success",
            }
            if c_sol.get("detected") and c_sol.get("solved"):
                step_record["captcha_solved"] = c_sol
            steps_log.append(step_record)
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
        "page_content_snippet": await _extract_page_content(page),
    }


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
    reflex, engine = get_services()
    page = await engine.get_page()

    t_start = time.perf_counter()
    await engine.auto_dismiss_overlays(page)

    if target_ref or target_name:
        chosen_candidate = None
        if target_ref and _last_candidates:
            chosen_candidate = next((c for c in _last_candidates if c.get("ref") == target_ref), None)

        if not chosen_candidate and target_name and _last_candidates:
            clean_tname = (
                re.sub(r"[\ue000-\uf8ff\uf000-\uffff]|\\u[e-fE-F][0-9a-fA-F]{3}", "", target_name).strip().lower()
            )
            for c in _last_candidates:
                cname = (c.get("name") or "").lower()
                if clean_tname in cname or cname in clean_tname:
                    chosen_candidate = c
                    break

        if chosen_candidate:
            role = chosen_candidate["role"]
            name = chosen_candidate["name"]
            nth = chosen_candidate.get("nth", 0)
        else:
            role = "textbox" if fill_value else "link"
            name = target_name or instruction
            nth = 0

        action_plan = ReflexCore.infer_action_plan(
            instruction, {"role": role, "name": name, "nth": nth}, explicit_fill_value=fill_value
        )
        if file_paths is not None:
            action_plan["action"] = "upload_file"
            action_plan["file_paths"] = file_paths
        elif fill_value is not None and action_plan["action"] not in ("fill", "fill_and_submit", "select"):
            action_plan["action"] = "fill"
            action_plan["value"] = fill_value

        action_desc = await _dispatch_step_action(engine, page, action_plan, instruction=instruction)

        await page.wait_for_timeout(300)
        return {
            "status": "success",
            "action": action_desc,
            "target": f"[{role}] {name}",
            "current_url": page.url,
            "current_title": await page.title(),
            "execution_seconds": round(time.perf_counter() - t_start, 2),
        }

    snapshot_yaml = await page.aria_snapshot()
    candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=150, require_input_only=bool(fill_value))
    _last_candidates = candidates

    decision = reflex.evaluate_candidates(intent=instruction, candidates=candidates)
    if decision.get("should_escalate") or decision.get("target_ref") == "NONE":
        return {
            "status": "clarification_needed",
            "current_url": page.url,
            "pending_instruction": instruction,
            "confidence": decision.get("confidence", 0.0),
            "top_candidates": _compact_candidates(decision.get("ranked_candidates", [])[:5]),
        }

    best = decision["ranked_candidates"][0]
    role = best["role"]
    name = best["name"]

    action_plan = ReflexCore.infer_action_plan(instruction, best, explicit_fill_value=fill_value)
    if file_paths is not None:
        action_plan["action"] = "upload_file"
        action_plan["file_paths"] = file_paths
    elif fill_value is not None and action_plan["action"] not in ("fill", "fill_and_submit", "select"):
        action_plan["action"] = "fill"
        action_plan["value"] = fill_value

    action_desc = await _dispatch_step_action(engine, page, action_plan, instruction=instruction)

    await page.wait_for_timeout(300)
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
    _, engine = get_services()
    page = await engine.get_page()

    t_start = time.perf_counter()

    if url:
        validated_url = validate_url(url)
        if str(page.url).rstrip("/") != validated_url.rstrip("/"):
            try:
                await page.goto(validated_url, wait_until="domcontentloaded", timeout=15000)
            except Exception as nav_err:
                if "blockedbyclient" in str(nav_err).lower():
                    raise
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
            await page.wait_for_timeout(250)
            await engine.auto_dismiss_overlays(page)

    loc = page.locator(selector)
    if await loc.count() == 0:
        loc = page.locator("body")

    snapshot_yaml = await loc.aria_snapshot()
    candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=max_candidates)
    _last_candidates = candidates

    shown_candidates = candidates[:15] if include_aria_tree else candidates
    res: Dict[str, Any] = {
        "current_url": page.url,
        "current_title": await page.title(),
        "candidates_count": len(candidates),
        "candidates": _compact_candidates(shown_candidates),
        "execution_seconds": round(time.perf_counter() - t_start, 2),
    }

    if include_aria_tree:
        res["aria_tree"] = _clean_aria_lines(
            snapshot_yaml, max_lines=min(max_tree_lines, 140), max_chars=5200
        )

    return res


def main():
    mcp.run()


if __name__ == "__main__":
    main()
