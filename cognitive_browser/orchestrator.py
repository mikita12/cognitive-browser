"""
BrowserOrchestrator: Coordinates multi-hop autonomous web navigation, planned step workflows,
universal parameter binding, RouteCache lookups with self-healing, and CAPTCHA resolution.
Unified engine used by both FastMCP and FastAPI servers.
"""

import asyncio
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Sequence, Union

from .browser import BrowserEngine
from .cache import RouteCache
from .models import ActionCommand, CandidateElement
from .parser import AriaElementParser
from .reflex import ReflexCore
from .security import mask_sensitive_value, validate_url

MIN_GOAL_CONFIDENCE = 0.35
HIGH_CONFIDENCE_THRESHOLD = 0.55


class BrowserOrchestrator:
    """
    Central orchestration engine for autonomous and planned browser interactions.
    Unifies the multi-hop decision loop across MCP and REST interfaces using `ActionCommand` contracts.
    """

    def __init__(
        self,
        engine: BrowserEngine,
        reflex: ReflexCore,
        route_cache: Optional[RouteCache] = None,
    ):
        self.engine = engine
        self.reflex = reflex
        self.route_cache = route_cache or getattr(engine, "route_cache", None)
        self.last_candidates: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    @staticmethod
    def compact_candidates(candidates: Sequence[Union[CandidateElement, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Strips redundant fields from candidates to keep JSON payloads small (<8KB)."""
        compact: List[Dict[str, Any]] = []
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

    @staticmethod
    def clean_aria_lines(raw_aria: str, max_lines: int = 140, max_chars: int = 5800) -> str:
        """
        Removes decorative empty ARIA primitives (`- img`, `- separator`, `- group`)
        and collapses long runs of repetitive filter checkboxes.
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
            ):
                continue

            indent = len(line) - len(line.lstrip(" "))

            if skip_banner:
                if stripped.startswith("- banner"):
                    in_banner = True
                    banner_indent = indent
                    continue
                elif in_banner:
                    if indent > banner_indent:
                        continue
                    in_banner = False

            if stripped.startswith("- checkbox"):
                consecutive_checkboxes += 1
                if consecutive_checkboxes > 4:
                    continue
            elif not stripped.startswith("- text:"):
                consecutive_checkboxes = 0

            if stripped.startswith("- /url:") and len(line) > 130:
                line = line.split("?utm_")[0].split("&utm_")[0][:140]

            if total_chars + len(line) + 1 > max_chars or len(cleaned) >= max_lines:
                remaining = len(raw_lines) - idx
                cleaned.append(f"... (truncated {remaining} remaining lines)")
                break

            cleaned.append(line)
            total_chars += len(line) + 1

            if len(cleaned) >= max_lines:
                break

        return "\n".join(cleaned)

    async def safe_resolve_captcha(self, page: Any) -> Dict[str, Any]:
        """Safely invokes auto_resolve_captcha_if_present if implemented on engine."""
        resolver = getattr(self.engine, "auto_resolve_captcha_if_present", None)
        if callable(resolver):
            try:
                res = resolver(page)
                if asyncio.iscoroutine(res):
                    res = await res
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
        return {}

    async def _check_and_resolve_captcha(self, page: Any, snapshot_yaml: str, current_title: str) -> tuple[str, str]:
        """Evaluates CAPTCHA presence and resolves it if detected, returning updated (snapshot_yaml, title)."""
        if hasattr(self.reflex, "evaluate_captcha_presence") and callable(self.reflex.evaluate_captcha_presence):
            c_eval = self.reflex.evaluate_captcha_presence(
                candidates=AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=50),
                page_title=current_title,
                page_url=str(page.url),
                snapshot_yaml=snapshot_yaml,
            )
            if isinstance(c_eval, dict) and c_eval.get("is_captcha"):
                c_res = await self.safe_resolve_captcha(page)
                if c_res.get("solved"):
                    await page.wait_for_timeout(1000)
                    snapshot_yaml = await page.aria_snapshot()
                    current_title = await page.title()
        return snapshot_yaml, current_title

    async def extract_page_content(
        self,
        page: Any,
        max_lines: int = 140,
        ranked_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Extracts clean, noise-free semantic ARIA content from the destination page.
        Combines summary with top-scoring deep chunks if ranked_chunks are available.
        """
        try:
            for sel in ("main", "article", "body"):
                raw_loc = page.locator(sel)
                if asyncio.iscoroutine(raw_loc):
                    raw_loc = await raw_loc
                loc = raw_loc.first
                cnt = loc.count()
                if asyncio.iscoroutine(cnt):
                    cnt = await cnt
                if isinstance(cnt, int) and cnt > 0:
                    raw_aria = await loc.aria_snapshot()
                    if isinstance(raw_aria, str) and len(raw_aria.strip()) > 60:
                        raw_lines = raw_aria.splitlines()
                        if ranked_chunks and len(raw_lines) > max_lines:
                            head_lines = raw_lines[: max_lines // 2]
                            deep_chunks = [c for c in ranked_chunks[:3] if int(c.get("chunk_index", 0)) >= 2]
                            deep_chunks.sort(key=lambda x: int(x.get("chunk_index", 0)))
                            combined_lines: List[str] = list(head_lines)
                            for ch in deep_chunks:
                                combined_lines.extend(ch.get("lines", []))
                            return self.clean_aria_lines(
                                "\n".join(combined_lines), max_lines=max_lines, max_chars=max(5800, max_lines * 45)
                            )
                        return self.clean_aria_lines(raw_aria, max_lines=max_lines, max_chars=max(5800, max_lines * 45))
        except Exception:
            pass
        return ""

    async def dispatch_step_action(
        self,
        page: Any,
        action_plan: Union[ActionCommand, Dict[str, Any]],
        instruction: str = "",
    ) -> str:
        """Executes an ActionCommand on BrowserEngine while preserving test mocks and sensitive value masking."""
        cmd = ActionCommand.from_dict(action_plan)
        act = cmd.action
        role = cmd.role
        name = cmd.name
        nth = cmd.nth
        val = cmd.value
        url = cmd.url

        if act == "scroll" and hasattr(self.engine, "scroll_page"):
            direction = str(val or cmd.direction or "down")
            await self.engine.scroll_page(page, direction=direction)
            return f"scroll('{direction}')"

        if act == "press" and hasattr(self.engine, "press_key"):
            key = str(val or cmd.key or "Enter")
            await self.engine.press_key(page, key=key)
            return f"press('{key}')"

        if act == "hover" and hasattr(self.engine, "hover_element"):
            await self.engine.hover_element(page, role=role, name=name, nth=nth)
            return "hover()"

        if act == "select" and val is not None and hasattr(self.engine, "select_option"):
            await self.engine.select_option(page, role=role, name=name, value=str(val), nth=nth)
            return f"select('{val}')"

        if act in ("check", "uncheck") and hasattr(self.engine, "toggle_checkbox"):
            await self.engine.toggle_checkbox(page, role=role, name=name, checked=(act == "check"), nth=nth)
            return f"{act}()"

        if act == "upload_file" and hasattr(self.engine, "upload_file"):
            files = cmd.files or val
            if files:
                await self.engine.upload_file(page, role=role, name=name, file_paths=files, nth=nth)
                return f"upload_file('{files}')"

        if act in ("fill", "fill_and_submit") and val is not None:
            submit = act == "fill_and_submit"
            masked_val = mask_sensitive_value(str(val), instruction=instruction)
            if submit:
                try:
                    await self.engine.fill_element(page, role=role, name=name, value=str(val), nth=nth, submit=True)
                except TypeError:
                    await self.engine.fill_element(page, role=role, name=name, value=str(val))
                return f"fill_and_submit('{masked_val}')"
            if nth > 0:
                await self.engine.fill_element(page, role=role, name=name, value=str(val), nth=nth)
            else:
                await self.engine.fill_element(page, role=role, name=name, value=str(val))
            return f"fill('{masked_val}')"

        if url and str(url).startswith("http"):
            valid_target = validate_url(str(url))
            await page.goto(valid_target, wait_until="domcontentloaded", timeout=15000)
            return f"goto('{valid_target}')"

        if nth > 0:
            await self.engine.click_element(page, role=role, name=name, nth=nth)
        else:
            await self.engine.click_element(page, role=role, name=name)
        return "click()"

    async def browse(
        self,
        page: Any,
        *,
        url: Optional[str] = None,
        goal: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        payload: Optional[Dict[str, Any]] = None,
        browser_context: Optional[str] = "ephemeral",
        timeout_ms: int = 20000,
    ) -> Dict[str, Any]:
        async with self._lock:
            return await self._browse_unlocked(
                page,
                url=url,
                goal=goal,
                steps=steps,
                payload=payload,
                browser_context=browser_context,
                timeout_ms=timeout_ms,
            )

    async def _browse_unlocked(
        self,
        page: Any,
        *,
        url: Optional[str] = None,
        goal: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        payload: Optional[Dict[str, Any]] = None,
        browser_context: Optional[str] = "ephemeral",
        timeout_ms: int = 20000,
    ) -> Dict[str, Any]:
        """
        Unified multi-hop execution flow supporting:
        - Mode A: Autonomous Goal Navigation on GPU (0 LLM tokens)
        - Mode B: Planned Steps execution with RouteCache self-healing and candidate calibration
        - Universal Parameter Binding for complex forms and file uploads
        """
        t_start = time.perf_counter()
        has_cache = isinstance(self.route_cache, RouteCache)

        # 1. Navigate to target URL if provided
        if url:
            validated_url = validate_url(url)
            if browser_context and browser_context != "ephemeral":
                from .config import is_cookie_extraction_enabled

                if is_cookie_extraction_enabled():
                    parsed = urllib.parse.urlparse(validated_url)
                    target_domain = parsed.hostname or validated_url
                    if hasattr(self.engine, "import_cookies_for_domain"):
                        await self.engine.import_cookies_for_domain(
                            page, domain=target_domain, browser_type=browser_context
                        )

            if str(page.url).rstrip("/") != validated_url.rstrip("/"):
                try:
                    await page.goto(validated_url, wait_until="domcontentloaded", timeout=timeout_ms)
                except Exception as nav_err:
                    if "blockedbyclient" in str(nav_err).lower():
                        raise
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                await page.wait_for_timeout(300)
                await self.engine.auto_dismiss_overlays(page)
                await self.safe_resolve_captcha(page)

        # 2. Universal Payload Binding
        payload_steps_log: List[Dict[str, Any]] = []
        if payload:
            snapshot_yaml = await page.aria_snapshot()
            all_cands = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=400)
            self.last_candidates = all_cands
            if hasattr(self.reflex, "bind_payload_to_elements"):
                bound_actions = self.reflex.bind_payload_to_elements(payload=payload, candidates=all_cands)
                for b_act in bound_actions:
                    t_b = time.perf_counter()
                    cmd = ActionCommand.from_dict(b_act)
                    action_desc = await self.dispatch_step_action(
                        page,
                        cmd,
                        instruction=f"Fill payload field: {cmd.payload_key}",
                    )
                    b_ms = round((time.perf_counter() - t_b) * 1000, 2)
                    payload_steps_log.append(
                        {
                            "step": len(payload_steps_log) + 1,
                            "instruction": f"Set payload field '{cmd.payload_key}'",
                            "executed_element": f"[{cmd.role}] {cmd.name}",
                            "action": action_desc,
                            "confidence": cmd.confidence,
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
                    "page_content_snippet": await self.extract_page_content(page),
                }

        # 3. MODE A: Autonomous Multi-Hop Goal Navigation
        if goal and not steps:
            start_url = str(page.url)
            steps_log: List[Dict[str, Any]] = list(payload_steps_log)
            total_reflex_gpu_ms = 0.0

            if has_cache and self.route_cache is not None:
                cached_dest = self.route_cache.get(start_url, f"GOAL::{goal}")
                if cached_dest and cached_dest.get("name", "").startswith("http"):
                    dest_url = validate_url(cached_dest["name"])
                    t_c = time.perf_counter()
                    try:
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
                            "page_content_snippet": await self.extract_page_content(page),
                        }
                    except Exception:
                        self.route_cache.invalidate(start_url, f"GOAL::{goal}")

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
            last_ranked_candidates: List[Dict[str, Any]] = []
            stop_reason: Optional[str] = None

            for hop in range(1, 6):
                snapshot_yaml = await page.aria_snapshot()
                current_title = await page.title()
                snapshot_yaml, current_title = await self._check_and_resolve_captcha(page, snapshot_yaml, current_title)

                if hasattr(self.reflex, "evaluate_page_satisfaction") and callable(
                    self.reflex.evaluate_page_satisfaction
                ):
                    sat_res = self.reflex.evaluate_page_satisfaction(goal, snapshot_yaml, current_title, str(page.url))
                    if isinstance(sat_res, dict):
                        total_reflex_gpu_ms += float(sat_res.get("latency_ms", 0.0))
                        last_satisfaction_rate = float(sat_res.get("goal_satisfaction_rate", 0.0))
                        last_ranked_chunks = sat_res.get("ranked_chunks")
                        if sat_res.get("is_satisfied") and not (extracted_query and not did_fill_search):
                            stop_reason = f"satisfaction_met (rate={last_satisfaction_rate:.3f})"
                            break

                if not (extracted_query and not did_fill_search) and ReflexCore.is_goal_satisfied_on_page(
                    goal, snapshot_yaml, current_title, str(page.url)
                ):
                    last_satisfaction_rate = max(last_satisfaction_rate, 0.90)
                    stop_reason = f"goal_satisfied_on_page (rate={last_satisfaction_rate:.3f})"
                    break

                all_cands = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=400)
                self.last_candidates = all_cands

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
                    stop_reason = "no_available_candidates"
                    break

                if extracted_query and not did_fill_search:
                    input_cands = [c for c in avail if c["role"] in ("searchbox", "textbox", "combobox")]
                    if input_cands:
                        avail = input_cands

                decision = self.reflex.evaluate_candidates(intent=goal, candidates=avail)
                total_reflex_gpu_ms += decision.get("latency_ms", 0.0)
                confidence = float(decision.get("confidence", 0.0))
                last_ranked_candidates = decision.get("ranked_candidates", [])

                if last_satisfaction_rate >= 0.65 and last_satisfaction_rate > confidence + 0.05:
                    stop_reason = f"satisfaction_higher_than_confidence (sat={last_satisfaction_rate:.3f}, conf={confidence:.3f})"
                    break

                if confidence < MIN_GOAL_CONFIDENCE and not (
                    extracted_query and not did_fill_search and avail[0]["role"] in ("searchbox", "textbox")
                ):
                    stop_reason = f"confidence_below_threshold (conf={confidence:.3f} < {MIN_GOAL_CONFIDENCE})"
                    break

                best = decision["ranked_candidates"][0]
                role = best["role"]
                name = best["name"]
                visited.add((role, name))

                cmd = ReflexCore.infer_action_command(
                    goal,
                    best,
                    explicit_fill_value=extracted_query if not did_fill_search else None,
                    confidence=confidence,
                )
                if cmd.action in ("fill", "fill_and_submit"):
                    did_fill_search = True

                try:
                    action_desc = await self.dispatch_step_action(page, cmd, instruction=goal)
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
                            "selection_source": "reranker",
                            "top_candidates": self.compact_candidates(decision.get("ranked_candidates", [])[:5]),
                            "status": "success",
                        }
                    )

                    if cmd.action == "fill_and_submit":
                        await page.wait_for_timeout(400)
                        continue

                    title_lower = (await page.title()).lower()
                    name_lower = name.lower()
                    cand_url = best.get("url")
                    matches_in_title = sum(1 for gw in goal_words if gw[:5] in title_lower or gw[:5] in name_lower)
                    if (
                        cand_url and str(cand_url).startswith("http") and matches_in_title >= 2
                    ) or matches_in_title >= 3:
                        stop_reason = "target_title_reached"
                        break
                except Exception as e:
                    stop_reason = f"action_dispatch_error: {e}"
                    continue

            final_url_str = str(page.url)
            if (
                has_cache
                and self.route_cache is not None
                and final_url_str != start_url
                and len(steps_log) > 0
                and not any(k in final_url_str.lower() for k in ("captcha", "sorry", "error", "login"))
            ):
                self.route_cache.store(start_url, f"GOAL::{goal}", role="url", name=final_url_str, action="goto")

            final_status = "success"
            final_message = ""
            if stop_reason and ("confidence_below_threshold" in stop_reason or stop_reason == "no_available_candidates"):
                if len(steps_log) == 0:
                    final_status = "needs_fallback"
                    if "confidence_below_threshold" in stop_reason:
                        final_message = f"Autonomous navigation stopped: {stop_reason}. Candidates confidence below threshold."
                    else:
                        final_message = "Autonomous navigation stopped: no available interactive candidates found on page."
                else:
                    final_status = "partial_needs_fallback"
                    final_message = (
                        f"Wykonano {len(steps_log)} kroków, ale dalsza autonomiczna nawigacja utknęła (powód: {stop_reason}). "
                        f"Wybierz kolejny krok z top_candidates lub użyj smart_action / get_aria_snapshot."
                    )
            elif len(steps_log) == 0 and stop_reason and "action_dispatch_error" in stop_reason:
                final_status = "error"
                final_message = f"Autonomous navigation error: {stop_reason}"

            top_cands = last_ranked_candidates[:5] if last_ranked_candidates else (self.last_candidates[:5] if self.last_candidates else [])

            return {
                "status": final_status,
                "final_url": page.url,
                "final_title": await page.title(),
                "steps_executed": len(steps_log),
                "goal_satisfaction_rate": last_satisfaction_rate,
                "stop_reason": stop_reason,
                "message": final_message,
                "selection_source": "reranker",
                "top_candidates": self.compact_candidates(top_cands),
                "telemetry": {
                    "total_execution_seconds": round(time.perf_counter() - t_start, 2),
                    "reflex_gpu_ms": round(total_reflex_gpu_ms, 2),
                    "tokens_consumed_by_server": 0,
                },
                "steps_log": steps_log,
                "page_content_snippet": await self.extract_page_content(page, ranked_chunks=last_ranked_chunks),
            }

        # 4. If neither goal nor steps provided, return compact ARIA state
        if not steps:
            snapshot_yaml = await page.aria_snapshot()
            candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=60)
            self.last_candidates = candidates
            return {
                "status": "ready_for_plan",
                "current_url": page.url,
                "current_title": await page.title(),
                "candidates_count": len(candidates),
                "candidates": self.compact_candidates(candidates),
                "aria_tree": self.clean_aria_lines(snapshot_yaml, max_lines=90),
                "message": "Page loaded. Pass `goal` or `steps` to cognitive_browse to execute autonomously on GPU.",
                "execution_seconds": round(time.perf_counter() - t_start, 2),
            }

        # 5. MODE B: Explicit Step-by-Step Execution
        steps_log = list(payload_steps_log)
        total_reflex_gpu_ms = 0.0

        for idx, s in enumerate(steps):
            instruction = s.get("instruction", "")
            fill_val = s.get("fill_value")
            explicit_action = s.get("action")
            current_page_url = str(page.url)

            if explicit_action in ("scroll", "press"):
                t_act = time.perf_counter()
                cmd = ActionCommand(
                    action=explicit_action,
                    value=fill_val or ("down" if explicit_action == "scroll" else "Enter"),
                    source="reflex_direct",
                )
                action_desc = await self.dispatch_step_action(page, cmd, instruction=instruction)
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

            # RouteCache fast path with automatic self-healing on failure
            if has_cache and self.route_cache is not None:
                cached = self.route_cache.get(current_page_url, instruction)
                if cached:
                    c_role = cached["role"]
                    c_name = cached["name"]
                    c_nth = cached.get("nth", 0)
                    c_act = explicit_action or cached.get("action") or ("fill" if fill_val else "click")
                    t_cache = time.perf_counter()
                    try:
                        cmd = ActionCommand(
                            action=c_act,
                            role=c_role,
                            name=c_name,
                            nth=c_nth,
                            value=fill_val,
                            source="route_cache",
                        )
                        action_desc = await self.dispatch_step_action(page, cmd, instruction=instruction)
                        cache_ms = round((time.perf_counter() - t_cache) * 1000, 2)
                        await page.wait_for_timeout(250)
                        self.route_cache.store(
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
                        # Self-healing: evict stale route and fall through to neural evaluation
                        self.route_cache.invalidate(current_page_url, instruction)

            snapshot_yaml = await page.aria_snapshot()
            snapshot_yaml, _ = await self._check_and_resolve_captcha(page, snapshot_yaml, await page.title())

            candidates = AriaElementParser.parse_snapshot(
                snapshot_yaml, max_candidates=400, require_input_only=bool(fill_val)
            )
            self.last_candidates = candidates

            if not candidates:
                return {
                    "status": "execution_error",
                    "failed_step": idx + 1,
                    "instruction": instruction,
                    "error": "No interactive candidate elements found on page.",
                    "current_url": page.url,
                    "steps_log": steps_log,
                }

            decision = self.reflex.evaluate_candidates(intent=instruction, candidates=candidates)
            total_reflex_gpu_ms += decision.get("latency_ms", 0.0)

            if decision.get("should_escalate") or decision.get("target_ref") == "NONE":
                top_candidates = self.compact_candidates(decision.get("ranked_candidates", [])[:5])
                remaining_steps = [s.get("instruction") for s in steps[idx:]]
                return {
                    "status": "clarification_needed",
                    "current_url": page.url,
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

            best = decision["ranked_candidates"][0]
            role = best["role"]
            name = best["name"]
            nth = best.get("nth", 0)

            if explicit_action:
                cmd = ActionCommand(
                    action=explicit_action,
                    role=role,
                    name=name,
                    nth=nth,
                    value=fill_val if fill_val is not None else ReflexCore.extract_fill_value(instruction),
                    confidence=decision["confidence"],
                )
            elif fill_val is not None:
                cmd = ActionCommand(
                    action="fill",
                    role=role,
                    name=name,
                    nth=nth,
                    value=fill_val,
                    confidence=decision["confidence"],
                )
            else:
                cmd = ReflexCore.infer_action_command(instruction, best, confidence=decision["confidence"])
                cmd.url = None

            try:
                action_desc = await self.dispatch_step_action(page, cmd, instruction=instruction)

                if has_cache and self.route_cache is not None:
                    self.route_cache.store(
                        current_page_url,
                        instruction,
                        role=role,
                        name=name,
                        nth=nth,
                        action=cmd.action,
                        context=best.get("context", ""),
                    )

                await page.wait_for_timeout(300)
                c_sol = await self.safe_resolve_captcha(page)
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
            "page_content_snippet": await self.extract_page_content(page),
        }

    async def smart_action(
        self,
        page: Any,
        *,
        instruction: str,
        target_ref: Optional[str] = None,
        target_name: Optional[str] = None,
        fill_value: Optional[str] = None,
        file_paths: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Executes a single directed or escalated action on page."""
        t_start = time.perf_counter()
        await self.engine.auto_dismiss_overlays(page)

        files_list = [file_paths] if isinstance(file_paths, str) else (list(file_paths) if file_paths else None)

        if target_ref or target_name:
            chosen_candidate = None
            if target_ref and self.last_candidates:
                chosen_candidate = next((c for c in self.last_candidates if c.get("ref") == target_ref), None)

            if not chosen_candidate and target_name and self.last_candidates:
                clean_tname = (
                    re.sub(r"[\ue000-\uf8ff\uf000-\uffff]|\\u[e-fE-F][0-9a-fA-F]{3}", "", target_name).strip().lower()
                )
                for c in self.last_candidates:
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

            cmd = ReflexCore.infer_action_command(
                instruction, {"role": role, "name": name, "nth": nth}, explicit_fill_value=fill_value
            )
            if files_list is not None:
                cmd.action = "upload_file"
                cmd.files = files_list
            elif fill_value is not None and cmd.action not in ("fill", "fill_and_submit", "select"):
                cmd.action = "fill"
                cmd.value = fill_value

            action_desc = await self.dispatch_step_action(page, cmd, instruction=instruction)
            await page.wait_for_timeout(300)
            return {
                "status": "success",
                "action": action_desc,
                "target": f"[{role}] {name}",
                "selection_source": "llm_target_name",
                "top_candidates": self.compact_candidates(self.last_candidates[:5]) if self.last_candidates else [],
                "current_url": page.url,
                "current_title": await page.title(),
                "execution_seconds": round(time.perf_counter() - t_start, 2),
            }

        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(
            snapshot_yaml, max_candidates=150, require_input_only=bool(fill_value)
        )
        self.last_candidates = candidates

        decision = self.reflex.evaluate_candidates(intent=instruction, candidates=candidates)
        if decision.get("should_escalate") or decision.get("target_ref") == "NONE":
            return {
                "status": "clarification_needed",
                "current_url": page.url,
                "pending_instruction": instruction,
                "selection_source": "reranker",
                "confidence": decision.get("confidence", 0.0),
                "top_candidates": self.compact_candidates(decision.get("ranked_candidates", [])[:5]),
            }

        best = decision["ranked_candidates"][0]
        role = best["role"]
        name = best["name"]

        cmd = ReflexCore.infer_action_command(
            instruction, best, explicit_fill_value=fill_value, confidence=decision["confidence"]
        )
        if files_list is not None:
            cmd.action = "upload_file"
            cmd.files = files_list
        elif fill_value is not None and cmd.action not in ("fill", "fill_and_submit", "select"):
            cmd.action = "fill"
            cmd.value = fill_value

        action_desc = await self.dispatch_step_action(page, cmd, instruction=instruction)
        await page.wait_for_timeout(300)
        return {
            "status": "success",
            "action": action_desc,
            "target": f"[{role}] {name}",
            "selection_source": "reranker",
            "confidence": decision["confidence"],
            "latency_ms": decision["latency_ms"],
            "top_candidates": self.compact_candidates(decision.get("ranked_candidates", [])[:5]),
            "current_url": page.url,
            "current_title": await page.title(),
            "execution_seconds": round(time.perf_counter() - t_start, 2),
        }

    async def get_aria_snapshot(
        self,
        page: Any,
        *,
        url: Optional[str] = None,
        max_candidates: int = 80,
        selector: str = "body",
        include_aria_tree: bool = True,
        max_tree_lines: int = 120,
    ) -> Dict[str, Any]:
        """Captures page ARIA state with semantic filtering."""
        async with self._lock:
            return await self._get_aria_snapshot_unlocked(
                page,
                url=url,
                max_candidates=max_candidates,
                selector=selector,
                include_aria_tree=include_aria_tree,
                max_tree_lines=max_tree_lines,
            )

    async def _get_aria_snapshot_unlocked(
        self,
        page: Any,
        *,
        url: Optional[str] = None,
        max_candidates: int = 80,
        selector: str = "body",
        include_aria_tree: bool = True,
        max_tree_lines: int = 120,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        if url:
            valid_url = validate_url(url)
            if str(page.url).rstrip("/") != valid_url.rstrip("/"):
                try:
                    await page.goto(valid_url, wait_until="domcontentloaded", timeout=15000)
                except Exception as nav_err:
                    if "blockedbyclient" in str(nav_err).lower():
                        raise
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                await page.wait_for_timeout(250)
                await self.engine.auto_dismiss_overlays(page)

        loc = page.locator(selector)
        if await loc.count() == 0:
            loc = page.locator("body")

        snapshot_yaml = await loc.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=max_candidates)
        self.last_candidates = candidates

        shown_candidates = candidates[:15] if include_aria_tree else candidates
        result: Dict[str, Any] = {
            "current_url": page.url,
            "current_title": await page.title(),
            "candidates_count": len(candidates),
            "candidates": self.compact_candidates(shown_candidates),
            "execution_seconds": round(time.perf_counter() - t_start, 2),
        }
        if include_aria_tree:
            eff_lines = min(max(int(max_tree_lines or 120), 20), 600)
            result["aria_tree"] = self.clean_aria_lines(
                snapshot_yaml, max_lines=eff_lines, max_chars=max(5200, eff_lines * 45)
            )
        return result
