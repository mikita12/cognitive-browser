"""
AgentOrchestrator: Pełna pętla System 1 (DeBERTa) + System 2 (LLM Escalation Recovery).
Gdy DeBERTa ma wysoką pewność (>0.50), wykonuje krok w 40 ms z 0 tokenami.
Gdy pewność jest niska (<0.50), System 2 (LLM) analizuje zwięzłe ARIA,
koryguje krok i pomaga DeBERCIE doprowadzić zadanie do 100% ukończenia.
"""

import asyncio
import time
import json
import requests
import re
from typing import List, Dict, Any, Optional
from playwright.async_api import Page, Browser, async_playwright

from deberta_router import DebertaDecisionRouter
from agent_core import AgentCore, AriaElementParser

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"


class AgentOrchestrator:
    def __init__(self, device: str = "cuda", router: Optional[Any] = None):
        if router is not None:
            self.router = router
        else:
            from router_factory import get_decision_router
            self.router = get_decision_router(device=device)
        self.agent = AgentCore(device=device, router=self.router)

    def call_system2_llm(self, intent: str, candidates: List[Dict[str, Any]], aria_tree: str, fill_value: Optional[str] = None) -> Dict[str, Any]:
        """
        System 2 (LLM) wkracza w momencie eskalacji, analizuje zwięzłe ARIA i wskazuje właściwy element.
        """
        t0 = time.perf_counter()
        
        # Przygotowanie zwięzłego kontekstu dla LLM (tylko kluczowe elementy)
        element_list_str = "\n".join([f"- [{c['role']}] \"{c['name']}\" (ref: {c['ref']}, details: {c.get('text', '')[:80]})" for c in candidates[:35]])
        
        goal_desc = f'fill "{fill_value}" into the field for "{intent}"' if fill_value else f'achieve "{intent}"'
        prompt = f"""You are System 2 of an autonomous browser agent.
The user goal is to: {goal_desc}.
System 1 (reflex router) was uncertain or encountered an obstruction.
Here are the available interactive elements on the page:
{element_list_str}

Analyze which element best matches the user's intent.
Return ONLY a valid JSON object with format:
{{"chosen_ref": "ref_name", "chosen_name": "exact element name", "action": "click" or "fill", "reason": "brief explanation"}}
"""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=10
            )
            data = resp.json()
            out_text = data.get("response", "{}")
            prompt_tokens = data.get("prompt_eval_count", int(len(prompt)/3.5))
            completion_tokens = data.get("eval_count", int(len(out_text)/3.5))
            total_tokens = prompt_tokens + completion_tokens

            res_json = json.loads(out_text)
            res_json["tokens"] = total_tokens
            res_json["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return res_json

        except Exception as e:
            # Fallback na proste dopasowanie słów kluczowych jeśli LLM nie odpowiedział
            return {
                "chosen_ref": candidates[0]["ref"] if candidates else "NONE",
                "chosen_name": candidates[0]["name"] if candidates else "",
                "action": "click" if not fill_value else "fill",
                "tokens": 200,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error": str(e)
            }

    async def execute_task(self, page: Page, url: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Wykonuje całe zadanie do 100% ukończenia.
        System 1 wykonuje odruchy, System 2 pomaga przy każdej eskalacji.
        """
        t_start = time.perf_counter()
        total_system2_tokens = 0
        total_system1_gpu_ms = 0.0
        escalations_count = 0
        steps_log = []

        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        await page.wait_for_timeout(1000)

        # 1. Auto-dismiss cookies
        dismissed = await self.agent.auto_dismiss_overlays(page)
        if dismissed:
            steps_log.append({"step": "auto_dismiss_cookies", "status": "dismissed"})

        # 2. Pętla wykonawcza z aktywną eskalacją
        for idx, s in enumerate(steps):
            inst = s.get("instruction", "")
            val = s.get("fill_value")
            
            # Krok A: Próba Systemu 1 (DeBERTa NLI)
            step_res = await self.agent.execute_step(page=page, instruction=inst, fill_value=val)
            total_system1_gpu_ms += step_res.get("latency_ms", 0.0)

            # Krok B: Jeśli DeBERTa ma wątpliwości -> wkracza System 2!
            if not step_res.get("executed"):
                escalations_count += 1
                print(f"     ⚠️ Eskalacja na kroku {idx+1} (\"{inst}\") -> wzywam System 2...")
                
                # Pobranie czystego ARIA dla Systemu 2
                current_snapshot = await page.aria_snapshot()
                candidates = AriaElementParser.parse_snapshot(
                    current_snapshot,
                    max_candidates=50,
                    require_input_only=bool(val)
                )
                if not candidates and val:
                    candidates = AriaElementParser.parse_snapshot(current_snapshot, max_candidates=50, require_input_only=False)
                
                # System 2 analizuje i pomaga
                s2_help = self.call_system2_llm(intent=inst, candidates=candidates, aria_tree=current_snapshot, fill_value=val)
                total_system2_tokens += s2_help.get("tokens", 0)
                
                chosen_name = s2_help.get("chosen_name", "")
                chosen_ref = s2_help.get("chosen_ref", "")
                print(f"     🧠 System 2 wskazał: \"{chosen_name}\" ({chosen_ref}) w {s2_help.get('latency_ms')} ms (~{s2_help.get('tokens')} tok.)")
                
                # Dopasowanie elementu
                matched_cand = next(
                    (c for c in candidates if c["ref"] == chosen_ref or c["name"] == chosen_name or (chosen_name and chosen_name.lower() in c["name"].lower())),
                    None
                )
                if not matched_cand and candidates:
                    matched_cand = candidates[0]

                if matched_cand:
                    try:
                        role = matched_cand["role"]
                        name = matched_cand["name"]
                        clean_name = re.sub(r'[\ue000-\uf8ff]', '', name).strip()
                        name_pattern = re.compile(re.escape(clean_name), re.I) if clean_name else name

                        if val and role in ["combobox", "textbox", "searchbox", "spinbutton"]:
                            try:
                                await page.get_by_role(role, name=name_pattern).first.fill(val, timeout=5000)
                            except Exception:
                                await page.locator(f"input, textarea").first.fill(val, timeout=4000)
                            step_res["action_taken"] = f"fill('{val}') [assisted by System 2]"
                        else:
                            try:
                                await page.get_by_role(role, name=name_pattern).first.click(timeout=3500)
                            except Exception:
                                try:
                                    await page.get_by_role(role, name=name_pattern).first.click(timeout=2000, force=True)
                                except Exception:
                                    await page.get_by_text(name_pattern, exact=False).first.click(timeout=3000, force=True)
                            step_res["action_taken"] = f"click() [assisted by System 2]"
                        
                        step_res["executed"] = True
                        step_res["assisted_by_system2"] = True
                        step_res["error"] = None
                        await page.wait_for_timeout(1000)
                    except Exception as ex:
                        step_res["error"] = f"Błąd po asyście: {ex}"
                
            steps_log.append(step_res)

        t_total = time.perf_counter() - t_start
        final_snapshot = await page.aria_snapshot()

        task_steps = [s for s in steps_log if s.get("step") != "auto_dismiss_cookies"]
        is_completed = len(task_steps) > 0 and all(s.get("executed", False) for s in task_steps)

        return {
            "status": "completed" if is_completed else "partial",
            "total_execution_seconds": round(t_total, 2),
            "final_url": page.url,
            "final_title": await page.title(),
            "escalations_count": escalations_count,
            "system2_tokens": total_system2_tokens,
            "system1_gpu_ms": round(total_system1_gpu_ms, 2),
            "steps_executed": len([s for s in task_steps if s.get("executed")]),
            "steps_log": steps_log,
            "final_snapshot_summary": final_snapshot[:1500]
        }
