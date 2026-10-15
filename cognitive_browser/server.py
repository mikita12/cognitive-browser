"""
Cognitive Browser Microservice (System 1 + Playwright + DeBERTa-v3).
Uniwersalny kontenerowy 'klocek' udostępniający lokalny silnik decyzyjny
zarówno przez REST API, jak i protokół MCP dla agentów AI.
"""

import os
import time
import asyncio
import torch
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, Browser, BrowserContext

try:
    from .deberta_router import DebertaDecisionRouter
    from .agent_core import AgentCore, AriaElementParser
except ImportError:
    from deberta_router import DebertaDecisionRouter
    from agent_core import AgentCore, AriaElementParser

# Konfiguracja środowiska
MODEL_NAME = os.getenv("MODEL_NAME", "cross-encoder/nli-deberta-v3-small")
DEVICE = os.getenv(
    "DEVICE",
    "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)
PORT = int(os.getenv("PORT", "8000"))
SESSION_DIR = os.getenv("SESSION_DIR", "/data/browser_sessions")

os.makedirs(SESSION_DIR, exist_ok=True)

# Globalne instancje
router_instance: Optional[DebertaDecisionRouter] = None
agent_instance: Optional[AgentCore] = None
playwright_manager = None
browser_instance: Optional[Browser] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_instance, agent_instance, playwright_manager, browser_instance
    print(f"\n🚀 [CognitiveBrowser] Uruchamianie serwera na urządzeniu: {DEVICE.upper()}...")
    
    # Inicjalizacja modelu NLI
    router_instance = DebertaDecisionRouter(model_name=MODEL_NAME, device=DEVICE)
    agent_instance = AgentCore(model_name=MODEL_NAME, device=DEVICE)
    
    # Inicjalizacja Playwrighta
    playwright_manager = await async_playwright().start()
    browser_instance = await playwright_manager.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
    )
    print("✅ [CognitiveBrowser] Gotowy do przyjmowania zadań!\n")
    yield
    
    # Cleanup
    if browser_instance:
        await browser_instance.close()
    if playwright_manager:
        await playwright_manager.stop()
    print("🛑 [CognitiveBrowser] Serwer zatrzymany.")


app = FastAPI(
    title="Cognitive Browser Decision Microservice",
    description="Ultraszybki, lokalny router System 1 do automatyzacji Playwrighta na GPU",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modele danych Pydantic
class EvaluateRequest(BaseModel):
    intent: str = Field(..., description="Cel lub intencja użytkownika w języku naturalnym")
    candidates: List[Dict[str, Any]] = Field(..., description="Lista kandydatów ARIA ze snapshotu")
    confidence_threshold: float = Field(0.50, description="Próg pewności NLI")


class StepInstruction(BaseModel):
    instruction: str
    fill_value: Optional[str] = None


class BrowseTaskRequest(BaseModel):
    url: str
    steps: Optional[List[StepInstruction]] = None
    goal: Optional[str] = None
    session_id: Optional[str] = "default"
    timeout_ms: int = 20000


class SnapshotRequest(BaseModel):
    url: str
    max_candidates: int = 50
    wait_time_ms: int = 1500


# Endpointy API
@app.get("/health")
async def health_check():
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None (CPU Mode)"
    vram_alloc = round(torch.cuda.memory_allocated(0) / (1024**2), 2) if cuda_ok else 0
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "vram_allocated_mb": vram_alloc,
        "model": MODEL_NAME,
        "playwright_ready": browser_instance is not None
    }


@app.post("/v1/evaluate")
async def evaluate_candidates(req: EvaluateRequest):
    """Błyskawiczne wnioskowanie NLI (System 1) na przekazanych elementach."""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Model NLI nie został załadowany")
    
    t0 = time.perf_counter()
    decision = router_instance.evaluate_candidates(
        intent=req.intent,
        candidates=req.candidates,
        confidence_threshold=req.confidence_threshold
    )
    decision["endpoint_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return decision


@app.post("/v1/snapshot")
async def take_aria_snapshot(req: SnapshotRequest):
    """Pobiera stronę i zwraca czyste semantyczne drzewo ARIA oraz listę elementów."""
    if not browser_instance:
        raise HTTPException(status_code=503, detail="Przeglądarka nie jest gotowa")
        
    t0 = time.perf_counter()
    page = await browser_instance.new_page()
    try:
        await page.goto(req.url, wait_until="domcontentloaded", timeout=req.wait_time_ms + 10000)
        await page.wait_for_timeout(req.wait_time_ms)
        
        # Auto-dismiss cookies jeśli obecne
        if agent_instance:
            await agent_instance.auto_dismiss_overlays(page)
            
        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=req.max_candidates)
        
        return {
            "url": page.url,
            "title": await page.title(),
            "raw_snapshot": snapshot_yaml,
            "candidates_count": len(candidates),
            "candidates": candidates,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
        }
    finally:
        await page.close()


@app.post("/v1/browse")
async def execute_browse_task(req: BrowseTaskRequest):
    """
    Wielokrokowe wykonanie zadania w przeglądarce przy użyciu System 1.
    Obsługuje sekwencję kroków lub cel wysokopoziomowy.
    """
    if not browser_instance or not agent_instance:
        raise HTTPException(status_code=503, detail="Silnik nie jest gotowy")
        
    t0 = time.perf_counter()
    page = await browser_instance.new_page()
    steps_log = []
    
    try:
        print(f"\n[BrowseTask] Otwieram: {req.url}")
        await page.goto(req.url, wait_until="domcontentloaded", timeout=req.timeout_ms)
        await page.wait_for_timeout(1000)
        
        # Krok 0: Automatyczne zamykanie modali RODO/cookies
        dismissed = await agent_instance.auto_dismiss_overlays(page)
        if dismissed:
            steps_log.append({"step": "auto_dismiss_cookies", "status": "dismissed"})
            
        # Wykonanie zdefiniowanych kroków
        if req.steps:
            for s in req.steps:
                print(f"  -> Krok: \"{s.instruction}\" (fill: {s.fill_value})")
                res = await agent_instance.execute_step(
                    page=page,
                    instruction=s.instruction,
                    fill_value=s.fill_value
                )
                steps_log.append(res)
                if not res.get("executed"):
                    print(f"  ⚠️ Zatrzymanie na kroku: {res.get('error')}")
                    break
                    
        # Aktualny stan po wykonaniu
        current_snapshot = await page.aria_snapshot()
        return {
            "status": "success",
            "initial_url": req.url,
            "final_url": page.url,
            "final_title": await page.title(),
            "steps_executed": len(steps_log),
            "steps_log": steps_log,
            "final_aria_snapshot": current_snapshot[:1500] + "... (truncated)",
            "total_task_time_ms": round((time.perf_counter() - t0) * 1000, 2)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "steps_log": steps_log,
            "total_task_time_ms": round((time.perf_counter() - t0) * 1000, 2)
        }
    finally:
        await page.close()


# Narzędzia MCP w formacie JSON (kompatybilne ze specyfikacją Model Context Protocol)
@app.get("/v1/mcp/tools")
async def list_mcp_tools():
    """Zwraca listę narzędzi MCP udostępnianych przez ten mikroserwis."""
    return {
        "tools": [
            {
                "name": "cognitive_browse",
                "description": "Autonomiczna nawigacja po stronie WWW na bazie Systemu 1 (DeBERTa-v3) z zerowym zużyciem tokenów podczas klikania.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Adres strony WWW"},
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "instruction": {"type": "string", "description": "Intencja/akcja do wykonania"},
                                    "fill_value": {"type": "string", "description": "Opcjonalna wartość do wpisania w pole"}
                                },
                                "required": ["instruction"]
                            }
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "get_aria_snapshot",
                "description": "Pobiera czyste drzewo dostępności (ARIA YAML) strony WWW z pominięciem 98% szumu HTML.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Adres strony WWW"}
                    },
                    "required": ["url"]
                }
            }
        ]
    }


def main():
    import uvicorn
    uvicorn.run("cognitive_browser.server:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()
