"""
Cognitive Browser Microservice (v0.3.0).
Universal containerized microservice exposing the local decision engine (ModernBERT/Reflex Core)
via both REST API and Model Context Protocol (MCP) for AI agents with automated CAPTCHA resolution.
"""

import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from playwright.async_api import Browser, async_playwright
from pydantic import BaseModel, Field

from .browser import BrowserEngine
from .config import get_reflex_model
from .orchestrator import BrowserOrchestrator
from .reflex import AriaElementParser, ReflexCore
from .security import SecurityError, setup_ssrf_route_guard, validate_url

# Environment configuration
MODEL_NAME = get_reflex_model()
DEVICE = os.getenv(
    "DEVICE",
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
)
PORT = int(os.getenv("PORT", "8000"))
API_TOKEN = os.getenv("API_TOKEN", "").strip()
SESSION_DIR = os.getenv("SESSION_DIR")
if not SESSION_DIR:
    SESSION_DIR = (
        "/data/browser_sessions"
        if os.path.exists("/data") and os.access("/data", os.W_OK)
        else os.path.expanduser("~/.cognitive_browser/sessions")
    )

try:
    os.makedirs(SESSION_DIR, exist_ok=True)
except Exception:
    SESSION_DIR = "/tmp/cognitive_browser_sessions"
    os.makedirs(SESSION_DIR, exist_ok=True)

# Global instances
router_instance: Optional[ReflexCore] = None
engine_instance: Optional[BrowserEngine] = None
playwright_manager = None
browser_instance: Optional[Browser] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_instance, engine_instance, playwright_manager, browser_instance
    print(f"\n🚀 [CognitiveBrowser] Starting server on device: {DEVICE.upper()}...")

    if not os.getenv("API_TOKEN", "").strip():
        print("⚠️  SECURITY WARNING: No API_TOKEN configured. REST API and SSE endpoints are unauthenticated.")
        print("    Do NOT expose port 8000 to public networks without setting API_TOKEN!")
    else:
        print("🔒 [CognitiveBrowser] API token authentication enabled.")

    # Initialize NLI model
    router_instance = ReflexCore(model_name=MODEL_NAME, device=DEVICE)
    engine_instance = BrowserEngine(reflex=router_instance)

    # Initialize Playwright
    playwright_manager = await async_playwright().start()
    launch_args = ["--disable-dev-shm-usage"]
    if os.getenv("COGNITIVE_BROWSER_NO_SANDBOX", "0").strip() in ("1", "true", "yes"):
        launch_args.extend(["--no-sandbox", "--disable-setuid-sandbox"])

    browser_instance = await playwright_manager.chromium.launch(
        headless=True,
        args=launch_args,
    )
    print("✅ [CognitiveBrowser] Ready to accept tasks!\n")
    yield

    # Cleanup
    if browser_instance:
        await browser_instance.close()
    if playwright_manager:
        await playwright_manager.stop()
    print("🛑 [CognitiveBrowser] Server stopped.")


app = FastAPI(
    title="Cognitive Browser Decision Microservice",
    description="Ultra-fast local Reflex Core router for Playwright browser automation on GPU",
    version="1.0.0",
    lifespan=lifespan,
)

cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()] if cors_env else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def verify_api_token_middleware(request: Request, call_next):
    token_configured = os.getenv("API_TOKEN", "").strip()
    if token_configured:
        # Public health check and schema documentation
        if request.url.path in ("/health", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

        if not token or not secrets.compare_digest(token, token_configured):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: Missing or invalid API token."},
            )

    return await call_next(request)


# Pydantic schemas
class EvaluateRequest(BaseModel):
    intent: str = Field(..., description="Natural language intent or user goal")
    candidates: List[Dict[str, Any]] = Field(..., description="List of ARIA snapshot candidate elements")
    confidence_threshold: float = Field(0.50, description="NLI confidence threshold")


class StepInstruction(BaseModel):
    instruction: str
    fill_value: Optional[str] = None


class BindPayloadRequest(BaseModel):
    payload: Dict[str, Any] = Field(..., description="Key-value pairs to bind to form elements")
    candidates: List[Dict[str, Any]] = Field(..., description="ARIA snapshot candidates")
    min_confidence: float = Field(0.30, description="Minimum confidence threshold")


class BrowseTaskRequest(BaseModel):
    url: str
    steps: Optional[List[StepInstruction]] = None
    goal: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    browser_context: Optional[str] = "ephemeral"
    session_id: Optional[str] = "default"
    timeout_ms: int = 20000


class SnapshotRequest(BaseModel):
    url: str
    max_candidates: int = 50
    wait_time_ms: int = 1500


# API endpoints
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
        "playwright_ready": browser_instance is not None,
    }


@app.post("/v1/evaluate")
async def evaluate_candidates(req: EvaluateRequest):
    """Fast NLI inference (Reflex Core) on provided candidate elements."""
    if not router_instance:
        raise HTTPException(status_code=503, detail="NLI model is not initialized")

    t0 = time.perf_counter()
    decision = router_instance.evaluate_candidates(
        intent=req.intent, candidates=req.candidates, confidence_threshold=req.confidence_threshold
    )
    decision["endpoint_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return decision


@app.post("/v1/bind_payload")
async def bind_payload(req: BindPayloadRequest):
    """Neural semantic bipartite matching of user payload to form elements."""
    if not router_instance:
        raise HTTPException(status_code=503, detail="NLI model is not initialized")

    t0 = time.perf_counter()
    actions = router_instance.bind_payload_to_elements(
        payload=req.payload, candidates=req.candidates, min_confidence=req.min_confidence
    )
    return {
        "matched_actions": actions,
        "actions_count": len(actions),
        "endpoint_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.post("/v1/snapshot")
async def take_aria_snapshot(req: SnapshotRequest):
    """Navigates to URL and returns clean semantic ARIA accessibility tree, candidates, and CAPTCHA state."""
    try:
        clean_url = validate_url(req.url)
    except SecurityError as se:
        raise HTTPException(status_code=400, detail=str(se)) from se

    if not browser_instance:
        raise HTTPException(status_code=503, detail="Browser instance is not ready")

    t0 = time.perf_counter()
    page = await browser_instance.new_page(service_workers="block")
    await setup_ssrf_route_guard(page.context)
    try:
        await page.goto(clean_url, wait_until="domcontentloaded", timeout=req.wait_time_ms + 10000)
        await page.wait_for_timeout(req.wait_time_ms)

        # Auto-dismiss cookie overlays if present
        if engine_instance:
            await engine_instance.auto_dismiss_overlays(page)

        snapshot_yaml = await page.aria_snapshot()
        candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=req.max_candidates)

        # Evaluate CAPTCHA state & auto-resolve if present
        captcha_state: Optional[Dict[str, Any]] = None
        if router_instance and hasattr(router_instance, "evaluate_captcha_presence"):
            captcha_state = router_instance.evaluate_captcha_presence(
                candidates=candidates,
                page_title=await page.title(),
                page_url=page.url,
                snapshot_yaml=snapshot_yaml,
            )
            if captcha_state.get("is_captcha") and engine_instance:
                resolve_res = await engine_instance.auto_resolve_captcha_if_present(page)
                captcha_state["resolve_result"] = resolve_res
                if resolve_res.get("solved"):
                    await page.wait_for_timeout(1000)
                    snapshot_yaml = await page.aria_snapshot()
                    candidates = AriaElementParser.parse_snapshot(snapshot_yaml, max_candidates=req.max_candidates)

        return {
            "url": page.url,
            "title": await page.title(),
            "raw_snapshot": snapshot_yaml,
            "candidates_count": len(candidates),
            "candidates": candidates,
            "captcha_state": captcha_state,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
    finally:
        await page.close()


@app.post("/v1/browse")
async def execute_browse_task(req: BrowseTaskRequest):
    """
    Multi-step browser execution using Reflex Core decision engine.
    Supports sequences of instructions, high-level goals, and automated CAPTCHA resolution.
    Delegates multi-hop execution loop to BrowserOrchestrator.
    """
    try:
        clean_url = validate_url(req.url)
    except SecurityError as se:
        raise HTTPException(status_code=400, detail=str(se)) from se

    if not browser_instance or not engine_instance or not router_instance:
        raise HTTPException(status_code=503, detail="Browser or agent engine is not ready")

    t0 = time.perf_counter()
    page = await browser_instance.new_page(service_workers="block")
    await setup_ssrf_route_guard(page.context)

    try:
        dict_steps = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in req.steps] if req.steps else None
        orchestrator = BrowserOrchestrator(engine=engine_instance, reflex=router_instance)
        res = await orchestrator.browse(
            page,
            url=clean_url,
            goal=req.goal,
            steps=dict_steps,
            payload=req.payload,
            browser_context=req.browser_context,
            timeout_ms=req.timeout_ms,
        )

        final_snapshot = await page.aria_snapshot()
        return {
            "status": res.get("status", "success"),
            "initial_url": req.url,
            "final_url": res.get("final_url", page.url),
            "final_title": res.get("final_title", await page.title()),
            "captcha_state": None,
            "steps_executed": res.get("steps_executed", len(res.get("steps_log", []))),
            "steps_log": res.get("steps_log", []),
            "final_aria_snapshot": final_snapshot[:1500] + "... (truncated)",
            "page_content_snippet": res.get("page_content_snippet", ""),
            "total_task_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
    except SecurityError as se:
        raise HTTPException(status_code=400, detail=str(se)) from se
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "captcha_state": None,
            "steps_log": [],
            "total_task_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
    finally:
        await page.close()


# MCP tools schema (compatible with Model Context Protocol specification)
@app.get("/v1/mcp/tools")
async def list_mcp_tools():
    """Returns the list of MCP tools exposed by this microservice."""
    return {
        "tools": [
            {
                "name": "cognitive_browse",
                "description": "Autonomous web browsing powered by Reflex Core (DeBERTa-v3) with 0 token consumption during interaction.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Target web page URL"},
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "instruction": {
                                        "type": "string",
                                        "description": "User intent or action to execute",
                                    },
                                    "fill_value": {
                                        "type": "string",
                                        "description": "Optional string value to fill into field",
                                    },
                                },
                                "required": ["instruction"],
                            },
                        },
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "get_aria_snapshot",
                "description": "Extracts clean semantic ARIA accessibility tree from web page, removing ~98% of HTML noise.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "Target web page URL"}},
                    "required": ["url"],
                },
            },
        ]
    }


def main():
    import uvicorn

    uvicorn.run("cognitive_browser.server:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()
