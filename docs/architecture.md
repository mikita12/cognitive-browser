# Cognitive Browser - System Architecture & Engineering Guide

## 1. Executive Summary & Design Philosophy

**Cognitive Browser** is a high-speed, zero-token browser automation system designed for autonomous AI agents. Unlike traditional LLM-based web browsing approaches that stream multi-megabyte raw HTML snapshots into external LLM context windows (costing tokens, latency, and losing spatial coherence), Cognitive Browser executes navigation loops entirely locally on GPU:

1. **Zero-Token Autonomous Navigation:** The agent issues a single high-level intent (e.g. `cognitive_browse(url="...", goal="Find contact email and pricing")`).
2. **Local ModernBERT / DeBERTa-v3 Cross-Encoder (Reflex Core):** Evaluates semantic relevance between intent and candidate interactive elements directly on local GPU in milliseconds (~3.3 ms per element; 1000 elements processed in under 3.5s).
3. **RouteCache:** Caches deterministic execution paths in SQLite; recurring tasks complete with 0ms neural latency.
4. **Stealth & Bot Resistance:** Integrates passive CAPTCHA heuristics, CapSolver automated resolution, and authenticated Firefox/System cookie synchronization.
5. **Agent-in-the-Loop Escalation:** Escalates back to the orchestrating LLM *only* when local confidence falls below calibrated thresholds or explicit ambiguity is detected.

---

## 2. Component Pipeline & Architecture Diagram

```mermaid
flowchart TD
    subgraph ClientInterfaces ["Client Interfaces"]
        MCP["MCP Client (Claude / Antigravity / Cursor)"]
        REST["REST API Clients (FastAPI :8000)"]
    end

    subgraph EntryAdapters ["Interface Adapters"]
        MCPServer["cognitive_browser.mcp_server<br/>(FastMCP Tools)"]
        FastAPIServer["cognitive_browser.server<br/>(REST Endpoints)"]
    end

    subgraph OrchestrationLayer ["Orchestration Engine"]
        Orchestrator["cognitive_browser.orchestrator<br/><b>BrowserOrchestrator</b><br/>• Multi-hop navigation (up to 5 hops)<br/>• State vs Action satisfaction competition<br/>• Fallback & confidence braking"]
    end

    subgraph DomainModules ["Specialized Domain Modules"]
        Parser["cognitive_browser.parser<br/><b>AriaElementParser</b><br/>• ARIA snapshot parser<br/>• Section flags (is_main, is_footer)<br/>• Uncapped candidate scaling"]
        Payload["cognitive_browser.payload<br/><b>PayloadBinder</b><br/>• Neural parameter binding<br/>• Multi-file uploads<br/>• Combobox & checkbox matching"]
        Captcha["cognitive_browser.captcha<br/><b>CaptchaResolver</b><br/>• Heuristic presence evaluation<br/>• Turnstile & reCAPTCHA v2 solving<br/>• CapSolver integration"]
        Cache["cognitive_browser.cache<br/><b>RouteCache</b><br/>• SQLite path cache<br/>• Goal & Step direct jumps"]
        Reflex["cognitive_browser.reflex<br/><b>ReflexCore</b><br/>• ModernBERT / DeBERTa NLI Cross-Encoder<br/>• GPU mini-batching<br/>• Goal satisfaction score"]
        Security["cognitive_browser.security<br/>• SSRF guard & DNS rebinding<br/>• Sensitive data masking"]
        Cookies["cognitive_browser.cookies<br/>• Firefox/System profile cookie extractor"]
    end

    subgraph BrowserEngineLayer ["Browser Execution Engine"]
        Engine["cognitive_browser.browser<br/><b>BrowserEngine</b><br/>• Playwright stealth execution<br/>• Overlay auto-dismiss<br/>• DOM action dispatching"]
    end

    MCP --> MCPServer
    REST --> FastAPIServer
    MCPServer --> Orchestrator
    FastAPIServer --> Orchestrator

    Orchestrator --> Parser
    Orchestrator --> Payload
    Orchestrator --> Captcha
    Orchestrator --> Cache
    Orchestrator --> Reflex
    Orchestrator --> Security
    Orchestrator --> Cookies
    Orchestrator --> Engine

    Payload --> Reflex
    Engine --> Security
```

---

## 3. Core Modules & Responsibilities

### 3.1 [`cognitive_browser.parser.AriaElementParser`](file://./cognitive_browser/parser.py)
- **Role:** Extracts interactive candidate elements from Playwright ARIA snapshots.
- **Key Capabilities:**
  - Fast regex-based parsing of ARIA tree YAML (`- role "name"`).
  - PUA icon character cleaning (`\ue000-\uf8ff`) to prevent token pollution.
  - Section context flags: distinguishes elements inside `<main>`, `<article>`, `<header>`, and `<footer>` (`is_main`, `is_footer`, `is_header`).
  - Scalability: Supports 1,000+ candidates without artificial cutoffs; handles large infinite-scroll pages seamlessly.

### 3.2 [`cognitive_browser.payload.PayloadBinder`](file://./cognitive_browser/payload.py)
- **Role:** Maps arbitrary structured JSON payload dictionaries to DOM elements using neural cross-encoding.
- **Key Capabilities:**
  - Matches keys (e.g. `first_name`, `cv_file`, `job_type`) to candidate form fields (`textbox`, `combobox`, `file`, `checkbox`).
  - Distinguishes file upload inputs (`file_paths`) from regular text inputs.
  - Generates atomic action plans for automatic submission or field-by-field filling.

### 3.3 [`cognitive_browser.captcha.CaptchaResolver`](file://./cognitive_browser/captcha.py)
- **Role:** Automated CAPTCHA detection and third-party solving (CapSolver).
- **Key Capabilities:**
  - `evaluate_presence(...)`: Dual heuristic and semantic check. Differentiates passive legal footers ("Protected by reCAPTCHA") from active challenge screens.
  - Detects Cloudflare Turnstile `cf-turnstile` wrappers and reCAPTCHA v2 iframe tokens.
  - Integrates with CapSolver API asynchronously; falls back gracefully when API keys are absent.

### 3.4 [`cognitive_browser.cache.RouteCache`](file://./cognitive_browser/cache.py)
- **Role:** Persistent SQLite cache storing successful navigation decisions.
- **Key Capabilities:**
  - Indexed by `(page_url, instruction)` or `(page_url, "GOAL::" + goal)`.
  - Enables instant direct jumps to target URLs or interactive elements for routine queries, dropping latency to <15ms.

### 3.5 [`cognitive_browser.reflex.ReflexCore`](file://./cognitive_browser/reflex.py)
- **Role:** High-speed semantic decision core powered by local HuggingFace Cross-Encoder models (`modernbert-embed-base` or `deberta-v3-small`).
- **Key Capabilities:**
  - Mini-batched GPU inference across candidate pools.
  - Page goal satisfaction scoring: breaks long page text into sliding semantic chunks and scores content match.
  - State vs Action Competition: If page satisfaction rate exceeds action candidate confidence, the agent stops navigating and extracts the content immediately.
  - Calibration: Calibrated confidence threshold (0.35 - 0.55) to trigger Agent-in-the-Loop escalation when ambiguous.

### 3.6 [`cognitive_browser.orchestrator.BrowserOrchestrator`](file://./cognitive_browser/orchestrator.py)
- **Role:** Unified state machine and loop coordinator.
- **Key Capabilities:**
  - Consolidates multi-hop autonomous exploration (`browse(goal=...)`), planned step execution (`browse(steps=[...])`), and universal parameter binding (`browse(payload=...)`).
  - Manages cookie context synchronization, overlay auto-dismissal, CAPTCHA resolution, RouteCache updates, and clean ARIA content extraction.
  - Decouples server interfaces (FastMCP, FastAPI) from browser automation internals.

### 3.7 [`cognitive_browser.browser.BrowserEngine`](file://./cognitive_browser/browser.py)
- **Role:** Low-level Playwright automation wrapper.
- **Key Capabilities:**
  - Stealth flags (`navigator.webdriver` spoofing, randomized viewport sizes).
  - Overlay auto-dismissal for standard cookie banners and privacy dialogs.
  - Direct execution of clicks, text typing, combobox selection, checkbox toggling, and file uploads.

### 3.8 [`cognitive_browser.security`](file://./cognitive_browser/security.py) & [`cookies.py`](file://./cognitive_browser/cookies.py)
- **Security:** Strict URL validation, DNS rebinding mitigation, loopback/cloud-metadata blocking, and sensitive data masking in telemetry.
- **Cookies:** Reads local Firefox SQLite databases or exported JSON cookie jars to import authenticated sessions directly into browser context.

---

## 4. Test Verification & Code Quality

The entire architecture is verified by a comprehensive automated test suite (`pytest tests/`):
- **Parser Tests:** ARIA snapshot parsing, context resolution, PUA icon stripping.
- **Payload Tests:** Neural parameter binding, file uploads, combobox selection.
- **Captcha Tests:** Active challenge detection, passive footer filtering, mock solving.
- **Orchestrator & MCP Tests:** High-confidence autonomous paths, escalation flows, smart actions.
- **Security Tests:** SSRF guards, private IP blocking, redirect hop limit enforcement.
- **Calibration Tests:** Mini-batching on large candidate pools (1000+ items), adversarial intent suppression.
