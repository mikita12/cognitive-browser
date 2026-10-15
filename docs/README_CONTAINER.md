# 🧱 Cognitive Browser – Standalone Container Microservice

Autonomous, ultra-lightweight web navigation engine powered by **Reflex Core (Playwright ARIA Snapshot + DeBERTa-v3 NLI)** packaged inside a Docker container.

Designed as a self-contained, drop-in building block for any AI agent (via MCP), workflow automation platform (n8n, Make), or custom Python/Node.js application (via REST API).

---

## 🚀 Quickstart (Running the Microservice)

### 1. Launch with GPU (NVIDIA CUDA recommended, ~40 ms decision latency)
```bash
docker compose up -d --build
```

### 2. Verify Service Health (Healthcheck)
```bash
curl http://localhost:8000/health
```
Example response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
  "vram_allocated_mb": 420.5,
  "model": "cross-encoder/nli-deberta-v3-small",
  "playwright_ready": true
}
```

---

## 🔌 Available Interfaces

### A. REST API (for scripts, bots, n8n, cURL)

#### 1. Autonomous multi-step navigation (`POST /v1/browse`)
Executes a sequence of natural language steps. Automatically handles cookie consent banners and interacts with target ARIA elements:
```bash
curl -X POST http://localhost:8000/v1/browse \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://wikipedia.org",
    "steps": [
      {"instruction": "type search query into the search field", "fill_value": "Artificial intelligence"},
      {"instruction": "click the search button"}
    ]
  }'
```

#### 2. Clean ARIA tree without HTML noise (`POST /v1/snapshot`)
Returns page accessibility tree condensed by ~98% relative to raw HTML:
```bash
curl -X POST http://localhost:8000/v1/snapshot \
  -H "Content-Type: application/json" \
  -d '{"url": "https://wikipedia.org", "max_candidates": 30}'
```

#### 3. Sub-50ms semantic NLI evaluation (`POST /v1/evaluate`)
Evaluates candidate elements on GPU with 0 LLM tokens consumed:
```bash
curl -X POST http://localhost:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "accept cookie consent",
    "candidates": [
      {"ref": "c-1", "role": "button", "name": "Accept all cookies", "text": "Accept all"},
      {"ref": "c-2", "role": "link", "name": "Privacy policy", "text": "Privacy policy"}
    ]
  }'
```

---

## B. MCP Protocol (for AI Agents: Antigravity, Claude Desktop, Cursor)

The container exposes tool definitions at:
`GET http://localhost:8000/v1/mcp/tools`

Tools exposed to the agent:
* `cognitive_browse`: Autonomously completes navigation goals on web pages with 0 token consumption during UI execution.
* `get_aria_snapshot`: Extracts condensed semantic page state.

---

## 📁 Persistence & Session Storage (Cookies / Authentication)

Browser sessions and state are persisted inside the mounted volume:
`./data/browser_sessions:/data/browser_sessions`

This ensures logged-in sessions (marketplaces, portals, accounts) persist across container restarts.
