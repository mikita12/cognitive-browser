# Cognitive Browser

> **Zero-token autonomous browser agent & MCP server powered by pluggable local GPU Cross-Encoder / NLI routing (`ReflexCore`), universal cross-lingual form binding (`PayloadBinder`), Stealth Playwright (`BrowserEngine`), and self-healing `RouteCache`.**

Cognitive Browser enables AI coding assistants (Antigravity, Claude Desktop, Cursor) to navigate the web, search dynamic SPAs, and fill complex multi-field forms with **0 LLM tokens** consumed during DOM exploration and action execution.

---

## Highlights

- ⚡ **Zero-Token Autonomous Navigation:** Navigates multi-hop websites on GPU in milliseconds (~3.3ms per candidate; 400+ DOM elements evaluated in <25ms).
- 🔌 **Pluggable Neural Models (Rerankers & NLI):** Works out-of-the-box with any HuggingFace `AutoModelForSequenceClassification` — including multilingual Cross-Encoder Rerankers (`mmarco-mMiniLMv2`, `bge-reranker-v2-m3`, `gte-reranker-modernbert-base`) and NLI models (`DeBERTa-v3`, `mDeBERTa-v3`, `ModernBERT-nli`).
- 🧠 **Dictionary-Free Cross-Lingual `PayloadBinder`:** Partitions DOM controls by structural role (`textbox`, `combobox`, `checkbox`, `upload_file`) and binds arbitrary JSON payloads across languages (e.g., English keys to Polish forms or vice versa) with zero hardcoded synonym tables.
- 🚀 **Self-Healing `RouteCache`:** Learns deterministic shortcut routes in SQLite for near-instant repeat execution and automatically invalidates stale entries if a site's DOM changes.
- 🛡️ **SSRF & Security Guards:** Restricts navigation to public destinations, blocks loopback/private networks and cloud metadata APIs (`169.254.169.254`), and protects against DNS rebinding.
- 🍪 **Opt-In Cookie Sync & CAPTCHA Resolution:** Can import authenticated sessions from local Firefox profiles (disabled by default; enabled via `--enable-cookies` flag) and integrates automated Turnstile / reCAPTCHA resolution.
- 🧩 **Strongly-Typed Clean Architecture:** Built with decoupled single-responsibility modules (`models`, `parser`, `payload`, `captcha`, `cache`, `reflex`, `orchestrator`, `browser`, `security`).

---

## Architecture Overview

<p align="center">
  <img src="docs/architecture.svg" alt="Cognitive Browser System Architecture" width="100%" />
</p>

For complete technical specifications, see the [Architecture & Engineering Guide](docs/architecture.md) (compiled from [`docs/architecture.d2`](docs/architecture.d2) via [D2](https://d2lang.com/)).

---

## Supported Models

`ReflexCore` automatically supports Cross-Encoder Rerankers (`num_labels=1`), NLI models (`num_labels=3`), and remote `/v1/rerank` APIs (0 MB local RAM/VRAM):

| Model (`COGNITIVE_BROWSER_MODEL`) | Type | Size / RAM | Speed (CPU / GPU) | Best For |
| :--- | :--- | :---: | :---: | :--- |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` *(Default)* | Reranker (14+ langs) | 470 MB | 20 ms / 3.6 ms | Default for GPU & CPU (PL, EN, DE...) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker (EN) | **90 MB** | **14 ms** / 2.6 ms | Weak laptops / CPU-only |
| `api:jina-reranker-v2-base-multilingual` | Cloud API (100+ langs) | **0 MB** | ~80 ms (HTTP) | No GPU & no free RAM ([free API key](https://jina.ai/reranker/)) |
| `tasksource/ModernBERT-base-nli` | NLI (EN) | 598 MB | 120 ms / 7.8 ms | ModernBERT on GPU |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | NLI (100+ langs) | 1.1 GB | 190 ms / 14 ms | Multilingual NLI on GPU |
| `BAAI/bge-reranker-v2-m3` | Reranker (100+ langs) | 2.2 GB | 350 ms / 18 ms | Max accuracy on 8GB+ VRAM GPU |

### How to Switch Models

Set `COGNITIVE_BROWSER_MODEL` and `COGNITIVE_BROWSER_DEVICE` (`cuda`, `mps`, `cpu`, or `api`):

```bash
# 1. Default (Local GPU or CPU auto-detected)
export COGNITIVE_BROWSER_MODEL="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# 2. Ultra-light CPU Mode (~90 MB RAM, no GPU needed)
export COGNITIVE_BROWSER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
export COGNITIVE_BROWSER_DEVICE="cpu"

# 3. Zero-RAM Cloud Mode via Jina Reranker API (free key from https://jina.ai/reranker/)
export COGNITIVE_BROWSER_DEVICE="api"
export JINA_API_KEY="jina_your_api_key"
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/mikita12/cognitive-browser.git
cd cognitive-browser
pip install -e .
playwright install chromium
```

### 2. Configure MCP Server

Add to `~/.gemini/antigravity/mcp_config.json` or `claude_desktop_config.json` (pass `"--enable-cookies"` in `"args"` only if you want to allow importing local browser cookies):

```json
{
  "mcpServers": {
    "cognitive-browser": {
      "command": "python3",
      "args": ["-m", "cognitive_browser.mcp_server"],
      "cwd": "/path/to/cognitive-browser",
      "env": {
        "COGNITIVE_BROWSER_MODEL": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "COGNITIVE_BROWSER_DEVICE": "cuda"
      }
    }
  }
}
```

### 3. Usage Examples

#### Single-Call Autonomous Goal Navigation:
```python
cognitive_browse(
    url="https://pl.wikipedia.org",
    goal="Wpisz w wyszukiwarkę: 'Sztuczna inteligencja' i otwórz artykuł"
)
```

#### Cross-Lingual Form Binding & File Upload:
```python
cognitive_browse(
    url="https://company.com/careers/apply",
    payload={
        "first_name": "Jan",
        "last_name": "Kowalski",
        "email": "jan@example.com",
        "cv": "/home/user/documents/cv.pdf",
        "rodo": True
    }
)
```

---

## Running the Test Suite

```bash
pytest tests/
```

All 62 automated unit and integration tests pass with comprehensive coverage across domain models, ARIA parsing, cross-lingual payload binding, CAPTCHA handling, RouteCache self-healing, and SSRF security controls.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
