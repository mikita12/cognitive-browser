# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-15

### Added
- **Reflex Core (DeBERTa-v3 NLI)**: Discriminative cross-encoder for sub-50ms local element resolution without consuming generative LLM tokens.
- **Cortex Engine Escalation**: Automatic fallback to generative LLMs (Groq, OpenAI, or custom endpoint) when Reflex confidence falls below threshold or when interaction errors occur.
- **ApiDecisionRouter**: Cloud-hosted decision routing supporting Hugging Face Inference API (DeBERTa NLI), Cohere Rerank v3.5, and fast LLMs.
- **Model Context Protocol (MCP) Server**: FastMCP-based standard tools (`cognitive_browse`, `smart_action`, `get_aria_snapshot`).
- **Configurable Confidence Threshold**: Environment variable `COGNITIVE_BROWSER_CONFIDENCE_THRESHOLD` and programmatic configuration support.
- **Security Protections**:
  - Strict URL scheme validation (HTTP/HTTPS only).
  - SSRF protection: blocking loopback (127.0.0.0/8, `::1`), private ranges (RFC 1918, `fc00::/7`), link-local (`fe80::/10`), IPv4-mapped IPv6, and cloud metadata endpoints (169.254.169.254) by default, with opt-in override `ALLOW_PRIVATE_NETWORKS`.
  - Post-navigation final landing URL verification against redirect SSRF attacks.
  - Domain allowlisting via `ALLOWED_DOMAINS`.
  - Sensitive value masking enabled by default in logs and telemetry, with opt-out via `DISABLE_FILL_MASKING=true`.
  - API Token bearer authentication (`API_TOKEN`) for REST API and SSE on port 8000, with loopback binding (`127.0.0.1:8000`) in `docker-compose.yml`.
- **Benchmarking Suite**:
  - Real `@playwright/mcp` execution harness (`benchmarks/run_playwright_mcp_baseline.py`) supporting local Ollama and cloud LLMs.
  - Consistent per-step Reflex inference latency percentiles (24.9 ms median, 64.6 ms p95).
  - Multi-run statistical reporting (`--runs N` with mean ± standard deviation).
  - Offline local HTML fixtures mode (`--mode fixture`) to evaluate workflows without live internet access.
  - Threshold sweep evaluation utility (`benchmarks/threshold_sweep.py`) with empirical precision/recall/escalation sweep across thresholds 0.30–0.90.
- **Documentation**:
  - `docs/benchmark-methodology.md`: Complete accounting rules, ReAct baseline simulation disclosure, and benchmark limits.
  - `docs/threshold-analysis.md`: Empirical threshold calibration and trade-off analysis.
  - `docs/architecture.md`: Single-page architectural walkthrough with accurate Mermaid diagrams.
  - `docs/threat-model.md` and `SECURITY.md`: Vulnerability reporting process and threat mitigations (including DNS rebinding limits and prompt injection disclosures).
  - `docs/demo.md`: Step-by-step terminal demo GIF recording guide.
- **Developer Tooling & CI**:
  - Ruff linter and formatter configuration.
  - Mypy static type checking across core library and test suite.
  - Pytest test suite with code coverage (`pytest-cov`) operating with mocked network and Playwright instances.

### Changed
- Standardized architecture naming: renamed legacy System 1 / System 2 concepts to Reflex Core and Cortex Engine.
- Replaced legacy root entrypoints `mcp_server.py` and `server.py` with thin deprecation shims pointing to `cognitive_browser.mcp_server` and `cognitive_browser.server`.
- Replaced subjective comparison ratings with an objective, neutral comparison table and explicit user requirements.
- Translated all repository code comments and documentation to English.
