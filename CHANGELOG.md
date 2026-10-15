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
  - SSRF protection: blocking loopback (127.0.0.0/8), private ranges (RFC 1918, RFC 4193), and cloud metadata endpoints (169.254.169.254) by default, with opt-in override `COGNITIVE_BROWSER_ALLOW_PRIVATE_NETWORKS`.
  - Domain allowlisting via `COGNITIVE_BROWSER_ALLOWED_DOMAINS`.
  - Sensitive value masking in logs and traces for credentials, API tokens, and passwords.
- **Benchmarking Suite**:
  - Multi-run statistical reporting (`--runs N` with mean ± standard deviation).
  - Offline local HTML fixtures mode (`--use-fixtures`) to evaluate workflows without live internet access.
  - Threshold sweep evaluation utility (`benchmarks/threshold_sweep.py`).
- **Documentation**:
  - `docs/benchmark-methodology.md`: Complete accounting rules, ReAct baseline simulation disclosure, and benchmark limits.
  - `docs/architecture.md`: Single-page architectural walkthrough with accurate Mermaid diagrams.
  - `docs/threat-model.md` and `SECURITY.md`: Vulnerability reporting process and threat mitigations.
- **Developer Tooling & CI**:
  - Ruff linter and formatter configuration.
  - Mypy static type checking across core library and test suite.
  - Pytest test suite with code coverage (`pytest-cov`) operating with mocked network and Playwright instances.

### Changed
- Standardized architecture naming: renamed legacy System 1 / System 2 concepts to Reflex Core and Cortex Engine.
- Replaced legacy root entrypoints `mcp_server.py` and `server.py` with thin deprecation shims pointing to `cognitive_browser.mcp_server` and `cognitive_browser.server`.
- Translated all repository code comments and documentation to English.
