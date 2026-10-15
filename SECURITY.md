# Security Policy

## Reporting Security Issues

We take the security of Cognitive Browser seriously. If you discover a vulnerability or potential security flaw, please report it responsibly rather than opening a public GitHub issue.

### Reporting Channels

To report a vulnerability:
- **Email:** Send details to `security@cognitive-browser.local` or contact the repository maintainer directly via GitHub Security Advisories.
- **Details to Include:**
  - Description of the vulnerability and its potential impact.
  - Minimal reproducible example or proof-of-concept steps.
  - Affected components (`mcp_server`, `server.py`, `agent_core`, etc.).
  - Proposed remediation or patch if available.

### Response Timeline
- **Initial Acknowledgement:** Within 48 hours.
- **Triage & Status Update:** Within 5 business days.
- **Fix & Advisory Release:** Coordinated with the reporter before public disclosure.

---

## Supported Versions

| Version | Supported |
|---------|:---------:|
| 0.1.x   | Yes       |
| < 0.1.0 | No        |

---

## Security Architecture Principles

1. **SSRF Guardrails:** Access to RFC 1918 private networks, loopback interfaces (`127.0.0.1`, `localhost`), and cloud instance metadata endpoints (`169.254.169.254`) is strictly blocked by default.
2. **Credential Sanitization:** Fill values targeting authentication fields (passwords, tokens, PINs) are masked in execution logs and telemetry.
3. **Indirect Prompt Injection Defense:** All candidate names, attributes, and text parsed from external web pages are sanitized and explicitly treated as inert untrusted data before insertion into reasoning prompts.
4. **Zero-PII Local Inference:** Reflex Core runs cross-encoder inferences on your local GPU/CPU; interaction data is not transmitted to external cloud servers unless cloud router mode is explicitly configured.
