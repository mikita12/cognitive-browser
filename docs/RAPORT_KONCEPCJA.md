# Project: Cognitive Browser (Reflex Core + Cortex Engine in Playwright Automation)

## 1. Introduction and Research Problem

Modern autonomous web agents (built on frameworks like Playwright and directed by Large Language Models) face two fundamental bottlenecks:
1. **Severe Token Bloat:** Raw DOM dumps or accessibility trees generate tens of thousands of tokens per browser interaction step.
2. **High Latency & Costs:** Invoking heavy generative frontier models (e.g., Claude 3.5 Sonnet / GPT-4o) for routine decision-making (e.g., *"Is this button 'Next'?"*, *"Is the cookie banner present?"*, *"Did the page finish loading?"*) takes 2–5 seconds per roundtrip and rapidly exhausts API budgets.

---

## 2. Core Architectural Concept

The architecture is inspired by dual-process cognitive theory: **Reflex Core** (rapid, subconscious, zero-token reflex decisions) and **Cortex Engine** (deliberative, slower planning and complex reasoning).

Instead of piping raw page dumps directly into the primary LLM loop, we place a **local discriminative decision router** (DeBERTa-v3 NLI or fine-tuned cross-encoder) directly in front of the browser execution engine:

```
                    ┌─────────────────────────┐
                    │  Playwright DOM / A11y  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Reflex Core Layer    │
                    │ DeBERTa-v3 Cross-Encoder│ <── Sub-50ms parallel evaluation
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
         High Confidence (>0.50)         Low Confidence / Ambiguity (<0.50)
         (Deterministic Action)          (Requires Deliberation)
                 │                               │
                 ▼                               ▼
       ┌───────────────────┐           ┌───────────────────┐
       │ Direct Playwright │           │ Compact Context   │
       │ Action in ~40 ms  │           │ to Primary LLM    │
       │ [0 LLM TOKENS]    │           │ (Cortex Engine)   │
       └───────────────────┘           └───────────────────┘
```

---

## 3. Key Roles of the Decision Router

### A. Element Tree Pruning
* The decision router evaluates interactive candidates concurrently against the active user intent.
* Decorative elements, ad banners, and repetitive noise are stripped locally.
* **Result:** The primary LLM receives only the top 5–10% most relevant candidates, reducing context size by 85–90%.

### B. Fast-Path Routine Handling
* **Consent modal dismissal:** *"Is a GDPR / cookie banner visible on screen?"* $\rightarrow$ if YES, dismissed instantly via predefined routine in ~30 ms.
* **Goal completion check:** *"Is the order confirmation displayed?"* $\rightarrow$ loop terminates without querying the cloud LLM.
* **Blocker detection:** Captcha / Cloudflare challenge detection triggers immediate notification.

### C. Speculative Verification
* Following a Playwright action, the router evaluates state changes.
* Binary validation: *"Did the interaction yield the intended UI state transition?"*.
* If state remains unchanged (e.g., click failed due to an ongoing animation), a local retry is executed without wasting an API call.

---

## 4. Expected Benefits

| Metric | Traditional LLM Agent | Agent with Decision Router | Gain |
| :--- | :--- | :--- | :--- |
| **Token Consumption / Step** | 10,000 – 25,000 tokens | 0 – 400 tokens | **~98% Reduction** |
| **Routine Step Latency** | 2.5 – 5.0 s | 30 – 60 ms | **Up to 40x Faster** |
| **API Cost per 100 Steps** | High ($0.50 – $2.00) | Near Zero ($0.00 – $0.05) | **~95% Cost Savings** |
| **Locator Reliability** | Prone to selector hallucination | Native ARIA roles & labels | **100% Type-Safe Locators** |

---

## 5. Technology Stack & Implementation

1. **Reflex Core (Discriminative Decision Model - DeBERTa-v3):**
   * Model: `cross-encoder/nli-deberta-v3-small` running natively in PyTorch on GPU (`cuda:0`), Apple Silicon (`mps`), or CPU.
   * Inference method: **Zero-Shot NLI (Softmax Entailment vs Contradiction)**.
   * Decision latency: **~39 ms**.
   * Tokens generated: **0**.
2. **Web Automation Engine:**
   * Playwright ARIA Snapshot extraction (`page.aria_snapshot()`).
   * Protection against hallucinated selectors: semantic ARIA role/name targeting.
3. **Cortex Engine (Primary Reasoning Engine):**
   * Engaged exclusively when `should_escalate = True` (confidence below threshold or obstruction encountered).

---

## 6. Real-World Benchmark Telemetry

```
================================================================================
Method                         | Model Type         | Latency       | Token Usage  
--------------------------------------------------------------------------------
Traditional Agent (Claude/GPT) | Cloud Generative   | ~3000-5000 ms | ~12,000 tok./step
Local LLM (Qwen2.5-1.5B)       | Generative LLM     | ~400-500 ms   | Local (0 cost)
DeBERTa-v3 NLI (Our Router)    | Discriminative NLI | 39.96 ms      | 0 tokens ($0.00)
================================================================================
```
The Playwright action is executed immediately with 0 token expenditure.
