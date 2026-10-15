# Benchmark Methodology & Measurement Disclosures

This document details the setup, evaluation criteria, token accounting, and limitations of the 15-case web automation benchmark published in this repository.

---

## 1. Objective and Overview

The benchmark evaluates the performance of **Cognitive Browser** (Reflex Core DeBERTa-v3 + Cortex Engine LLM escalation) against a standard agent loop model (Playwright MCP with repeated accessibility snapshots sent to a frontier LLM on every step).

---

## 2. Experimental Setup & Model Versions

- **Hardware Platform:**
  - GPU: NVIDIA GeForce RTX 4070 Ti (12GB VRAM)
  - Driver & CUDA: Linux x86_64, CUDA 12.x
  - Host CPU: AMD Ryzen 9
- **Reflex Core Model:**
  - `cross-encoder/nli-deberta-v3-small` (140M parameters)
  - Hosted locally via PyTorch (`transformers` AutoModelForSequenceClassification)
  - Precision: FP32 / FP16 depending on torch defaults
  - Inference mode: Zero-shot NLI Softmax between Entailment and Contradiction logits
- **Cortex Escalation Model:**
  - `qwen2.5:1.5b` served locally via Ollama (`http://localhost:11434/api/generate`)
  - Invoked strictly when Reflex confidence is below the escalation threshold ($< 0.50$) or when an element match cannot be found.
- **Browser Automation:**
  - Playwright Chromium (headless mode)
  - Default viewport: 1280x720

---

## 3. Evaluated Tasks (n = 15)

The benchmark includes 15 realistic multi-step web workflows across e-commerce, travel, civic portals, search, and job boards:

1. **Koleo (`koleo.pl`)**: Train ticket search (Warszawa Centralna -> Kraków Główny).
2. **OLX (`olx.pl`)**: Real estate rental search in Wrocław.
3. **Otomoto (`otomoto.pl`)**: Vehicle search (Toyota Yaris).
4. **Wikipedia (`wikipedia.org`)**: Search for "Artificial intelligence" article.
5. **Allegro (`allegro.pl`)**: Consumer electronics search for laptops.
6. **Pracuj.pl (`pracuj.pl`)**: Job search for Python Developer in Warszawa.
7. **GitHub (`github.com`)**: Repository search for "playwright".
8. **Hacker News (`news.ycombinator.com`)**: Algolia search for "deberta".
9. **Gov.pl (`gov.pl`)**: Public administration search for ID card renewal.
10. **Castorama (`castorama.pl`)**: Hardware search for rotary hammer drill.
11. **Empik (`empik.com`)**: Book search for "Wiedźmin".
12. **Ceneo (`ceneo.pl`)**: Price aggregator search for "PlayStation 5".
13. **Booking.com (`booking.com`)**: Accommodation search in Gdańsk.
14. **DuckDuckGo (`duckduckgo.com`)**: Search for "local NLI agent".
15. **Medonet (`medonet.pl`)**: Medical information search for "grypa objawy".

---

## 4. Definition of "Task Completed"

A task is classified as **completed** if and only if:
1. Every sequential instruction in the workflow was executed (`executed=True`).
2. Target inputs were filled with specified values without locator timeouts.
3. Target buttons or links were clicked without unhandled DOM exceptions.
4. The page transitioned to the expected state or URL.

If any intermediate step fails or times out without successful escalation recovery, the task is marked as **partial** or **failed**.

---

## 5. Token Accounting: What is Included and Excluded

> [!IMPORTANT]
> **Planning Tokens Disclosure:** The benchmark measures the execution of pre-specified multi-step instruction arrays passed to the browser tool. It **does NOT** include tokens that an upstream planning agent (e.g., Claude 3.5 Sonnet or GPT-4o) would expend to initially formulate the list of steps.

### What is INCLUDED in Cognitive Browser Token Count:
- **MCP tool call overhead:** An estimate of the JSON parameter payload for the tool invocation:
  $$\text{JSON overhead} = 150 + \left\lfloor \frac{\text{len}(\text{json\_steps})}{3.5} \right\rfloor$$
- **Cortex Escalation Tokens:** When Reflex confidence falls below the threshold, Cortex is invoked. All tokens (`prompt_eval_count` + `eval_count` reported by Ollama) are tracked and added directly to the total token count.
- **Reflex Core Steps:** Consume **0 LLM tokens** (local cross-encoder tensor forward pass).

### What is EXCLUDED from Cognitive Browser Token Count:
- Upstream agent prompt tokens (system prompts, user conversation history, and high-level reasoning prior to tool calling).
- Downstream synthesis tokens (summarizing the final page content back to the user).

---

## 6. Baseline Comparison Disclosure (Analytical Simulation)

> [!NOTE]
> The numbers for **Base Playwright MCP (`@playwright/mcp`)** reported in the summary tables are based on an **analytical simulation model**, not live API calls against proprietary frontier LLMs.

The simulation models standard industry ReAct agent behavior:
- Initial ARIA snapshot token cost: $\left\lfloor \frac{\text{len}(\text{raw\_snapshot})}{3.5} \right\rfloor + 350$.
- Step-by-step context growth: The agent context window accumulates prior observations ($400$ tokens per roundtrip).
- Network latency: Each LLM reasoning turn adds an estimated $2.8\text{ s}$ cloud network + inference roundtrip time.

While this accurately reflects typical ReAct overhead observed in frameworks like `browser-use` or Playwright MCP, it is an analytical projection rather than an empirical trial against a specific commercial provider.

---

## 7. Known Benchmark Limitations

1. **Volatile Live Websites:** Real-world websites change layouts, display A/B tests, or present regional variations. Results on live endpoints can drift over time.
2. **Single-Run Sample Size in Initial Dataset:** The historical dataset (`benchmark_15_cases_results.json`) was generated from single runs ($N=1$) without standard deviations. Use the new `--runs N` CLI parameter to gather statistical averages.
3. **No Dynamic Replanning:** If a website changes structure completely, a static step list will fail unless Cortex escalates. Cognitive Browser relies on an upstream agent or Cortex to recover from unexpected structural shifts.
4. **Local Hardware Dependency:** Reflex latency depends directly on the inference accelerator (NVIDIA CUDA Tensor Cores vs Apple Silicon MPS vs CPU).
