# Benchmark Methodology & Measurement Disclosures

This document details the setup, evaluation criteria, token accounting, and limitations of the 15-case web automation benchmark published in this repository.

---

## 1. Objective and Overview

The benchmark evaluates the performance of **Cognitive Browser** (Reflex Core DeBERTa-v3 + Cortex Engine LLM escalation) across structured web automation workflows.

> [!NOTE]
> **Scope & Nature of Benchmark Suite:**
> The 15 scripted tasks in this suite serve as an automated **regression benchmark** to empirically verify semantic routing and locator accuracy across diverse web DOM architectures, **not** as proof of general autonomous browsing competence. In production, formulating an open-ended goal into concrete step actions requires an upstream reasoning agent.

---

## 2. Experimental Setup & Model Versions

- **Hardware Platform:**
  - GPU: NVIDIA GeForce RTX 4070 Ti (12GB VRAM)
  - Driver & CUDA: Linux x86_64, Driver 580.x, CUDA 13.0
  - Host CPU: AMD Ryzen 9
- **Reflex Core Model:**
  - `cross-encoder/nli-deberta-v3-small` (140M parameters)
  - Hosted locally via PyTorch (`transformers` AutoModelForSequenceClassification)
  - Precision: FP32 / FP16 depending on torch defaults
  - Inference mode: Zero-shot NLI Softmax between Entailment and Contradiction logits
  - Measured inference latency on CUDA: **24.9 ms median, 64.6 ms p95** (range: 5.6–95.5 ms, $n = 32$ interaction steps; note that at $n = 32$, $p95$ is approximate)
- **Cortex Escalation Model:**
  - `qwen2.5:1.5b` served locally via Ollama (`http://localhost:11434/api/generate`)
  - Invoked strictly when Reflex confidence is below the escalation threshold ($< 0.50$) or when an element match cannot be found.
- **Browser Automation:**
  - Playwright Chromium (headless mode)
  - Default viewport: 1280x720

---

## 3. Evaluated Tasks (n = 15)

The benchmark includes 15 realistic multi-step web workflows across e-commerce, travel, civic portals, search, and job boards:

1. **Koleo (`koleo.pl`)**: Train ticket search (Wrocław Główny -> Poznań Główny).
2. **InPost (`inpost.pl`)**: Navigation to parcel locker pricing tab.
3. **Books to Scrape (`books.toscrape.com`)**: Category selection and product interaction.
4. **HTTPBin (`httpbin.org/forms/post`)**: Multi-field HTML5 form completion (name, phone, email, radio, checkbox).
5. **OLX (`olx.pl`)**: Search for electronics offers ("PlayStation 5") with consent modal handling.
6. **Hacker News (`news.ycombinator.com`)**: Navigation to Show HN and discussion comments.
7. **GitHub Trending (`github.com/trending`)**: Navigation to developer and spoken language filters.
8. **Morele.net (`morele.net`)**: Search for computer peripherals with consent banner handling.
9. **Quotes to Scrape (`quotes.toscrape.com`)**: Tag filtering and pagination navigation.
10. **Express Pizza (`expresspizza.pl`)**: Online food ordering delivery location selection.
11. **Mania Smaku (`mania.pizza`)**: Online pizza menu exploration.
12. **Pizzu Wrocław (`pizzu.pl`)**: Lunch menu specials navigation.
13. **E-Podróżnik (`e-podroznik.pl`)**: Intercity transit route search (Wrocław -> Kraków).
14. **Gov.pl (`gov.pl`)**: Public administration search for ID card renewal procedure.
15. **MPK Wrocław (`wroclaw.pl/komunikacja/rozklady-jazdy`)**: Urban tram timetable verification.

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
- **Reflex Core Steps:** Consume **0 LLM tokens** (local cross-encoder tensor forward pass). **13 out of 15 tasks completed consuming 0 LLM tokens**.
- **Cortex Escalation Tokens:** When Reflex confidence falls below the threshold, Cortex is invoked. All tokens (`prompt_eval_count` + `eval_count` reported by Ollama) are tracked and added directly to the total token count. In the 15-task suite, exactly 2 tasks triggered Cortex escalation:
  - **Hacker News (Task 06):** 1 escalation turn, consuming **285 tokens** (450 total with step representation).
  - **MPK Wrocław (Task 15):** 1 escalation turn, consuming **1,347 tokens** (1,513 total with step representation).
- **JSON parameter overhead:** An estimate of the JSON parameter payload for the tool invocation:
  $$\text{JSON overhead} = 150 + \left\lfloor \frac{\text{len}(\text{json\_steps})}{3.5} \right\rfloor$$

### What is EXCLUDED from Cognitive Browser Token Count:
- Upstream agent prompt tokens (system prompts, user conversation history, and high-level reasoning prior to tool calling).
- Downstream synthesis tokens (summarizing the final page content back to the user).

---

## 6. Baseline Estimation Model & Real Harness Disclosure

> [!NOTE]
> The numbers for **Simulated Baseline ReAct Agent** reported in earlier references are based on an **analytical simulation model**, NOT live API execution against a proprietary frontier LLM.
> 
> Because an analytical simulation does not execute real browser interactions, **no empirical task completion rate (e.g. "15/15") exists for it**, and it has been excluded from all performance summary tables.

The analytical estimation models standard industry ReAct agent overhead:
- Initial ARIA snapshot token cost: $\left\lfloor \frac{\text{len}(\text{raw\_snapshot})}{3.5} \right\rfloor + 350$.
- Step-by-step context growth: The agent context window accumulates prior observations ($400$ tokens per roundtrip).
- Network latency: Each LLM reasoning turn adds an estimated $2.8\text{ s}$ cloud network + inference roundtrip time.

### Real Playwright MCP Baseline Harness (`benchmarks/run_playwright_mcp_baseline.py`)
To obtain empirical comparisons against an actual MCP server, the repository provides a dedicated harness:
```bash
python benchmarks/run_playwright_mcp_baseline.py --provider openai --model gpt-4o-mini
```
**Important Accounting Difference:** When running `run_playwright_mcp_baseline.py`, planning tokens are included in the baseline measurement because the external LLM must autonomously formulate tool calls, inspect DOM snapshots, and recover from failures. In contrast, Cognitive Browser measures execution of pre-scripted workflow steps via local discriminative routing.

---

## 7. Known Benchmark Limitations

1. **Volatile Live Websites:** Real-world websites change layouts, display A/B tests, or present regional variations. Results on live endpoints can drift over time.
2. **Single-Run Sample Size in Initial Dataset:** The historical dataset (`benchmark_15_cases_results.json`) was generated from single runs ($N=1$) without standard deviations. Use the new `--runs N` CLI parameter to gather statistical averages.
3. **No Dynamic Replanning:** If a website changes structure completely, a static step list will fail unless Cortex escalates. Cognitive Browser relies on an upstream agent or Cortex to recover from unexpected structural shifts.
4. **Local Hardware Dependency:** Reflex latency depends directly on the inference accelerator (NVIDIA CUDA Tensor Cores vs Apple Silicon MPS vs CPU).
