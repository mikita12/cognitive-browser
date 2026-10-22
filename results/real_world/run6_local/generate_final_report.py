"""
Run 6 Final Report Generator (100% Offline / Self-Hosted with Ornith 1.5).
Computes:
1. Task Success Rate per arm with 95% Wilson Score Confidence Intervals.
2. Token usage distribution for Ornith 1.5.
3. Latency metrics (p50, p95, mean).
4. McNemar's exact/continuity-corrected test for statistical significance (A vs B, A vs C) calculated from RAW data.
5. Detailed error analysis using stop_reason and partial_needs_fallback.
6. Infra errors logged separately.
Generates results/real_world/run6_local/REPORT.md.
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes the 95% Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z**2) / (4 * (total**2)))
    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return low, high


def mcnemar_test(contingency_table: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """
    Computes McNemar's test with Edwards continuity correction on raw paired 2x2 table.
    Table format: (n_00, n_01, n_10, n_11) where:
    n_01: Arm 1 fails, Arm 2 succeeds (b)
    n_10: Arm 1 succeeds, Arm 2 fails (c)
    """
    n_00, b, c, n_11 = contingency_table
    diff = abs(b - c)
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False, "b": b, "c": c}

    chi2 = ((diff - 1.0) ** 2) / (b + c) if diff > 0 else 0.0

    try:
        from scipy.stats import chi2 as chi2_dist
        p_val = 1.0 - chi2_dist.cdf(chi2, df=1)
    except Exception:
        z = math.sqrt(chi2)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_val, 5),
        "significant": p_val < 0.05,
        "b": b,
        "c": c,
    }


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p)
    return data_sorted[min(idx, len(data_sorted) - 1)]


def generate_report():
    base_dir = Path("results/real_world/run6_local")
    key_path = base_dir / "anonymization_key.json"
    verdicts_path = base_dir / "judge_verdicts.jsonl"

    if not key_path.exists() or not verdicts_path.exists():
        print(f"Error: Missing {key_path} or {verdicts_path}")
        sys.exit(1)

    with open(key_path, "r", encoding="utf-8") as f:
        key_map = json.load(f)

    verdicts = {}
    with open(verdicts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    v = json.loads(line)
                    verdicts[v["sample_id"]] = v
                except Exception:
                    pass

    print(f"Loaded {len(key_map)} anonymized keys, {len(verdicts)} judge verdicts.")

    # Load raw episode logs
    infra_errors = defaultdict(list)
    episodes = {}
    for arm, p in [
        ("cognitive_browser", base_dir / "runner_cognitive_browser.jsonl"),
        ("playwright_mcp", base_dir / "baseline_playwright_mcp.jsonl"),
        ("cognitive_bm25", base_dir / "baseline_cognitive_bm25.jsonl"),
    ]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ep = json.loads(line)
                        episodes[ep["sample_id"]] = ep
                        if ep.get("status") == "infra_error":
                            infra_errors[arm].append(ep)

    arm_data = defaultdict(list)
    task_arm_results = defaultdict(lambda: defaultdict(list))

    for blind_id, info in key_map.items():
        v = verdicts.get(blind_id, {})
        arm = info["arm"]
        task_id = info["task_id"]
        rep = info["repetition"]
        raw_ep = episodes.get(info["original_sample_id"], {})

        status = raw_ep.get("status", info.get("status", ""))
        if status == "infra_error":
            infra_errors[arm].append({
                "sample_id": blind_id,
                "task_id": task_id,
                "error": raw_ep.get("error_message", info.get("error_message", "")),
            })
            continue

        success = v.get("task_success", False)
        reason = v.get("failure_reason", "")

        sample_record = {
            "blind_id": blind_id,
            "task_id": task_id,
            "repetition": rep,
            "success": success,
            "failure_reason": reason,
            "tokens_in": raw_ep.get("agent_input_tokens", 0),
            "tokens_out": raw_ep.get("agent_output_tokens", 0),
            "tokens_total": raw_ep.get("total_agent_tokens", 0),
            "latency_ms": raw_ep.get("e2e_latency_ms", 0.0),
            "stop_reason": raw_ep.get("stop_reason", ""),
            "steps_count": raw_ep.get("steps_count", 0),
            "fallback_count": raw_ep.get("partial_fallback_count", 0),
            "interventions": raw_ep.get("agent_interventions", 0),
        }
        arm_data[arm].append(sample_record)
        task_arm_results[task_id][arm].append(success)

    # 1. Success Rates and Confidence Intervals
    stats_summary = {}
    for arm in ["cognitive_browser", "playwright_mcp", "cognitive_bm25"]:
        samples = arm_data[arm]
        total = len(samples)
        succ = sum(1 for s in samples if s["success"])
        rate = succ / max(1, total)
        ci_low, ci_high = wilson_score_interval(succ, total)

        toks = [s["tokens_total"] for s in samples if s["tokens_total"] > 0]
        lats = [s["latency_ms"] / 1000.0 for s in samples]

        # Majority task successes (>=2 out of 3 reps)
        maj_succ = 0
        all_task_ids = set(s["task_id"] for s in samples)
        for tid in all_task_ids:
            task_reps = task_arm_results[tid][arm]
            if task_reps and (sum(task_reps) >= 2 or (len(task_reps) < 3 and sum(task_reps) >= len(task_reps)/2)):
                maj_succ += 1

        stats_summary[arm] = {
            "total_episodes": total,
            "successes": succ,
            "rate": rate,
            "ci_95": (ci_low, ci_high),
            "majority_tasks": maj_succ,
            "majority_rate": maj_succ / max(1, len(all_task_ids)),
            "tokens_mean": sum(toks) / max(1, len(toks)) if toks else 0,
            "tokens_median": percentile(toks, 0.5) if toks else 0,
            "lat_mean": sum(lats) / max(1, len(lats)),
            "lat_p50": percentile(lats, 0.5),
            "lat_p95": percentile(lats, 0.95),
            "fallbacks_total": sum(s["fallback_count"] for s in samples),
            "interventions_total": sum(s["interventions"] for s in samples),
            "stop_reasons": Counter(s["stop_reason"] for s in samples if s["stop_reason"]),
        }

    # 2. McNemar's Tests on RAW paired data
    all_tasks_sorted = sorted(list(task_arm_results.keys()))

    def compute_contingency(arm1: str, arm2: str) -> Tuple[int, int, int, int]:
        n_00 = n_01 = n_10 = n_11 = 0
        for tid in all_tasks_sorted:
            s1 = sum(task_arm_results[tid][arm1]) >= 2
            s2 = sum(task_arm_results[tid][arm2]) >= 2
            if not s1 and not s2:
                n_00 += 1
            elif not s1 and s2:
                n_01 += 1
            elif s1 and not s2:
                n_10 += 1
            else:
                n_11 += 1
        return n_00, n_01, n_10, n_11

    mcnemar_ab = mcnemar_test(compute_contingency("cognitive_browser", "playwright_mcp"))
    mcnemar_ac = mcnemar_test(compute_contingency("cognitive_browser", "cognitive_bm25"))

    # 3. Generate Markdown Report
    report_md = rf"""# Raport Końcowy Benchmarku: Run 6 (100% Offline / Self-Hosted)

**Data ewaluacji:** 2026-10-20  
**Zbiór testowy:** `results/real_world/new_benchmark_80/tasks.jsonl` (80 zadań, hash SHA-256: `22f7130f439e59fde9337e969599112c10630d169eccceaeb331cb360b96fc6d`)  
**Architektura testowa:** 3 powtórzenia per zadanie w losowej kolejności, max 1 req/s na domenę.  
**Agent sterujący:** `ornith-1.5:35b` (35B MoE) via tunel SSH `http://127.0.0.1:11440/v1` (temp=0.1, max_tokens=1024, limit 12 kroków).  
**Sędzia:** Niezależny LLM (Gemini Flash), ocena w 100% ślepa (brak dostępu do logów narzędzi i tożsamości ramion).

---

## 1. Tabela Skuteczności (Task Success Rate)

| Ramię badawcze | Skuteczność per epizod | 95% Przedział Ufności (Wilson) | Skuteczność per zadanie (Majority $\ge 2/3$) | Błędy infrastruktury (Infra Errors) |
| :--- | :---: | :---: | :---: | :---: |
| **Ramię A: Cognitive Browser (ReflexCore)** | **{stats_summary['cognitive_browser']['rate']*100:.1f}%** ({stats_summary['cognitive_browser']['successes']}/{stats_summary['cognitive_browser']['total_episodes']}) | **[{stats_summary['cognitive_browser']['ci_95'][0]*100:.1f}% - {stats_summary['cognitive_browser']['ci_95'][1]*100:.1f}%]** | **{stats_summary['cognitive_browser']['majority_rate']*100:.1f}%** ({stats_summary['cognitive_browser']['majority_tasks']}/80) | {len(infra_errors['cognitive_browser'])} |
| **Ramię B: Playwright MCP (Standard)** | **{stats_summary['playwright_mcp']['rate']*100:.1f}%** ({stats_summary['playwright_mcp']['successes']}/{stats_summary['playwright_mcp']['total_episodes']}) | **[{stats_summary['playwright_mcp']['ci_95'][0]*100:.1f}% - {stats_summary['playwright_mcp']['ci_95'][1]*100:.1f}%]** | **{stats_summary['playwright_mcp']['majority_rate']*100:.1f}%** ({stats_summary['playwright_mcp']['majority_tasks']}/80) | {len(infra_errors['playwright_mcp'])} |
| **Ramię C: Cognitive Browser (BM25 $\\tau=0.02$)** | **{stats_summary['cognitive_bm25']['rate']*100:.1f}%** ({stats_summary['cognitive_bm25']['successes']}/{stats_summary['cognitive_bm25']['total_episodes']}) | **[{stats_summary['cognitive_bm25']['ci_95'][0]*100:.1f}% - {stats_summary['cognitive_bm25']['ci_95'][1]*100:.1f}%]** | **{stats_summary['cognitive_bm25']['majority_rate']*100:.1f}%** ({stats_summary['cognitive_bm25']['majority_tasks']}/80) | {len(infra_errors['cognitive_bm25'])} |

---

## 2. Tabela Latencji (Czas Wykonania)

| Ramię badawcze | Czas p50 (s) | Czas p95 (s) | Średni czas na epizod (s) |
| :--- | :---: | :---: | :---: |
| **Ramię A: Cognitive Browser** | **{stats_summary['cognitive_browser']['lat_p50']:.1f}s** | **{stats_summary['cognitive_browser']['lat_p95']:.1f}s** | **{stats_summary['cognitive_browser']['lat_mean']:.1f}s** |
| **Ramię B: Playwright MCP** | {stats_summary['playwright_mcp']['lat_p50']:.1f}s | {stats_summary['playwright_mcp']['lat_p95']:.1f}s | {stats_summary['playwright_mcp']['lat_mean']:.1f}s |
| **Ramię C: Cognitive Browser (BM25)** | {stats_summary['cognitive_bm25']['lat_p50']:.1f}s | {stats_summary['cognitive_bm25']['lat_p95']:.1f}s | {stats_summary['cognitive_bm25']['lat_mean']:.1f}s |

---

## 3. Tabela Użycia Tokenów (Ornith 1.5)

| Ramię badawcze | Średnia liczba tokenów / ep | Mediana tokenów / ep | Oszczędność tokenów vs Playwright MCP |
| :--- | :---: | :---: | :---: |
| **Ramię A: Cognitive Browser** | **{stats_summary['cognitive_browser']['tokens_mean']:.0f}** | **{stats_summary['cognitive_browser']['tokens_median']:.0f}** | **{(1.0 - stats_summary['cognitive_browser']['tokens_mean']/max(1.0, stats_summary['playwright_mcp']['tokens_mean']))*100:.1f}% mniej tokenów** |
| **Ramię B: Playwright MCP** | {stats_summary['playwright_mcp']['tokens_mean']:.0f} | {stats_summary['playwright_mcp']['tokens_median']:.0f} | Baseline (1.0x) |
| **Ramię C: Cognitive Browser (BM25)** | {stats_summary['cognitive_bm25']['tokens_mean']:.0f} | {stats_summary['cognitive_bm25']['tokens_median']:.0f} | {(1.0 - stats_summary['cognitive_bm25']['tokens_mean']/max(1.0, stats_summary['playwright_mcp']['tokens_mean']))*100:.1f}% mniej tokenów |

---

## 4. Statystyki Testu McNemara (Na podstawie surowych par danych)

Test McNemara z poprawką na ciągłość Edwardsa wyliczony na wynikach większościowych (Majority Vote $\ge 2/3$) na 80 zadaniach:

### A vs B (Cognitive Browser vs Playwright MCP)
- **Tabela kontyngencji:**
  - Zadania zaliczone przez oba ramiona: **{compute_contingency('cognitive_browser', 'playwright_mcp')[3]}**
  - Tylko Cognitive Browser (A zaliczył, B oblał): **{mcnemar_ab['c']}**
  - Tylko Playwright MCP (B zaliczył, A oblał): **{mcnemar_ab['b']}**
  - Oblały oba ramiona: **{compute_contingency('cognitive_browser', 'playwright_mcp')[0]}**
- **Statystyka $\chi^2$ (z poprawką Edwardsa):** **{mcnemar_ab['chi2']}**
- **Wartość p ($p$-value):** **{mcnemar_ab['p_value']}**
- **Wniosek:** **{"Różnica jest statystycznie istotna (p < 0.05)" if mcnemar_ab['significant'] else "Brak statystycznie istotnej różnicy (p >= 0.05)"}**.

### A vs C (Cognitive Browser ReflexCore vs BM25)
- **Tabela kontyngencji:**
  - Zadania zaliczone przez oba ramiona: **{compute_contingency('cognitive_browser', 'cognitive_bm25')[3]}**
  - Tylko ReflexCore (A zaliczył, C oblał): **{mcnemar_ac['c']}**
  - Tylko BM25 (C zaliczył, A oblał): **{mcnemar_ac['b']}**
  - Oblały oba ramiona: **{compute_contingency('cognitive_browser', 'cognitive_bm25')[0]}**
- **Statystyka $\chi^2$ (z poprawką Edwardsa):** **{mcnemar_ac['chi2']}**
- **Wartość p ($p$-value):** **{mcnemar_ac['p_value']}**
- **Wniosek:** **{"Różnica jest statystycznie istotna (p < 0.05)" if mcnemar_ac['significant'] else "Brak statystycznie istotnej różnicy"}**.

---

## 5. Analiza Błędów i Mechanizmów Fallbacku (Ramię A)

- **Liczba wyzwoleń `partial_needs_fallback` / `needs_fallback`:** **{stats_summary['cognitive_browser']['fallbacks_total']}**
- **Interwencje agenta przez `smart_action` po fallbacku:** **{stats_summary['cognitive_browser']['interventions_total']}**
- **Rozkład przyczyn zatrzymania (`stop_reason`):**
"""
    for sr, count in stats_summary['cognitive_browser']['stop_reasons'].most_common():
        report_md += f"  - `{sr}`: {count} razy\n"

    report_md += f"""
### Błędy Infrastruktury (Infra Errors)
- Ramię A: **{len(infra_errors['cognitive_browser'])}** zdarzeń
- Ramię B: **{len(infra_errors['playwright_mcp'])}** zdarzeń
- Ramię C: **{len(infra_errors['cognitive_bm25'])}** zdarzeń
"""

    report_path = base_dir / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Report generated successfully at: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
