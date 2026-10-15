#!/usr/bin/env python3
"""
Summarize latency and execution metrics from Cognitive Browser benchmark JSON results.
Computes median, p95, mean, min, and max step-level Reflex inference latencies.
"""

import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional


def summarize_benchmark(results_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    step_latencies: List[float] = []

    # Extract step latencies from runs
    for run in data.get("runs", []):
        for case in run:
            cog = case.get("cognitive", {})
            steps_log = cog.get("steps_log", [])
            for s in steps_log:
                if "latency_ms" in s and s.get("step") != "auto_dismiss_cookies":
                    step_latencies.append(float(s["latency_ms"]))

    # If steps_log wasn't serialized, fall back to aggregate step data or metadata
    if not step_latencies:
        # Check pre-computed or recorded step latencies in aggregate section
        agg_step = data.get("aggregate_statistics", {}).get("reflex_step_latency_ms")
        if agg_step:
            return agg_step

    if not step_latencies:
        raise ValueError(f"No step-level latencies found in {results_path}")

    sorted_lats = sorted(step_latencies)
    median_val = round(statistics.median(sorted_lats), 1)
    p95_idx = int(len(sorted_lats) * 0.95)
    p95_val = round(sorted_lats[p95_idx], 1)
    mean_val = round(statistics.mean(sorted_lats), 1)
    min_val = round(min(sorted_lats), 1)
    max_val = round(max(sorted_lats), 1)

    summary = {
        "metric": "Reflex Core per-step inference latency (CUDA)",
        "sample_size": len(step_latencies),
        "median_ms": median_val,
        "p95_ms": p95_val,
        "mean_ms": mean_val,
        "min_ms": min_val,
        "max_ms": max_val,
    }

    print("\n" + "=" * 60)
    print("📊 REFLEX CORE STEP-LEVEL INFERENCE LATENCY SUMMARY")
    print("=" * 60)
    print(f"Sample size : {summary['sample_size']} steps")
    print(f"Median      : {summary['median_ms']} ms")
    print(f"p95         : {summary['p95_ms']} ms")
    print(f"Mean        : {summary['mean_ms']} ms")
    print(f"Range       : {summary['min_ms']} ms – {summary['max_ms']} ms")
    print("=" * 60 + "\n")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved latency summary to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize Cognitive Browser benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_15_cases_results.json"),
        help="Path to benchmark results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_summary.json"),
        help="Path to output summary JSON",
    )
    args = parser.parse_args()
    summarize_benchmark(args.input, args.output)


if __name__ == "__main__":
    main()
