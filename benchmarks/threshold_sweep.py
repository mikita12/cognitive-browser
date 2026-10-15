#!/usr/bin/env python3
"""
Threshold Sweep Analysis Tool for Cognitive Browser Reflex Core.
Calculates Precision, Recall, F1 Score, and Escalation Rate across varying confidence thresholds.

Usage:
    python benchmarks/threshold_sweep.py --dataset path/to/dataset.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from cognitive_browser import get_decision_router
except ImportError:
    from router_factory import get_decision_router


# Reference labeled evaluation samples for offline threshold calibration
DEFAULT_CALIBRATION_SAMPLES: List[Dict[str, Any]] = [
    {
        "intent": "Search for flights to London",
        "ground_truth_ref": "c-search-btn",
        "candidates": [
            {"ref": "c-search-btn", "role": "button", "name": "Search flights", "text": "Search flights"},
            {"ref": "c-hotel-btn", "role": "button", "name": "Hotels & Stays", "text": "Hotels"},
            {"ref": "c-faq-link", "role": "link", "name": "Customer Support FAQ", "text": "FAQ"},
        ],
    },
    {
        "intent": "Enter departure city",
        "ground_truth_ref": "c-from-input",
        "candidates": [
            {"ref": "c-from-input", "role": "textbox", "name": "Departure city or airport", "text": "From"},
            {"ref": "c-to-input", "role": "textbox", "name": "Destination city", "text": "To"},
            {"ref": "c-submit-btn", "role": "button", "name": "Continue", "text": "Continue"},
        ],
    },
    {
        "intent": "Accept all cookies and privacy terms",
        "ground_truth_ref": "c-accept-cookie",
        "candidates": [
            {"ref": "c-accept-cookie", "role": "button", "name": "Accept all cookies", "text": "Accept"},
            {"ref": "c-reject-cookie", "role": "button", "name": "Reject non-essential", "text": "Reject"},
            {"ref": "c-terms-link", "role": "link", "name": "Privacy Policy", "text": "Policy"},
        ],
    },
    {
        "intent": "Purchase VIP diamond subscription for $1000",
        "ground_truth_ref": "NONE",  # Impossible task - should escalate
        "candidates": [
            {"ref": "c-doc-link", "role": "link", "name": "Wikipedia Main Page", "text": "Main Page"},
            {"ref": "c-about-link", "role": "link", "name": "About Wikipedia", "text": "About"},
            {"ref": "c-donate-link", "role": "link", "name": "Donate to Wikimedia", "text": "Donate"},
        ],
    },
    {
        "intent": "Log in with username and password",
        "ground_truth_ref": "c-login-btn",
        "candidates": [
            {"ref": "c-login-btn", "role": "button", "name": "Sign In to your account", "text": "Sign In"},
            {"ref": "c-register-btn", "role": "button", "name": "Create new account", "text": "Register"},
            {"ref": "c-forgot-link", "role": "link", "name": "Forgot password?", "text": "Forgot password"},
        ],
    },
    # Fixture: Local E-Commerce
    {
        "intent": "Search Products",
        "ground_truth_ref": "c-search-box",
        "candidates": [
            {"ref": "c-search-box", "role": "searchbox", "name": "Search Products", "text": ""},
            {"ref": "c-search-btn", "role": "button", "name": "Search", "text": "Search"},
            {"ref": "c-cart-link", "role": "link", "name": "Shopping Cart", "text": "Shopping Cart"},
        ],
    },
    {
        "intent": "Search",
        "ground_truth_ref": "c-search-btn",
        "candidates": [
            {"ref": "c-search-box", "role": "searchbox", "name": "Search Products", "text": ""},
            {"ref": "c-search-btn", "role": "button", "name": "Search", "text": "Search"},
            {"ref": "c-login-link", "role": "link", "name": "Sign In", "text": "Sign In"},
        ],
    },
    {
        "intent": "Add laptop to shopping cart",
        "ground_truth_ref": "c-add-cart",
        "candidates": [
            {"ref": "c-add-cart", "role": "button", "name": "Add to Cart", "text": "Add to Cart"},
            {"ref": "c-search-btn", "role": "button", "name": "Search", "text": "Search"},
            {"ref": "c-cart-link", "role": "link", "name": "Shopping Cart", "text": "Shopping Cart"},
        ],
    },
    # Fixture: Local Travel Booking
    {
        "intent": "Departure station",
        "ground_truth_ref": "c-from-station",
        "candidates": [
            {"ref": "c-from-station", "role": "textbox", "name": "Departure station", "text": ""},
            {"ref": "c-to-station", "role": "textbox", "name": "Destination station", "text": ""},
            {"ref": "c-search-conn", "role": "button", "name": "Search connections", "text": "Search connections"},
        ],
    },
    {
        "intent": "Destination station",
        "ground_truth_ref": "c-to-station",
        "candidates": [
            {"ref": "c-from-station", "role": "textbox", "name": "Departure station", "text": ""},
            {"ref": "c-to-station", "role": "textbox", "name": "Destination station", "text": ""},
            {"ref": "c-search-conn", "role": "button", "name": "Search connections", "text": "Search connections"},
        ],
    },
    {
        "intent": "Search connections",
        "ground_truth_ref": "c-search-conn",
        "candidates": [
            {"ref": "c-from-station", "role": "textbox", "name": "Departure station", "text": ""},
            {"ref": "c-to-station", "role": "textbox", "name": "Destination station", "text": ""},
            {"ref": "c-search-conn", "role": "button", "name": "Search connections", "text": "Search connections"},
        ],
    },
    # Ambiguous / impossible tasks (must escalate -> GT: "NONE")
    {
        "intent": "Download corporate salary report spreadsheet",
        "ground_truth_ref": "NONE",
        "candidates": [
            {"ref": "c-from-station", "role": "textbox", "name": "Departure station", "text": ""},
            {"ref": "c-to-station", "role": "textbox", "name": "Destination station", "text": ""},
            {"ref": "c-search-conn", "role": "button", "name": "Search connections", "text": "Search connections"},
        ],
    },
    {
        "intent": "Switch system kernel execution profile to superuser",
        "ground_truth_ref": "NONE",
        "candidates": [
            {"ref": "c-search-box", "role": "searchbox", "name": "Search Products", "text": ""},
            {"ref": "c-search-btn", "role": "button", "name": "Search", "text": "Search"},
            {"ref": "c-cart-link", "role": "link", "name": "Shopping Cart", "text": "Shopping Cart"},
        ],
    },
]


def evaluate_threshold(evaluated_predictions: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """
    Evaluates predictions at a given confidence threshold.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for item in evaluated_predictions:
        gt = item["ground_truth_ref"]
        conf = item["confidence"]
        pred = item["predicted_ref"]

        if conf >= threshold:
            if pred == gt:
                tp += 1
            else:
                fp += 1
        else:
            # Below threshold -> Escalated to Cortex
            if gt == "NONE":
                tn += 1  # Correctly refused/escalated impossible case
            else:
                fn += 1  # Unnecessarily escalated resolvable case

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    total = tp + fp + fn + tn
    escalation_rate = (fn + tn) / total if total > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "escalation_rate": round(escalation_rate, 4),
    }


def run_sweep(
    dataset_path: Optional[str] = None,
    thresholds: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    live_results_path: Optional[str] = None,
):
    if thresholds is None:
        thresholds = [round(0.30 + i * 0.05, 2) for i in range(13)]

    # Load dataset
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} evaluation samples from: {dataset_path}")
    else:
        samples = DEFAULT_CALIBRATION_SAMPLES
        print(f"Using default calibration dataset ({len(samples)} labeled samples).")

    # Load live benchmark step confidences if available
    live_confidences: List[float] = []
    if not live_results_path:
        default_live = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_15_cases_results.json")
        if os.path.exists(default_live):
            live_results_path = default_live

    if live_results_path and os.path.exists(live_results_path):
        with open(live_results_path, "r", encoding="utf-8") as f:
            live_data = json.load(f)
        runs = live_data.get("runs", [])
        if runs and len(runs) > 0:
            for case in runs[0]:
                for s in case.get("cognitive", {}).get("steps_log", []):
                    if "confidence" in s:
                        live_confidences.append(float(s["confidence"]))
        print(f"Loaded {len(live_confidences)} step confidences from live benchmark: {live_results_path}")

    router = get_decision_router()

    # Pre-compute model outputs once (independent of threshold)
    print("Running Reflex Core model inference on evaluation dataset...")
    predictions = []
    for item in samples:
        intent = item["intent"]
        candidates = item["candidates"]
        gt = item.get("ground_truth_ref", "NONE")

        # Evaluate with 0.0 threshold to get raw top prediction and confidence
        res = router.evaluate_candidates(intent=intent, candidates=candidates, confidence_threshold=0.0)
        predictions.append(
            {
                "intent": intent,
                "ground_truth_ref": gt,
                "predicted_ref": res.get("target_ref", "NONE"),
                "confidence": res.get("confidence", 0.0),
                "latency_ms": res.get("latency_ms", 0.0),
            }
        )

    # Sweep thresholds
    sweep_results = []
    print("\n" + "=" * 95)
    print("🎯 REFLEX CORE THRESHOLD SWEEP ANALYSIS")
    print("=" * 95)
    print(
        f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Labeled Esc%':<13} | {'Live Esc%':<10} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4}"
    )
    print("-" * 95)

    for th in thresholds:
        metrics = evaluate_threshold(predictions, threshold=th)
        if live_confidences:
            live_esc_count = sum(1 for c in live_confidences if c < th)
            live_esc_rate = round(live_esc_count / len(live_confidences), 4)
        else:
            live_esc_rate = 0.0
        metrics["live_escalation_rate"] = live_esc_rate

        sweep_results.append(metrics)
        print(
            f"{metrics['threshold']:<10.2f} | "
            f"{metrics['precision']:<10.2%} | "
            f"{metrics['recall']:<10.2%} | "
            f"{metrics['f1']:<10.4f} | "
            f"{metrics['escalation_rate']:<13.2%} | "
            f"{metrics['live_escalation_rate']:<10.2%} | "
            f"{metrics['tp']:<4} | "
            f"{metrics['fp']:<4} | "
            f"{metrics['fn']:<4} | "
            f"{metrics['tn']:<4}"
        )
    print("=" * 95)

    output_file = output_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "threshold_sweep_results.json"
    )
    payload = {
        "metadata": {
            "labeled_samples_count": len(samples),
            "live_steps_count": len(live_confidences),
            "thresholds_evaluated": thresholds,
            "device": getattr(router, "device", "unknown"),
        },
        "results": sweep_results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved threshold sweep report to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Reflex Core Confidence Threshold Sweep Tool")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON dataset of labeled samples")
    parser.add_argument("--live", type=str, default=None, help="Path to live benchmark results JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(dataset_path=args.dataset, live_results_path=args.live, output_path=args.output)
