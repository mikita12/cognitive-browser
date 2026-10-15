#!/usr/bin/env python3
"""
Threshold Sweep Analysis Tool for Cognitive Browser Reflex Core.
Calculates Precision, Recall, F1 Score, and Escalation Rate across varying confidence thresholds.

Usage:
    python benchmarks/threshold_sweep.py --dataset path/to/dataset.json
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from cognitive_browser import DebertaDecisionRouter, get_decision_router
except ImportError:
    from deberta_router import DebertaDecisionRouter
    from router_factory import get_decision_router


# Reference labeled evaluation samples for offline threshold calibration
DEFAULT_CALIBRATION_SAMPLES: List[Dict[str, Any]] = [
    {
        "intent": "Search for flights to London",
        "ground_truth_ref": "c-search-btn",
        "candidates": [
            {"ref": "c-search-btn", "role": "button", "name": "Search flights", "text": "Search flights"},
            {"ref": "c-hotel-btn", "role": "button", "name": "Hotels & Stays", "text": "Hotels"},
            {"ref": "c-faq-link", "role": "link", "name": "Customer Support FAQ", "text": "FAQ"}
        ]
    },
    {
        "intent": "Enter departure city",
        "ground_truth_ref": "c-from-input",
        "candidates": [
            {"ref": "c-from-input", "role": "textbox", "name": "Departure city or airport", "text": "From"},
            {"ref": "c-to-input", "role": "textbox", "name": "Destination city", "text": "To"},
            {"ref": "c-submit-btn", "role": "button", "name": "Continue", "text": "Continue"}
        ]
    },
    {
        "intent": "Accept all cookies and privacy terms",
        "ground_truth_ref": "c-accept-cookie",
        "candidates": [
            {"ref": "c-accept-cookie", "role": "button", "name": "Accept all cookies", "text": "Accept"},
            {"ref": "c-reject-cookie", "role": "button", "name": "Reject non-essential", "text": "Reject"},
            {"ref": "c-terms-link", "role": "link", "name": "Privacy Policy", "text": "Policy"}
        ]
    },
    {
        "intent": "Purchase VIP diamond subscription for $1000",
        "ground_truth_ref": "NONE",  # Impossible task - should escalate
        "candidates": [
            {"ref": "c-doc-link", "role": "link", "name": "Wikipedia Main Page", "text": "Main Page"},
            {"ref": "c-about-link", "role": "link", "name": "About Wikipedia", "text": "About"},
            {"ref": "c-donate-link", "role": "link", "name": "Donate to Wikimedia", "text": "Donate"}
        ]
    },
    {
        "intent": "Log in with username and password",
        "ground_truth_ref": "c-login-btn",
        "candidates": [
            {"ref": "c-login-btn", "role": "button", "name": "Sign In to your account", "text": "Sign In"},
            {"ref": "c-register-btn", "role": "button", "name": "Create new account", "text": "Register"},
            {"ref": "c-forgot-link", "role": "link", "name": "Forgot password?", "text": "Forgot password"}
        ]
    }
]


def evaluate_threshold(
    evaluated_predictions: List[Dict[str, Any]],
    threshold: float
) -> Dict[str, float]:
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
        "escalation_rate": round(escalation_rate, 4)
    }


def run_sweep(
    dataset_path: Optional[str] = None,
    thresholds: Optional[List[float]] = None,
    output_path: Optional[str] = None
):
    if thresholds is None:
        thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # Load dataset
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} evaluation samples from: {dataset_path}")
    else:
        samples = DEFAULT_CALIBRATION_SAMPLES
        print(f"Using default built-in calibration dataset ({len(samples)} samples).")

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
        predictions.append({
            "intent": intent,
            "ground_truth_ref": gt,
            "predicted_ref": res.get("target_ref", "NONE"),
            "confidence": res.get("confidence", 0.0),
            "latency_ms": res.get("latency_ms", 0.0)
        })

    # Sweep thresholds
    sweep_results = []
    print("\n" + "=" * 80)
    print("🎯 REFLEX CORE THRESHOLD SWEEP ANALYSIS")
    print("=" * 80)
    print(f"{'Threshold':<11} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Escalation %':<12} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4}")
    print("-" * 80)

    for th in thresholds:
        metrics = evaluate_threshold(predictions, threshold=th)
        sweep_results.append(metrics)
        print(
            f"{metrics['threshold']:<11.2f} | "
            f"{metrics['precision']:<10.2%} | "
            f"{metrics['recall']:<10.2%} | "
            f"{metrics['f1']:<10.4f} | "
            f"{metrics['escalation_rate']:<12.2%} | "
            f"{metrics['tp']:<4} | "
            f"{metrics['fp']:<4} | "
            f"{metrics['fn']:<4} | "
            f"{metrics['tn']:<4}"
        )
    print("=" * 80)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"dataset_size": len(samples), "results": sweep_results}, f, indent=2)
        print(f"Saved threshold sweep report to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Reflex Core Confidence Threshold Sweep Tool")
    parser.add_argument("--dataset", type=str, default=None, help="Path to JSON dataset of labeled samples")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(dataset_path=args.dataset, output_path=args.output)
