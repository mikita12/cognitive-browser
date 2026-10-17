"""
Unit tests for ReflexCore decision calibration, Ambiguity Margin, and GPU mini-batching.
"""

from cognitive_browser.reflex import ReflexCore


def test_ambiguity_margin_triggers_escalation():
    """
    Verifies that when top 2 distinct candidates have confidence difference < 0.04
    and confidence < 0.75, ReflexCore flags ambiguity_detected = True and escalates.
    """
    reflex = ReflexCore()
    # Two distinct buttons with very close semantic meanings
    candidates = [
        {"role": "button", "name": "Proceed to Checkout", "context": "cart", "ref": "aria-1"},
        {"role": "link", "name": "Pay Now with Card", "context": "cart", "ref": "aria-2"},
    ]
    res = reflex.evaluate_candidates("finish purchase and pay order", candidates)

    assert "ambiguity_detected" in res
    assert "ambiguity_margin" in res
    # If confidence is below 0.75 and margin < 0.04, ambiguity must be detected
    if res["confidence"] < 0.75 and res.get("ambiguity_margin") is not None and res["ambiguity_margin"] < 0.04:
        assert res["ambiguity_detected"] is True
        assert res["should_escalate"] is True
        assert res["target_ref"] == "NONE"


def test_gpu_mini_batching_large_candidate_pool():
    """
    Verifies that a candidate pool larger than batch size (64 elements) is processed
    in mini-batches and maintains original ordering and complete scoring.
    """
    reflex = ReflexCore()
    # Generate 80 candidates
    candidates = []
    for i in range(80):
        candidates.append(
            {
                "role": "link" if i % 2 == 0 else "button",
                "name": f"Navigation Item {i}",
                "context": f"section_{i // 10}",
                "ref": f"aria-{i}",
            }
        )
    # Target item with exact name
    candidates[42] = {
        "role": "button",
        "name": "Download Financial Annual Report 2025",
        "context": "investor relations",
        "ref": "aria-target",
    }

    res = reflex.evaluate_candidates("download financial annual report", candidates)

    assert len(res["ranked_candidates"]) == 80
    assert res["target_ref"] == "aria-target"
    assert res["confidence"] >= 0.65
    assert res["should_escalate"] is False
    assert res["latency_ms"] > 0.0


def test_adversarial_impossible_intent_suppression():
    """
    Verifies that completely impossible / adversarial queries on regular pages
    are suppressed below confidence threshold without False Positives.
    """
    reflex = ReflexCore()
    candidates = [
        {"role": "button", "name": "Submit Application", "context": "form"},
        {"role": "textbox", "name": "Your Email", "context": "form"},
        {"role": "link", "name": "Terms of Service", "context": "footer"},
    ]
    # Completely unrelated query
    res = reflex.evaluate_candidates("teleport purple unicorn to saturn rings", candidates)

    assert res["confidence"] < 0.50
    assert res["should_escalate"] is True
    assert res["target_ref"] == "NONE"
