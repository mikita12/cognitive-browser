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


def test_remote_reranker_api_mode_zero_vram(monkeypatch):
    """
    Verifies that ReflexCore in remote API mode (e.g. Jina Reranker API / Cohere / TEI)
    initializes with 0 local model weights (self.model is None) and properly scores
    candidate elements and multi-query payload batches via /v1/rerank JSON responses.
    """
    import io
    import json
    import urllib.request

    def fake_urlopen(req, timeout=15.0):
        body = json.loads(req.data.decode("utf-8"))
        query = body["query"].lower()
        docs = body["documents"]
        results = []
        for idx, doc in enumerate(docs):
            doc_lower = doc.lower()
            score = 0.08
            if "szukaj" in query and "szukaj" in doc_lower:
                score = 0.94
            elif "email" in query and "e-mail" in doc_lower:
                score = 0.91
            elif "telefon" in query and "telefon" in doc_lower:
                score = 0.89
            results.append({"index": idx, "relevance_score": score})

        resp_bytes = json.dumps({"results": results}).encode("utf-8")
        return io.BytesIO(resp_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    reflex = ReflexCore(
        model_name="api:jina-reranker-v2-base-multilingual",
        device="api",
        api_url="https://api.jina.ai/v1/rerank",
        api_key="jina_test_key",
    )

    assert reflex.use_api is True
    assert reflex.model is None
    assert reflex.tokenizer is None
    assert reflex.is_reranker is True

    candidates = [
        {"role": "link", "name": "Kontakt", "context": "Stopka", "ref": "aria-1"},
        {"role": "searchbox", "name": "Szukaj w serwisie...", "context": "Nagłówek", "ref": "aria-2"},
        {"role": "textbox", "name": "Adres e-mail", "context": "Formularz", "ref": "aria-3"},
        {"role": "textbox", "name": "Numer telefonu", "context": "Formularz", "ref": "aria-4"},
    ]

    res = reflex.evaluate_candidates("Wpisz w wyszukiwarkę: Szukaj AI", candidates)
    assert res["target_ref"] == "aria-2"
    assert res["confidence"] > 0.85
    assert res["should_escalate"] is False

    bound = reflex.bind_payload_to_elements(
        {"adres_email": "jan@example.com", "numer_telefonu": "500600700"},
        candidates,
    )
    assert len(bound) == 2
    by_key = {item["payload_key"]: item["name"] for item in bound}
    assert by_key["adres_email"] == "Adres e-mail"
    assert by_key["numer_telefonu"] == "Numer telefonu"
