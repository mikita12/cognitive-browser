#!/usr/bin/env python3
"""
Test Client for Cognitive Browser microservice (REST + MCP).
Demonstrates interaction with the container over HTTP.
"""

import time
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    print("\n🔍 1. Testing /health endpoint...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def test_evaluate():
    print("\n⚡ 2. Testing Reflex Core inference (/v1/evaluate)...")
    payload = {
        "intent": "accept all cookies and terms to close the consent banner",
        "candidates": [
            {"ref": "c-1", "role": "link", "name": "Privacy policy", "text": "Privacy policy"},
            {"ref": "c-2", "role": "button", "name": "Accept all cookies", "text": "Accept all cookies"},
            {"ref": "c-3", "role": "button", "name": "Reject optional", "text": "Reject optional"},
            {"ref": "c-4", "role": "link", "name": "Contact us", "text": "Contact us"}
        ]
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/v1/evaluate", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Status: {resp.status_code} | HTTP Roundtrip: {elapsed_ms:.1f} ms")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def test_snapshot():
    print("\n🌐 3. Testing ARIA snapshot extraction (/v1/snapshot)...")
    payload = {
        "url": "https://example.com",
        "max_candidates": 10
    }
    resp = requests.post(f"{BASE_URL}/v1/snapshot", json=payload)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Page title: {data.get('title')}")
    print(f"Generation time: {data.get('latency_ms')} ms")
    print(f"Candidates count: {data.get('candidates_count')}")
    print(f"ARIA snapshot:\n{data.get('raw_snapshot')}")


def test_browse():
    print("\n🤖 4. Testing autonomous multi-step navigation (/v1/browse)...")
    payload = {
        "url": "https://wikipedia.org",
        "steps": [
            {
                "instruction": "type search query into the search field",
                "fill_value": "Artificial intelligence"
            },
            {
                "instruction": "click the search button"
            }
        ]
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/v1/browse", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Status: {resp.status_code} | Total task duration: {elapsed_ms:.1f} ms")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("=" * 60)
    print("COGNITIVE BROWSER MICROSERVICE CLIENT TEST")
    print("=" * 60)
    try:
        test_health()
        test_evaluate()
        test_snapshot()
        test_browse()
    except Exception as e:
        print(f"\n❌ Connection error to ({BASE_URL}): {e}")
        print("Make sure the microservice container or server is running (`docker compose up -d` or `python3 server.py`)")
