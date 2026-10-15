#!/usr/bin/env python3
"""
Test Client dla mikroserwisu Cognitive Browser (REST + MCP).
Demonstruje komunikację z kontenerem z poziomu Pythona.
"""

import time
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    print("\n🔍 1. Test endpointu /health...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def test_evaluate():
    print("\n⚡ 2. Test wnioskowania System 1 (/v1/evaluate)...")
    payload = {
        "intent": "zaakceptuj zgody na przetwarzanie danych i ciasteczka",
        "candidates": [
            {"ref": "c-1", "role": "link", "name": "Polityka prywatności", "text": "Polityka prywatności"},
            {"ref": "c-2", "role": "button", "name": "Zgadzam się na wszystkie", "text": "Zgadzam się na wszystkie"},
            {"ref": "c-3", "role": "button", "name": "Odrzuć opcjonalne", "text": "Odrzuć opcjonalne"},
            {"ref": "c-4", "role": "link", "name": "Kontakt", "text": "Kontakt"}
        ]
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/v1/evaluate", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Status: {resp.status_code} | Roundtrip HTTP: {elapsed_ms:.1f} ms")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def test_snapshot():
    print("\n🌐 3. Test pobierania czystego drzewa ARIA (/v1/snapshot)...")
    payload = {
        "url": "https://example.com",
        "max_candidates": 10
    }
    resp = requests.post(f"{BASE_URL}/v1/snapshot", json=payload)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Tytuł strony: {data.get('title')}")
    print(f"Czas generowania: {data.get('latency_ms')} ms")
    print(f"Liczba kandydatów: {data.get('candidates_count')}")
    print(f"Snapshot ARIA:\n{data.get('raw_snapshot')}")


def test_browse():
    print("\n🤖 4. Test autonomicznej nawigacji wielokrokowej (/v1/browse)...")
    payload = {
        "url": "https://wikipedia.org",
        "steps": [
            {
                "instruction": "wpisz wyszukiwane hasło do pola wyszukiwania",
                "fill_value": "Wrocław"
            },
            {
                "instruction": "kliknij przycisk wyszukiwania"
            }
        ]
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/v1/browse", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Status: {resp.status_code} | Czas całego zadania: {elapsed_ms:.1f} ms")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("=" * 60)
    print("TEST KLIENTA COGNITIVE BROWSER MICROSERVICE")
    print("=" * 60)
    try:
        test_health()
        test_evaluate()
        test_snapshot()
        test_browse()
    except Exception as e:
        print(f"\n❌ Błąd połączenia z serwerem ({BASE_URL}): {e}")
        print("Upewnij się, że kontener lub serwer działa (`docker compose up -d` lub `python3 server.py`)")
