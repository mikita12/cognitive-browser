# 🧱 Cognitive Browser – Samodzielny Kontenerowy Klocek (Microservice)

Autonomiczny, ultralekki silnik nawigacji webowej oparty o **System 1 (Playwright ARIA Snapshot + DeBERTa-v3 NLI)** zamknięty w kontenerze Docker. 

Zaprojektowany jako gotowy, samowystarczalny moduł do podpięcia pod dowolnego agenta AI (przez MCP), system automatyzacji (n8n, Make) lub własny kod w Pythonie/Node.js (przez REST API).

---

## 🚀 Szybki Start (Uruchomienie Klocka)

### 1. Uruchomienie z GPU (RTX 4070 Ti – rekomendowane, czas decyzji ~40 ms)
```bash
cd /home/mikita/Desktop/agentic-decision-router
docker compose up -d --build
```

### 2. Sprawdzenie stanu (Healthcheck)
```bash
curl http://localhost:8000/health
```
Odpowiedź:
```json
{
  "status": "healthy",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
  "vram_allocated_mb": 420.5,
  "model": "cross-encoder/nli-deberta-v3-small",
  "playwright_ready": true
}
```

---

## 🔌 Dostępne Interfejsy

### A. REST API (dla skryptów, botów, n8n, cURL)

#### 1. Autonomiczna wielokrokowa nawigacja (`POST /v1/browse`)
Wykonuje sekwencję kroków w języku naturalnym. Samodzielnie zamyka modale cookies i klika odpowiednie elementy ARIA:
```bash
curl -X POST http://localhost:8000/v1/browse \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://wikipedia.org",
    "steps": [
      {"instruction": "wpisz wyszukiwane hasło do pola wyszukiwania", "fill_value": "Wrocław"},
      {"instruction": "kliknij przycisk wyszukiwania"}
    ]
  }'
```

#### 2. Czyste drzewo ARIA bez szumu HTML (`POST /v1/snapshot`)
Zwraca strukturę strony zredukowaną o 98% względem surowego HTML:
```bash
curl -X POST http://localhost:8000/v1/snapshot \
  -H "Content-Type: application/json" \
  -d '{"url": "https://olx.pl", "max_candidates": 30}'
```

#### 3. Błyskawiczna ewaluacja semantyczna NLI (`POST /v1/evaluate`)
Wycena kandydatów w 40 ms na GPU bez użycia tokenów:
```bash
curl -X POST http://localhost:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "zaakceptuj zgody na cookies",
    "candidates": [
      {"ref": "c-1", "role": "button", "name": "Zgadzam się na wszystkie", "text": "Zgadzam się"},
      {"ref": "c-2", "role": "link", "name": "Polityka prywatności", "text": "Polityka"}
    ]
  }'
```

---

### B. Protokół MCP (dla Agentów AI: Antigravity, Claude Desktop, Cursor)

Kontener wystawia specyfikację narzędzi pod adresem:
`GET http://localhost:8000/v1/mcp/tools`

Narzędzia widoczne dla agenta:
* `cognitive_browse`: Autonomiczne wykonanie celu na stronie bez zużywania tokenów w pętli.
* `get_aria_snapshot`: Pobranie skompresowanego stanu strony.

---

## 📁 Trwałość Danych i Sesji (Cookies / Logowania)

Wszystkie sesje przeglądarki są zapisywane w podmontowanym wolumenie:
`./data/browser_sessions:/data/browser_sessions`

Dzięki temu kontener po restarcie pamięta zalogowane konta (Facebook, OLX, portale ogłoszeniowe).
