# Raport Końcowy Benchmarku: Run 6 (100% Offline / Self-Hosted)

**Data ewaluacji:** 2026-10-20  
**Zbiór testowy:** `results/real_world/new_benchmark_80/tasks.jsonl` (80 zadań, hash SHA-256: `22f7130f439e59fde9337e969599112c10630d169eccceaeb331cb360b96fc6d`)  
**Architektura testowa:** 3 powtórzenia per zadanie w losowej kolejności, max 1 req/s na domenę.  
**Agent sterujący:** `ornith-1.5:35b` (35B MoE) via tunel SSH `http://127.0.0.1:11440/v1` (temp=0.1, max_tokens=1024, limit 12 kroków).  
**Sędzia:** Niezależny LLM (Gemini Flash), ocena w 100% ślepa (brak dostępu do logów narzędzi i tożsamości ramion).

---

## 1. Tabela Skuteczności (Task Success Rate)

| Ramię badawcze | Skuteczność per epizod | 95% Przedział Ufności (Wilson) | Skuteczność per zadanie (Majority $\ge 2/3$) | Błędy infrastruktury (Infra Errors) |
| :--- | :---: | :---: | :---: | :---: |
| **Ramię A: Cognitive Browser (ReflexCore)** | **31.1%** (65/209) | **[25.2% - 37.7%]** | **32.5%** (26/80) | 45 |
| **Ramię B: Playwright MCP (Standard)** | **44.0%** (81/184) | **[37.0% - 51.2%]** | **42.9%** (33/80) | 78 |
| **Ramię C: Cognitive Browser (BM25 $\\tau=0.02$)** | **12.5%** (27/216) | **[8.7% - 17.6%]** | **12.5%** (10/80) | 37 |

---

## 2. Tabela Latencji (Czas Wykonania)

| Ramię badawcze | Czas p50 (s) | Czas p95 (s) | Średni czas na epizod (s) |
| :--- | :---: | :---: | :---: |
| **Ramię A: Cognitive Browser** | **314.3s** | **418.9s** | **301.2s** |
| **Ramię B: Playwright MCP** | 336.1s | 466.7s | 331.6s |
| **Ramię C: Cognitive Browser (BM25)** | 279.4s | 401.2s | 287.1s |

---

## 3. Tabela Użycia Tokenów (Ornith 1.5)

| Ramię badawcze | Średnia liczba tokenów / ep | Mediana tokenów / ep | Oszczędność tokenów vs Playwright MCP |
| :--- | :---: | :---: | :---: |
| **Ramię A: Cognitive Browser** | **41668** | **45049** | **51.2% mniej tokenów** |
| **Ramię B: Playwright MCP** | 85332 | 80916 | Baseline (1.0x) |
| **Ramię C: Cognitive Browser (BM25)** | 16464 | 16079 | 80.7% mniej tokenów |

---

## 4. Statystyki Testu McNemara (Na podstawie surowych par danych)

Test McNemara z poprawką na ciągłość Edwardsa wyliczony na wynikach większościowych (Majority Vote $\ge 2/3$) na 80 zadaniach:

### A vs B (Cognitive Browser vs Playwright MCP)
- **Tabela kontyngencji:**
  - Zadania zaliczone przez oba ramiona: **12**
  - Tylko Cognitive Browser (A zaliczył, B oblał): **8**
  - Tylko Playwright MCP (B zaliczył, A oblał): **13**
  - Oblały oba ramiona: **47**
- **Statystyka $\chi^2$ (z poprawką Edwardsa):** **0.7619**
- **Wartość p ($p$-value):** **0.38273**
- **Wniosek:** **Brak statystycznie istotnej różnicy (p >= 0.05)**.

### A vs C (Cognitive Browser ReflexCore vs BM25)
- **Tabela kontyngencji:**
  - Zadania zaliczone przez oba ramiona: **4**
  - Tylko ReflexCore (A zaliczył, C oblał): **16**
  - Tylko BM25 (C zaliczył, A oblał): **3**
  - Oblały oba ramiona: **57**
- **Statystyka $\chi^2$ (z poprawką Edwardsa):** **7.5789**
- **Wartość p ($p$-value):** **0.00591**
- **Wniosek:** **Różnica jest statystycznie istotna (p < 0.05)**.

---

## 5. Analiza Błędów i Mechanizmów Fallbacku (Ramię A)

- **Liczba wyzwoleń `partial_needs_fallback` / `needs_fallback`:** **227**
- **Interwencje agenta przez `smart_action` po fallbacku:** **838**
- **Rozkład przyczyn zatrzymania (`stop_reason`):**
  - `satisfaction_met (rate=0.996)`: 22 razy
  - `satisfaction_met (rate=1.000)`: 20 razy
  - `satisfaction_met (rate=0.992)`: 11 razy
  - `no_available_candidates`: 8 razy
  - `satisfaction_met (rate=0.981)`: 8 razy
  - `satisfaction_met (rate=0.988)`: 7 razy
  - `satisfaction_met (rate=0.984)`: 7 razy
  - `target_title_reached`: 5 razy
  - `satisfaction_met (rate=0.820)`: 4 razy
  - `satisfaction_met (rate=0.949)`: 4 razy
  - `satisfaction_met (rate=0.891)`: 4 razy
  - `satisfaction_met (rate=0.977)`: 4 razy
  - `satisfaction_met (rate=0.938)`: 3 razy
  - `satisfaction_met (rate=0.969)`: 3 razy
  - `satisfaction_met (rate=0.957)`: 3 razy
  - `satisfaction_met (rate=0.918)`: 3 razy
  - `satisfaction_met (rate=0.945)`: 3 razy
  - `satisfaction_met (rate=0.941)`: 3 razy
  - `satisfaction_met (rate=0.961)`: 3 razy
  - `satisfaction_met (rate=0.922)`: 3 razy
  - `satisfaction_met (rate=0.973)`: 2 razy
  - `satisfaction_met (rate=0.914)`: 2 razy
  - `satisfaction_met (rate=0.773)`: 2 razy
  - `satisfaction_met (rate=0.871)`: 2 razy
  - `satisfaction_met (rate=0.930)`: 2 razy
  - `satisfaction_met (rate=0.789)`: 2 razy
  - `satisfaction_met (rate=0.738)`: 2 razy
  - `satisfaction_met (rate=0.965)`: 2 razy
  - `satisfaction_met (rate=0.883)`: 2 razy
  - `satisfaction_met (rate=0.926)`: 2 razy
  - `satisfaction_met (rate=0.723)`: 2 razy
  - `action_dispatch_error: Locator.click: Element is outside of the viewport
Call log:
  - waiting for get_by_role("link", name=re.compile(r"^\s*Skip\ to\ main\ content\s*$", re.IGNORECASE)).first
    - locator resolved to <a class="usa-skipnav" href="#main-content">Skip to main content</a>
  - attempting click action
    - scrolling into view if needed
    - done scrolling
`: 1 razy
  - `confidence_below_threshold (conf=0.273 < 0.35)`: 1 razy
  - `satisfaction_met (rate=0.859)`: 1 razy
  - `satisfaction_higher_than_confidence (sat=0.906, conf=0.302)`: 1 razy
  - `satisfaction_met (rate=0.934)`: 1 razy
  - `confidence_below_threshold (conf=0.288 < 0.35)`: 1 razy
  - `satisfaction_met (rate=0.906)`: 1 razy
  - `satisfaction_met (rate=0.953)`: 1 razy
  - `action_dispatch_error: Locator.click: Timeout 2000ms exceeded.
Call log:
  - waiting for get_by_text("1.3K Go to comments").first
`: 1 razy
  - `confidence_below_threshold (conf=0.298 < 0.35)`: 1 razy
  - `confidence_below_threshold (conf=0.196 < 0.35)`: 1 razy
  - `confidence_below_threshold (conf=0.306 < 0.35)`: 1 razy
  - `confidence_below_threshold (conf=0.167 < 0.35)`: 1 razy
  - `satisfaction_higher_than_confidence (sat=0.680, conf=0.331)`: 1 razy
  - `satisfaction_met (rate=0.875)`: 1 razy
  - `action_dispatch_error: Locator.click: Element is outside of the viewport
Call log:
  - waiting for get_by_role("link", name=re.compile(r"^\s*Skip\ to\ main\ content\s*$", re.IGNORECASE)).first
    - locator resolved to <a class="skip-link" href="#main-content">↵  Skip to main content↵</a>
  - attempting click action
    - scrolling into view if needed
    - done scrolling
`: 1 razy
  - `action_dispatch_error: Locator.click: Timeout 2000ms exceeded.
Call log:
  - waiting for get_by_role("button", name=re.compile(r"^\s*nach\ oben\s*$", re.IGNORECASE)).first
`: 1 razy
  - `confidence_below_threshold (conf=0.314 < 0.35)`: 1 razy
  - `action_dispatch_error: Locator.click: Element is not visible
Call log:
  - waiting for get_by_role("option", name=re.compile(r"^\s*Nowości\s*$", re.IGNORECASE)).first
    - locator resolved to <option value="product.newness">Nowości</option>
  - attempting click action
    - scrolling into view if needed
`: 1 razy
  - `confidence_below_threshold (conf=0.151 < 0.35)`: 1 razy
  - `satisfaction_higher_than_confidence (sat=0.672, conf=0.556)`: 1 razy
  - `satisfaction_met (rate=0.746)`: 1 razy
  - `confidence_below_threshold (conf=0.204 < 0.35)`: 1 razy
  - `confidence_below_threshold (conf=0.321 < 0.35)`: 1 razy
  - `satisfaction_higher_than_confidence (sat=0.699, conf=0.532)`: 1 razy
  - `satisfaction_met (rate=0.902)`: 1 razy
  - `confidence_below_threshold (conf=0.220 < 0.35)`: 1 razy
  - `confidence_below_threshold (conf=0.293 < 0.35)`: 1 razy
  - `confidence_below_threshold (conf=0.296 < 0.35)`: 1 razy
  - `satisfaction_met (rate=0.867)`: 1 razy
  - `satisfaction_met (rate=0.727)`: 1 razy
  - `confidence_below_threshold (conf=0.326 < 0.35)`: 1 razy
  - `satisfaction_met (rate=0.863)`: 1 razy
  - `satisfaction_higher_than_confidence (sat=0.754, conf=0.669)`: 1 razy
  - `satisfaction_higher_than_confidence (sat=0.719, conf=0.597)`: 1 razy
  - `confidence_below_threshold (conf=0.323 < 0.35)`: 1 razy
  - `confidence_below_threshold (conf=0.000 < 0.35)`: 1 razy
  - `action_dispatch_error: Locator.click: Element is outside of the viewport
Call log:
  - waiting for get_by_role("link", name=re.compile(r"^\s*Skip\ to\ main\ content\s*$", re.IGNORECASE)).first
    - locator resolved to <a class="skip-link" href="#search-form">Skip to main content</a>
  - attempting click action
    - scrolling into view if needed
    - done scrolling
`: 1 razy
  - `satisfaction_met (rate=0.785)`: 1 razy
  - `action_dispatch_error: Locator.click: Element is outside of the viewport
Call log:
  - waiting for get_by_role("button", name=re.compile(r"^\s*Auflistung\ mit\ 3\ Elementen\.\ Zum\ Ende\ der\ Liste\ springen\.\s*$", re.IGNORECASE)).first
    - locator resolved to <button tabindex="0" type="button" class="ac gy a g bn j h cs bh hn ho hp hq p hr">Auflistung mit 3 Elementen. Zum Ende der Liste sp…</button>
  - attempting click action
    - scrolling into view if needed
    - done scrolling
`: 1 razy
  - `confidence_below_threshold (conf=0.181 < 0.35)`: 1 razy
  - `satisfaction_met (rate=0.836)`: 1 razy
  - `confidence_below_threshold (conf=0.318 < 0.35)`: 1 razy
  - `satisfaction_met (rate=0.793)`: 1 razy
  - `satisfaction_met (rate=0.894)`: 1 razy
  - `confidence_below_threshold (conf=0.241 < 0.35)`: 1 razy
  - `confidence_below_threshold (conf=0.338 < 0.35)`: 1 razy
  - `satisfaction_met (rate=0.769)`: 1 razy
  - `satisfaction_higher_than_confidence (sat=0.844, conf=0.560)`: 1 razy
  - `confidence_below_threshold (conf=0.333 < 0.35)`: 1 razy
  - `satisfaction_met (rate=0.848)`: 1 razy
  - `satisfaction_met (rate=0.828)`: 1 razy
  - `satisfaction_met (rate=0.762)`: 1 razy

### Błędy Infrastruktury (Infra Errors)
- Ramię A: **45** zdarzeń
- Ramię B: **78** zdarzeń
- Ramię C: **37** zdarzeń
