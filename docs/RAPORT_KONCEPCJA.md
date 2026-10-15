# Projekt: Agentic Decision Router (System 1 + System 2 w automatyzacji Playwright)

## 1. Wprowadzenie i Problem Badawczy

Współczesne autonomiczne agenty webowe (oparte na narzędziach takich jak Playwright i sterowane dużymi modelami językowymi – LLM) mierzą się z dwoma kluczowymi wąskimi gardłami:
1. **Ogromne zużycie kontekstu (Token Bloat):** Zrzuty drzewa DOM lub drzewa dostępności (Accessibility Tree) stron internetowych generują od kilkunastu do kilkudziesięciu tysięcy tokenów na jeden krok przeglądarki.
2. **Wysokie opóźnienia i koszty (High Latency & Costs):** Angażowanie potężnego, generatywnego modelu (np. Claude Opus / GPT-4o) do trywialnych decyzji binarnych (np. *„Czy ten przycisk to 'Dalej'?”*, *„Czy modal z cookies jest widoczny?”*, *„Czy strona się załadowała?”*) trwa 2–5 sekund na akcję i drastycznie drenuje budżet na API.

---

## 2. Istota Pomysłu (Koncepcja Architektury)

Inspiracją jest podział kognitywny na **System 1** (szybkie, intuicyjne, zero-generatywne decyzje) oraz **System 2** (głębokie, powolne rozumowanie i planowanie).

Zamiast przekazywać pełny stan przeglądarki bezpośrednio do głównego LLM-a, w pętli sterowania umieszczamy **lokalny model dyskryminacyjny / decyzyjny** (np. `nimble` w Ollamie lub model NLI typu `DeBERTa`).

```
                    ┌─────────────────────────┐
                    │  Playwright DOM / A11y  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     System 1 Layer      │
                    │ Model Decyzyjny (Ollama)│ <── Błyskawiczna analiza (20-50 ms)
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
         Decyzja jednoznaczna            Sytuacja złożona
         (Reguła / Prosty krok)         (Wymaga planowania)
                 │                               │
                 ▼                               ▼
       ┌───────────────────┐           ┌───────────────────┐
       │ Bezpośrednia Akcja│           │ Skompresowany Stan│
       │ w Playwrighcie    │           │ do Głównego LLM   │
       │ (Zero kosztu LLM) │           │ (System 2)        │
       └───────────────────┘           └───────────────────┘
```

---

## 3. Kluczowe Role Modelu Decyzyjnego

### A. Filtr Szumu (Pruning Drzewa Elementów)
* Model decyzyjny równolegle ewaluuje węzły strony pod kątem przydatności do aktualnego celu.
* Elementy dekoracyjne, banery reklamowe i powtarzalny szum są odcinane lokalnie.
* **Efekt:** Główny LLM otrzymuje tylko 5–10% najistotniejszych elementów, redukując okno kontekstu o 85–90%.

### B. Fast-Path (Obsługa zdarzeń rutynowych i przeszkód)
* **Obsługa modali / zgód:** *„Czy na ekranie pojawił się modal RODO/Cookies? TAK/NIE”* $\rightarrow$ jeśli TAK, kliknięcie następuje natychmiast z poziomu predefiniowanej procedury Playwrighta.
* **Wykrywanie końca zadania:** *„Czy na ekranie widnieje potwierdzenie sukcesu? TAK/NIE”* $\rightarrow$ przerwanie pętli bez pytania LLM-a.
* **Wykrywanie błędów blokujących:** *„Czy pojawiła się Captcha / blokada Cloudflare? TAK/NIE”*.

### C. Weryfikacja Akcji (Speculative Verification)
* Po wykonaniu kliknięcia przez Playwrighta model decyzyjny porównuje stan przed i po akcji.
* Pytanie binarne: *„Czy akcja przyniosła zamierzoną zmianę w interfejsie? TAK/NIE”*.
* Jeśli stan się nie zmienił (np. kliknięcie nie trafiło w element z powodu animacji), pętla lokalna ponawia akcję bez angażowania drogiego modelu.

---

## 4. Przewidywane Korzyści

| Metryka | Klasyczny Agent LLM | Agent z Routerem Decyzyjnym | Zysk |
| :--- | :--- | :--- | :--- |
| **Zużycie tokenów / krok** | 10 000 – 25 000 tokenów | 500 – 2 000 tokenów | **Oszczędność ~85-90%** |
| **Średni czas kroku rutynowego** | 2.5 – 5.0 s | 30 – 150 ms | **Przyspieszenie do 20x** |
| **Koszt API na 100 kroków** | Wysoki ($0.50 - $2.00) | Niski ($0.05 - $0.15) | **Redukcja kosztów ~80%** |
| **Niezawodność pętli** | Ryzyko halucynacji formatu | Deterministyczny JSON / Enum | **100% poprawności typu** |

---

## 5. Zrealizowany Stos Technologiczny (RTX 4070 Ti 12 GB VRAM)

1. **System 1 (Dyskryminacyjny Model Decyzyjny - DeBERTa-v3):**
   * Model: `cross-encoder/nli-deberta-v3-small` uruchomiony natywnie w PyTorch na GPU (`cuda:0`).
   * Metoda wnioskowania: **Zero-Shot NLI (Softmax Entailment vs Contradiction)**.
   * Czas reakcji: **39.9 ms**.
   * Wygenerowane tokeny: **0**.
2. **Automatyzacja Webowa:**
   * Ekstraktor `DomCandidateExtractor` z Playwrightem (`extractor.py`).
   * Zabezpieczenie przed halucynacjami: znaczniki `data-s1-ref`.
3. **System 2 (Główny Planista):**
   * Angażowany tylko w przypadku `should_escalate = True` (gdy żaden element nie osiągnie progu pewności).

---

## 6. Realne Wyniki z Benchmarku na Żywo

```
================================================================================
Metoda                         | Typ Modelu         | Opóźnienie   | Koszt tokenów  
--------------------------------------------------------------------------------
Tradycyjny Agent (Claude/GPT)  | Generatywny API    | ~3000 - 5000 ms | ~12 000 tok./krok
Ollama (Qwen2.5-1.5B)          | Generatywny LLM    | ~400 - 450 ms | Lokalny (0 zł) 
DeBERTa-v3 NLI (Nasz Router)   | Dyskryminacyjny    | 39.96 ms     | 0 tokenów (0 zł)
================================================================================
```
Akcja w Playwrighcie została wykonana natychmiastowo z zerowym błędem selektora.
