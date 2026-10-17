# Raport z Etapu 5: Uniwersalny Mechanizm Wiązania Parametrów (`payload`), Inteligentny Upload Plików oraz Import Sesji Systemowych (`browser_context`)

**Data raportu**: 16 października 2026 r.  
**Środowisko**: Linux (Fedora 41), NVIDIA GeForce RTX 4070 Ti, CUDA 12.8, PyTorch 2.6.0+cu124, Playwright 1.50+  
**Model lokalny**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (ModernBERT / Cross-Encoder GPU)

---

## 1. Wstęp i Cel Architektoniczny

Głównym problemem w integracji modeli językowych z serwerem MCP `cognitive-browser` była konieczność ręcznego eksplorowania stron, szukania formularzy i wyciągania selektorów przez LLM (co powodowało zapętlanie się modeli w inspekcji kodu źródłowego). 

W Etapie 5 wdrożono beztokenowy, uniwersalny mechanizm deklaratywnego przekazywania danych biznesowych:
1. **Uniwersalny parametr `payload: Dict[str, Any]`**: Umożliwia przekazanie dowolnych danych formularza (CV, dane kandydata, filtry samochodowe Otomoto, parametry logistyczne, koszyk e-commerce, galeria zdjęć).
2. **Semantyczne mapowanie dwudzielne (Neural Bipartite Matching)**: `ReflexCore.bind_payload_to_elements` dokonuje optymalnego dopasowania 1-do-1 między polami `payload` a elementami drzewa ARIA snapshotu bez sztywnych reguł regex czy schematów bazodanowych.
3. **Inteligentny Upload Plików (`upload_file`)**: Obsługuje zarówno klasyczne `<input type="file">`, jak i nowoczesne dropzony React/Vue/Angular otwierające okno wyboru pliku po kliknięciu przycisku (`page.expect_file_chooser()`), ze wsparciem dla pojedynczych plików (`str`) i list wielu plików (`List[str]`).
4. **Import sesji i ciasteczek przeglądarki (`browser_context`)**: Odczytuje autentykowane sesje i ciasteczka bez szyfrowania z bazy `cookies.sqlite` profilu Firefox (`~/.mozilla/firefox/*.default*`), pozwalając natychmiast omijać blokady Cloudflare / DataDome na stronach takich jak Pracuj.pl czy Otomoto.

---

## 2. Kluczowe Zmiany w Kodzie

### 2.1. Moduł Ciasteczek Systemowych: `cognitive_browser/cookies.py`
- `find_firefox_profiles()`: Automatycznie wykrywa domyślne profile Firefox (w tym instalacje standardowe, Flatpak i Snap).
- `extract_firefox_cookies(domain, profile_path)`: Bezpiecznie kopiuje bazę `cookies.sqlite` do pliku tymczasowego (zapobiegając konfliktom blokady bazy z działającą przeglądarką) i wyciąga ciasteczka pasujące do żądanej domeny w formacie Playwright.
- `import_system_cookies(context_or_page, domain, browser_type)`: Wstrzykuje ciasteczka do kontekstu Playwright (`context.add_cookies()`).

### 2.2. Rozbudowa Silnika Przeglądarki: `cognitive_browser/browser.py`
- `upload_file(page, role, name, file_paths, nth)`:
  1. Sprawdza, czy element to bezpośredni `<input type="file">` i wywołuje `set_input_files`.
  2. Jeśli to przycisk lub dropzone, przechwytuje zdarzenie wyboru pliku: `async with page.expect_file_chooser(): await locator.click()`.
  3. Obsługuje zagnieżdżone elementy w kontenerze oraz fallback do pierwszego pola pliku na stronie.
- `import_cookies_for_domain(page, domain, browser_type)`: Automatyczny import ciasteczek sesyjnych przed nawigacją.
- `select_option(page, role, name, value, nth)`: Inteligentna obsługa wyboru opcji w kontrolkach `<select>` oraz customowych dropdownach (wyszukiwanie po `combobox`, `ancestor::select`, fallback do page locator).
- `execute_reflex_action(page, action_plan)`: Obsługa akcji `upload_file`.

### 2.3. Semantyczny Bipartite Matcher: `cognitive_browser/reflex.py`
- `bind_payload_to_elements(payload, candidates, min_confidence=0.30)`:
  - Automatycznie klasyfikuje pola w `payload` na pliki (po rozszerzeniach `.pdf`, `.png`, `.docx` lub słowach kluczowych `cv`, `plik`, `file`, `foto`), flagi boolean oraz tekst/select.
  - Generuje wielojęzyczne zapytania semantyczne (PL/EN) dla Cross-Encodera GPU.
  - Wylicza affinities: penalizuje próby przypisania emaila/imienia do przycisków uploadu pliku (-0.50) oraz promuje powiązania ról (checkbox dla boolean, combobox dla select).
  - Wykonuje zachłanne dopasowanie dwudzielne o najwyższym scorze (1-to-1).
  - Sortuje akcje według kolejności DOM (od góry do dołu strony), gwarantując stabilne wypełnianie formularza.

### 2.4. Serwer MCP i Microservice: `mcp_server.py`, `server.py`, `cognitive_browse.json`
- Do `cognitive_browse` dodano opcjonalne parametry:
  - `payload: Optional[Dict[str, Any]] = None`
  - `browser_context: Optional[str] = "ephemeral"` (`"firefox"`, `"system"` lub ścieżka do pliku `.json`)
- W trybie autonomicznym: jeśli przekazano `payload`, serwer sam wypełnia wszystkie pola formularza w jednym kroku GPU, a następnie (jeśli podano `goal`) klika wyślij i weryfikuje sukces.
- Zaktualizowano schemat JSON narzędzia w `~/.gemini/antigravity-cli/mcp/cognitive-browser/cognitive_browse.json`.
- Dodano endpoint `/v1/bind_payload` w REST API `cognitive_browser/server.py`.

---

## 3. Wyniki Testów Jednostkowych i Integracyjnych

Uruchomiono pełny pakiet testów projektu za pomocą `pytest`:

```text
================== 60 passed, 1 xfailed, 2 warnings in 28.61s ==================
```

Wszystkie nowe testy w `tests/test_payload_binding.py` zakończyły się pełnym sukcesem (9/9):
1. `test_job_application_payload_binding` – PASSED
2. `test_multi_file_gallery_upload` – PASSED
3. `test_combobox_select_binding` – PASSED
4. `test_upload_file_direct_input` – PASSED
5. `test_upload_file_chooser_event` – PASSED
6. `test_import_system_cookies_into_context` – PASSED
7. `test_cognitive_browse_with_payload` – PASSED
8. `test_extract_firefox_cookies_from_mock_db` – PASSED
9. `test_load_cookies_from_json` – PASSED

Statyczny linter `ruff` oraz kontrola typów `mypy`:
- `uv run ruff check cognitive_browser tests`: **All checks passed!**
- `uv run mypy cognitive_browser/`: **Success: no issues found in 10 source files.**

---

## 4. Wyniki Testu End-to-End w Przeglądarce Chromium

Przeprowadzono pełny test E2E na lokalnym serwerze HTTP (`benchmarks/run_payload_e2e_benchmark.py`) z symulowanym zaawansowanym formularzem rekrutacyjnym:
- Imię, Nazwisko, E-mail, Telefon
- Dropdown select: Poziom doświadczenia ("Senior")
- Upload pojedynczego pliku CV (`.pdf`) przez przycisk ukrytego inputu (`onclick="document.getElementById('cv_input').click()"`)
- Upload wielu plików certyfikatów (`.png`) przez przycisk `<input multiple>`
- Checkbox zgody RODO
- Weryfikacja odczytu ciasteczek profilu Firefox (`pracuj.pl`)

### Raport z Wykonania (`benchmarks/results_payload_e2e.json`):

| Parametr / Pole | Wykryty Element ARIA | Wykonana Akcja | Czas Wykonania | Status DOM |
| :--- | :--- | :--- | :--- | :--- |
| `first_name` | `textbox "Imię"` | `fill('Antoni')` | 9.24 ms | ✅ Poprawny |
| `last_name` | `textbox "Nazwisko"` | `fill('Zieliński')` | 4.63 ms | ✅ Poprawny |
| `email` | `textbox "Adres e-mail"` | `fill('antoni.zielinski@example.com')` | 6.49 ms | ✅ Poprawny |
| `phone` | `textbox "Numer telefonu"` | `fill('+48 501 234 567')` | 4.44 ms | ✅ Poprawny |
| `experience` | `combobox "Poziom doświadczenia"` | `select('Senior')` | 16.56 ms | ✅ Wybrano 'Senior' |
| `cv` | `button "Załącz dokument CV"` | `upload_file(['tmpwp6ovsif.pdf'])` | 49.87 ms | ✅ Plik załączony |
| `certificates` | `button "Dodaj certyfikaty..."` | `upload_file(['*.png', '*.png'])` | 46.78 ms | ✅ 2 pliki załączone |
| `rodo` | `checkbox "Zgadzam się..."` | `check()` | 29.57 ms | ✅ Zaznaczony (True) |
| **Przesłanie** | `button "Wyślij aplikację"` | `click()` | 12.10 ms | ✅ Sukces formularza |

- **Czas dopasowania neuronowego 8 pól na GPU**: **56.25 ms**
- **Ciasteczka pobrane z Firefox dla Pracuj.pl**: **24 ciasteczka** (w tym `REDACTED`, clearance)
- **Zero błędnych przypisań**: E-mail i telefon nie trafiły do przycisków uploadu; CV nie trafiło do galerii.
