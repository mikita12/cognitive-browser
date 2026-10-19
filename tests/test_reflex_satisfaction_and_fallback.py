"""
Unit tests for Reflex page satisfaction requiring specific goal phrases
and cognitive_browse returning explicit 'needs_fallback' status on 0 steps.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from cognitive_browser.reflex import ReflexCore
from cognitive_browser.orchestrator import BrowserOrchestrator


def test_extract_specific_goal_phrases():
    """Verify that specific substantive phrases are extracted while generic words are excluded."""
    goal_pl = "Sprawdź aktualnie podawaną stopę bezrobocia rejestrowanego w Polsce"
    phrases_pl = ReflexCore.extract_specific_goal_phrases(goal_pl)
    assert any("bezroboc" in p for p in phrases_pl)
    assert "sprawdź" not in phrases_pl
    assert "strona" not in phrases_pl

    goal_en = "On vuejs.org, open the 'Guide', navigate via the sidebar to 'Reactivity Fundamentals'"
    phrases_en = ReflexCore.extract_specific_goal_phrases(goal_en)
    assert "reactivity fundamentals" in phrases_en or "reactivity" in phrases_en
    assert "open" not in phrases_en
    assert "page" not in phrases_en

    goal_de = "Auf der Startseite der Bundeszentrale für politische Bildung das 'Politik-Lexikon' öffnen"
    phrases_de = ReflexCore.extract_specific_goal_phrases(goal_de)
    assert "politik-lexikon" in phrases_de or "politik" in phrases_de
    assert "öffnen" not in phrases_de
    assert "startseite" not in phrases_de


def test_page_satisfaction_rejects_generic_numbers_without_specific_phrases():
    """
    Synthetic snapshot with lots of numbers, currencies, and macroeconomic data
    must NOT satisfy a goal seeking 'stopa bezrobocia rejestrowanego' because specific
    goal phrases are missing from the page.
    """
    reflex = ReflexCore()
    goal = "Sprawdź aktualnie podawaną stopę bezrobocia rejestrowanego w Polsce"
    generic_macro_snapshot = (
        '- heading "Główny Urząd Statystyczny - Portal Informacyjny"\n'
        '- text "Podstawowe wskaźniki makroekonomiczne"\n'
        '- text "Produkt Krajowy Brutto: wzrost o 3.2% r/r"\n'
        '- text "Wskaźnik cen towarów i usług konsumpcyjnych: 4.8%"\n'
        '- text "Przeciętne miesięczne wynagrodzenie: 8200 zł"\n'
        '- text "Wskaźnik produkcji sprzedanej przemysłu: 102.5"\n'
        '- link "Aktualności i komunikaty"\n'
        '- link "Baza danych"\n'
        '- text "Stopa referencyjna NBP: 5.75%"\n'
    )

    # 1. Structural check must be False
    assert not ReflexCore.is_goal_satisfied_on_page(
        goal, generic_macro_snapshot, "GUS - Główny Urząd Statystyczny", "https://stat.gov.pl"
    )

    # 2. evaluate_page_satisfaction must NOT declare satisfaction
    sat_res = reflex.evaluate_page_satisfaction(
        goal, generic_macro_snapshot, "GUS - Główny Urząd Statystyczny", "https://stat.gov.pl"
    )
    assert sat_res["is_satisfied"] is False
    assert sat_res["goal_satisfaction_rate"] <= 0.40


def test_page_satisfaction_accepts_when_specific_phrases_present():
    """
    Synthetic snapshot containing the specific goal phrase ('stopa bezrobocia rejestrowanego')
    and targeted data points must satisfy the goal.
    """
    reflex = ReflexCore()
    goal = "Sprawdź aktualnie podawaną stopę bezrobocia rejestrowanego w Polsce"
    target_snapshot = (
        '- heading "Stopa bezrobocia rejestrowanego w Polsce"\n'
        '- text "Komunikat Prezesa Głównego Urzędu Statystycznego"\n'
        '- text "Stopa bezrobocia rejestrowanego w końcu miesiąca wyniosła 5.0%"\n'
        '- text "Liczba bezrobotnych zarejestrowanych w urzędach pracy: 780.2 tys."\n'
        '- text "Przeciętna stopa bezrobocia w województwach: od 3.1% do 8.4%"\n'
    )

    assert ReflexCore.has_specific_goal_match(goal, target_snapshot)
    sat_res = reflex.evaluate_page_satisfaction(
        goal, target_snapshot, "Stopa bezrobocia rejestrowanego | GUS", "https://stat.gov.pl/rynek-pracy/bezrobocie"
    )
    assert sat_res["is_satisfied"] is True
    assert sat_res["goal_satisfaction_rate"] >= 0.70


@pytest.mark.anyio
async def test_cognitive_browse_returns_needs_fallback_on_confidence_below_threshold():
    """
    Verifies that when no candidates exceed MIN_GOAL_CONFIDENCE and 0 steps are executed,
    cognitive_browse returns status='needs_fallback' with explicit stop_reason and message,
    NOT status='success'.
    """
    mock_engine = MagicMock()
    mock_engine.auto_dismiss_overlays = AsyncMock()

    mock_reflex = MagicMock()
    mock_reflex.evaluate_page_satisfaction.return_value = {
        "is_satisfied": False,
        "goal_satisfaction_rate": 0.20,
        "latency_ms": 5.0,
        "ranked_chunks": [],
    }
    # Return low confidence below MIN_GOAL_CONFIDENCE (0.50)
    mock_reflex.evaluate_candidates.return_value = {
        "confidence": 0.32,
        "latency_ms": 8.0,
        "ranked_candidates": [
            {"role": "link", "name": "Irrelevant Footer Link", "ref": "aria-1"}
        ],
        "should_escalate": True,
        "target_ref": "NONE",
    }

    orchestrator = BrowserOrchestrator(engine=mock_engine, reflex=mock_reflex)

    mock_page = AsyncMock()
    mock_page.url = "https://example.com/portal"
    mock_page.title.return_value = "Example Portal"
    # Snapshot with some links so candidates exist
    mock_page.aria_snapshot.return_value = (
        '- heading "Portal Title"\n'
        '- link "Irrelevant Footer Link"\n'
        '- link "Privacy Policy"\n'
    )

    res = await orchestrator.browse(
        mock_page,
        url="https://example.com/portal",
        goal="Find specific target section that is not present here",
        browser_context="ephemeral",
    )

    assert res["status"] == "needs_fallback"
    assert res["steps_executed"] == 0
    assert "confidence_below_threshold" in res["stop_reason"]
    assert "Autonomous navigation stopped" in res["message"]
    assert res["selection_source"] == "reranker"


@pytest.mark.anyio
async def test_cognitive_browse_returns_needs_fallback_on_no_candidates():
    """
    Verifies that when a page has 0 interactive candidates and goal is not satisfied,
    cognitive_browse returns status='needs_fallback' instead of 'success'.
    """
    mock_engine = MagicMock()
    mock_engine.auto_dismiss_overlays = AsyncMock()

    mock_reflex = MagicMock()
    mock_reflex.evaluate_page_satisfaction.return_value = {
        "is_satisfied": False,
        "goal_satisfaction_rate": 0.10,
        "latency_ms": 2.0,
        "ranked_chunks": [],
    }

    orchestrator = BrowserOrchestrator(engine=mock_engine, reflex=mock_reflex)

    mock_page = AsyncMock()
    mock_page.url = "https://example.com/empty"
    mock_page.title.return_value = "Empty Page"
    # Snapshot with only text, no clickable candidates
    mock_page.aria_snapshot.return_value = '- heading "Just Static Text"\n- text "Nothing to click here"\n'

    res = await orchestrator.browse(
        mock_page,
        url="https://example.com/empty",
        goal="Search for data",
        browser_context="ephemeral",
    )

    assert res["status"] == "needs_fallback"
    assert res["steps_executed"] == 0
    assert res["stop_reason"] == "no_available_candidates"


def test_has_specific_goal_match_prefix_stemming():
    """
    Verifies that prefix-stemming handles inflections in Polish/German (e.g. 'kawalerki' vs 'kawalerka')
    and requires at least min(2, len(specific)) stems when multiple specific terms exist.
    """
    goal = "znajdź tanie kawalerki w Gdańsku"
    # Specific terms: ['kawalerki', 'gdańsku'] (len 2)
    # Stems: 'kawalerk', 'gdańsk'

    # Inflected Polish on page: "Kawalerka, Gdańsk - Ogłoszenia"
    assert ReflexCore.has_specific_goal_match(goal, "Kawalerka, Gdańsk - Ogłoszenia nieruchomości")

    # Only one stem matches: "Mieszkanie w Gdańsku" (only 'gdańsk', 'kawalerk' missing) -> False
    assert not ReflexCore.has_specific_goal_match(goal, "Mieszkanie w Gdańsku - Ogłoszenia")

    # Neither matches -> False
    assert not ReflexCore.has_specific_goal_match(goal, "Polska - portal główny")

    # Goal with single specific word
    single_word_goal = "otwórz dokumentację Pythona"
    assert ReflexCore.has_specific_goal_match(single_word_goal, "Witaj na stronie Python")


def test_page_satisfaction_requires_multiple_specific_stems():
    """
    Verifies that a generic page with only one matching word (e.g. country name 'Polsce')
    does not satisfy a multi-term goal.
    """
    goal = "Sprawdź bezrobocie w Polsce"
    # Page only mentions Polska, not bezrobocie
    portal_snapshot = (
        '- heading "Rzeczpospolita Polska - Portal Rządowy"\n'
        '- text "Oficjalny serwis informacyjny dla obywateli Polski"\n'
        '- link "Aktualności"\n'
        '- text "Dziś w Polsce odnotowano wzrost gospodarczy"\n'
    )
    assert not ReflexCore.is_goal_satisfied_on_page(goal, portal_snapshot, "Gov.pl - Portal", "https://gov.pl")


def test_footnote_filtering_and_penalization():
    """
    Verifies that footnote/citation links (e.g. [1], [2], 42) are filtered out as noise
    unless the goal explicitly asks for citations/footnotes, and penalized in candidate scoring.
    """
    intent_general = "znajdź informację o historii komputera"
    intent_citation = "zobacz przypis [1] do bibliografii"

    c_bracket_footnote = {"role": "link", "name": "[1]", "ref": "aria-1"}
    c_num_footnote = {"role": "link", "name": "42", "ref": "aria-2"}
    c_article_link = {"role": "link", "name": "Historia informatyki", "ref": "aria-3"}

    # General intent: footnotes are noise
    assert ReflexCore.is_noise_candidate(c_bracket_footnote, intent_general) is True
    assert ReflexCore.is_noise_candidate(c_num_footnote, intent_general) is True
    assert ReflexCore.is_noise_candidate(c_article_link, intent_general) is False

    # Citation intent: footnotes are NOT noise
    assert ReflexCore.is_noise_candidate(c_bracket_footnote, intent_citation) is False

    # Empty links are noise
    assert ReflexCore.is_noise_candidate({"role": "link", "name": "", "ref": "aria-4"}, intent_general) is True


@pytest.mark.anyio
async def test_cognitive_browse_returns_partial_needs_fallback_when_steps_executed_gt_0():
    """
    Verifies that when cognitive_browse executes >= 1 step and then encounters
    confidence_below_threshold on a subsequent step, it returns 'partial_needs_fallback',
    a descriptive Polish message, and top_candidates from the current page.
    """
    mock_engine = MagicMock()
    mock_engine.auto_dismiss_overlays = AsyncMock()

    mock_reflex = MagicMock()
    mock_reflex.evaluate_page_satisfaction.return_value = {
        "is_satisfied": False,
        "goal_satisfaction_rate": 0.20,
        "latency_ms": 5.0,
        "ranked_chunks": [],
    }

    # First call: high confidence (executes step 1)
    # Second call: low confidence (< 0.50), triggers stop_reason confidence_below_threshold
    mock_reflex.evaluate_candidates.side_effect = [
        {
            "confidence": 0.88,
            "latency_ms": 10.0,
            "ranked_candidates": [
                {"role": "link", "name": "Kategoria Samochody", "ref": "aria-1"}
            ],
            "should_escalate": False,
            "target_ref": "aria-1",
        },
        {
            "confidence": 0.20,
            "latency_ms": 10.0,
            "ranked_candidates": [
                {"role": "link", "name": "Niejasny link", "ref": "aria-2"},
                {"role": "link", "name": "Pomoc", "ref": "aria-3"},
            ],
            "should_escalate": True,
            "target_ref": "NONE",
        },
    ]

    orchestrator = BrowserOrchestrator(engine=mock_engine, reflex=mock_reflex)
    # Stub dispatch_step_action to avoid real browser calls
    orchestrator.dispatch_step_action = AsyncMock(return_value="click('Kategoria Samochody')")

    mock_page = AsyncMock()
    mock_page.url = "https://example.com/start"
    mock_page.title.return_value = "Example Marketplace"
    mock_page.aria_snapshot.return_value = (
        '- heading "Marketplace"\n'
        '- link "Kategoria Samochody"\n'
        '- link "Niejasny link"\n'
    )

    res = await orchestrator.browse(
        mock_page,
        url="https://example.com/start",
        goal="Wyszukaj ogłoszenia samochodowe",
        browser_context="ephemeral",
    )

    assert res["status"] == "partial_needs_fallback"
    assert res["steps_executed"] == 1
    assert "Wykonano 1 kroków, ale dalsza autonomiczna nawigacja utknęła" in res["message"]
    assert "confidence_below_threshold" in res["stop_reason"]
    assert len(res["top_candidates"]) > 0
    assert res["top_candidates"][0]["name"] == "Niejasny link"

