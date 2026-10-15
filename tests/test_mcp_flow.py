"""
Test MCP server tools and Agent-in-the-Loop escalation workflow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognitive_browser.mcp_server import cognitive_browse, smart_action


@pytest.mark.anyio
async def test_cognitive_browse_high_confidence():
    """Verify that high-confidence steps execute autonomously without LLM escalation."""
    mock_reflex = MagicMock()
    mock_reflex.evaluate_candidates.return_value = {
        "target_ref": "aria-1",
        "confidence": 0.95,
        "should_escalate": False,
        "latency_ms": 22.5,
        "ranked_candidates": [{"role": "link", "name": "Documentation", "confidence_score": 0.95}],
    }

    mock_engine = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://example.com/docs"
    mock_page.title.return_value = "Documentation"
    mock_page.aria_snapshot.return_value = '- link "Documentation"'
    mock_engine.get_page = AsyncMock(return_value=mock_page)
    mock_engine.click_element = AsyncMock(return_value=True)
    mock_engine.auto_dismiss_overlays = AsyncMock(return_value=False)

    with patch("cognitive_browser.mcp_server.get_services", return_value=(mock_reflex, mock_engine)):
        res = await cognitive_browse(url="https://example.com", steps=[{"instruction": "Open documentation"}])

        assert res["status"] == "success"
        assert res["final_url"] == "https://example.com/docs"
        assert res["steps_executed"] == 1
        assert res["telemetry"]["tokens_consumed_by_server"] == 0
        mock_engine.click_element.assert_awaited_once()


@pytest.mark.anyio
async def test_cognitive_browse_clarification_needed_escalation():
    """Verify that ambiguous steps return clarification_needed with candidate choices to the orchestrator."""
    mock_reflex = MagicMock()
    mock_reflex.evaluate_candidates.return_value = {
        "target_ref": "NONE",
        "confidence": 0.35,
        "should_escalate": True,
        "latency_ms": 25.0,
        "ranked_candidates": [
            {"role": "link", "name": "Article Discussion", "confidence_score": 0.35},
            {"role": "link", "name": "Comments", "confidence_score": 0.30},
        ],
    }

    mock_engine = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://news.ycombinator.com/newest"
    mock_page.title.return_value = "New Links | Hacker News"
    mock_page.aria_snapshot.return_value = '- link "Article Discussion"\n- link "Comments"'
    mock_engine.get_page = AsyncMock(return_value=mock_page)
    mock_engine.auto_dismiss_overlays = AsyncMock(return_value=False)

    with patch("cognitive_browser.mcp_server.get_services", return_value=(mock_reflex, mock_engine)):
        res = await cognitive_browse(
            url="https://news.ycombinator.com", steps=[{"instruction": "Open comments of the first story"}]
        )

        assert res["status"] == "clarification_needed"
        assert res["current_url"] == "https://news.ycombinator.com/newest"
        assert res["pending_step"] == "Open comments of the first story"
        assert len(res["top_candidates"]) == 2
        assert "call smart_action" in res["message"]


@pytest.mark.anyio
async def test_smart_action_resolves_via_orchestrator_choice():
    """Verify that smart_action directly executes the target chosen by the orchestrator."""
    mock_reflex = MagicMock()
    mock_engine = MagicMock()
    mock_page = AsyncMock()
    mock_page.url = "https://news.ycombinator.com/item?id=123"
    mock_page.title.return_value = "Discussion"
    mock_engine.get_page = AsyncMock(return_value=mock_page)
    mock_engine.click_element = AsyncMock(return_value=True)
    mock_engine.auto_dismiss_overlays = AsyncMock(return_value=False)

    with patch("cognitive_browser.mcp_server.get_services", return_value=(mock_reflex, mock_engine)):
        res = await smart_action(instruction="Open comments", target_name="Comments")

        assert res["status"] == "success"
        assert res["target"] == "[link] Comments"
        assert res["current_url"] == "https://news.ycombinator.com/item?id=123"
        mock_engine.click_element.assert_awaited_once_with(mock_page, role="link", name="Comments")
