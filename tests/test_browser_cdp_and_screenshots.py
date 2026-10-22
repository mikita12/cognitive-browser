"""
Unit tests for BrowserEngine CDP attach authorization guards and error screenshot capture.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognitive_browser.browser import BrowserEngine
from cognitive_browser.security import SecurityError


@pytest.mark.anyio
async def test_cdp_attach_rejected_without_allow_flag(monkeypatch):
    """Connecting over CDP without COGNITIVE_BROWSER_ALLOW_CDP=1 raises SecurityError."""
    monkeypatch.setenv("COGNITIVE_BROWSER_CDP_URL", "http://localhost:9222")
    monkeypatch.delenv("COGNITIVE_BROWSER_ALLOW_CDP", raising=False)

    engine = BrowserEngine()
    with patch("cognitive_browser.browser.async_playwright") as mock_pw_start:
        mock_pw = MagicMock()
        mock_pw_start.return_value.start = AsyncMock(return_value=mock_pw)

        with pytest.raises(SecurityError) as exc_info:
            await engine.get_page()

        assert "requires explicit permission" in str(exc_info.value)
        assert "COGNITIVE_BROWSER_ALLOW_CDP=1" in str(exc_info.value)


@pytest.mark.anyio
async def test_cdp_attach_allowed_with_flag(monkeypatch, caplog):
    """Connecting over CDP with COGNITIVE_BROWSER_ALLOW_CDP=1 succeeds and logs security warning."""
    import logging

    monkeypatch.setenv("COGNITIVE_BROWSER_CDP_URL", "http://localhost:9222")
    monkeypatch.setenv("COGNITIVE_BROWSER_ALLOW_CDP", "1")

    engine = BrowserEngine()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_context.pages = [mock_page]
    mock_browser.contexts = [mock_context]

    with (
        patch("cognitive_browser.browser.async_playwright") as mock_pw_start,
        patch("cognitive_browser.browser.setup_ssrf_route_guard", new_callable=AsyncMock),
        caplog.at_level(logging.WARNING),
    ):
        mock_pw = MagicMock()
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_pw_start.return_value.start = AsyncMock(return_value=mock_pw)

        page = await engine.get_page()
        assert page is mock_page
        mock_pw.chromium.connect_over_cdp.assert_awaited_once_with("http://localhost:9222")
        assert any("SECURITY WARNING: Attaching to browser via CDP" in record.message for record in caplog.records)


@pytest.mark.anyio
async def test_capture_screenshot_disabled_by_default(monkeypatch):
    """By default (no explicit path, no env dir), screenshot capture is disabled and returns None."""
    monkeypatch.delenv("COGNITIVE_BROWSER_ERROR_SCREENSHOTS_DIR", raising=False)
    monkeypatch.delenv("COGNITIVE_BROWSER_ERROR_SCREENSHOTS_FULL_PAGE", raising=False)

    engine = BrowserEngine()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    engine._active_page = mock_page

    res = await engine.capture_screenshot()
    assert res is None
    mock_page.screenshot.assert_not_called()


@pytest.mark.anyio
async def test_capture_screenshot_enabled_via_env(monkeypatch, tmp_path):
    """When COGNITIVE_BROWSER_ERROR_SCREENSHOTS_DIR is set, saves viewport screenshot in that directory."""
    monkeypatch.setenv("COGNITIVE_BROWSER_ERROR_SCREENSHOTS_DIR", str(tmp_path))
    monkeypatch.delenv("COGNITIVE_BROWSER_ERROR_SCREENSHOTS_FULL_PAGE", raising=False)

    engine = BrowserEngine()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.screenshot = AsyncMock()
    engine._active_page = mock_page

    saved_path = await engine.capture_screenshot()
    assert saved_path is not None
    assert str(tmp_path) in saved_path
    assert saved_path.endswith(".png")
    mock_page.screenshot.assert_awaited_once()
    _, kwargs = mock_page.screenshot.call_args
    assert kwargs["full_page"] is False


@pytest.mark.anyio
async def test_capture_screenshot_full_page_flag(monkeypatch, tmp_path):
    """When COGNITIVE_BROWSER_ERROR_SCREENSHOTS_FULL_PAGE=1, passes full_page=True."""
    monkeypatch.setenv("COGNITIVE_BROWSER_ERROR_SCREENSHOTS_DIR", str(tmp_path))
    monkeypatch.setenv("COGNITIVE_BROWSER_ERROR_SCREENSHOTS_FULL_PAGE", "1")

    engine = BrowserEngine()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.screenshot = AsyncMock()
    engine._active_page = mock_page

    saved_path = await engine.capture_screenshot()
    assert saved_path is not None
    mock_page.screenshot.assert_awaited_once()
    _, kwargs = mock_page.screenshot.call_args
    assert kwargs["full_page"] is True
