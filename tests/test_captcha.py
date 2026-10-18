"""
Unit tests for automated CAPTCHA detection and solving module (CaptchaResolver & ReflexCore).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognitive_browser.captcha import CaptchaResolver
from cognitive_browser.config import get_capsolver_api_key
from cognitive_browser.reflex import ReflexCore


def test_capsolver_key_loading():
    """Verifies that the CapSolver key is retrieved from env or file."""
    key = get_capsolver_api_key()
    assert isinstance(key, str)
    # If .env or env var exists, it should start with CAP-
    if key:
        assert key.startswith("CAP-")


def test_evaluate_captcha_presence_heuristic():
    """Verifies ReflexCore detects CAPTCHA via page title and snapshot heuristics."""
    reflex = ReflexCore()
    res = reflex.evaluate_captcha_presence(
        candidates=[],
        page_title="Just a moment... | Cloudflare",
        page_url="https://example.com/login",
        snapshot_yaml="- heading 'Verify you are human'",
    )
    assert res["is_captcha"] is True
    assert res["confidence"] >= 0.80
    assert res["heuristic_trigger"] is True
    assert res["captcha_type"] == "cloudflare_interstitial"
    assert "neural_score" in res
    assert "latency_ms" in res


def test_evaluate_captcha_presence_clean_page():
    """Verifies ReflexCore returns low confidence on normal content pages."""
    reflex = ReflexCore()
    candidates = [
        {"role": "link", "name": "Documentation", "context": "nav"},
        {"role": "button", "name": "Download PDF", "context": "main"},
    ]
    res = reflex.evaluate_captcha_presence(
        candidates=candidates,
        page_title="Python Documentation",
        page_url="https://docs.python.org",
        snapshot_yaml="- heading 'Welcome to Python'",
    )
    assert res["is_captcha"] is False
    assert res["captcha_type"] == "none"
    assert res["confidence"] < 0.50
    assert res["heuristic_trigger"] is False


def test_evaluate_captcha_presence_passive_footer_ignored():
    """Verifies that passive Google reCAPTCHA badge in footer does NOT trigger CAPTCHA challenge alarm."""
    reflex = ReflexCore()
    candidates = [
        {"role": "textbox", "name": "Username", "context": "form"},
        {"role": "textbox", "name": "Password", "context": "form"},
        {"role": "button", "name": "Sign In", "context": "form"},
        {"role": "link", "name": "Privacy Policy", "context": "footer (Protected by reCAPTCHA)"},
        {"role": "link", "name": "Terms of Service", "context": "footer"},
    ]
    snap = (
        "- heading 'Sign In to Account'\n"
        "- textbox 'Username'\n"
        "- textbox 'Password'\n"
        "- button 'Sign In'\n"
        "- text 'This site is protected by reCAPTCHA and the Google Privacy Policy and Terms of Service apply.'\n"
        "- link 'Privacy Policy'\n"
        "- link 'Terms of Service'"
    )
    res = reflex.evaluate_captcha_presence(
        candidates=candidates,
        page_title="Login to Portal",
        page_url="https://example.com/login",
        snapshot_yaml=snap,
    )
    assert res["is_captcha"] is False
    assert res["captcha_type"] == "none"
    assert res["confidence"] < 0.35
    assert res["heuristic_trigger"] is False
    assert "neural_score" in res
    assert "latency_ms" in res


def test_evaluate_captcha_presence_active_recaptcha_challenge():
    """Verifies that interactive checkbox 'I'm not a robot' is identified as active reCAPTCHA."""
    reflex = ReflexCore()
    candidates = [
        {"role": "checkbox", "name": "I'm not a robot", "context": "recaptcha-anchor"},
    ]
    snap = "- heading 'Security Verification'\n- checkbox \"I'm not a robot\"\n- text 'Protected by reCAPTCHA'"
    res = reflex.evaluate_captcha_presence(
        candidates=candidates,
        page_title="Verify you are human",
        page_url="https://example.com/verify",
        snapshot_yaml=snap,
    )
    assert res["is_captcha"] is True
    assert res["captcha_type"] == "recaptcha"
    assert res["confidence"] >= 0.85
    assert res["heuristic_trigger"] is True
    assert res["matched_candidate"] is not None


@pytest.mark.anyio
async def test_detect_turnstile_from_page():
    """Verifies Turnstile detection from DOM state."""
    page_mock = AsyncMock()
    page_mock.url = "https://apply.workable.com/job/apply/"
    page_mock.evaluate.return_value = {
        "has_turnstile": True,
        "sitekey": "0x4AAAAAAAVY8hH3nz6RxaK0",
        "action": "application-form-submit",
        "container": "#turnstile-container",
    }

    resolver = CaptchaResolver(api_key="TEST_KEY")
    detected = await resolver.detect_captcha(page_mock)

    assert detected is not None
    assert detected["type"] == "turnstile"
    assert detected["sitekey"] == "0x4AAAAAAAVY8hH3nz6RxaK0"
    assert detected["action"] == "application-form-submit"


@pytest.mark.anyio
async def test_detect_recaptcha_v2():
    """Verifies reCAPTCHA detection from DOM state."""
    page_mock = AsyncMock()
    page_mock.url = "https://example.com/form"
    # First turnstile returns None, then recaptcha returns data
    page_mock.evaluate.side_effect = [
        {"has_turnstile": False},
        {"has_recaptcha": True, "sitekey": "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-", "is_v3": False},
    ]

    resolver = CaptchaResolver(api_key="TEST_KEY")
    detected = await resolver.detect_captcha(page_mock)

    assert detected is not None
    assert detected["type"] == "recaptcha_v2"
    assert detected["sitekey"] == "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-"


@pytest.mark.anyio
async def test_solve_turnstile_mock():
    """Verifies CapSolver task creation and polling logic."""
    resolver = CaptchaResolver(api_key="CAP-TEST-123")
    captcha_info = {
        "type": "turnstile",
        "sitekey": "0x4AAAAAAAVY8hH3nz6RxaK0",
        "url": "https://apply.workable.com",
    }

    create_resp_mock = MagicMock()
    create_resp_mock.__enter__.return_value = create_resp_mock
    create_resp_mock.read.return_value = json.dumps({"taskId": "task-test-456", "errorId": 0}).encode()

    poll_resp_mock = MagicMock()
    poll_resp_mock.__enter__.return_value = poll_resp_mock
    poll_resp_mock.read.return_value = json.dumps(
        {
            "status": "ready",
            "solution": {"token": "1.test_valid_turnstile_token_mock"},
            "errorId": 0,
        }
    ).encode()

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [create_resp_mock, poll_resp_mock]
        res = await resolver.solve(captcha_info, timeout=5.0)

        assert res["status"] == "ready"
        assert res["token"] == "1.test_valid_turnstile_token_mock"
        assert res["type"] == "turnstile"
        assert res["latency_ms"] >= 0.0


@pytest.mark.anyio
async def test_inject_token_turnstile():
    """Verifies client callback invocation on page."""
    page_mock = AsyncMock()
    page_mock.evaluate.return_value = {"called": True, "inputs": 1}

    resolver = CaptchaResolver(api_key="TEST_KEY")
    res = await resolver.inject_token(page_mock, {"type": "turnstile"}, "token_abc")
    assert res is True
    page_mock.evaluate.assert_called_once()
