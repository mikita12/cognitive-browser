"""
Unit and integration tests for secure file upload allowlisting.
Verifies UPLOAD_ALLOWED_DIRS enforcement, symlink resolution, traversal mitigation, and SecurityError propagation.
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from cognitive_browser.browser import BrowserEngine
from cognitive_browser.config import get_upload_allowed_dirs
from cognitive_browser.security import SecurityError


@pytest.fixture
def temp_upload_dirs():
    """Creates isolated allowed and forbidden directories with test files and symlinks."""
    with tempfile.TemporaryDirectory() as base_dir:
        allowed_dir = os.path.join(base_dir, "allowed")
        forbidden_dir = os.path.join(base_dir, "forbidden")
        os.makedirs(allowed_dir, exist_ok=True)
        os.makedirs(forbidden_dir, exist_ok=True)

        valid_file = os.path.join(allowed_dir, "valid.pdf")
        with open(valid_file, "w") as f:
            f.write("Valid document")

        secret_file = os.path.join(forbidden_dir, "secret.txt")
        with open(secret_file, "w") as f:
            f.write("Sensitive info")

        # Symlink inside allowed pointing outside to forbidden
        escaping_symlink = os.path.join(allowed_dir, "symlink_escape.txt")
        os.symlink(secret_file, escaping_symlink)

        # Normal symlink inside allowed pointing to a file inside allowed
        safe_symlink = os.path.join(allowed_dir, "safe_symlink.pdf")
        os.symlink(valid_file, safe_symlink)

        yield {
            "base_dir": base_dir,
            "allowed_dir": allowed_dir,
            "forbidden_dir": forbidden_dir,
            "valid_file": valid_file,
            "secret_file": secret_file,
            "escaping_symlink": escaping_symlink,
            "safe_symlink": safe_symlink,
        }


def test_empty_allowlist_blocks_upload(temp_upload_dirs, monkeypatch):
    """Empty UPLOAD_ALLOWED_DIRS must block all file uploads by default."""
    monkeypatch.setenv("UPLOAD_ALLOWED_DIRS", "")
    assert get_upload_allowed_dirs() == []

    with pytest.raises(SecurityError) as exc_info:
        BrowserEngine.validate_upload_path(temp_upload_dirs["valid_file"])
    assert "UPLOAD_ALLOWED_DIRS is empty" in str(exc_info.value)


def test_path_outside_allowlist_blocked(temp_upload_dirs, monkeypatch):
    """File outside allowed directory must raise SecurityError."""
    monkeypatch.setenv("UPLOAD_ALLOWED_DIRS", temp_upload_dirs["allowed_dir"])

    with pytest.raises(SecurityError) as exc_info:
        BrowserEngine.validate_upload_path(temp_upload_dirs["secret_file"])
    assert "outside allowed upload directories" in str(exc_info.value)


def test_symlink_escaping_allowlist_blocked(temp_upload_dirs, monkeypatch):
    """Symlink residing in allowed dir but pointing outside must be rejected after realpath resolution."""
    monkeypatch.setenv("UPLOAD_ALLOWED_DIRS", temp_upload_dirs["allowed_dir"])

    with pytest.raises(SecurityError) as exc_info:
        BrowserEngine.validate_upload_path(temp_upload_dirs["escaping_symlink"])
    assert "outside allowed upload directories" in str(exc_info.value)


def test_dot_dot_traversal_blocked(temp_upload_dirs, monkeypatch):
    """Path traversal attempting ../ into another directory must be rejected."""
    monkeypatch.setenv("UPLOAD_ALLOWED_DIRS", temp_upload_dirs["allowed_dir"])
    traversal_path = os.path.join(temp_upload_dirs["allowed_dir"], "..", "forbidden", "secret.txt")

    with pytest.raises(SecurityError) as exc_info:
        BrowserEngine.validate_upload_path(traversal_path)
    assert "outside allowed upload directories" in str(exc_info.value)


def test_valid_file_and_safe_symlink_allowed(temp_upload_dirs, monkeypatch):
    """Legitimate files and safe intra-directory symlinks must pass validation and return resolved paths."""
    monkeypatch.setenv("UPLOAD_ALLOWED_DIRS", temp_upload_dirs["allowed_dir"])

    res1 = BrowserEngine.validate_upload_path(temp_upload_dirs["valid_file"])
    assert res1 == os.path.realpath(temp_upload_dirs["valid_file"])

    res2 = BrowserEngine.validate_upload_path(temp_upload_dirs["safe_symlink"])
    assert res2 == os.path.realpath(temp_upload_dirs["valid_file"])


def test_nonexistent_file_rejected(temp_upload_dirs, monkeypatch):
    """Non-existent files must be rejected even if under allowed directory."""
    monkeypatch.setenv("UPLOAD_ALLOWED_DIRS", temp_upload_dirs["allowed_dir"])
    fake_path = os.path.join(temp_upload_dirs["allowed_dir"], "ghost.txt")

    with pytest.raises(SecurityError) as exc_info:
        BrowserEngine.validate_upload_path(fake_path)
    assert "does not exist" in str(exc_info.value)


def test_directory_target_rejected(temp_upload_dirs, monkeypatch):
    """Passing a directory instead of a regular file must be rejected."""
    monkeypatch.setenv("UPLOAD_ALLOWED_DIRS", temp_upload_dirs["allowed_dir"])

    with pytest.raises(SecurityError) as exc_info:
        BrowserEngine.validate_upload_path(temp_upload_dirs["allowed_dir"])
    assert "is a directory, not a file" in str(exc_info.value)


@pytest.mark.anyio
async def test_upload_file_raises_security_error(temp_upload_dirs, monkeypatch):
    """BrowserEngine.upload_file raises SecurityError when upload target violates allowlist."""
    monkeypatch.setenv("UPLOAD_ALLOWED_DIRS", temp_upload_dirs["allowed_dir"])
    engine = BrowserEngine()
    mock_page = MagicMock()

    with pytest.raises(SecurityError):
        await engine.upload_file(mock_page, file_paths=temp_upload_dirs["secret_file"])
