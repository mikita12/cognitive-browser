"""
Unit tests for Cognitive Browser FastAPI REST API and authentication controls.
"""

import os
import secrets
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from cognitive_browser.server import app


class TestServerAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_health_check_unauthenticated(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")

    def test_mcp_tools_list_without_token(self):
        # When API_TOKEN is empty, endpoints are publicly accessible
        with patch.dict(os.environ, {"API_TOKEN": ""}):
            response = self.client.get("/v1/mcp/tools")
            self.assertEqual(response.status_code, 200)
            self.assertIn("tools", response.json())

    def test_auth_token_enforcement(self):
        secret_token = "test-secret-token-12345"
        with patch.dict(os.environ, {"API_TOKEN": secret_token}):
            # 1. Health check is always public
            resp_health = self.client.get("/health")
            self.assertEqual(resp_health.status_code, 200)

            # 2. Unauthenticated request to protected endpoint -> 401
            resp_no_token = self.client.get("/v1/mcp/tools")
            self.assertEqual(resp_no_token.status_code, 401)
            self.assertIn("Unauthorized", resp_no_token.json().get("detail", ""))

            # 3. Invalid token -> 401
            resp_bad_token = self.client.get("/v1/mcp/tools", headers={"Authorization": "Bearer wrong-token"})
            self.assertEqual(resp_bad_token.status_code, 401)

            # 4. Valid Bearer token header -> 200
            resp_valid = self.client.get("/v1/mcp/tools", headers={"Authorization": f"Bearer {secret_token}"})
            self.assertEqual(resp_valid.status_code, 200)
            self.assertIn("tools", resp_valid.json())

            # 5. Token in query parameter must be rejected (only Authorization: Bearer is accepted) -> 401
            resp_query = self.client.get(f"/v1/mcp/tools?token={secret_token}")
            self.assertEqual(resp_query.status_code, 401)
            self.assertIn("Unauthorized", resp_query.json().get("detail", ""))

            # 6. Verify constant-time comparison via secrets.compare_digest
            with patch("secrets.compare_digest", wraps=secrets.compare_digest) as mock_digest:
                resp_mock = self.client.get("/v1/mcp/tools", headers={"Authorization": f"Bearer {secret_token}"})
                self.assertEqual(resp_mock.status_code, 200)
                mock_digest.assert_called_once_with(secret_token, secret_token)

    def test_cors_middleware_configuration(self):
        # Default CORS settings: no allowed origins, allow_credentials=False
        from fastapi.middleware.cors import CORSMiddleware
        cors_middlewares = [m for m in app.user_middleware if m.cls == CORSMiddleware]
        self.assertTrue(len(cors_middlewares) > 0)
        options = cors_middlewares[0].kwargs
        self.assertFalse(options.get("allow_credentials", True))
        self.assertEqual(options.get("allow_origins"), [])

    def test_sandbox_flag_configuration(self):
        # Verifies that sandbox args are only added when COGNITIVE_BROWSER_NO_SANDBOX=1
        with patch.dict(os.environ, {"COGNITIVE_BROWSER_NO_SANDBOX": "0"}):
            self.assertEqual(os.getenv("COGNITIVE_BROWSER_NO_SANDBOX"), "0")

        with patch.dict(os.environ, {"COGNITIVE_BROWSER_NO_SANDBOX": "1"}):
            self.assertEqual(os.getenv("COGNITIVE_BROWSER_NO_SANDBOX"), "1")

    def test_model_and_device_env_aliases_with_deprecation_warning(self):
        import warnings
        from cognitive_browser.config import get_device, get_reflex_model

        # 1. COGNITIVE_BROWSER_MODEL takes precedence
        with patch.dict(os.environ, {"COGNITIVE_BROWSER_MODEL": "custom/model-cb", "MODEL_NAME": "old/model-legacy"}):
            self.assertEqual(get_reflex_model(), "custom/model-cb")

        # 2. MODEL_NAME emits deprecation warning
        with patch.dict(os.environ, {"MODEL_NAME": "legacy/model-only"}, clear=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = get_reflex_model()
                self.assertEqual(model, "legacy/model-only")
                self.assertTrue(any(issubclass(item.category, DeprecationWarning) and "MODEL_NAME" in str(item.message) for item in w))

        # 3. COGNITIVE_BROWSER_DEVICE takes precedence
        with patch.dict(os.environ, {"COGNITIVE_BROWSER_DEVICE": "cpu", "DEVICE": "cuda"}):
            self.assertEqual(get_device(), "cpu")

        # 4. DEVICE emits deprecation warning
        with patch.dict(os.environ, {"DEVICE": "mps"}, clear=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                device = get_device()
                self.assertEqual(device, "mps")
                self.assertTrue(any(issubclass(item.category, DeprecationWarning) and "DEVICE" in str(item.message) for item in w))


if __name__ == "__main__":
    unittest.main()
