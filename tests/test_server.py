"""
Unit tests for Cognitive Browser FastAPI REST API and authentication controls.
"""

import os
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

            # 5. Valid token query parameter -> 200
            resp_query = self.client.get(f"/v1/mcp/tools?token={secret_token}")
            self.assertEqual(resp_query.status_code, 200)
            self.assertIn("tools", resp_query.json())


if __name__ == "__main__":
    unittest.main()
