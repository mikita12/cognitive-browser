"""
Test RouterFactory logic and fallback mechanism.
"""

import os
import unittest
from unittest.mock import patch

from cognitive_browser import ApiDecisionRouter, get_decision_router


class TestRouterFactory(unittest.TestCase):
    def test_explicit_api_mode(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test123"}):
            router = get_decision_router(mode="api")
            self.assertIsInstance(router, ApiDecisionRouter)
            assert isinstance(router, ApiDecisionRouter)
            self.assertEqual(router.provider, "groq")

    def test_explicit_api_mode_cohere(self):
        with patch.dict(os.environ, {"COHERE_API_KEY": "coh_test123"}, clear=True):
            router = get_decision_router(mode="api")
            self.assertIsInstance(router, ApiDecisionRouter)
            assert isinstance(router, ApiDecisionRouter)
            self.assertEqual(router.provider, "cohere")


if __name__ == "__main__":
    unittest.main()
