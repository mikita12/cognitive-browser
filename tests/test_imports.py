"""
Test package imports and version definition.
"""

import unittest


class TestPackageImports(unittest.TestCase):
    def test_package_exports(self):
        import cognitive_browser

        self.assertTrue(hasattr(cognitive_browser, "__version__"))
        self.assertEqual(cognitive_browser.__version__, "0.1.0")

    def test_core_classes_exported(self):
        from cognitive_browser import (
            AgentCore,
            AgentOrchestrator,
            ApiDecisionRouter,
            AriaElementParser,
            DebertaDecisionRouter,
            get_decision_router,
        )

        self.assertIsNotNone(DebertaDecisionRouter)
        self.assertIsNotNone(ApiDecisionRouter)
        self.assertIsNotNone(get_decision_router)
        self.assertIsNotNone(AgentCore)
        self.assertIsNotNone(AriaElementParser)
        self.assertIsNotNone(AgentOrchestrator)


if __name__ == "__main__":
    unittest.main()
