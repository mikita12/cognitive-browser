"""
Test package imports and version definition.
"""

import unittest


class TestPackageImports(unittest.TestCase):
    def test_package_exports(self):
        import cognitive_browser
        from cognitive_browser.server import app

        self.assertTrue(hasattr(cognitive_browser, "__version__"))
        self.assertEqual(cognitive_browser.__version__, "0.3.0")
        self.assertEqual(app.version, "0.3.0")
        self.assertEqual(app.version, cognitive_browser.__version__)

    def test_core_classes_exported(self):
        from cognitive_browser import (
            AgentCore,
            AriaElementParser,
            BrowserEngine,
            DebertaDecisionRouter,
            ReflexCore,
        )

        self.assertIsNotNone(ReflexCore)
        self.assertIsNotNone(BrowserEngine)
        self.assertIsNotNone(AriaElementParser)
        self.assertIsNotNone(DebertaDecisionRouter)
        self.assertIsNotNone(AgentCore)


if __name__ == "__main__":
    unittest.main()
