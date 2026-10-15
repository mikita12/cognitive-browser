"""
Test package imports and version definition.
"""

import unittest


class TestPackageImports(unittest.TestCase):
    def test_package_exports(self):
        import cognitive_browser

        self.assertTrue(hasattr(cognitive_browser, "__version__"))
        self.assertEqual(cognitive_browser.__version__, "0.2.0")

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
