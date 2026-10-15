"""
Test AriaElementParser parsing functionality.
"""

import unittest

from cognitive_browser import AriaElementParser


class TestAriaElementParser(unittest.TestCase):
    def setUp(self):
        self.sample_snapshot = """
- banner:
  - heading "Marketplace" [level=1]
  - link "Sign In":
    - /url: "/account/login"
- main:
  - searchbox "What are you looking for?"
  - combobox "Location":
    - /placeholder: "Enter city..."
  - button "Search"
  - list:
    - listitem:
      - link "iPhone 15 Pro 128GB - Mint Condition":
        - /url: "/item/iphone-15-pro-123.html"
        - text: "$850 negotiable"
"""

    def test_parse_candidates(self):
        candidates = AriaElementParser.parse_snapshot(self.sample_snapshot)
        self.assertGreater(len(candidates), 0)

        # Check searchbox parsed
        searchbox = next((c for c in candidates if c["role"] == "searchbox"), None)
        self.assertIsNotNone(searchbox)
        assert searchbox is not None
        self.assertEqual(searchbox["name"], "What are you looking for?")

        # Check button parsed
        button = next((c for c in candidates if c["role"] == "button"), None)
        self.assertIsNotNone(button)
        assert button is not None
        self.assertEqual(button["name"], "Search")

        # Check link with URL parsed
        login_link = next((c for c in candidates if c["name"] == "Sign In"), None)
        self.assertIsNotNone(login_link)
        assert login_link is not None
        self.assertEqual(login_link["url"], "/account/login")

    def test_require_input_only_filter(self):
        candidates = AriaElementParser.parse_snapshot(self.sample_snapshot, require_input_only=True)
        roles = {c["role"] for c in candidates}
        # Only input roles should remain
        for role in roles:
            self.assertIn(role, {"textbox", "searchbox", "combobox"})


if __name__ == "__main__":
    unittest.main()
