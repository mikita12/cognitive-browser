"""
Test AriaElementParser parsing functionality.
"""
import unittest
from cognitive_browser import AriaElementParser


class TestAriaElementParser(unittest.TestCase):
    def setUp(self):
        self.sample_snapshot = """
- banner:
  - heading "OLX.pl" [level=1]
  - link "Zaloguj się":
    - /url: "/mojolx/"
- main:
  - searchbox "Czego szukasz?"
  - combobox "Lokalizacja":
    - /placeholder: "Wpisz miejscowość..."
  - button "Szukaj"
  - list:
    - listitem:
      - link "iPhone 15 Pro 128GB - Stan Idealny":
        - /url: "/d/oferta/iphone-15-pro-CID99.html"
        - text: "3 400 zł do negocjacji"
"""

    def test_parse_candidates(self):
        candidates = AriaElementParser.parse_snapshot(self.sample_snapshot)
        self.assertGreater(len(candidates), 0)

        # Check searchbox parsed
        searchbox = next((c for c in candidates if c["role"] == "searchbox"), None)
        self.assertIsNotNone(searchbox)
        self.assertEqual(searchbox["name"], "Czego szukasz?")

        # Check button parsed
        button = next((c for c in candidates if c["role"] == "button"), None)
        self.assertIsNotNone(button)
        self.assertEqual(button["name"], "Szukaj")

        # Check link with URL parsed
        login_link = next((c for c in candidates if c["name"] == "Zaloguj się"), None)
        self.assertIsNotNone(login_link)
        self.assertEqual(login_link["url"], "/mojolx/")

    def test_require_input_only_filter(self):
        candidates = AriaElementParser.parse_snapshot(
            self.sample_snapshot,
            require_input_only=True
        )
        roles = {c["role"] for c in candidates}
        # Only input roles should remain
        for role in roles:
            self.assertIn(role, {"textbox", "searchbox", "combobox"})


if __name__ == "__main__":
    unittest.main()
