"""
Test AriaElementParser parsing functionality and RouteCache.
"""

import os
import tempfile
import unittest

from cognitive_browser import AriaElementParser, RouteCache


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
      - heading "iPhone 15"
      - link "iPhone 15 Pro 128GB - Mint Condition":
        - /url: "/item/iphone-15-pro-123.html"
        - text: "$850 negotiable"
      - button "Buy Now"
    - listitem:
      - heading "Galaxy S24"
      - button "Buy Now"
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
        button = next((c for c in candidates if c["role"] == "button" and c["name"] == "Search"), None)
        self.assertIsNotNone(button)
        assert button is not None
        self.assertEqual(button["name"], "Search")

        # Check link with URL parsed
        login_link = next((c for c in candidates if c["name"] == "Sign In"), None)
        self.assertIsNotNone(login_link)
        assert login_link is not None
        self.assertEqual(login_link["url"], "/account/login")

    def test_hierarchical_context_and_nth(self):
        candidates = AriaElementParser.parse_snapshot(self.sample_snapshot)
        buy_buttons = [c for c in candidates if c["role"] == "button" and c["name"] == "Buy Now"]
        self.assertEqual(len(buy_buttons), 2)
        self.assertEqual(buy_buttons[0]["nth"], 0)
        self.assertEqual(buy_buttons[1]["nth"], 1)
        self.assertIn("iPhone 15", buy_buttons[0]["context"])
        self.assertIn("Galaxy S24", buy_buttons[1]["context"])

    def test_require_input_only_filter(self):
        candidates = AriaElementParser.parse_snapshot(self.sample_snapshot, require_input_only=True)
        roles = {c["role"] for c in candidates}
        # Only input roles should remain
        for role in roles:
            self.assertIn(role, {"textbox", "searchbox", "combobox"})

    def test_pua_icon_stripping(self):
        snapshot_with_icons = """
- main:
  - button "\ue0c3 Wyszukaj"
  - button "Zakładanie firmy \uea23"
"""
        candidates = AriaElementParser.parse_snapshot(snapshot_with_icons)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["name"], "Wyszukaj")
        self.assertEqual(candidates[1]["name"], "Zakładanie firmy")

    def test_domain_models_and_parse_elements(self):
        from cognitive_browser.models import ActionCommand, CandidateElement

        elements = AriaElementParser.parse_elements(self.sample_snapshot)
        self.assertGreater(len(elements), 0)
        first = elements[0]
        self.assertIsInstance(first, CandidateElement)
        self.assertEqual(first.role, "link")
        self.assertEqual(first.name, "Sign In")
        self.assertEqual(first.url, "/account/login")
        self.assertEqual(first.section, "header")
        self.assertTrue(first.is_header)
        # Dict compatibility check
        self.assertEqual(first["role"], "link")
        self.assertEqual(first.get("url"), "/account/login")

        cmd = ActionCommand(action="click", target=first, confidence=0.95)
        self.assertEqual(cmd.role, "link")
        self.assertEqual(cmd.name, "Sign In")
        self.assertEqual(cmd.url, "/account/login")
        self.assertEqual(cmd["action"], "click")

    def test_route_cache_store_and_get(self):
        tmp_file = os.path.join(tempfile.gettempdir(), "test_route_cache_unit.json")
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        cache = RouteCache(cache_file=tmp_file, enabled=True)
        cache.store("https://www.olx.pl/oferty/12345", "Kliknij Kup Teraz", role="button", name="Kup Teraz", nth=1)

        # Should match normalized URL even with different numeric ID
        hit = cache.get("https://olx.pl/oferty/99999", "  kliknij kup teraz ")
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit["role"], "button")
        self.assertEqual(hit["name"], "Kup Teraz")
        self.assertEqual(hit["nth"], 1)

        cache.invalidate("https://olx.pl/oferty/99999", "Kliknij Kup Teraz")
        self.assertIsNone(cache.get("https://olx.pl/oferty/99999", "Kliknij Kup Teraz"))
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    def test_reflex_action_inference_and_smart_stop(self):
        from cognitive_browser.reflex import ReflexCore

        # 1. Extract fill value from natural Polish & English instructions
        q1 = ReflexCore.extract_fill_value(
            'Wpisz w pole wyszukiwania: site:olx.pl "rtx 4070 ti" wrocław i kliknij szukaj'
        )
        self.assertEqual(q1, 'site:olx.pl "rtx 4070 ti" wrocław')

        q2 = ReflexCore.extract_fill_value('Wpisz "RTX 4070 Ti" w wyszukiwarkę')
        self.assertEqual(q2, "RTX 4070 Ti")

        # 2. Infer action plan for searchbox (fill_and_submit), checkbox (check), combobox (select)
        plan_search = ReflexCore.infer_action_plan(
            'Wpisz "RTX 4070 Ti" i wyszukaj', {"role": "searchbox", "name": "Czego szukasz?", "nth": 0}
        )
        self.assertEqual(plan_search["action"], "fill_and_submit")
        self.assertEqual(plan_search["value"], "RTX 4070 Ti")

        plan_check = ReflexCore.infer_action_plan(
            "Zaznacz filtr używane", {"role": "checkbox", "name": "Używane", "nth": 0}
        )
        self.assertEqual(plan_check["action"], "check")

        # 3. Noise link filtering
        self.assertTrue(
            ReflexCore.is_noise_candidate(
                {"role": "link", "name": "Polityka Prywatności", "context": "contentinfo"},
                "Znajdź oferty RTX 4070 Ti",
            )
        )
        self.assertFalse(
            ReflexCore.is_noise_candidate(
                {"role": "link", "name": "Polityka Prywatności", "context": "contentinfo"},
                "Otwórz Polityka Prywatności",
            )
        )

        # 4. Smart goal stop when page already lists prices/offers
        sample_offers_aria = "\n".join(
            [
                "- main:",
                '  - heading "Wyniki wyszukiwania RTX 4070 Ti" [level=1]',
                '  - link "Gigabyte RTX 4070 Ti Gaming OC":',
                "    - text: 2 850 zł",
                '  - link "ASUS TUF RTX 4070 Ti":',
                "    - text: 3 100 zł",
                '  - link "MSI Ventus RTX 4070 Ti":',
                "    - text: 2 799 zł",
            ]
            * 5
        )
        self.assertTrue(
            ReflexCore.is_goal_satisfied_on_page(
                "Znajdź oferty używanych kart RTX 4070 Ti i sprawdź ceny",
                sample_offers_aria,
                "RTX 4070 Ti - Wyniki",
                "https://www.olx.pl/wroclaw/q-rtx-4070-ti/",
            )
        )


if __name__ == "__main__":
    unittest.main()
