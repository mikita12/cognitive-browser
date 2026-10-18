"""
Unit and integration tests for universal payload parameter binding,
intelligent file upload support, and system cookie importation.
"""

import json
import os
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognitive_browser.browser import BrowserEngine
from cognitive_browser.cookies import (
    extract_firefox_cookies,
    import_system_cookies,
    load_cookies_from_file,
)
from cognitive_browser.mcp_server import cognitive_browse
from cognitive_browser.reflex import ReflexCore


@pytest.fixture(scope="module")
def reflex():
    return ReflexCore()


class TestPayloadBinding:
    def test_job_application_payload_binding(self, reflex):
        """Tests that a standard job application with CV and Cover Letter binds 1-to-1 without crosstalk."""
        candidates = [
            {"ref": "aria-0", "role": "textbox", "name": "Imię", "context": "Dane kandydata", "nth": 0},
            {"ref": "aria-1", "role": "textbox", "name": "Nazwisko", "context": "Dane kandydata", "nth": 0},
            {"ref": "aria-2", "role": "textbox", "name": "Adres e-mail", "context": "Kontakt", "nth": 0},
            {"ref": "aria-3", "role": "textbox", "name": "Numer telefonu", "context": "Kontakt", "nth": 0},
            {"ref": "aria-4", "role": "button", "name": "Załącz CV (PDF, DOCX)", "context": "Dokumenty", "nth": 0},
            {"ref": "aria-5", "role": "button", "name": "Załącz list motywacyjny", "context": "Dokumenty", "nth": 0},
            {
                "ref": "aria-6",
                "role": "checkbox",
                "name": "Akceptuję regulamin i politykę prywatności",
                "context": "Zgody",
                "nth": 0,
            },
        ]

        payload = {
            "imie": "Jan",
            "nazwisko": "Kowalski",
            "email": "jan.kowalski@example.com",
            "telefon": "+48 600 123 456",
            "cv": "/tmp/cv_jan_kowalski.pdf",
            "list_motywacyjny": "/tmp/cover_letter.pdf",
            "regulamin": True,
        }

        actions = reflex.bind_payload_to_elements(payload, candidates)
        assert len(actions) == 7

        action_map = {act["payload_key"]: act for act in actions}

        # Verify text fields
        assert action_map["imie"]["action"] == "fill"
        assert action_map["imie"]["name"] == "Imię"
        assert action_map["imie"]["value"] == "Jan"

        assert action_map["nazwisko"]["action"] == "fill"
        assert action_map["nazwisko"]["name"] == "Nazwisko"
        assert action_map["nazwisko"]["value"] == "Kowalski"

        assert action_map["email"]["action"] == "fill"
        assert action_map["email"]["name"] == "Adres e-mail"
        assert action_map["email"]["value"] == "jan.kowalski@example.com"

        assert action_map["telefon"]["action"] == "fill"
        assert action_map["telefon"]["name"] == "Numer telefonu"
        assert action_map["telefon"]["value"] == "+48 600 123 456"

        # Verify dedicated file uploads
        assert action_map["cv"]["action"] == "upload_file"
        assert action_map["cv"]["name"] == "Załącz CV (PDF, DOCX)"
        assert action_map["cv"]["file_paths"] == ["/tmp/cv_jan_kowalski.pdf"]

        assert action_map["list_motywacyjny"]["action"] == "upload_file"
        assert action_map["list_motywacyjny"]["name"] == "Załącz list motywacyjny"
        assert action_map["list_motywacyjny"]["file_paths"] == ["/tmp/cover_letter.pdf"]

        # Verify checkbox
        assert action_map["regulamin"]["action"] == "check"
        assert action_map["regulamin"]["role"] == "checkbox"

        # Verify strict DOM ordering (from index 0 to 6)
        names_order = [act["name"] for act in actions]
        expected_order = [
            "Imię",
            "Nazwisko",
            "Adres e-mail",
            "Numer telefonu",
            "Załącz CV (PDF, DOCX)",
            "Załącz list motywacyjny",
            "Akceptuję regulamin i politykę prywatności",
        ]
        assert names_order == expected_order

    def test_multi_file_gallery_upload(self, reflex):
        """Tests that a list of file paths correctly targets a photo gallery button/dropzone."""
        candidates = [
            {"ref": "aria-0", "role": "textbox", "name": "Tytuł ogłoszenia", "context": "Opis", "nth": 0},
            {"ref": "aria-1", "role": "button", "name": "Dodaj zdjęcia pojazdu", "context": "Galeria", "nth": 0},
            {"ref": "aria-2", "role": "textbox", "name": "Cena", "context": "Finanse", "nth": 0},
        ]

        photos = ["/tmp/photo1.jpg", "/tmp/photo2.jpg", "/tmp/photo3.png"]
        payload = {
            "tytul": "BMW Seria 3 2.0d M-Pakiet",
            "zdjecia": photos,
            "cena": "95000",
        }

        actions = reflex.bind_payload_to_elements(payload, candidates)
        assert len(actions) == 3

        photo_act = next(a for a in actions if a["payload_key"] == "zdjecia")
        assert photo_act["action"] == "upload_file"
        assert photo_act["name"] == "Dodaj zdjęcia pojazdu"
        assert photo_act["file_paths"] == photos

    def test_combobox_select_binding(self, reflex):
        """Tests that dropdowns and select fields trigger the select action."""
        candidates = [
            {"ref": "aria-0", "role": "combobox", "name": "Marka", "context": "Filtry", "nth": 0},
            {"ref": "aria-1", "role": "combobox", "name": "Model", "context": "Filtry", "nth": 0},
            {"ref": "aria-2", "role": "textbox", "name": "Cena do", "context": "Filtry", "nth": 0},
        ]

        payload = {
            "marka": "Audi",
            "model": "A6",
            "cena_do": "200000",
        }

        actions = reflex.bind_payload_to_elements(payload, candidates)
        assert len(actions) == 3

        marka_act = next(a for a in actions if a["payload_key"] == "marka")
        assert marka_act["action"] == "select"
        assert marka_act["value"] == "Audi"

        model_act = next(a for a in actions if a["payload_key"] == "model")
        assert model_act["action"] == "select"
        assert model_act["value"] == "A6"

    def test_cross_lingual_payload_binding_without_dictionary(self, reflex):
        """Verifies that English keys bind 1-to-1 to Polish form labels purely via structural partitioning + Cross-Encoder."""
        candidates = [
            {"ref": "aria-0", "role": "textbox", "name": "Imię", "context": "Formularz"},
            {"ref": "aria-1", "role": "textbox", "name": "Nazwisko", "context": "Formularz"},
            {"ref": "aria-2", "role": "textbox", "name": "Adres e-mail", "context": "Formularz"},
            {"ref": "aria-3", "role": "textbox", "name": "Numer telefonu", "context": "Formularz"},
            {"ref": "aria-4", "role": "combobox", "name": "Poziom doświadczenia", "context": "Formularz"},
            {"ref": "aria-5", "role": "button", "name": "Załącz dokument CV", "context": "Formularz"},
            {"ref": "aria-6", "role": "button", "name": "Dodaj certyfikaty (wiele plików)", "context": "Formularz"},
            {
                "ref": "aria-7",
                "role": "checkbox",
                "name": "Zgadzam się na przetwarzanie danych osobowych (RODO)",
                "context": "Formularz",
            },
        ]
        payload = {
            "first_name": "Antoni",
            "last_name": "Zieliński",
            "email": "antoni.zielinski@example.com",
            "phone": "+48 501 234 567",
            "experience": "Senior",
            "cv": "/tmp/cv.pdf",
            "certificates": ["/tmp/c1.png", "/tmp/c2.png"],
            "rodo": True,
        }
        actions = reflex.bind_payload_to_elements(payload, candidates)
        assert len(actions) == 8
        by_key = {a["payload_key"]: a for a in actions}
        assert by_key["first_name"]["name"] == "Imię"
        assert by_key["last_name"]["name"] == "Nazwisko"
        assert by_key["email"]["name"] == "Adres e-mail"
        assert by_key["phone"]["name"] == "Numer telefonu"
        assert by_key["experience"]["name"] == "Poziom doświadczenia"
        assert by_key["experience"]["action"] == "select"
        assert by_key["cv"]["name"] == "Załącz dokument CV"
        assert by_key["cv"]["action"] == "upload_file"
        assert by_key["certificates"]["name"] == "Dodaj certyfikaty (wiele plików)"
        assert by_key["rodo"]["action"] == "check"


class TestBrowserEngineUpload:
    @pytest.mark.anyio
    async def test_upload_file_direct_input(self):
        """Tests uploading files directly when the locator is an input[type='file']."""
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_locator = AsyncMock()

        # Mock element is input:file
        mock_locator.evaluate = AsyncMock(return_value="input:file")
        mock_locator.set_input_files = AsyncMock()
        engine._resolve_locator = AsyncMock(return_value=mock_locator)  # type: ignore[method-assign]

        success = await engine.upload_file(mock_page, role="button", name="Wybierz plik", file_paths="/tmp/test.pdf")
        assert success is True
        mock_locator.set_input_files.assert_awaited_once()

    @pytest.mark.anyio
    async def test_upload_file_chooser_event(self):
        """Tests modern file upload triggered by expect_file_chooser."""
        engine = BrowserEngine()
        mock_page = MagicMock()

        # Locator is a custom dropzone/button (not input:file)
        mock_locator = AsyncMock()
        mock_locator.evaluate = AsyncMock(return_value="button:")
        mock_locator.click = AsyncMock()
        engine._resolve_locator = AsyncMock(return_value=mock_locator)  # type: ignore[method-assign]

        # Mock expect_file_chooser context manager
        mock_fc = AsyncMock()
        mock_fc.set_files = AsyncMock()
        mock_fc_info = MagicMock()
        mock_fc_info.value = mock_fc

        mock_context_manager = MagicMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_fc_info)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_page.expect_file_chooser.return_value = mock_context_manager

        success = await engine.upload_file(
            mock_page, role="button", name="Załącz dokument", file_paths=["/tmp/a.png", "/tmp/b.png"]
        )
        assert success is True
        mock_fc.set_files.assert_awaited_once()


class TestSystemCookies:
    def test_extract_firefox_cookies_from_mock_db(self):
        """Tests reading cookies from a real SQLite database file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "cookies.sqlite")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE moz_cookies (
                    id INTEGER PRIMARY KEY,
                    host TEXT,
                    name TEXT,
                    value TEXT,
                    path TEXT,
                    expiry INTEGER,
                    isSecure INTEGER,
                    isHttpOnly INTEGER,
                    sameSite INTEGER
                )
                """
            )
            cur.execute(
                """
                INSERT INTO moz_cookies (host, name, value, path, expiry, isSecure, isHttpOnly, sameSite)
                VALUES ('.pracuj.pl', 'cf_clearance', 'abc123secret', '/', 1802137937547, 1, 1, 1)
                """
            )
            cur.execute(
                """
                INSERT INTO moz_cookies (host, name, value, path, expiry, isSecure, isHttpOnly, sameSite)
                VALUES ('.other.com', 'session', 'xyz', '/', 1802137937, 0, 0, 0)
                """
            )
            conn.commit()
            conn.close()

            cookies = extract_firefox_cookies(domain="pracuj.pl", profile_path=tmp_dir)
            assert len(cookies) == 1
            assert cookies[0]["name"] == "cf_clearance"
            assert cookies[0]["value"] == "abc123secret"
            assert cookies[0]["domain"] == ".pracuj.pl"
            assert cookies[0]["httpOnly"] is True
            assert cookies[0]["secure"] is True
            assert cookies[0]["sameSite"] == "Lax"
            # Expiry should be converted to seconds
            assert cookies[0]["expires"] < 1e11

    def test_load_cookies_from_json(self):
        """Tests loading exported cookies from a JSON file."""
        sample_cookies = [
            {"name": "token", "value": "12345", "domain": ".otomoto.pl", "path": "/"},
            {"name": "other", "value": "99999", "domain": ".allegro.pl", "path": "/"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(sample_cookies, f)
            tmp_json = f.name

        try:
            cookies = load_cookies_from_file(tmp_json, domain="otomoto.pl")
            assert len(cookies) == 1
            assert cookies[0]["name"] == "token"
            assert cookies[0]["domain"] == ".otomoto.pl"
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)

    @pytest.mark.anyio
    async def test_import_system_cookies_into_context(self):
        """Tests injecting cookies into a Playwright BrowserContext."""
        mock_context = MagicMock(spec=["add_cookies"])
        mock_context.add_cookies = AsyncMock()

        sample_cookies = [{"name": "cf_clearance", "value": "val", "domain": ".pracuj.pl", "path": "/"}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(sample_cookies, f)
            tmp_json = f.name

        try:
            count = await import_system_cookies(mock_context, domain="pracuj.pl", browser_type=tmp_json)
            assert count == 1
            mock_context.add_cookies.assert_awaited_once()
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)


class TestMCPPayloadIntegration:
    @pytest.mark.anyio
    async def test_cognitive_browse_with_payload(self):
        """Tests that cognitive_browse binds payload and fills form elements."""
        mock_reflex = MagicMock()
        mock_reflex.bind_payload_to_elements.return_value = [
            {
                "payload_key": "email",
                "action": "fill",
                "role": "textbox",
                "name": "Adres e-mail",
                "nth": 0,
                "value": "user@example.com",
                "file_paths": None,
                "confidence": 0.95,
            },
            {
                "payload_key": "cv",
                "action": "upload_file",
                "role": "button",
                "name": "Załącz CV",
                "nth": 0,
                "value": None,
                "file_paths": ["/tmp/cv.pdf"],
                "confidence": 0.94,
            },
        ]

        mock_engine = MagicMock()
        mock_page = AsyncMock()
        mock_page.url = "https://example.com/apply"
        mock_page.title.return_value = "Apply Now"
        mock_page.aria_snapshot.return_value = '- textbox "Adres e-mail"\n- button "Załącz CV"'
        mock_engine.get_page = AsyncMock(return_value=mock_page)
        mock_engine.fill_element = AsyncMock(return_value=True)
        mock_engine.upload_file = AsyncMock(return_value=True)
        mock_engine.auto_dismiss_overlays = AsyncMock(return_value=False)

        with patch("cognitive_browser.mcp_server.get_services", return_value=(mock_reflex, mock_engine)):
            res = await cognitive_browse(
                url="https://example.com/apply",
                payload={"email": "user@example.com", "cv": "/tmp/cv.pdf"},
            )

            assert res["status"] == "success"
            assert res["steps_executed"] == 2
            assert len(res["steps_log"]) == 2
            assert res["steps_log"][0]["source"] == "payload_binding"
            assert res["steps_log"][1]["source"] == "payload_binding"
            mock_engine.fill_element.assert_awaited_once()
            mock_engine.upload_file.assert_awaited_once()
