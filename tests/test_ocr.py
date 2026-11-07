"""Простые тесты для модуля src/ocr."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf as fitz
from PIL import Image
import pytesseract

from src.ocr.detect import analyze_pdf, has_text_layer
from src.ocr.extract import extract_text_page, extract_text_pdf
from src.ocr.models import DocumentResult, PageResult
from src.ocr.ocr_engine import ocr_image, OcrEngineError
from src.ocr.render import render_page, render_pages, render_pdf_pages


class TestPageResult(unittest.TestCase):
    def test_char_count_auto(self):
        page = PageResult(page_number=1, text="Привет", source="text_layer")
        self.assertEqual(page.char_count, 6)

    def test_char_count_explicit(self):
        page = PageResult(
            page_number=1, text="Привет", source="text_layer", char_count=10
        )
        self.assertEqual(page.char_count, 10)

    def test_confidence_none_for_text_layer(self):
        page = PageResult(page_number=1, text="текст", source="text_layer")
        self.assertIsNone(page.confidence)


class TestDocumentResult(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = DocumentResult(
            file_name="test.pdf",
            pages=[
                PageResult(page_number=1, text="Первая страница", source="text_layer", char_count=17),
                PageResult(page_number=2, text="Вторая страница", source="ocr", confidence=95.5, char_count=17),
            ],
        )

    def test_full_text(self):
        expected = "Первая страница\n\nВторая страница"
        self.assertEqual(self.doc.full_text(), expected)

    def test_to_dict(self):
        data = self.doc.to_dict()
        self.assertEqual(data["file_name"], "test.pdf")
        self.assertEqual(len(data["pages"]), 2)
        self.assertEqual(data["pages"][0]["page_number"], 1)
        self.assertEqual(data["pages"][0]["source"], "text_layer")
        self.assertEqual(data["pages"][1]["source"], "ocr")
        self.assertEqual(data["pages"][1]["confidence"], 95.5)

    def test_to_json(self):
        result = self.doc.to_json()
        parsed = json.loads(result)
        self.assertEqual(parsed["file_name"], "test.pdf")
        self.assertEqual(len(parsed["pages"]), 2)


class TestHasTextLayer(unittest.TestCase):
    def test_page_with_enough_text(self):
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.get_text.return_value = "Это страница с достаточным количеством текста для проверки."
        self.assertTrue(has_text_layer(mock_page, min_chars=30))

    def test_page_with_little_text(self):
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.get_text.return_value = "Мало текста."
        self.assertFalse(has_text_layer(mock_page, min_chars=30))

    def test_page_with_no_text(self):
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.get_text.return_value = ""
        self.assertFalse(has_text_layer(mock_page, min_chars=30))

    def test_custom_threshold(self):
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.get_text.return_value = "Привет мир"
        self.assertTrue(has_text_layer(mock_page, min_chars=5))
        self.assertFalse(has_text_layer(mock_page, min_chars=20))


class TestAnalyzePdf(unittest.TestCase):
    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            analyze_pdf("/nonexistent/path/file.pdf")

    def test_not_a_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                analyze_pdf(tmpdir)

    def test_not_a_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not a pdf")
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                analyze_pdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pdf_with_text_page(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This page has enough text for the text layer check.")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            doc.save(tmp_path)
            doc.close()
            results = analyze_pdf(tmp_path)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_mixed_pdf(self):
        doc = fitz.open()
        # Страница с текстом
        page1 = doc.new_page()
        page1.insert_text((72, 72), "This page has a full text layer with enough characters to pass the threshold.")
        # Страница без текста
        page2 = doc.new_page()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            doc.save(tmp_path)
            doc.close()
            results = analyze_pdf(tmp_path)
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0])
            self.assertFalse(results[1])
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestExtractTextPage(unittest.TestCase):
    def test_clean_text_page(self):
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.get_text.return_value = "Это текст со  \n  лишними   пробелами и переносом-\nслов."
        result = extract_text_page(mock_page, page_number=1)
        self.assertEqual(result.source, "text_layer")
        self.assertIsNone(result.confidence)
        self.assertEqual(result.page_number, 1)
        self.assertEqual(result.char_count, len(result.text))
        self.assertNotIn("  ", result.text)
        self.assertNotIn("\n", result.text)
        # Перенос со дефисом соединён: "переносом-\nслов" → "переносомслов"
        self.assertIn("переносомслов", result.text)

    def test_empty_page(self):
        mock_page = MagicMock(spec=fitz.Page)
        mock_page.get_text.return_value = ""
        result = extract_text_page(mock_page, page_number=3)
        self.assertEqual(result.text, "")
        self.assertEqual(result.char_count, 0)
        self.assertEqual(result.page_number, 3)


class TestExtractTextPdf(unittest.TestCase):
    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            extract_text_pdf("/nonexistent/path/file.pdf")

    def test_not_a_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                extract_text_pdf(tmpdir)

    def test_not_a_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not a pdf")
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                extract_text_pdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pdf_text_pages(self):
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((72, 72), "First page text with enough characters here.")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Second page also has sufficient text content.")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            doc.save(tmp_path)
            doc.close()
            result = extract_text_pdf(tmp_path)
            self.assertEqual(result.file_name, Path(tmp_path).name)
            self.assertEqual(len(result.pages), 2)
            self.assertEqual(result.pages[0].page_number, 1)
            self.assertEqual(result.pages[1].page_number, 2)
            self.assertEqual(result.pages[0].source, "text_layer")
            self.assertEqual(result.pages[1].source, "text_layer")
            self.assertIn("First page", result.pages[0].text)
            self.assertIn("Second page", result.pages[1].text)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_full_text_joins_pages(self):
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((72, 72), "First page content here.")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Second page content here.")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            doc.save(tmp_path)
            doc.close()
            result = extract_text_pdf(tmp_path)
            full = result.full_text()
            self.assertIn("First page content here.", full)
            self.assertIn("Second page content here.", full)
            self.assertIn("\n\n", full)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestRenderPage(unittest.TestCase):
    def test_render_default_dpi(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Render test page.")
        image = render_page(page)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")
        doc.close()

    def test_render_custom_dpi(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Custom DPI render test.")
        image_72 = render_page(page, dpi=72)
        image_144 = render_page(page, dpi=144)
        self.assertEqual(image_144.width, image_72.width * 2)
        self.assertEqual(image_144.height, image_72.height * 2)
        doc.close()

    def test_render_max_side_limits(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Max side limit render test.")
        image = render_page(page, dpi=300, max_side=100)
        self.assertLessEqual(max(image.width, image.height), 100)
        doc.close()

    def test_render_page_number_tracking(self):
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((72, 72), "Page one.")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Page two.")
        results = list(render_pages(doc, dpi=72))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[1][0], 2)
        self.assertIsInstance(results[0][1], Image.Image)
        self.assertIsInstance(results[1][1], Image.Image)
        doc.close()

    def test_render_pdf_pages_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            list(render_pdf_pages("/nonexistent/path/file.pdf"))

    def test_render_pdf_pages_not_a_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not a pdf")
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                list(render_pdf_pages(tmp_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_render_pdf_pages_yields_all(self):
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1} content.")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            doc.save(tmp_path)
            doc.close()
            pages = list(render_pdf_pages(tmp_path, dpi=72))
            self.assertEqual(len(pages), 3)
            self.assertEqual([p[0] for p in pages], [1, 2, 3])
            for _, image in pages:
                self.assertIsInstance(image, Image.Image)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestOcrEngine(unittest.TestCase):
    def setUp(self) -> None:
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            self.skipTest("Tesseract не установлен в системе")

    def test_ocr_blank_image(self):
        image = Image.new("RGB", (200, 100), color="white")
        text, confidence = ocr_image(image)
        self.assertIsInstance(text, str)
        self.assertIsInstance(confidence, float)
        self.assertEqual(text, "")
        self.assertEqual(confidence, 0.0)

    def test_ocr_engine_error_class(self):
        self.assertTrue(issubclass(OcrEngineError, RuntimeError))

    def test_ocr_min_conf_filters_low_confidence_words(self):
        image = Image.new("RGB", (100, 50), color="white")
        fake_data = {
            "text": ["хорошее", "плохое", "отличное", ""],
            "conf": [90.0, 10.0, 80.0, -1],
        }
        with patch.object(
            pytesseract, "image_to_data", return_value=fake_data
        ):
            text, confidence = ocr_image(image, min_conf=30.0)
        self.assertIn("хорошее", text)
        self.assertIn("отличное", text)
        self.assertNotIn("плохое", text)
        self.assertEqual(confidence, (90.0 + 80.0) / 2)

    def test_ocr_min_conf_all_filtered(self):
        image = Image.new("RGB", (100, 50), color="white")
        fake_data = {
            "text": ["плохое", "очень плохое"],
            "conf": [10.0, 5.0],
        }
        with patch.object(
            pytesseract, "image_to_data", return_value=fake_data
        ):
            text, confidence = ocr_image(image, min_conf=30.0)
        self.assertEqual(text, "")
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

