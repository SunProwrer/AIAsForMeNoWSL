"""Тесты для конвейера обработки PDF src/ocr/pipeline.py."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

import pymupdf as fitz
from PIL import Image

from src.ocr.pipeline import process_pdf


def _make_text_pdf(tmp_path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This page has enough text for the text layer check.")
    doc.save(tmp_path)
    doc.close()


def _make_scan_pdf(tmp_path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    pil_img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    page.insert_image(fitz.Rect(0, 0, 100, 100), stream=buf.read())
    doc.save(tmp_path)
    doc.close()


def _make_mixed_pdf(tmp_path: str) -> None:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72), "This page has a full text layer with enough characters."
    )
    page2 = doc.new_page()
    pil_img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    page2.insert_image(fitz.Rect(0, 0, 100, 100), stream=buf.read())
    doc.save(tmp_path)
    doc.close()


class TestProcessPdf(unittest.TestCase):
    def test_text_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _make_text_pdf(tmp_path)
            result = process_pdf(tmp_path)
            self.assertEqual(result.file_name, Path(tmp_path).name)
            self.assertEqual(len(result.pages), 1)
            self.assertEqual(result.pages[0].page_number, 1)
            self.assertEqual(result.pages[0].source, "text_layer")
            self.assertIsNone(result.pages[0].confidence)
            self.assertIn("This page", result.pages[0].text)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_scan_pdf_does_not_crash(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _make_scan_pdf(tmp_path)
            result = process_pdf(tmp_path)
            self.assertEqual(len(result.pages), 1)
            self.assertEqual(result.pages[0].page_number, 1)
            # OCR может вернуть пустой текст с нулевой уверенностью
            # (если Tesseract не установлен) или реальный результат
            self.assertEqual(result.pages[0].source, "ocr")
            self.assertIsInstance(result.pages[0].text, str)
            self.assertIsInstance(result.pages[0].confidence, float)
            self.assertEqual(result.pages[0].char_count, len(result.pages[0].text))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_ocr_error_on_page_is_handled(self):
        """Ошибка OCR на одной странице не ломает весь документ."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _make_scan_pdf(tmp_path)
            with patch(
                "src.ocr.pipeline.ocr_image",
                side_effect=RuntimeError("OCR failed"),
            ):
                result = process_pdf(tmp_path)
            self.assertEqual(len(result.pages), 1)
            self.assertEqual(result.pages[0].page_number, 1)
            self.assertEqual(result.pages[0].source, "ocr")
            self.assertEqual(result.pages[0].text, "")
            self.assertIsNone(result.pages[0].confidence)
            self.assertEqual(result.pages[0].char_count, 0)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_mixed_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _make_mixed_pdf(tmp_path)
            result = process_pdf(tmp_path)
            self.assertEqual(len(result.pages), 2)
            self.assertEqual(result.pages[0].page_number, 1)
            self.assertEqual(result.pages[0].source, "text_layer")
            self.assertIn("full text layer", result.pages[0].text)
            self.assertEqual(result.pages[1].page_number, 2)
            self.assertEqual(result.pages[1].source, "ocr")
            self.assertEqual(result.pages[1].text, "")
            self.assertIsInstance(result.pages[1].confidence, float)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            process_pdf("/nonexistent/path/file.pdf")

    def test_not_a_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not a pdf")
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                process_pdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
