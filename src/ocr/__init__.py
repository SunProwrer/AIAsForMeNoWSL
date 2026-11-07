"""Модуль извлечения текста из PDF (текстовый слой + OCR для сканов)."""

from .detect import analyze_pdf, has_text_layer
from .extract import extract_text_page, extract_text_pdf
from .models import DocumentResult, PageResult
from .ocr_engine import ocr_image, OcrEngineError
from .pipeline import process_pdf
from .preprocess import preprocess
from .render import render_page, render_pages, render_pdf_pages

__all__ = [
    "analyze_pdf",
    "has_text_layer",
    "extract_text_page",
    "extract_text_pdf",
    "process_pdf",
    "ocr_image",
    "OcrEngineError",
    "preprocess",
    "render_page",
    "render_pages",
    "render_pdf_pages",
    "DocumentResult",
    "PageResult",
]
