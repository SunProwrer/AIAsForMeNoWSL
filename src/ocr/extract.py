"""Извлечение текста из PDF с текстовым слоем (без OCR)."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz

from .models import DocumentResult, PageResult


def _clean_text(text: str) -> str:
    """Очистить извлечённый текст.

    Убирает переносы слов с дефисом через строку, заменяет остальные
    переносы строк на пробелы и схлопывает множественные пробелы.

    Args:
        text: Исходный текст страницы.

    Returns:
        Очищенный текст.
    """
    # Соединяем слова, разделённые переносом с дефисом: "при-\nвет" → "привет"
    text = re.sub(r"-\n", "", text)
    # Заменяем оставшиеся переносы строк на пробелы
    text = text.replace("\n", " ")
    # Схлопываем множественные пробелы и табуляции в один пробел
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_text_page(page: fitz.Page, page_number: int = 1) -> PageResult:
    """Извлечь текст с одной страницы PDF с текстовым слоем.

    Сохраняет порядок чтения страницы. Убирает лишние переносы
    внутри слов и множественные пробелы.

    Args:
        page: Страница PyMuPDF.
        page_number: Номер страницы (нумерация с 1).

    Returns:
        PageResult с текстом страницы, source="text_layer", confidence=None.
    """
    raw_text = page.get_text() or ""
    cleaned = _clean_text(raw_text)
    return PageResult(
        page_number=page_number,
        text=cleaned,
        source="text_layer",
        confidence=None,
        char_count=len(cleaned),
    )


def extract_text_pdf(path: str | Path) -> DocumentResult:
    """Извлечь текст из всего PDF с текстовым слоем.

    Страницы обрабатываются по порядку (нумерация с 1). Страницы без
    текста включаются с пустой строкой. OCR не используется.

    Args:
        path: Путь к PDF-файлу.

    Returns:
        DocumentResult с результатами по всем страницам.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если путь не к файлу, не PDF или файл повреждён.
    """
    pdf_path = Path(path).expanduser()

    if not pdf_path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Нужно указать путь к файлу, а не к папке: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Допускаются только файлы с расширением .pdf.")

    try:
        doc = fitz.open(pdf_path)
    except Exception as error:
        raise ValueError(f"Повреждённый PDF-файл: {pdf_path}") from error

    pages: list[PageResult] = []
    for page_index in range(len(doc)):
        page_number = page_index + 1
        page = doc[page_index]
        pages.append(extract_text_page(page, page_number=page_number))

    doc.close()

    return DocumentResult(file_name=pdf_path.name, pages=pages)
