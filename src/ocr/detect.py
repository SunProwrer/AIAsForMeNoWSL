"""Определение типа страниц PDF: текстовый слой или скан."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz


def has_text_layer(page: fitz.Page, min_chars: int = 30) -> bool:
    """Проверяет, есть ли на странице текстовый слой.

    Страница считается имеющей текстовый слой, если на ней достаточно
    алфавитно-цифровых символов.

    Args:
        page: Страница PyMuPDF.
        min_chars: Минимальное количество алфавитно-цифровых символов
            для признания текстового слоя присутствующим.

    Returns:
        True, если текста на странице не меньше min_chars, иначе False.
    """
    text = page.get_text() or ""
    alnum_count = sum(ch.isalnum() for ch in text)
    return alnum_count >= min_chars


def _page_has_images(page: fitz.Page) -> bool:
    """Проверяет, есть ли на странице встроенные изображения.

    Args:
        page: Страница PyMuPDF.

    Returns:
        True, если на странице есть изображения, иначе False.
    """
    return len(page.get_images(full=True)) > 0


def analyze_pdf(path: str | Path) -> list[bool]:
    """Анализирует каждую страницу PDF на наличие текстового слоя.

    Страница считается сканом (False в результате), если текста меньше
    порога min_chars, но на ней есть изображения. Иначе — True.

    Args:
        path: Путь к PDF-файлу.

    Returns:
        Список булевых значений по страницам (нумерация с 0).
        True — страница имеет текстовый слой, False — скан.

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

    results: list[bool] = []
    for page in doc:
        if has_text_layer(page, min_chars=30):
            results.append(True)
        elif _page_has_images(page):
            results.append(False)  # скан: мало текста, есть изображения
        else:
            results.append(False)  # пустая страница без текста и изображений

    doc.close()
    return results
