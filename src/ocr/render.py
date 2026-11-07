"""Рендеринг страниц PDF в изображения через PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
from PIL import Image


def render_page(page: fitz.Page, dpi: int = 300, max_side: int | None = None) -> Image.Image:
    """Отрендерить страницу PDF в изображение PIL.

    Страница рендерится по одной, без удержания всего документа в памяти.

    Args:
        page: Страница PyMuPDF.
        dpi: Разрешение рендеринга в точках на дюйм (по умолчанию 300).
        max_side: Максимальный размер большей стороны изображения в пикселях.
            Если после рендеринга ширина или высота превышает max_side, DPI
            снижается пропорционально, чтобы изображение поместилось в лимит.
            Это экономит память на больших страницах.

    Returns:
        Изображение страницы в формате PIL.Image.Image (RGB).
    """
    zoom = dpi / 72.0

    if max_side is not None:
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        # Размер страницы в пикселях при заданном DPI
        render_width = page_width * zoom
        render_height = page_height * zoom
        needed_zoom = zoom
        if render_width > max_side:
            needed_zoom = min(needed_zoom, max_side / page_width)
        if render_height > max_side:
            needed_zoom = min(needed_zoom, max_side / page_height)
        zoom = needed_zoom

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    try:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        pix = None  # Освобождаем пиксельную карту сразу после создания изображения

    return img


def render_pages(
    doc: fitz.Document, dpi: int = 300, max_side: int | None = None
):
    """Генератор: рендерит страницы документа по одной.

    Каждая страница рендерится и возвращается как PIL.Image.Image, после чего
    пиксельная карта освобождается. Весь документ не удерживается в памяти.

    Args:
        doc: Открытый документ PyMuPDF.
        dpi: Разрешение рендеринга в точках на дюйм (по умолчанию 300).
        max_side: Максимальный размер большей стороны изображения в пикселях.
            Снижает DPI для больших страниц, чтобы экономить память.

    Yields:
        Кортеж (page_number, image), где page_number — номер страницы с 1,
        image — PIL.Image.Image.
    """
    for page_index in range(len(doc)):
        page = doc[page_index]
        image = render_page(page, dpi=dpi, max_side=max_side)
        yield page_index + 1, image


def render_pdf_pages(
    pdf_path: str | Path, dpi: int = 300, max_side: int | None = None
):
    """Генератор: открывает PDF и рендерит его страницы по одной.

    Документ открывается, каждая страница рендерится и возвращается, затем
    документ закрывается. Весь документ не удерживается в памяти.

    Args:
        pdf_path: Путь к PDF-файлу.
        dpi: Разрешение рендеринга в точках на дюйм (по умолчанию 300).
        max_side: Максимальный размер большей стороны изображения в пикселях.

    Yields:
        Кортеж (page_number, image), где page_number — номер страницы с 1,
        image — PIL.Image.Image.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если путь не к файлу, не PDF или файл повреждён.
    """
    path = Path(pdf_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Нужно указать путь к файлу, а не к папке: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Допускаются только файлы с расширением .pdf.")

    try:
        doc = fitz.open(path)
    except Exception as error:
        raise ValueError(f"Повреждённый PDF-файл: {path}") from error

    try:
        yield from render_pages(doc, dpi=dpi, max_side=max_side)
    finally:
        doc.close()
